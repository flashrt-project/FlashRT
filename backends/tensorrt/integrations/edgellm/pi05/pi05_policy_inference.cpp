/*
 * pi0.5 policy inference on a FlashRT TensorRT engine, as a TensorRT Edge-LLM
 * example (overlay: experimental_models/pi05).
 *
 * The engine comes from FlashRT backends/tensorrt/tools/build_pi05_engine.sh: inputs
 * images fp16 [views, 224, 224, 3], lang_tokens int32 [n], noise fp16 [10, 32];
 * output actions fp16 [10, 32] (raw model space). It needs the FlashRT plugin
 * library, which is loaded into the TensorRT plugin registry before the engine
 * is deserialized.
 *
 *   pi05_policy_inference --engine pi05.engine --plugin libflashrt_trt_pi05.so \
 *       --tokenizer <dir with tokenizer.json> --images base.png,wrist.png \
 *       --prompt "put the bowl on the plate" [--noise noise.safetensors] \
 *       [--norm_stats norm_stats.json --action_dim 7] [--iters 100] \
 *       [--pixel_norm openpi|flashrt] [--no_cuda_graph] [--output actions.json]
 */
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm;
namespace fs = std::filesystem;

namespace
{

constexpr int64_t kImageSize = 224;
constexpr int64_t kTokensPerView = 256;

struct Args
{
    std::string engine, plugin, tokenizerDir, prompt, noisePath, normStats, output, pixelNorm{"openpi"};
    std::vector<std::string> images;
    int32_t actionDim{7};
    int32_t iters{0};
    uint32_t seed{0};
    bool cudaGraph{true};
};

std::vector<std::string> splitComma(std::string const& s)
{
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size())
    {
        size_t const comma = s.find(',', start);
        out.push_back(s.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    return out;
}

Args parseArgs(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string const k = argv[i];
        auto next = [&]() -> std::string {
            ELLM_CHECK(i + 1 < argc, "missing value for " + k);
            return argv[++i];
        };
        if (k == "--engine") a.engine = next();
        else if (k == "--plugin") a.plugin = next();
        else if (k == "--tokenizer") a.tokenizerDir = next();
        else if (k == "--images") a.images = splitComma(next());
        else if (k == "--prompt") a.prompt = next();
        else if (k == "--noise") a.noisePath = next();
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (k == "--norm_stats") a.normStats = next();
        else if (k == "--action_dim") a.actionDim = std::stoi(next());
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--pixel_norm") a.pixelNorm = next();
        else if (k == "--no_cuda_graph") a.cudaGraph = false;
        else if (k == "--output") a.output = next();
        else ELLM_CHECK(false, "unknown argument " + k);
    }
    ELLM_CHECK(!a.engine.empty() && !a.plugin.empty() && !a.tokenizerDir.empty() && !a.images.empty(),
        "--engine, --plugin, --tokenizer and --images are required");
    ELLM_CHECK(a.pixelNorm == "openpi" || a.pixelNorm == "flashrt", "--pixel_norm is openpi or flashrt");
    return a;
}

// openpi tokenization for pi0.5 without state tokens: BOS + cleaned text + "\n".
std::vector<int32_t> tokenizePrompt(tokenizer::Tokenizer const& tok, std::string prompt)
{
    std::replace(prompt.begin(), prompt.end(), '_', ' ');
    std::replace(prompt.begin(), prompt.end(), '\n', ' ');
    size_t const b = prompt.find_first_not_of(" \t\r");
    size_t const e = prompt.find_last_not_of(" \t\r");
    prompt = b == std::string::npos ? std::string() : prompt.substr(b, e - b + 1);
    std::vector<int32_t> ids{2};  // <bos>
    for (auto id : tok.encode(prompt, false, false)) ids.push_back(static_cast<int32_t>(id));
    for (auto id : tok.encode("\n", false, false)) ids.push_back(static_cast<int32_t>(id));
    return ids;
}

struct DeviceBuffer
{
    void* ptr{nullptr};
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(bytes, 1))); }
    ~DeviceBuffer() { cudaFree(ptr); }
};

} // namespace

int main(int argc, char** argv)
{
    Args const args = parseArgs(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));

    // Plugin library first: the engine cannot deserialize without its creators.
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ELLM_CHECK(runtime->getPluginRegistry().loadLibrary(args.plugin.c_str()) != nullptr,
        "failed to load plugin library " + args.plugin);
    auto engine = deserializeCudaEngineFromFile(*runtime, args.engine);
    ELLM_CHECK(engine != nullptr, "failed to deserialize " + args.engine);
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    int64_t const views = engine->getTensorShape("images").d[0];
    nvinfer1::Dims const noiseDims = engine->getTensorShape("noise");
    int64_t const horizon = noiseDims.d[0], modelDim = noiseDims.d[1];
    ELLM_CHECK(static_cast<int64_t>(args.images.size()) == views,
        "engine takes " + std::to_string(views) + " images, got " + std::to_string(args.images.size()));

    // Images: HWC uint8 -> [-1, 1] -> fp16. openpi normalizes x / 255 * 2 - 1 in
    // float32; FlashRT's uint8 table is x / 127.5 - 1.
    std::vector<__fp16> pixels(static_cast<size_t>(views * kImageSize * kImageSize * 3));
    for (int64_t v = 0; v < views; ++v)
    {
        rt::imageUtils::ImageData const img = rt::imageUtils::loadImageFromFile(args.images[v]);
        rt::imageUtils::ImageData resized(rt::Tensor(
            {1, kImageSize, kImageSize, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "pi05::resized"));
        rt::imageUtils::ImageData const& frame = (img.width == kImageSize && img.height == kImageSize)
            ? img
            : rt::imageUtils::resizeImage(img, resized, kImageSize, kImageSize, rt::imageUtils::InterpolationMode::kLINEAR);
        unsigned char const* src = frame.data();
        __fp16* dst = pixels.data() + v * kImageSize * kImageSize * 3;
        for (int64_t i = 0; i < kImageSize * kImageSize * 3; ++i)
        {
            float const x = args.pixelNorm == "openpi" ? static_cast<float>(src[i]) / 255.0F * 2.0F - 1.0F
                                                       : static_cast<float>(src[i]) / 127.5F - 1.0F;
            dst[i] = static_cast<__fp16>(x);
        }
    }

    tokenizer::Tokenizer tok;
    ELLM_CHECK(tok.loadFromHF(args.tokenizerDir, /*requireChatTemplate=*/false),
        "failed to load tokenizer.json from " + args.tokenizerDir);
    std::vector<int32_t> tokens = tokenizePrompt(tok, args.prompt);
    if ((views * kTokensPerView + static_cast<int64_t>(tokens.size())) % 2 != 0)
    {
        tokens.push_back(tokens.back());  // FlashRT keeps the prefix length even
    }

    std::vector<__fp16> noise(static_cast<size_t>(horizon * modelDim));
    if (!args.noisePath.empty())
    {
        // A small host-side safetensors read: one fp16 tensor named noise_in or noise.
        std::ifstream f(args.noisePath, std::ios::binary);
        ELLM_CHECK(f.good(), "cannot open " + args.noisePath);
        uint64_t headerLen = 0;
        f.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));
        std::string header(headerLen, '\0');
        f.read(header.data(), static_cast<std::streamsize>(headerLen));
        nlohmann::json const h = nlohmann::json::parse(header);
        std::string const key = h.contains("noise_in") ? "noise_in" : "noise";
        ELLM_CHECK(h.contains(key), "no noise_in tensor in " + args.noisePath);
        std::vector<uint64_t> const off = h[key]["data_offsets"];
        ELLM_CHECK(h[key]["dtype"] == "F16" && off[1] - off[0] == noise.size() * sizeof(__fp16),
            "noise must be fp16 [" + std::to_string(horizon) + ", " + std::to_string(modelDim) + "]");
        f.seekg(static_cast<std::streamoff>(sizeof(headerLen) + headerLen + off[0]));
        f.read(reinterpret_cast<char*>(noise.data()), static_cast<std::streamsize>(noise.size() * sizeof(__fp16)));
    }
    else
    {
        std::mt19937 gen(args.seed);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        for (auto& n : noise) n = static_cast<__fp16>(dist(gen));
    }

    // Persistent device I/O: CUDA graph replay reuses the captured addresses.
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    DeviceBuffer imagesDev(pixels.size() * 2), tokensDev(tokens.size() * 4), noiseDev(noise.size() * 2),
        actionsDev(noise.size() * 2);
    CUDA_CHECK(cudaMemcpy(imagesDev.ptr, pixels.data(), pixels.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tokensDev.ptr, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice));
    ELLM_CHECK(context->setInputShape("lang_tokens", nvinfer1::Dims{1, {static_cast<int64_t>(tokens.size())}}),
        "lang_tokens length " + std::to_string(tokens.size()) + " outside the engine profile");
    context->setTensorAddress("images", imagesDev.ptr);
    context->setTensorAddress("lang_tokens", tokensDev.ptr);
    context->setTensorAddress("noise", noiseDev.ptr);
    context->setTensorAddress("actions", actionsDev.ptr);

    auto uploadNoise = [&]() {
        CUDA_CHECK(cudaMemcpy(noiseDev.ptr, noise.data(), noise.size() * 2, cudaMemcpyHostToDevice));
    };
    uploadNoise();
    ELLM_CHECK(context->enqueueV3(stream), "enqueueV3 failed");  // warmup outside capture
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraphExec_t graphExec = nullptr;
    if (args.cudaGraph)
    {
        auto captured = captureTRTCudaGraph(context.get(), stream);
        if (captured.has_value())
        {
            graphExec = captured->second;
        }
        else
        {
            LOG_WARNING("CUDA graph capture failed; running enqueueV3");
        }
    }
    auto run = [&]() {
        if (graphExec != nullptr)
        {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        }
        else
        {
            ELLM_CHECK(context->enqueueV3(stream), "enqueueV3 failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    uploadNoise();
    run();
    std::vector<__fp16> raw(noise.size());
    CUDA_CHECK(cudaMemcpy(raw.data(), actionsDev.ptr, raw.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<double> ms;
    for (int32_t i = 0; i < args.iters; ++i)
    {
        auto const t0 = std::chrono::steady_clock::now();
        run();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }

    nlohmann::json out;
    out["tokens"] = tokens;
    std::vector<std::vector<float>> rawRows(static_cast<size_t>(horizon), std::vector<float>(modelDim));
    for (int64_t r = 0; r < horizon; ++r)
        for (int64_t c = 0; c < modelDim; ++c) rawRows[r][c] = static_cast<float>(raw[r * modelDim + c]);
    out["raw_actions"] = rawRows;
    if (!args.normStats.empty())
    {
        // openpi pi0.5 quantile normalization of actions: x -> (x + 1) / 2 * (q99 - q01) + q01
        std::ifstream f(args.normStats);
        nlohmann::json const stats = nlohmann::json::parse(f)["norm_stats"]["actions"];
        std::vector<double> const q01 = stats["q01"], q99 = stats["q99"];
        std::vector<std::vector<double>> actions(static_cast<size_t>(horizon), std::vector<double>(args.actionDim));
        for (int64_t r = 0; r < horizon; ++r)
            for (int32_t c = 0; c < args.actionDim; ++c)
                actions[r][c] = (rawRows[r][c] + 1.0) / 2.0 * (q99[c] - q01[c] + 1e-6) + q01[c];
        out["actions"] = actions;
    }
    if (!ms.empty())
    {
        std::sort(ms.begin(), ms.end());
        out["latency_ms"] = {{"median", ms[ms.size() / 2]}, {"p90", ms[ms.size() * 9 / 10]}, {"iters", ms.size()},
            {"cuda_graph", graphExec != nullptr}};
    }
    if (args.output.empty())
    {
        std::cout << out.dump(1) << std::endl;
    }
    else
    {
        std::ofstream(args.output) << out.dump(1);
        std::cout << "wrote " << args.output << std::endl;
        if (out.contains("latency_ms")) std::cout << "latency " << out["latency_ms"].dump() << std::endl;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
