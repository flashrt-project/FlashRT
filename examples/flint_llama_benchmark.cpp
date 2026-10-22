// SPDX-License-Identifier: Apache-2.0
// Optional reference driver. Build against the pinned llama.cpp in Flint usage.
// Same token files, context capacity, batch-one greedy loop and wall-clock
// boundaries as flint_benchmark.py. This is not linked into FlashRT.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char ** argv) {
    try {
        if (argc < 6) throw std::runtime_error("usage: flint_llama_benchmark MODEL N_GEN WARMUP REPS TOKENS...");
        const int n_gen = std::stoi(argv[2]), warmup = std::stoi(argv[3]), reps = std::stoi(argv[4]);
        if (n_gen < 1 || warmup < 1 || reps < 1) throw std::runtime_error("counts must be positive");
        std::vector<std::vector<llama_token>> prompts;
        size_t max_prompt = 0;
        for (int i = 5; i < argc; ++i) {
            std::ifstream f(argv[i], std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("cannot open token file");
            const auto bytes = static_cast<std::streamoff>(f.tellg());
            if (bytes <= 0 || bytes % sizeof(llama_token)) throw std::runtime_error("invalid int32 token file");
            std::vector<llama_token> ids(static_cast<size_t>(bytes) / sizeof(llama_token));
            f.seekg(0); f.read(reinterpret_cast<char *>(ids.data()), bytes);
            if (!f) throw std::runtime_error("truncated token file");
            max_prompt = std::max(max_prompt, ids.size());
            prompts.push_back(std::move(ids));
        }
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(max_prompt + n_gen + 64);
        cp.n_batch = 2048; cp.n_ubatch = 2048;
        cp.n_seq_max = 1; cp.n_outputs_max = 1; cp.n_outputs_max_per_seq = 1;
        cp.swa_full = false;
        cp.n_threads = 8; cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        llama_sampler_seq_config sampler_config{0, sampler};
        cp.samplers = &sampler_config; cp.n_samplers = 1;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        const auto * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        auto batch = llama_batch_init(2048, 0, 1);
        auto pick = [&]() {
            const auto token = llama_get_sampled_token_ith(ctx, -1);
            if (token == LLAMA_TOKEN_NULL) throw std::runtime_error("backend greedy sampling did not produce a token");
            return token;
        };
        auto put = [&](int i, llama_token token, int pos, bool logits) {
            batch.token[i] = token; batch.pos[i] = pos; batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0; batch.logits[i] = logits;
        };
        std::cout << "{\"batch\":1,\"backend_sampling\":true,\"n_gen\":" << n_gen << ",\"warmup\":" << warmup
                  << ",\"reps\":" << reps << ",\"results\":[";
        bool first_row = true;
        for (const auto & ids : prompts) {
            for (auto token : ids) if (token < 0 || token >= n_vocab) throw std::runtime_error("token outside vocabulary");
            if (!first_row) std::cout << ',';
            first_row = false;
            std::cout << "{\"prompt_tokens\":" << ids.size() << ",\"samples\":[";
            bool first_sample = true;
            for (int rep = 0; rep < warmup + reps; ++rep) {
                llama_synchronize(ctx);
                const auto t0 = Clock::now();
                llama_memory_clear(llama_get_memory(ctx), false);
                for (size_t start = 0; start < ids.size(); start += 2048) {
                    batch.n_tokens = static_cast<int>(std::min<size_t>(2048, ids.size() - start));
                    for (int i = 0; i < batch.n_tokens; ++i)
                        put(i, ids[start + i], static_cast<int>(start) + i, start + i + 1 == ids.size());
                    if (llama_decode(ctx, batch)) throw std::runtime_error("prefill decode failed");
                }
                auto next = pick();
                const auto t1 = Clock::now();
                for (int i = 0; i < n_gen; ++i) {
                    batch.n_tokens = 1;
                    put(0, next, static_cast<int>(ids.size()) + i, true);
                    if (llama_decode(ctx, batch)) throw std::runtime_error("generation decode failed");
                    next = pick();
                }
                llama_synchronize(ctx);
                const auto t2 = Clock::now();
                if (rep >= warmup) {
                    if (!first_sample) std::cout << ',';
                    first_sample = false;
                    std::cout << "{\"prefill_wall_ms\":" << ms(t0, t1)
                              << ",\"decode_wall_ms_per_token\":" << ms(t1, t2) / n_gen << '}';
                }
            }
            std::cout << "]}" << std::flush;
        }
        std::cout << "]}\n";
        llama_batch_free(batch); llama_free(ctx); llama_sampler_free(sampler);
        llama_model_free(model); llama_backend_free();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
