#include <flashrt/apple/pipeline.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace flashrt::apple;
namespace {
const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { float scale, bias; uint size; };
kernel void affine(device const float* x [[buffer(0)]],
                   device float* y [[buffer(1)]],
                   constant Params& p [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < p.size) y[i] = fma(x[i], p.scale, p.bias);
}
)MSL";
struct Params { float scale, bias; unsigned size; };
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
Pipeline build(const Context& context, Buffer a, Buffer b, Buffer parameters,
               unsigned count, unsigned size) {
    // Both kernel and dispatch description die here; pipeline must retain them.
    auto kernel = context.kernel(source, "affine");
    std::vector<Dispatch> nodes;
    for (unsigned i = 0; i < count; ++i)
        nodes.push_back({kernel,
            {{i % 2 ? b : a, 0, Access::read},
             {i % 2 ? a : b, 0, Access::write}, {parameters}},
            {(size + 63) / 64, 1, 1}, {64, 1, 1}});
    return context.pipeline(nodes);
}
}
int main() { try {
    Context context;
    std::cout << "device: " << context.device_name() << '\n';
    constexpr unsigned size = 1003; // Deliberately not divisible by threadgroup size.
    auto a = context.buffer(size * sizeof(float));
    auto b = context.buffer(size * sizeof(float));
    auto parameters = context.buffer(sizeof(Params));
    std::vector<float> input(size), expected(size), direct(size), indirect(size);
    for (unsigned count : {1u, 16u, 129u}) {
        auto pipeline = build(context, a, b, parameters, count, size);
        check(pipeline.dispatch_count() == count, "wrong dispatch count");
        auto output = count % 2 ? b : a;
        for (unsigned variant = 0; variant < 5; ++variant) {
            Params p{0.999f + variant * 0.0001f, 0.001f + variant * 0.0002f, size};
            for (unsigned i = 0; i < size; ++i)
                input[i] = (static_cast<int>(i % 257) - 128) * 0.01f + variant * 0.125f;
            expected = input;
            for (unsigned step = 0; step < count; ++step)
                for (float& value : expected) value = std::fma(value, p.scale, p.bias);
            parameters.write(&p, sizeof(p));
            a.write(input.data(), a.size());
            const auto d = pipeline.run(Execution::direct);
            output.read(direct.data(), output.size());
            a.write(input.data(), a.size());
            const auto g = pipeline.run(Execution::icb);
            output.read(indirect.data(), output.size());
            for (unsigned i = 0; i < size; ++i) {
                check(std::isfinite(indirect[i]) && std::abs(indirect[i] - expected[i]) < 2e-5f,
                      "CPU/native mismatch");
                check(direct[i] == indirect[i], "direct/ICB mismatch");
            }
            if (variant == 4)
                std::cout << "dispatches=" << count << " direct_encode_us=" << d.encode_us
                          << " icb_encode_us=" << g.encode_us << " PASS\n";
        }
    }
    rejects([&] { context.buffer(0); });
    rejects([&] { a.write(input.data(), 8, a.size() - 4); });
    rejects([&] { a.read(nullptr, 1); });
    rejects([&] { context.pipeline({}); });
    rejects([&] { context.kernel(source, "missing"); });
    auto kernel = context.kernel(source, "affine");
    // Nonzero binding offsets and partial host transfers preserve the prefix.
    auto padded = context.buffer(8);
    float pair[2] = {-99, 2};
    padded.write(pair, sizeof(pair));
    Params offset_params{3, 1, 1};
    parameters.write(&offset_params, sizeof(offset_params));
    auto offset_pipeline = context.pipeline({
        {kernel, {{padded, 4}, {b, 4, Access::write}, {parameters}}, {1}, {1}}
    });
    for (auto mode : {Execution::direct, Execution::icb}) {
        offset_pipeline.run(mode);
        float value = 0;
        b.read(&value, sizeof(value), 4);
        check(value == 7, "binding offset mismatch");
        padded.read(&value, sizeof(value));
        check(value == -99, "input prefix corrupted");
    }
    Dispatch node{kernel, {{a}, {b, 0, Access::write}, {parameters}}, {16}, {64}};
    node.threadgroups.x = 0;
    rejects([&] { context.pipeline({node}); });
    node.threadgroups.x = 16;
    node.threads_per_threadgroup.x = 65536;
    rejects([&] { context.pipeline({node}); });
    node.threads_per_threadgroup.x = 64;
    node.bindings[0].offset = a.size();
    rejects([&] { context.pipeline({node}); });
    node.bindings[0].offset = 1;
    rejects([&] { context.pipeline({node}); });
    node.bindings[0].offset = 0;
    Context other;
    rejects([&] { other.pipeline({node}); });
    node.bindings[0].buffer = other.buffer(a.size());
    rejects([&] { context.pipeline({node}); });
    // Blits before, between and after compute spans must preserve snapshots
    // even when later dispatches overwrite the original output.
    auto snapshot = context.buffer(8);
    auto copied = context.program({
        BufferCopy{{padded, 4}, {a, 4, Access::write}, 4, "input"},
        Dispatch{kernel, {{a, 4}, {b, 4, Access::write}, {parameters}}, {1}, {1}, "first"},
        BufferCopy{{b, 4}, {snapshot, 4, Access::write}, 4, "snapshot"},
        BufferCopy{{b, 4}, {padded, 4, Access::write}, 4, "feedback"},
        Dispatch{kernel, {{padded, 4}, {b, 4, Access::write}, {parameters}}, {1}, {1}, "second"},
        BufferCopy{{b, 4}, {a, 4, Access::write}, 4, "output"},
    });
    check(copied.dispatch_count() == 2, "blits counted as shader dispatches");
    for (auto mode : {Execution::direct, Execution::icb}) {
        padded.write(pair, sizeof(pair));
        snapshot.write(pair, sizeof(pair));
        copied.run(mode);
        float value = 0;
        snapshot.read(&value, 4, 4);
        check(value == 7, "snapshot did not preserve the intermediate output");
        a.read(&value, 4, 4);
        check(value == 22, "compute/blit dependency failed");
        snapshot.read(&value, 4);
        check(value == -99, "blit destination prefix corrupted");
    }
    BufferCopy copy{{a}, {b, 0, Access::write}, 4, "bounds"};
    copy.bytes = 0;
    rejects([&] { context.program({copy}); });
    copy.bytes = 8;
    copy.source.offset = a.size() - 4;
    rejects([&] { context.program({copy}); });
    copy.source.offset = 0;
    copy.destination.offset = b.size() - 4;
    rejects([&] { context.program({copy}); });
    copy.destination = {a, 4, Access::write};
    rejects([&] { context.program({copy}); });
    copy.destination = {other.buffer(8), 0, Access::write};
    rejects([&] { context.program({copy}); });
    copy.destination = {b, 0, Access::write};
    copy.source.buffer = other.buffer(8);
    rejects([&] { context.program({copy}); });
    auto retained_copy = [] {
        Context local;
        auto x = local.buffer(4), y = local.buffer(4);
        float value = 2;
        x.write(&value, 4);
        return local.program({BufferCopy{{x}, {y, 0, Access::write}, 4, "retained copy"}});
    }();
    retained_copy.run(Execution::direct);
    retained_copy.run(Execution::icb);
    // Destroy the public context and every public resource before launching.
    auto retained = [] {
        Context local;
        auto x = local.buffer(4), y = local.buffer(4), p = local.buffer(sizeof(Params));
        float value = 2;
        Params params{2, 1, 1};
        x.write(&value, 4);
        p.write(&params, sizeof(params));
        return build(local, x, y, p, 1, 1);
    }();
    retained.run(Execution::direct);
    retained.run(Execution::icb);
    std::cout << "compute/blit ordering, bounds, ownership, context isolation: PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}}
