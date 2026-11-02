// SPDX-License-Identifier: Apache-2.0
// Standalone launch/residency regression; no checkpoint or extension required.
#include "molmoact2_action_mk_sm120.cuh"
#include <cstdio>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "failed: %s at %d\n", #expr, __LINE__); return 1; } } while (0)

int main() {
    using namespace flash_rt::molmoact2;
    int device = 0;
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
    if (prop.major != 12) return 77;
    const int grid = action_mk_max_grid();
    CHECK(grid >= 33 && grid <= 64);
    if (prop.multiProcessorCount == 170) CHECK(grid == 64);
    if (prop.multiProcessorCount == 36) CHECK(grid == 36);
    ActionMkParams p{};
    p.rows = 30; p.prefetch = 1; p.ctx_len = 692; p.ctx_rows = 768;
    CHECK(cudaMalloc(&p.bar, 64 * sizeof(unsigned)) == cudaSuccess);
    CHECK(cudaMalloc(&p.counters, 16 * sizeof(unsigned)) == cudaSuccess);
    CHECK(cudaMemset(p.bar, 0, 64 * sizeof(unsigned)) == cudaSuccess);
    CHECK(cudaMemset(p.counters, 0, 16 * sizeof(unsigned)) == cudaSuccess);
    // Zero layers isolates actual cooperative launch and barrier setup from math.
    CHECK(action_mk_layers(p, grid, nullptr) == 0);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(action_mk_layers(p, 170, nullptr) != 0);
    CHECK(action_mk_layers(p, 32, nullptr) != 0);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(action_mk_layers(p, grid, stream) == 0);
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
    CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaGraphExecDestroy(exec) == cudaSuccess);
    CHECK(cudaGraphDestroy(graph) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    CHECK(cudaFree(p.bar) == cudaSuccess);
    CHECK(cudaFree(p.counters) == cudaSuccess);
    std::printf("SMs=%d grid=%d: cooperative launch, invalid-grid rejection and graph replay passed\n", prop.multiProcessorCount, grid);
    return 0;
}
