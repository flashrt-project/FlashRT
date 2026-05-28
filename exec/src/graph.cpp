/* FlashRT exec — Graph: a ShapeKey -> graph-exec variant table with LRU. */
#include "internal.h"
#include "backend.h"

void frt_graph_s::touch(frt_shape_key key) {
    for (auto it = lru.begin(); it != lru.end(); ++it) {
        if (*it == key) { lru.erase(it); break; }
    }
    lru.push_back(key);  // back = most recently used
}

void frt_graph_s::evict_one() {
    if (lru.empty()) return;
    frt_shape_key old = lru.front();
    lru.pop_front();
    auto it = variants.find(old);
    if (it != variants.end()) {
        frt::be::graph_exec_destroy(it->second);  // frees only the exec, not buffers
        variants.erase(it);
    }
}

frt_graph frt_graph_create(frt_ctx c, const char* name, size_t max_variants) {
    if (!c) return nullptr;
    auto* g = new frt_graph_s();
    g->ctx = c;
    g->name = name ? name : "";
    g->max_variants = max_variants;
    c->graphs.push_back(g);
    return g;
}

void frt_graph_destroy(frt_graph g) {
    if (!g) return;
    auto& gs = g->ctx->graphs;
    for (auto it = gs.begin(); it != gs.end(); ++it) {
        if (*it == g) { gs.erase(it); break; }
    }
    for (auto& kv : g->variants) frt::be::graph_exec_destroy(kv.second);
    delete g;
}

int frt_graph_capture(frt_graph g, frt_shape_key key,
                      void (*record)(void*, void*), void* user) {
    if (!g || !record) return FRT_ERR_INVALID;
    void* cap_stream = g->ctx->stream(0);  // capture on the default stream
    if (!cap_stream) return FRT_ERR_INVALID;

    if (!frt::be::capture_begin(cap_stream)) return FRT_ERR_CAPTURE;
    record(user, cap_stream);  // model enqueues its kernels onto cap_stream
    void* exec = frt::be::capture_end(cap_stream);
    if (!exec) return FRT_ERR_CAPTURE;

    // Replace an existing variant for this key.
    auto it = g->variants.find(key);
    if (it != g->variants.end()) {
        frt::be::graph_exec_destroy(it->second);
        it->second = exec;
    } else {
        g->variants.emplace(key, exec);
    }
    g->touch(key);
    if (g->max_variants > 0 && g->variants.size() > g->max_variants)
        g->evict_one();
    return FRT_OK;
}

int frt_graph_bind(frt_graph g, const char* port, frt_buffer b) {
    if (!g || !port || !b) return FRT_ERR_INVALID;
    g->bindings[port] = b;  // bookkeeping + lifetime ref; pointers were baked at capture
    return FRT_OK;
}

int frt_graph_replay(frt_graph g, frt_shape_key key, int stream_id) {
    if (!g) return FRT_ERR_INVALID;
    auto it = g->variants.find(key);
    if (it == g->variants.end()) return FRT_ERR_NO_VARIANT;  // never a silent no-op
    void* s = g->ctx->stream(stream_id);
    if (!s) return FRT_ERR_INVALID;
    g->touch(key);
    return frt::be::graph_launch(it->second, s) ? FRT_OK : FRT_ERR_BACKEND;
}

int frt_graph_has_variant(frt_graph g, frt_shape_key key) {
    if (!g) return 0;
    return g->variants.count(key) ? 1 : 0;
}
