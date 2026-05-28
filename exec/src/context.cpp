/* FlashRT exec — context, streams, events. */
#include "internal.h"
#include "backend.h"

frt_ctx frt_ctx_create(void) {
    auto* c = new frt_ctx_s();
    void* s0 = frt::be::stream_create(0);  // default stream = id 0
    if (!s0) { delete c; return nullptr; }
    c->streams.push_back(s0);
    c->stream_owned.push_back(1);
    return c;
}

void frt_ctx_destroy(frt_ctx c) {
    if (!c) return;
    // Order: plans ref graphs; graphs ref buffers/streams; events/buffers/streams last.
    for (auto* p : c->plans) delete p;
    for (auto* g : c->graphs) {
        if (g) {
            for (auto& kv : g->variants)
                if (kv.second.owned) frt::be::graph_exec_destroy(kv.second.exec);
            delete g;
        }
    }
    for (auto* b : c->buffers) {
        if (b) { if (b->owned) frt::be::free(b->dptr); delete b; }
    }
    for (auto* e : c->events) {
        if (e) { frt::be::event_destroy(e->handle); delete e; }
    }
    for (size_t i = 0; i < c->streams.size(); ++i)
        if (c->stream_owned[i]) frt::be::stream_destroy(c->streams[i]);
    delete c;
}

int frt_ctx_stream(frt_ctx c, int priority) {
    if (!c) return FRT_ERR_INVALID;
    void* s = frt::be::stream_create(priority);
    if (!s) return FRT_ERR_BACKEND;
    c->streams.push_back(s);
    c->stream_owned.push_back(1);
    return (int)c->streams.size() - 1;
}

int frt_ctx_wrap_stream(frt_ctx c, void* external_stream) {
    if (!c || !external_stream) return FRT_ERR_INVALID;
    c->streams.push_back(external_stream);
    c->stream_owned.push_back(0);  // non-owned; never destroyed by frt
    return (int)c->streams.size() - 1;
}

frt_event frt_ctx_event(frt_ctx c) {
    if (!c) return nullptr;
    void* h = frt::be::event_create();
    if (!h) return nullptr;
    auto* e = new frt_event_s();
    e->ctx = c;
    e->handle = h;
    c->events.push_back(e);
    return e;
}

void frt_event_destroy(frt_event e) {
    if (!e || !e->ctx) return;
    // Remove from ctx tracking and free now (avoid double-free at ctx_destroy).
    auto& ev = e->ctx->events;
    for (auto it = ev.begin(); it != ev.end(); ++it) {
        if (*it == e) { ev.erase(it); break; }
    }
    frt::be::event_destroy(e->handle);
    delete e;
}

int frt_event_record(frt_event e, int stream_id) {
    if (!e || !e->ctx) return FRT_ERR_INVALID;
    void* s = e->ctx->stream(stream_id);
    if (!s) return FRT_ERR_INVALID;
    return frt::be::event_record(e->handle, s) ? FRT_OK : FRT_ERR_BACKEND;
}

int frt_stream_wait(frt_ctx c, int stream_id, frt_event e) {
    if (!c || !e) return FRT_ERR_INVALID;
    void* s = c->stream(stream_id);
    if (!s) return FRT_ERR_INVALID;
    return frt::be::stream_wait_event(s, e->handle) ? FRT_OK : FRT_ERR_BACKEND;
}
