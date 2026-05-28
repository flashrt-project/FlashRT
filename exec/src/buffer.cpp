/* FlashRT exec — buffers (the only "state" primitive). */
#include "internal.h"
#include "backend.h"

frt_buffer frt_buffer_alloc(frt_ctx c, const char* name, size_t bytes) {
    if (!c || bytes == 0) return nullptr;
    void* d = frt::be::malloc(bytes);
    if (!d) return nullptr;
    auto* b = new frt_buffer_s();
    b->ctx = c;
    b->name = name ? name : "";
    b->dptr = d;
    b->bytes = bytes;
    b->owned = true;
    c->buffers.push_back(b);
    return b;
}

frt_buffer frt_buffer_wrap(frt_ctx c, const char* name, void* dptr, size_t bytes) {
    if (!c || !dptr) return nullptr;
    auto* b = new frt_buffer_s();
    b->ctx = c;
    b->name = name ? name : "";
    b->dptr = dptr;
    b->bytes = bytes;
    b->owned = false;  // external pointer; never freed by us
    c->buffers.push_back(b);
    return b;
}

void* frt_buffer_dptr(frt_buffer b)  { return b ? b->dptr : nullptr; }
size_t frt_buffer_bytes(frt_buffer b) { return b ? b->bytes : 0; }
const char* frt_buffer_name(frt_buffer b) { return b ? b->name.c_str() : ""; }

int frt_buffer_copy(frt_ctx c, frt_buffer dst, size_t dst_off,
                    frt_buffer src, size_t src_off, size_t bytes, int stream_id) {
    if (!c || !dst || !src) return FRT_ERR_INVALID;
    if (dst_off + bytes > dst->bytes || src_off + bytes > src->bytes)
        return FRT_ERR_INVALID;
    if (!c->has_stream(stream_id)) return FRT_ERR_INVALID;
    void* s = c->stream(stream_id);
    void* d = static_cast<char*>(dst->dptr) + dst_off;
    const void* sp = static_cast<const char*>(src->dptr) + src_off;
    return frt::be::memcpy_dtod_async(d, sp, bytes, s) ? FRT_OK : FRT_ERR_BACKEND;
}

// Note: no frt_buffer_destroy in the public ABI yet (Phase A) — the ctx owns
// all buffers and frees owned device memory at frt_ctx_destroy. Add per-buffer
// destroy when a real model needs finer lifetime than the ctx scope.
