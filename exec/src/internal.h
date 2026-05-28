/* FlashRT exec — internal object definitions backing the opaque C ABI handles.
 * Not installed / not public. */
#ifndef FLASHRT_EXEC_INTERNAL_H
#define FLASHRT_EXEC_INTERNAL_H

#include "flashrt/exec.h"

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

struct frt_buffer_s {
    frt_ctx     ctx   = nullptr;
    std::string name;
    void*       dptr  = nullptr;
    size_t      bytes = 0;
    bool        owned = false;   // true if we cudaMalloc'd it (free on destroy)
};

struct frt_event_s {
    frt_ctx ctx    = nullptr;    // for stream_id resolution
    void*   handle = nullptr;    // backend event
};

struct frt_variant {
    void* exec  = nullptr;   // graph-exec handle
    bool  owned = true;      // false if adopted from an external owner (torch)
};

struct frt_graph_s {
    frt_ctx     ctx = nullptr;
    std::string name;
    size_t      max_variants = 0;                       // 0 = unbounded
    std::unordered_map<frt_shape_key, frt_variant> variants;  // key -> exec
    std::list<frt_shape_key> lru;                       // front = oldest
    std::unordered_map<std::string, frt_buffer> bindings;  // port -> buffer (refs)

    void touch(frt_shape_key key);   // move key to MRU
    void evict_one();                // drop the oldest variant
    void put(frt_shape_key key, void* exec, bool owned);  // insert/replace + LRU
};

struct frt_plan_node {
    frt_graph     graph;
    frt_shape_key key;
    int           stream_id;
};

struct frt_plan_s {
    frt_ctx ctx = nullptr;
    std::vector<frt_plan_node> nodes;
    std::vector<std::pair<int, int>> deps;  // (node_idx, dep_node_idx)
};

struct frt_ctx_s {
    std::vector<void*> streams;            // stream_id -> backend stream; [0]=default
    std::vector<char>  stream_owned;       // parallel: 1 if frt created it (destroy), 0 if wrapped
    std::vector<frt_event_s*> events;      // tracked for cleanup safety
    std::vector<frt_buffer_s*> buffers;    // ctx owns all buffers (freed at destroy)
    std::vector<frt_graph_s*> graphs;      // tracked for cleanup safety
    std::vector<frt_plan_s*>  plans;       // tracked for cleanup safety

    bool has_stream(int id) const {
        return id >= 0 && id < (int)streams.size();
    }
    // Returns the backend stream handle for id; may legitimately be null
    // (handle 0 == the CUDA default stream). Validate the id with has_stream
    // first — do NOT treat a null return as "invalid", since 0 is a real stream.
    void* stream(int id) const {
        if (!has_stream(id)) return nullptr;
        return streams[id];
    }
};

#endif  /* FLASHRT_EXEC_INTERNAL_H */
