/* FlashRT exec — Plan: a dumb DAG of (graph, key) replays across streams.
 *
 * Data dependencies only. No priority / deadline / preemption — those are
 * upper-layer policy. Within one stream, add-order implies sequencing; across
 * streams, frt_plan_after declares an explicit event dependency.
 */
#include "internal.h"
#include "backend.h"

#include <vector>

frt_plan frt_plan_create(frt_ctx c) {
    if (!c) return nullptr;
    auto* p = new frt_plan_s();
    p->ctx = c;
    c->plans.push_back(p);
    return p;
}

void frt_plan_destroy(frt_plan p) {
    if (!p) return;
    auto& ps = p->ctx->plans;
    for (auto it = ps.begin(); it != ps.end(); ++it) {
        if (*it == p) { ps.erase(it); break; }
    }
    delete p;
}

int frt_plan_add(frt_plan p, frt_graph g, frt_shape_key key, int stream_id) {
    if (!p || !g) return FRT_ERR_INVALID;
    if (!p->ctx->has_stream(stream_id)) return FRT_ERR_INVALID;
    p->nodes.push_back(frt_plan_node{g, key, stream_id});
    return (int)p->nodes.size() - 1;
}

int frt_plan_after(frt_plan p, int node_idx, int dep_node_idx) {
    if (!p) return FRT_ERR_INVALID;
    int n = (int)p->nodes.size();
    if (node_idx < 0 || node_idx >= n || dep_node_idx < 0 || dep_node_idx >= n)
        return FRT_ERR_INVALID;
    if (node_idx == dep_node_idx) return FRT_ERR_INVALID;
    p->deps.emplace_back(node_idx, dep_node_idx);
    return FRT_OK;
}

int frt_plan_execute(frt_plan p, frt_shape_key inherit_key) {
    if (!p) return FRT_ERR_INVALID;
    const int n = (int)p->nodes.size();

    // One event per node, recorded right after that node launches, so a later
    // node on another stream can wait on it. Allocated from the backend (not
    // the public event ABI) and freed at the end of this execute.
    std::vector<void*> done(n, nullptr);
    for (int i = 0; i < n; ++i) {
        done[i] = frt::be::event_create();
        if (!done[i]) {
            for (int j = 0; j < i; ++j) frt::be::event_destroy(done[j]);
            return FRT_ERR_BACKEND;
        }
    }

    int rc = FRT_OK;
    // Execute in add order (assumed a valid topological order, per the ABI:
    // within-stream order is sequencing; cross-stream order via frt_plan_after).
    for (int i = 0; i < n && rc == FRT_OK; ++i) {
        const frt_plan_node& nd = p->nodes[i];
        void* s = p->ctx->stream(nd.stream_id);

        // Make this node's stream wait on every declared dependency.
        for (const auto& d : p->deps) {
            if (d.first == i) frt::be::stream_wait_event(s, done[d.second]);
        }

        frt_shape_key k = (nd.key == FRT_KEY_INHERIT) ? inherit_key : nd.key;
        rc = frt_graph_replay(nd.graph, k, nd.stream_id);
        if (rc == FRT_OK) frt::be::event_record(done[i], s);
    }

    for (int i = 0; i < n; ++i) frt::be::event_destroy(done[i]);
    return rc;
}

int frt_plan_sync(frt_plan p) {
    if (!p) return FRT_ERR_INVALID;
    // Sync every distinct stream the plan touches.
    std::vector<int> seen;
    for (const auto& nd : p->nodes) {
        bool found = false;
        for (int s : seen) if (s == nd.stream_id) { found = true; break; }
        if (!found) {
            seen.push_back(nd.stream_id);
            frt::be::stream_sync(p->ctx->stream(nd.stream_id));
        }
    }
    return FRT_OK;
}
