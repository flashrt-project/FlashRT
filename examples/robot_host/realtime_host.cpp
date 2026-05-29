/* examples/robot_host/realtime_host.cpp — illustrative C++ real-time VLA host.
 *
 * DEPLOYMENT hot path on the FlashRT execution contract: a native loop driving
 * adopted graphs across streams, with voice-interrupt + subgoal swap. This is
 * the scenario/policy layer (lives in examples, not in the contract).
 *
 * SETUP (capture + calibrate + autotune + adopt) is done once before this loop
 * — by the Python frontend in the SAME process (Python out of the hot loop),
 * or ported to C++ for a no-Python build. The `frt_*` handles below are the
 * product of that setup; this file shows only the hot loop + C-ABI usage and
 * is a reference structure, not a standalone runnable binary.
 *
 * Build (once libflashrt_exec is installed/available):
 *   g++ -std=c++17 realtime_host.cpp -I<exec>/include -lflashrt_exec -o realtime_host
 */
#include "flashrt/exec.h"

#include <atomic>
#include <cstring>

// Handles produced by setup (capture + adopt). In the pragmatic path these are
// created by the in-process Python frontend and handed to the C++ loop.
struct VlaHandles {
    frt_ctx    ctx;
    frt_graph  vision;       // writes enc_x
    frt_graph  action;       // reads enc_x (shared buffer), writes action_out
    frt_buffer img;          // sensor frame in
    frt_buffer enc_x;        // vision -> action hand-off (zero-copy)
    frt_buffer action_out;   // actions out
    frt_buffer subgoal_emb;  // mutable-at-interrupt state (overwrite, no recapture)
    int        s_action;     // high-priority action stream
    int        s_asr;        // concurrent lower-priority stream (voice)
};

// Change the subgoal WITHOUT recapture: overwrite the bound buffer; the next
// replay reads the new goal. (subgoal_emb was bound into the captured graph.)
static void on_voice_subgoal(VlaHandles& h, const void* new_emb, size_t n) {
    std::memcpy(frt_buffer_dptr(h.subgoal_emb), new_emb, n);  // host->device in real code
}

// One inference tick = vision -> action as a single DAG (zero-copy hand-off).
static int build_plan(VlaHandles& h, frt_plan& out) {
    out = frt_plan_create(h.ctx);
    int v = frt_plan_add(out, h.vision, /*key=*/0, h.s_action);
    int a = frt_plan_add(out, h.action, /*key=*/0, h.s_action);
    return frt_plan_after(out, a, v);  // action waits vision via an event
}

void realtime_loop(VlaHandles& h, std::atomic<bool>& running,
                   frt_graph asr /*optional, may be null*/) {
    frt_plan plan;
    build_plan(h, plan);

    while (running.load(std::memory_order_relaxed)) {
        // 1) write the fresh sensor frame into h.img (host memcpy / DMA).
        // 2) optional concurrent voice/ASR on its own stream — hardware overlaps;
        //    whether to run it (vs. protect action p99) is HOST policy.
        if (asr) frt_graph_replay(asr, /*key=*/0, h.s_asr);
        // 3) run vision -> action in one shot.
        if (frt_plan_execute(plan, /*key=*/0) != FRT_OK) break;
        frt_plan_sync(plan);
        // 4) read actions from h.action_out and send to the robot.
        // Interrupt granularity = one short replay: a hard interrupt simply does
        // not issue the next tick (or fires a higher-priority graph instead).
    }
    frt_plan_destroy(plan);
}

/* main() is intentionally omitted: in the pragmatic path, the in-process Python
 * frontend builds `VlaHandles` (setup+capture+adopt) and calls realtime_loop on
 * a dedicated thread. See README.md. */
