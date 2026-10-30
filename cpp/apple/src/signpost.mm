#include <os/signpost.h>
#include <stdint.h>

static os_log_t stage_log() {
    static os_log_t log = os_log_create("com.flashrt.apple", "pipeline");
    return log;
}

// Standalone bridge: does not create a Metal device or depend on PyTorch.
// The static interval name pairs begin/end; the public payload carries the
// model-stage label and the unique ID supports nested/concurrent intervals.
extern "C" __attribute__((visibility("default"))) uint64_t frt_apple_signpost_begin(const char* label) {
    auto log = stage_log();
    if (!os_signpost_enabled(log)) return 0;
    auto id = os_signpost_id_generate(log);
    os_signpost_interval_begin(log, id, "FlashRT stage", "%{public}s", label ? label : "");
    return id;
}
extern "C" __attribute__((visibility("default"))) void frt_apple_signpost_end(uint64_t id, const char* label) {
    if (id) os_signpost_interval_end(stage_log(), id, "FlashRT stage", "%{public}s", label ? label : "");
}
