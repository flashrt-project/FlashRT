/* FlashRT exec — pybind module (`_flashrt_exec`).
 *
 * Dev / research / migration only. The hot path for real deployment is the C
 * ABI linked directly by C++/Rust/robot hosts. Here we wrap the same C ABI in
 * thin Python-friendly classes; the Ctx owns everything and frees it on
 * destruction, so Buffer/Graph/Plan/Event Python objects do not self-destroy
 * (avoids GC-order use-after-free).
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "flashrt/exec.h"
#include "backend.h"

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

void check(int rc, const char* what) {
    if (rc < 0) throw std::runtime_error(std::string(what) + " failed: rc=" + std::to_string(rc));
}

// Trampoline: the C record callback forwards to a Python callable, passing the
// capture stream as an integer the caller can wrap (torch.cuda.ExternalStream).
void py_record_trampoline(void* user, void* stream) {
    auto* fn = reinterpret_cast<py::function*>(user);
    (*fn)(reinterpret_cast<std::uintptr_t>(stream));
}

struct PyBuffer { frt_buffer h; };
struct PyEvent  { frt_event  h; };
struct PyGraph  { frt_graph  h; };
struct PyPlan   { frt_plan   h; };

struct PyCtx {
    frt_ctx h;
    PyCtx() { h = frt_ctx_create(); if (!h) throw std::runtime_error("frt_ctx_create failed"); }
    ~PyCtx() { if (h) frt_ctx_destroy(h); }
};

}  // namespace

PYBIND11_MODULE(_flashrt_exec, m) {
    m.doc() = "FlashRT execution-contract C ABI (dev binding)";

    py::class_<PyBuffer>(m, "Buffer")
        .def("dptr",  [](PyBuffer& b) { return reinterpret_cast<std::uintptr_t>(frt_buffer_dptr(b.h)); })
        .def("nbytes",[](PyBuffer& b) { return frt_buffer_bytes(b.h); })
        .def("name",  [](PyBuffer& b) { return std::string(frt_buffer_name(b.h)); });

    py::class_<PyEvent>(m, "Event")
        .def("record", [](PyEvent& e, int stream_id) { check(frt_event_record(e.h, stream_id), "event_record"); });

    py::class_<PyGraph>(m, "Graph")
        .def("capture", [](PyGraph& g, std::uint64_t key, py::function record) {
            check(frt_graph_capture(g.h, key, &py_record_trampoline, &record), "graph_capture");
        }, py::arg("key"), py::arg("record"))
        .def("bind", [](PyGraph& g, const std::string& port, PyBuffer& b) {
            check(frt_graph_bind(g.h, port.c_str(), b.h), "graph_bind");
        })
        .def("replay", [](PyGraph& g, std::uint64_t key, int stream_id) {
            return frt_graph_replay(g.h, key, stream_id);  // return rc (e.g. NO_VARIANT) to caller
        }, py::arg("key"), py::arg("stream_id") = 0)
        .def("has_variant", [](PyGraph& g, std::uint64_t key) { return frt_graph_has_variant(g.h, key) != 0; });

    py::class_<PyPlan>(m, "Plan")
        .def("add", [](PyPlan& p, PyGraph& g, std::uint64_t key, int stream_id) {
            int idx = frt_plan_add(p.h, g.h, key, stream_id);
            check(idx, "plan_add"); return idx;
        }, py::arg("graph"), py::arg("key"), py::arg("stream_id") = 0)
        .def("after", [](PyPlan& p, int node_idx, int dep_idx) { check(frt_plan_after(p.h, node_idx, dep_idx), "plan_after"); })
        .def("execute", [](PyPlan& p, std::uint64_t key) { check(frt_plan_execute(p.h, key), "plan_execute"); }, py::arg("key") = (std::uint64_t)FRT_KEY_INHERIT)
        .def("sync", [](PyPlan& p) { check(frt_plan_sync(p.h), "plan_sync"); });

    py::class_<PyCtx>(m, "Ctx")
        .def(py::init<>())
        .def("stream", [](PyCtx& c, int priority) { int id = frt_ctx_stream(c.h, priority); check(id, "ctx_stream"); return id; }, py::arg("priority") = 0)
        .def("event", [](PyCtx& c) { PyEvent e; e.h = frt_ctx_event(c.h); if (!e.h) throw std::runtime_error("ctx_event failed"); return e; })
        .def("stream_wait", [](PyCtx& c, int stream_id, PyEvent& e) { check(frt_stream_wait(c.h, stream_id, e.h), "stream_wait"); })
        .def("buffer", [](PyCtx& c, const std::string& name, size_t nbytes) {
            PyBuffer b; b.h = frt_buffer_alloc(c.h, name.c_str(), nbytes);
            if (!b.h) throw std::runtime_error("buffer_alloc failed"); return b;
        })
        .def("wrap", [](PyCtx& c, const std::string& name, std::uintptr_t dptr, size_t nbytes) {
            PyBuffer b; b.h = frt_buffer_wrap(c.h, name.c_str(), reinterpret_cast<void*>(dptr), nbytes);
            if (!b.h) throw std::runtime_error("buffer_wrap failed"); return b;
        })
        .def("copy", [](PyCtx& c, PyBuffer& dst, size_t dst_off, PyBuffer& src, size_t src_off, size_t nbytes, int stream_id) {
            check(frt_buffer_copy(c.h, dst.h, dst_off, src.h, src_off, nbytes, stream_id), "buffer_copy");
        }, py::arg("dst"), py::arg("dst_off"), py::arg("src"), py::arg("src_off"), py::arg("nbytes"), py::arg("stream_id") = 0)
        .def("graph", [](PyCtx& c, const std::string& name, size_t max_variants) {
            PyGraph g; g.h = frt_graph_create(c.h, name.c_str(), max_variants);
            if (!g.h) throw std::runtime_error("graph_create failed"); return g;
        }, py::arg("name"), py::arg("max_variants") = 0)
        .def("plan", [](PyCtx& c) { PyPlan p; p.h = frt_plan_create(c.h); if (!p.h) throw std::runtime_error("plan_create failed"); return p; });

    // --- dev/test helpers: allocation-free, capture-safe ops on a raw stream
    //     (an integer cudaStream_t). Used by record callbacks in tests so we
    //     can validate the contract without a real model kernel. ---
    m.def("memset_async", [](std::uintptr_t dptr, int value, size_t nbytes, std::uintptr_t stream) {
        if (!frt::be::memset_async(reinterpret_cast<void*>(dptr), value, nbytes, reinterpret_cast<void*>(stream)))
            throw std::runtime_error("memset_async failed");
    });
    m.def("memcpy_async", [](std::uintptr_t dst, std::uintptr_t src, size_t nbytes, std::uintptr_t stream) {
        if (!frt::be::memcpy_dtod_async(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), nbytes, reinterpret_cast<void*>(stream)))
            throw std::runtime_error("memcpy_async failed");
    });
}
