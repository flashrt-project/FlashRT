// SPDX-License-Identifier: Apache-2.0
//
// pybind surface of the Inkling-2-Mini SM120 kernels (module flash_rt_inkling2). Pointers and streams are
// integers, as in the other FlashRT kernel modules, so the runtime hands stable raw pointers to captured graphs.

#include "kernels/inkling2/inkling2_kernels.cuh"

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
namespace ik = flash_rt::inkling2;

namespace {

inline cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
template <typename T>
inline T* ptr(uintptr_t p) { return reinterpret_cast<T*>(p); }

ik::Prefetch to_prefetch(const std::vector<std::tuple<uintptr_t, uint64_t>>& ranges) {
  if (ranges.size() > ik::kMaxPrefetch) throw std::invalid_argument("at most 8 prefetch ranges");
  ik::Prefetch pf = {};
  for (const auto& [p, n] : ranges) {
    pf.ptr[pf.count] = ptr<const void>(p);
    pf.bytes[pf.count] = n;
    ++pf.count;
  }
  return pf;
}

// Parameter structs from dicts: pointers as integers (absent = null), ints, floats, lists of pointers.
template <typename T>
T* dptr(const py::dict& d, const char* key) {
  return d.contains(key) ? reinterpret_cast<T*>(d[key].cast<uintptr_t>()) : nullptr;
}
int dint(const py::dict& d, const char* key, int def = 0) { return d.contains(key) ? d[key].cast<int>() : def; }
float dflt(const py::dict& d, const char* key, float def = 0.0f) { return d.contains(key) ? d[key].cast<float>() : def; }
ik::Prefetch dprefetch(const py::dict& d) {
  if (!d.contains("prefetch")) return ik::Prefetch{};
  return to_prefetch(d["prefetch"].cast<std::vector<std::tuple<uintptr_t, uint64_t>>>());
}
template <typename T, size_t N>
int dlist(const py::dict& d, const char* key, T* (&out)[N]) {
  if (!d.contains(key)) return 0;
  const auto v = d[key].cast<std::vector<uintptr_t>>();
  if (v.size() > N) throw std::invalid_argument(std::string(key) + ": too many entries");
  for (size_t i = 0; i < v.size(); ++i) out[i] = reinterpret_cast<T*>(v[i]);
  return int(v.size());
}

}  // namespace

PYBIND11_MODULE(flash_rt_inkling2, m) {
  m.doc() = "Inkling-2-Mini (SM120) kernels: batch-one decode GEMVs, attention, routed experts";

  m.attr("MAX_GEMV_TASKS") = ik::kMaxGemvTasks;
  m.attr("MAX_PREFETCH") = ik::kMaxPrefetch;

  m.attr("GEMV_PLAIN") = int(ik::kGemvPlain);
  m.attr("GEMV_NORM") = int(ik::kGemvNorm);
  m.attr("GEMV_GATE") = int(ik::kGemvGate);
  // tasks: (w, x, y, n, k, x_f32, y_f32, prologue, aux); prefetch: (ptr, bytes)
  m.def("gemv_multi",
        [](const std::vector<std::tuple<uintptr_t, uintptr_t, uintptr_t, int, int, int, int, int, uintptr_t>>& tasks,
           const std::vector<std::tuple<uintptr_t, uint64_t>>& prefetch, uintptr_t stream) {
          if (tasks.empty() || tasks.size() > ik::kMaxGemvTasks) throw std::invalid_argument("gemv_multi: 1 to 8 tasks");
          ik::GemvTask t[ik::kMaxGemvTasks] = {};
          for (size_t i = 0; i < tasks.size(); ++i) {
            const auto& [w, x, y, n, k, xf, yf, pro, aux] = tasks[i];
            if (pro != ik::kGemvPlain && !aux) throw std::invalid_argument("gemv_multi: prologue needs aux");
            t[i] = {ptr<const void>(w), ptr<const void>(x), ptr<void>(y), ptr<const void>(aux), n, k, xf, yf, pro};
          }
          ik::gemv_multi(t, int(tasks.size()), to_prefetch(prefetch), to_stream(stream));
        },
        py::arg("tasks"), py::arg("prefetch"), py::arg("stream"));

  m.attr("HIDDEN") = ik::kHidden;
  m.attr("TOPK") = ik::kTopK;
  m.attr("MAX_BLOCKS") = ik::kMaxBlocks;
  m.attr("ATTN_SPLIT_KEYS") = ik::attn_split_keys();

  m.def("visit_boundary", [](const py::dict& d, uintptr_t stream) {
    ik::BoundaryParams p = {};
    p.finish = dint(d, "finish");
    p.expert_out = dptr<const uint16_t>(d, "expert_out");
    p.route_w = dptr<const float>(d, "route_w");
    p.residual = dptr<const float>(d, "residual");
    p.mlp_conv_state = dptr<uint16_t>(d, "mlp_conv_state");
    p.mlp_conv_w = dptr<const uint16_t>(d, "mlp_conv_w");
    p.hidden_out = dptr<uint16_t>(d, "hidden_out");
    p.embed = dint(d, "embed");
    p.token = dptr<const int>(d, "token");
    p.embed_table = dptr<const uint16_t>(d, "embed_table");
    p.embed_norm = dptr<const uint16_t>(d, "embed_norm");
    p.bank_partial = dptr<float>(d, "bank_partial");
    p.next_hidden = dptr<const uint16_t>(d, "next_hidden");
    p.normed_res = dptr<float>(d, "normed_res");
    p.n_blocks = dlist(d, "blocks", p.blocks);
    p.att_query = dptr<const float>(d, "att_query");
    p.nq = dint(d, "nq");
    p.att_in = dptr<float>(d, "att_in");
    p.messages = dint(d, "messages");
    p.bank_block = dptr<const float>(d, "bank_block");
    p.bank_state = dptr<const float>(d, "bank_state");
    p.message_query = dptr<const float>(d, "message_query");
    p.messages_out = dptr<float>(d, "messages_out");
    p.gate_in = dptr<uint16_t>(d, "gate_in");
    p.prefetch = dprefetch(d);
    p.timing = dptr<unsigned long long>(d, "timing");
    ik::visit_boundary(p, to_stream(stream));
  });
  m.def("post_attention", [](const py::dict& d, uintptr_t stream) {
    ik::PostAttentionParams p = {};
    p.attn_out = dptr<const uint16_t>(d, "attn_out");
    p.block_residual = dptr<const uint16_t>(d, "block_residual");
    p.attn_conv_state = dptr<uint16_t>(d, "attn_conv_state");
    p.attn_conv_w = dptr<const uint16_t>(d, "attn_conv_w");
    p.boundary = dint(d, "boundary");
    p.residual = dptr<float>(d, "residual");
    p.n_blocks = dlist(d, "blocks", p.blocks);
    p.mlp_query = dptr<const float>(d, "mlp_query");
    p.mlp_in = dptr<float>(d, "mlp_in");
    p.message = dptr<const float>(d, "message");
    p.gate_in = dptr<uint16_t>(d, "gate_in");
    p.prefetch = dprefetch(d);
    ik::post_attention(p, to_stream(stream));
  });
  m.def("gate_mix", [](const py::dict& d, uintptr_t stream) {
    ik::GateMixParams p = {};
    p.count = dlist(d, "forget", p.forget);
    if (dlist(d, "gain", p.gain) != p.count || dlist(d, "states", p.states) != p.count ||
        dlist(d, "h", p.h) != p.count || dlist(d, "m", p.m) != p.count || dlist(d, "out", p.out) != p.count)
      throw std::invalid_argument("gate_mix: lists of equal length");
    ik::gate_mix(p, to_stream(stream));
  });
  m.def("attn_prologue", [](const py::dict& d, uintptr_t stream) {
    ik::AttnPrologueParams p = {};
    p.sliding = dint(d, "sliding");
    p.pos = dptr<const int>(d, "pos");
    p.q_lin = dptr<const uint16_t>(d, "q_lin");
    p.r_lin = dptr<const uint16_t>(d, "r_lin");
    p.k_lin = dptr<const uint16_t>(d, "k_lin");
    p.v_lin = dptr<const uint16_t>(d, "v_lin");
    p.k_dyn = dptr<const uint16_t>(d, "k_dyn");
    p.v_dyn = dptr<const uint16_t>(d, "v_dyn");
    p.k_conv_state = dptr<uint16_t>(d, "k_conv_state");
    p.v_conv_state = dptr<uint16_t>(d, "v_conv_state");
    p.k_conv_w = dptr<const uint16_t>(d, "k_conv_w");
    p.v_conv_w = dptr<const uint16_t>(d, "v_conv_w");
    p.q_clip = dptr<const uint16_t>(d, "q_clip");
    p.rel_clip = dptr<const uint16_t>(d, "rel_clip");
    p.rel_proj = dptr<const uint16_t>(d, "rel_proj");
    p.qk_threshold = dflt(d, "qk_threshold");
    p.rel_threshold = dflt(d, "rel_threshold");
    p.log_alpha = dflt(d, "log_alpha");
    p.log_floor = dflt(d, "log_floor", 1.0f);
    p.q_out = dptr<uint16_t>(d, "q_out");
    p.bias = dptr<uint16_t>(d, "bias");
    p.k_cache = dptr<uint16_t>(d, "k_cache");
    p.v_cache = dptr<uint16_t>(d, "v_cache");
    if (!p.sliding && (!p.k_dyn || !p.v_dyn)) throw std::invalid_argument("attn_prologue: full layers need k_dyn, v_dyn");
    p.prefetch = dprefetch(d);
    ik::attn_prologue(p, to_stream(stream));
  });
  m.def("attn_decode", [](const py::dict& d, uintptr_t stream) {
    ik::AttnDecodeParams p = {};
    p.sliding = dint(d, "sliding");
    p.pos = dptr<const int>(d, "pos");
    p.q_lin = dptr<const uint16_t>(d, "q_lin");
    p.r_lin = dptr<const uint16_t>(d, "r_lin");
    p.k_lin = dptr<const uint16_t>(d, "k_lin");
    p.v_lin = dptr<const uint16_t>(d, "v_lin");
    p.k_dyn = dptr<const uint16_t>(d, "k_dyn");
    p.v_dyn = dptr<const uint16_t>(d, "v_dyn");
    p.k_conv_state = dptr<uint16_t>(d, "k_conv_state");
    p.v_conv_state = dptr<uint16_t>(d, "v_conv_state");
    p.k_conv_w = dptr<const uint16_t>(d, "k_conv_w");
    p.v_conv_w = dptr<const uint16_t>(d, "v_conv_w");
    p.q_clip = dptr<const uint16_t>(d, "q_clip");
    p.rel_clip = dptr<const uint16_t>(d, "rel_clip");
    p.rel_proj = dptr<const uint16_t>(d, "rel_proj");
    p.qk_threshold = dflt(d, "qk_threshold");
    p.rel_threshold = dflt(d, "rel_threshold");
    p.log_alpha = dflt(d, "log_alpha");
    p.log_floor = dflt(d, "log_floor", 1.0f);
    p.k_cache = dptr<uint16_t>(d, "k_cache");
    p.v_cache = dptr<uint16_t>(d, "v_cache");
    if (!p.sliding && (!p.k_dyn || !p.v_dyn)) throw std::invalid_argument("attn_decode: full layers need k_dyn, v_dyn");
    p.sinks = dptr<const uint16_t>(d, "sinks");
    p.partial = dptr<float>(d, "partial");
    p.counters = dptr<int>(d, "counters");
    p.max_splits = dint(d, "max_splits");
    p.out = dptr<uint16_t>(d, "out");
    p.prefetch = dprefetch(d);
    ik::attn_decode(p, to_stream(stream));
  });
  m.def("router", [](const py::dict& d, uintptr_t stream) {
    ik::RouterParams p = {};
    p.in = dptr<const float>(d, "in");
    p.norm_w = dptr<const uint16_t>(d, "norm_w");
    p.weight = dptr<const uint16_t>(d, "weight");
    p.bias = dptr<const float>(d, "bias");
    p.route_scale = dflt(d, "route_scale");
    p.mxfp8 = dint(d, "mxfp8");
    p.logits = dptr<float>(d, "logits");
    p.counter = dptr<int>(d, "counter");
    p.ids = dptr<int>(d, "ids");
    p.weights = dptr<float>(d, "weights");
    p.x_codes = dptr<uint8_t>(d, "x_codes");
    p.x_scales = dptr<uint8_t>(d, "x_scales");
    p.x_amax = dptr<float>(d, "x_amax");
    ik::router(p, to_stream(stream));
  });
  auto expert = [](const py::dict& d) {
    ik::ExpertParams p = {};
    p.mxfp8 = dint(d, "mxfp8");
    p.ids = dptr<const int>(d, "ids");
    p.codes = dptr<const void>(d, "codes");
    p.scales = dptr<const void>(d, "scales");
    p.amax = dptr<const float>(d, "amax");
    p.denominator = dflt(d, "denominator", 1.0f);
    p.gate_cap = dflt(d, "gate_cap");
    p.up_cap = dflt(d, "up_cap");
    p.x_codes = dptr<const uint8_t>(d, "x_codes");
    p.x_scales = dptr<const uint8_t>(d, "x_scales");
    p.x_amax = dptr<const float>(d, "x_amax");
    p.act = dptr<uint16_t>(d, "act");
    p.out = dptr<uint16_t>(d, "out");
    p.logits = dptr<const float>(d, "logits");
    p.bias = dptr<const float>(d, "bias");
    p.route_scale = dflt(d, "route_scale");
    p.x_in = dptr<const float>(d, "x_in");
    p.norm_w = dptr<const uint16_t>(d, "norm_w");
    p.ids_out = dptr<int>(d, "ids_out");
    p.weights_out = dptr<float>(d, "weights_out");
    if (p.logits && (!p.bias || !p.x_in || !p.norm_w || !p.ids_out || !p.weights_out))
      throw std::invalid_argument("expert: folded routing needs bias, x_in, norm_w, ids_out, weights_out");
    if (d.contains("w2_prefetch")) {       // [(base, bytes per expert)] x 2: w2 codes, w2 scales
      const auto v = d["w2_prefetch"].cast<std::vector<std::tuple<uintptr_t, uint64_t>>>();
      if (v.size() != 2) throw std::invalid_argument("expert: w2_prefetch needs 2 (base, bytes) entries");
      for (int i = 0; i < 2; ++i) {
        p.w2_base[i] = reinterpret_cast<const uint8_t*>(std::get<0>(v[i]));
        p.w2_bytes[i] = std::get<1>(v[i]);
      }
    }
    p.prefetch = dprefetch(d);
    return p;
  };
  m.def("expert_w13", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w13(expert(d), to_stream(stream)); });
  m.def("expert_w2", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w2(expert(d), to_stream(stream)); });
  m.def("lm_head", [](const py::dict& d, uintptr_t stream) {
    ik::LmHeadParams p = {};
    p.in = dptr<const float>(d, "in");
    p.norm_w = dptr<const uint16_t>(d, "norm_w");
    p.weight = dptr<const uint16_t>(d, "weight");
    p.vocab = dint(d, "vocab");
    p.softcap = dflt(d, "softcap");
    p.blocks = dint(d, "blocks");
    p.logits = dptr<float>(d, "logits");
    p.block_best = dptr<float>(d, "block_best");
    p.block_index = dptr<int>(d, "block_index");
    p.counter = dptr<int>(d, "counter");
    p.token = dptr<int>(d, "token");
    p.prefetch = dprefetch(d);
    ik::lm_head(p, to_stream(stream));
  });
}
