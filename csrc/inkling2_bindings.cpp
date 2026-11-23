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
  // tasks: (w, x, y, n, k, x_f32, y_f32, prologue, aux[, rows]); prefetch: (ptr, bytes)
  m.def("gemv_multi",
        [](const std::vector<py::tuple>& tasks, const std::vector<std::tuple<uintptr_t, uint64_t>>& prefetch,
           uintptr_t stream) {
          if (tasks.empty() || tasks.size() > ik::kMaxGemvTasks) throw std::invalid_argument("gemv_multi: 1 to 8 tasks");
          ik::GemvTask t[ik::kMaxGemvTasks] = {};
          for (size_t i = 0; i < tasks.size(); ++i) {
            const py::tuple& a = tasks[i];
            if (a.size() < 9 || a.size() > 11) throw std::invalid_argument("gemv_multi: task of 9 to 11 fields");
            const int pro = a[7].cast<int>();
            const uintptr_t aux = a[8].cast<uintptr_t>();
            if (pro != ik::kGemvPlain && !aux) throw std::invalid_argument("gemv_multi: prologue needs aux");
            t[i] = {ptr<const void>(a[0].cast<uintptr_t>()), ptr<const void>(a[1].cast<uintptr_t>()),
                    ptr<void>(a[2].cast<uintptr_t>()), ptr<const void>(aux), a[3].cast<int>(), a[4].cast<int>(),
                    a[5].cast<int>(), a[6].cast<int>(), pro, a.size() >= 10 ? a[9].cast<int>() : 1,
                    a.size() == 11 ? a[10].cast<int>() : 0};
          }
          ik::gemv_multi(t, int(tasks.size()), to_prefetch(prefetch), to_stream(stream));
        },
        py::arg("tasks"), py::arg("prefetch"), py::arg("stream"));

  // tasks: (x, x_f32, x_stride, w, out, rows, k)
  m.def("norm_rows",
        [](const std::vector<std::tuple<uintptr_t, int, int, uintptr_t, uintptr_t, int, int>>& tasks, uintptr_t stream) {
          if (tasks.empty() || tasks.size() > ik::kMaxNormTasks) throw std::invalid_argument("norm_rows: 1 to 4 tasks");
          ik::NormRowsTask t[ik::kMaxNormTasks] = {};
          for (size_t i = 0; i < tasks.size(); ++i) {
            const auto& [x, x_f32, x_stride, w, out, rows, k] = tasks[i];
            t[i] = {ptr<const void>(x), ptr<const uint16_t>(w), ptr<uint16_t>(out), k, x_f32, rows, x_stride};
          }
          ik::norm_rows(t, int(tasks.size()), to_stream(stream));
        },
        py::arg("tasks"), py::arg("stream"));

  m.attr("HIDDEN") = ik::kHidden;
  m.attr("TOPK") = ik::kTopK;
  m.attr("MAX_BLOCKS") = ik::kMaxBlocks;
  m.attr("ATTN_SPLIT_KEYS") = ik::attn_split_keys();

  m.def("visit_boundary", [](const py::dict& d, uintptr_t stream) {
    ik::BoundaryParams p = {};
    p.rows = dint(d, "rows", 1);
    p.stage = dptr<uint16_t>(d, "stage");
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
    p.rows = dint(d, "rows", 1);
    p.stage = dptr<uint16_t>(d, "stage");
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
    p.rows = dint(d, "rows", 1);
    p.h_stride = dint(d, "h_stride", ik::kHidden);
    p.m_stride = dint(d, "m_stride", ik::kHidden);
    p.out_stride = dint(d, "out_stride", ik::kHidden);
    p.count = dlist(d, "forget", p.forget);
    if (dlist(d, "gain", p.gain) != p.count || dlist(d, "states", p.states) != p.count ||
        dlist(d, "h", p.h) != p.count || dlist(d, "m", p.m) != p.count || dlist(d, "out", p.out) != p.count)
      throw std::invalid_argument("gate_mix: lists of equal length");
    ik::gate_mix(p, to_stream(stream));
  });
  auto attn_params = [](const py::dict& d) {
    ik::AttnDecodeParams p = {};
    p.rows = dint(d, "rows", 1);
    p.stage_kv = dptr<uint16_t>(d, "stage_kv");
    p.stage_in = dptr<uint16_t>(d, "stage_in");
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
    p.gate = dptr<const uint16_t>(d, "gate");
    p.q_pre = dptr<float>(d, "q_pre");
    p.prefetch = dprefetch(d);
    return p;
  };
  m.def("attn_decode", [attn_params](const py::dict& d, uintptr_t stream) { ik::attn_decode(attn_params(d), to_stream(stream)); });
  m.def("kv_rows", [attn_params](const py::dict& d, uintptr_t stream) { ik::kv_rows(attn_params(d), to_stream(stream)); });
  auto expert = [](const py::dict& d) {
    ik::ExpertParams p = {};
    p.mxfp8 = dint(d, "mxfp8");
    p.rows = dint(d, "rows", 1);
    p.ids = dptr<const int>(d, "ids");
    p.codes = dptr<const void>(d, "codes");
    p.scales = dptr<const void>(d, "scales");
    p.amax = dptr<const float>(d, "amax");
    p.denominator = dflt(d, "denominator", 1.0f);
    p.gate_cap = dflt(d, "gate_cap");
    p.up_cap = dflt(d, "up_cap");
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
    p.prefetch = dprefetch(d);
    p.xq = dptr<void>(d, "xq");
    p.xq_scale = dptr<float>(d, "xq_scale");
    p.xq_amax = dptr<float>(d, "xq_amax");
    return p;
  };
  m.def("expert_w13", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w13(expert(d), to_stream(stream)); });
  m.def("expert_w2", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w2(expert(d), to_stream(stream)); });
  m.def("expert_route", [expert](const py::dict& d, uintptr_t stream) { ik::expert_route(expert(d), to_stream(stream)); });
  m.def("expert_w13_flat", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w13_flat(expert(d), to_stream(stream)); });
  m.def("expert_act_quant", [expert](const py::dict& d, uintptr_t stream) { ik::expert_act_quant(expert(d), to_stream(stream)); });
  m.def("expert_w2_flat", [expert](const py::dict& d, uintptr_t stream) { ik::expert_w2_flat(expert(d), to_stream(stream)); });
  auto mtp = [](const py::dict& d) {
    ik::MtpParams p = {};
    for (int i = 0; i < 4; ++i) {
      p.feat[i] = nullptr;
      p.norm[i] = nullptr;
    }
    if (d.contains("feat")) {
      const auto v = d["feat"].cast<std::vector<uintptr_t>>();
      if (v.size() != 4) throw std::invalid_argument("mtp: feat needs 4 pointers");
      for (int i = 0; i < 4; ++i) p.feat[i] = ptr<const uint16_t>(v[i]);
    }
    if (d.contains("norm")) {
      const auto v = d["norm"].cast<std::vector<uintptr_t>>();
      if (v.size() != 4) throw std::invalid_argument("mtp: norm needs 4 pointers");
      for (int i = 0; i < 4; ++i) p.norm[i] = ptr<const uint16_t>(v[i]);
    }
    if (d.contains("k_norm")) {
      const auto v = d["k_norm"].cast<std::vector<uintptr_t>>();
      if (v.size() != 3) throw std::invalid_argument("mtp: k_norm needs 3 pointers");
      for (int i = 0; i < 3; ++i) p.k_norm[i] = ptr<const uint16_t>(v[i]);
    }
    if (d.contains("alpha")) {
      const auto v = d["alpha"].cast<std::vector<float>>();
      if (v.size() != 16) throw std::invalid_argument("mtp: alpha needs 16 values");
      for (int i = 0; i < 16; ++i) p.alpha[i] = v[i];
    }
    p.count = dptr<const int>(d, "count");
    p.count_add = dint(d, "count_add");
    p.kvc = dptr<const uint16_t>(d, "kvc");
    p.pos = dptr<const int>(d, "pos");
    p.kv = dptr<uint16_t>(d, "kv");
    p.cap = dint(d, "cap");
    p.theta = dflt(d, "theta", 1.0e6f);
    p.layer = dint(d, "layer");
    p.window = dint(d, "window");
    p.x = dptr<uint16_t>(d, "x");
    p.frnn = dptr<uint16_t>(d, "frnn");
    p.anchor = dptr<const int>(d, "anchor");
    p.embed = dptr<const uint16_t>(d, "embed");
    p.embed_norm = dptr<const uint16_t>(d, "embed_norm");
    p.mask_embed = dptr<const uint16_t>(d, "mask_embed");
    p.dh = dptr<float>(d, "dh");
    p.add = dptr<const uint16_t>(d, "add");
    if (d.contains("add_part")) {
      const auto v = d["add_part"].cast<std::vector<uintptr_t>>();
      if (v.size() != 2) throw std::invalid_argument("mtp: add_part needs 2 pointers");
      for (int i = 0; i < 2; ++i) p.add_part[i] = ptr<const float>(v[i]);
    }
    p.norm_w = dptr<const uint16_t>(d, "norm_w");
    p.q_in = dptr<const uint16_t>(d, "q_in");
    p.k_in = dptr<const uint16_t>(d, "k_in");
    p.v_in = dptr<const uint16_t>(d, "v_in");
    p.q_norm = dptr<const uint16_t>(d, "q_norm");
    p.kb_norm = dptr<const uint16_t>(d, "kb_norm");
    p.q_out = dptr<uint16_t>(d, "q_out");
    p.attn_out = dptr<uint16_t>(d, "attn_out");
    p.gate = dptr<const uint16_t>(d, "gate");
    p.up = dptr<const uint16_t>(d, "up");
    p.act = dptr<uint16_t>(d, "act");
    p.step = dint(d, "step");
    p.drafts = dptr<int>(d, "drafts");
    p.markov_w1 = dptr<const uint16_t>(d, "markov_w1");
    p.hn = dptr<const uint16_t>(d, "hn");
    p.rnn_in = dptr<uint16_t>(d, "rnn_in");
    p.proj = dptr<const float>(d, "proj");
    p.rnn_bias = dptr<const uint16_t>(d, "rnn_bias");
    p.state = dptr<uint16_t>(d, "state");
    p.latent = dptr<uint16_t>(d, "latent");
    p.logits = dptr<const float>(d, "logits");
    p.vocab_stride = dint(d, "vocab_stride");
    p.vocab = dint(d, "vocab");
    p.fold_w = dptr<const uint16_t>(d, "fold_w");
    p.fold_b = dptr<const uint16_t>(d, "fold_b");
    p.fold_w8 = dptr<const uint8_t>(d, "fold_w8");
    p.fold_scale = dptr<const float>(d, "fold_scale");
    p.pick_best = dptr<float>(d, "pick_best");
    p.pick_index = dptr<int>(d, "pick_index");
    p.pick_counter = dptr<int>(d, "pick_counter");
    return p;
  };
#define MTP_DEF(name) \
  m.def(#name, [mtp](const py::dict& d, uintptr_t stream) { ik::name(mtp(d), to_stream(stream)); });
  MTP_DEF(mtp_context_mix)
  MTP_DEF(mtp_context_write)
  MTP_DEF(mtp_block_embed)
  MTP_DEF(mtp_norm)
  MTP_DEF(mtp_qk)
  MTP_DEF(mtp_attention)
  MTP_DEF(mtp_silu)
  MTP_DEF(mtp_rnn_in)
  MTP_DEF(mtp_gates)
  MTP_DEF(mtp_pick)
#undef MTP_DEF
  m.attr("MTP_PICK_BLOCKS") = ik::mtp_pick_blocks();

  m.def("lm_head", [](const py::dict& d, uintptr_t stream) {
    ik::LmHeadParams p = {};
    p.rows = dint(d, "rows", 1);
    p.in_stride = dint(d, "in_stride", 3 * ik::kHidden);
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
    p.normed_out = dptr<uint16_t>(d, "normed_out");
    p.weight_fp8 = dptr<const uint8_t>(d, "weight_fp8");
    p.weight_scale = dptr<const float>(d, "weight_scale");
    p.prefetch = dprefetch(d);
    ik::lm_head(p, to_stream(stream));
  });
  m.attr("COMMIT_FIELDS") = ik::kCommitFields;
  m.def("accept", [](const py::dict& d, uintptr_t stream) {
    ik::AcceptParams p = {};
    p.rows = dint(d, "rows");
    p.drafts = dptr<const int>(d, "drafts");
    p.target = dptr<const int>(d, "target");
    p.pos = dptr<const int>(d, "pos");
    p.accepted = dptr<int>(d, "accepted");
    p.tokens = dptr<int>(d, "tokens");
    p.next_token = dptr<int>(d, "next_token");
    p.next_pos = dptr<int>(d, "next_pos");
    p.log = dptr<int>(d, "log");
    p.log_count = dptr<int>(d, "log_count");
    ik::accept(p, to_stream(stream));
  });
  m.def("commit", [](const py::dict& d, uintptr_t stream) {
    ik::CommitParams p = {};
    p.visits = dint(d, "visits");
    p.table = dptr<const uint64_t>(d, "table");
    p.accepted = dptr<const int>(d, "accepted");
    p.pos = dptr<const int>(d, "pos");
    ik::commit(p, to_stream(stream));
  });
}
