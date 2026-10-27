"""FlashRT -- GR00T N1.5 policy forward: the stage sequence every target runs.

GR00T N1.5 (``GR00T_N1_5``) maps camera frames, a task prompt and the robot
state to an action chunk::

    pixel tiles [V, 3, 224, 224]
      -> SigLIP vision tower (27 layers, 1152 wide, 256 patches per tile)
      -> mlp1 (1152 -> 2048)                        image token rows
    prompt ids with the image rows merged in
      -> Qwen3 language model (12 layers, final RMSNorm)  backbone features
      -> vlln + 4-layer VL self-attention (2048 wide)     vl features
    state -> embodiment MLP                          state token
    noise -> 4 flow-matching steps of
      action encoder -> [state, 32 future tokens, actions]
      -> 16-layer DiT (cross-attention to the vl features on even layers,
         self-attention on odd ones, AdaLayerNorm on the step)
      -> action decoder -> Euler update               actions [H, action_dim]

The sequence is the same on every target. Hardware enters through a
:class:`GrootN15Target` (``models/groot_n15/pipeline_<hw>.py``), which names
the device the path was validated on and builds the attention bindings and
the projection lowering. Weights and buffers belong to the frontend and
arrive here as raw device pointers, so a call issues the same launches with
the same arguments every time and can be captured into a CUDA graph.

Numerical contract -- the upstream policy under bf16 autocast:

* In the BF16 tier every projection with a bias is torch's addmm algorithm
  (``groot_n15_linear_bias_bf16``) and every projection without one (the
  Qwen3 language model) torch's matmul algorithm
  (``bf16_matmul_cublaslt_ref_bf16``); q/k/v and gate/up stay separate
  projections, as in the reference. The embodiment MLPs are torch.bmm
  (``groot_n15_bmm_bf16``) followed by a bf16 bias add.
* The Qwen3 decoder runs the dense-Qwen3 kernels of
  ``models/qwen3/prefill_bf16`` (HF-rounding RMSNorm, q/k norm + RoPE, SiLU
  gate, FA2 causal attention) in the same order.
* LayerNorm computes in fp32 and is rounded to bf16 where the next Linear
  would cast it. The vision tower, language model and DiT keep a bf16
  residual stream; the VL self-attention keeps the fp32 one the reference
  builds from vlln's fp32 output.
* The timestep-dependent AdaLayerNorm tables, the sinusoidal step encoding,
  the position embeddings and the future tokens do not depend on the inputs;
  the frontend builds them once from the checkpoint with the reference's
  expressions.
* What is close to, rather than equal to, the reference: attention (the
  target's backend), the patch projection (an im2col GEMM where the
  reference runs cuDNN's convolution) and LayerNorm (another reduction
  order; 1-ulp differences on a small fraction of the elements).

Precision enters through the projection lowering: ``input(x, M, K)``
prepares an activation once for the projections that read it,
``linear(a, w, out, M)`` projects with a bias and ``matmul(a, w, out, M)``
without one. Which projections a tier quantizes is the target's
``fp8_projections``; the stage order is the same in every tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

#: flash_rt_kernels entries the forward launches (projection lowerings add
#: their own, see ``_linear``).
KERNEL_SYMBOLS = (
    "patch_im2col_nchw_bf16",
    "embedding_lookup_bf16",
    "rms_norm_hf_bf16",
    "residual_add_rms_norm_hf_bf16",
    "qk_rms_norm_rope_hf_bf16",
    "silu_mul_hf_bf16",
)

#: flash_rt_groot_n15 entries the forward launches.
GROOT_N15_SYMBOLS = (
    "groot_n15_linear_bias_bf16",
    "groot_n15_bmm_bf16",
    "groot_n15_layer_norm_bf16",
    "groot_n15_layer_norm_to_f32",
    "groot_n15_residual_layer_norm_f32",
    "groot_n15_residual_add_f32_to_bf16",
    "groot_n15_ada_layer_norm_bf16",
    "groot_n15_add_rows_bf16",
    "groot_n15_bias_act_bf16",
    "groot_n15_concat_bias_bf16",
    "groot_n15_gelu_tanh_bf16",
    "groot_n15_quant_rows_fp8",
    "groot_n15_rescale_rows_bf16",
    "groot_n15_actions_init",
    "groot_n15_euler_step",
)

BIAS_ONLY, BIAS_RELU, BIAS_SWISH = 0, 1, 2


@dataclass(frozen=True)
class GrootN15Dims:
    """Static shape contract of a GR00T N1.5 checkpoint (``GR00T_N1_5``).

    Defaults are the released GR00T-N1.5-3B geometry; the frontend reads
    every field from the checkpoint config and refuses configurations this
    sequence does not implement.
    """

    # SigLIP vision tower (no pixel shuffle, one-layer mlp1 connector)
    image_size: int = 224
    patch_size: int = 14
    vision_hidden: int = 1152
    vision_layers: int = 27
    vision_heads: int = 16
    vision_intermediate: int = 4304
    vision_eps: float = 1e-6
    # Qwen3 language model, truncated to select_layer layers
    llm_hidden: int = 2048
    llm_layers: int = 12
    llm_intermediate: int = 6144
    llm_q_heads: int = 16
    llm_kv_heads: int = 8
    llm_head_dim: int = 128
    llm_vocab: int = 151_680
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    # vlln + VL self-attention
    vl_layers: int = 4
    vl_heads: int = 32
    vl_head_dim: int = 64
    vl_eps: float = 1e-5
    # DiT action head
    dit_layers: int = 16
    dit_heads: int = 32
    dit_head_dim: int = 48
    dit_output: int = 1024
    dit_eps: float = 1e-5          # norm3 (the DiT config's norm_eps)
    dit_ada_eps: float = 1e-5      # AdaLayerNorm: fixed in the reference
    dit_out_eps: float = 1e-6      # norm_out: fixed in the reference
    # flow-matching head
    state_dim: int = 64
    action_dim: int = 32
    head_hidden: int = 1024
    future_tokens: int = 32
    action_horizon: int = 16
    denoising_steps: int = 4
    timestep_buckets: int = 1000

    @property
    def patches_per_tile(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_features(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @property
    def vision_head_dim(self) -> int:
        return self.vision_hidden // self.vision_heads

    @property
    def vl_hidden(self) -> int:
        return self.vl_heads * self.vl_head_dim

    @property
    def vl_intermediate(self) -> int:
        return 4 * self.vl_hidden

    @property
    def dit_hidden(self) -> int:
        return self.dit_heads * self.dit_head_dim

    @property
    def dit_intermediate(self) -> int:
        return 4 * self.dit_hidden

    @property
    def dit_tokens(self) -> int:
        """Rows of the DiT sequence: state token, future tokens, actions."""
        return 1 + self.future_tokens + self.action_horizon

    def step_bucket(self, step: int) -> int:
        """Discretized timestep of denoising step ``step`` (reference rule)."""
        return int(step / float(self.denoising_steps) * self.timestep_buckets)


@dataclass(frozen=True)
class GrootN15Target:
    """One device the GR00T N1.5 forward is bound to.

    ``name`` is the ``flash_rt.hardware`` arch key; ``compute_capability`` is
    the exact generation the path was validated on -- other devices are
    refused, not assumed to behave the same. ``make_attention(site, **kw)``
    builds the binding of one attention site; ``make_linear(precision, g,
    fvk, **scratch)`` returns the projection lowering of a precision tier and
    ``linear_symbols(precision)`` the kernel entries it launches.
    ``fp8_projections`` names the projection groups the FP8 tier packs to
    E4M3 (``vision``, ``vl``, ``cross_kv``, ``llm_attention``,
    ``llm_mlp``). ``pdl`` is whether the kernels may launch with
    programmatic dependent launch.
    """

    name: str
    compute_capability: tuple[int, int]
    attention: str
    precisions: tuple[str, ...]
    pdl: bool
    make_attention: Callable[..., Any]
    make_linear: Callable[..., Any]
    linear_symbols: Callable[[str], tuple[str, ...]]
    fp8_projections: frozenset[str]
    build_hint: str


def require_kernels(fvk: Any, g: Any, target: GrootN15Target, precision: str) -> None:
    """Raise unless both kernel modules carry every entry the forward and the
    precision tier's lowerings use."""
    missing = [n for n in KERNEL_SYMBOLS + target.linear_symbols(precision)
               if not hasattr(fvk, n)]
    missing += [n for n in GROOT_N15_SYMBOLS if not hasattr(g, n)]
    if missing:
        raise RuntimeError(
            f"GR00T N1.5 on {target.name} needs kernel entries {missing}, "
            f"which this build does not carry. {target.build_hint}")


class GrootN15Forward:
    """The policy stage sequence over frontend-owned buffers (batch 1).

    ``weights`` maps each stage to its pointer tables (see the frontend's
    checkpoint adapter for the key list); ``tables`` holds the input-
    independent tables; ``bufs`` the persistent activations, sized for the
    frontend's longest prompt. ``attention`` maps the site names ``vision``,
    ``language``, ``vl``, ``dit_cross`` and ``dit_self`` to their bindings;
    ``linear`` is the projection lowering of the tier.
    ``torch_reduce`` selects the reduction tree of the HF-rounding Qwen3
    kernels.
    """

    def __init__(self, g: Any, fvk: Any, dims: GrootN15Dims, num_views: int,
                 weights: Mapping[str, Any], tables: Mapping[str, Any],
                 bufs: Mapping[str, int], attention: Mapping[str, Any],
                 linear: Any, *, pdl: bool,
                 torch_reduce: int) -> None:
        for key, count in (("vision_layers", dims.vision_layers),
                           ("llm_layers", dims.llm_layers),
                           ("vl_layers", dims.vl_layers),
                           ("dit_layers", dims.dit_layers)):
            if len(weights[key]) != count:
                raise ValueError(f"{key}: {len(weights[key])} tables for {count} layers")
        if len(tables["ada"]) != dims.denoising_steps:
            raise ValueError("AdaLayerNorm tables do not cover every denoising step")
        self._g = g
        self._fvk = fvk
        self._d = dims
        self._views = int(num_views)
        self._w = weights
        self._t = tables
        self._b = dict(bufs)
        self._attn = dict(attention)
        self._lin = linear
        self._pdl = bool(pdl)
        self._tree = int(torch_reduce)

    # ------------------------------------------------------------- stages

    def vision(self, stream: int) -> None:
        """Pixel tiles ``bufs['pixels']`` -> image rows ``bufs['image_rows']``."""
        g, d, b, w, lin = self._g, self._d, self._b, self._w["vision"], self._lin
        P, D, F = d.patches_per_tile, d.vision_hidden, d.vision_intermediate
        M = self._views * P
        eps = d.vision_eps
        self._fvk.patch_im2col_nchw_bf16(b["pixels"], b["v_cols"], self._views, 3,
                                         d.image_size, d.image_size, d.patch_size, stream)
        lin(lin.input(b["v_cols"], M, d.patch_features, stream), w["patch"], b["v_t"], M, stream)
        g.groot_n15_add_rows_bf16(b["v_t"], 0, w["position"], b["v_h"], M, D, P, stream)
        layers = self._w["vision_layers"]
        ln = layers[0]["ln1"]
        g.groot_n15_layer_norm_bf16(b["v_h"], 0, 0, ln[0], ln[1], b["v_xn"], M, D, eps, stream)
        attn = self._attn["vision"]
        for i, lw in enumerate(layers):
            a = lin.input(b["v_xn"], M, D, stream)
            lin(a, lw["q"], b["v_q"], M, stream)
            lin(a, lw["k"], b["v_k"], M, stream)
            lin(a, lw["v"], b["v_v"], M, stream)
            attn.run(P, P, stream)
            lin(lin.input(b["v_o"], M, D, stream), lw["o"], b["v_t"], M, stream)
            ln = lw["ln2"]
            g.groot_n15_layer_norm_bf16(b["v_h"], b["v_t"], b["v_h"], ln[0], ln[1], b["v_xn"],
                                        M, D, eps, stream)
            lin(lin.input(b["v_xn"], M, D, stream), lw["fc1"], b["v_f"], M, stream)
            g.groot_n15_gelu_tanh_bf16(b["v_f"], M * F, stream)
            lin(lin.input(b["v_f"], M, F, stream), lw["fc2"], b["v_t"], M, stream)
            ln = layers[i + 1]["ln1"] if i + 1 < len(layers) else w["post_ln"]
            g.groot_n15_layer_norm_bf16(b["v_h"], b["v_t"], b["v_h"], ln[0], ln[1], b["v_xn"],
                                        M, D, eps, stream)
        lin(lin.input(b["v_xn"], M, D, stream), w["mlp1"], b["image_rows"], M, stream)

    def language(self, S: int, stream: int) -> None:
        """Prompt ids (image positions remapped to the image rows) -> backbone
        features ``bufs['features']``, final-normed: the Qwen3 decoder."""
        f, d, b, lin = self._fvk, self._d, self._b, self._lin
        H, I = d.llm_hidden, d.llm_intermediate
        NQ, NKV, HD = d.llm_q_heads, d.llm_kv_heads, d.llm_head_dim
        QD, KD = NQ * HD, NKV * HD
        eps, tree, pdl = d.rms_norm_eps, self._tree, self._pdl
        layers = self._w["llm_layers"]
        attn = self._attn["language"]
        f.embedding_lookup_bf16(b["ids"], self._w["llm"]["embed"], b["lm_h"], S, H, stream)
        f.rms_norm_hf_bf16(b["lm_h"], layers[0]["in_norm"], b["features"], S, H, eps, tree,
                           pdl, stream)
        last = len(layers) - 1
        for i, w in enumerate(layers):
            a = lin.input(b["features"], S, H, stream)
            lin.matmul(a, w["q"], b["lm_q"], S, stream)
            lin.matmul(a, w["k"], b["lm_k"], S, stream)
            lin.matmul(a, w["v"], b["lm_v"], S, stream)
            f.qk_rms_norm_rope_hf_bf16(
                b["lm_q"], QD, NQ, b["lm_k"], KD, NKV, w["q_norm"], w["k_norm"],
                b["cos"], b["sin"], S, HD, eps, tree, pdl, stream)
            attn.run(S, stream)
            lin.matmul(lin.input(b["lm_o"], S, QD, stream), w["o"], b["lm_t"], S, stream)
            f.residual_add_rms_norm_hf_bf16(b["lm_h"], b["lm_t"], w["post_norm"],
                                            b["features"], S, H, eps, tree, pdl, stream)
            a = lin.input(b["features"], S, H, stream)
            lin.matmul(a, w["gate"], b["lm_g"], S, stream)
            lin.matmul(a, w["up"], b["lm_u"], S, stream)
            f.silu_mul_hf_bf16(b["lm_g"], b["lm_u"], b["lm_g"], S * I, pdl, stream)
            lin.matmul(lin.input(b["lm_g"], S, I, stream), w["down"], b["lm_t"], S, stream)
            nxt = self._w["llm"]["final_norm"] if i == last else layers[i + 1]["in_norm"]
            f.residual_add_rms_norm_hf_bf16(b["lm_h"], b["lm_t"], nxt, b["features"], S, H,
                                            eps, tree, pdl, stream)

    def vl_adapter(self, S: int, stream: int) -> None:
        """Backbone features -> vl features ``bufs['vl']`` (bf16, the dtype the
        DiT's K/V projections read)."""
        g, d, b, lin = self._g, self._d, self._b, self._lin
        D, F, eps = d.vl_hidden, d.vl_intermediate, d.vl_eps
        w = self._w["vl"]
        g.groot_n15_layer_norm_to_f32(b["features"], w["vlln"][0], w["vlln"][1], b["l_h32"],
                                      S, D, eps, stream)
        layers = self._w["vl_layers"]
        ln = layers[0]["norm1"]
        g.groot_n15_residual_layer_norm_f32(b["l_h32"], 0, ln[0], ln[1], b["l_xn"], S, D, eps,
                                            stream)
        attn = self._attn["vl"]
        for i, lw in enumerate(layers):
            a = lin.input(b["l_xn"], S, D, stream)
            lin(a, lw["q"], b["l_q"], S, stream)
            lin(a, lw["k"], b["l_k"], S, stream)
            lin(a, lw["v"], b["l_v"], S, stream)
            attn.run(S, S, stream)
            lin(lin.input(b["l_o"], S, D, stream), lw["o"], b["l_t"], S, stream)
            ln = lw["norm3"]
            g.groot_n15_residual_layer_norm_f32(b["l_h32"], b["l_t"], ln[0], ln[1], b["l_xn"],
                                                S, D, eps, stream)
            lin(lin.input(b["l_xn"], S, D, stream), lw["ff1"], b["l_f"], S, stream)
            g.groot_n15_gelu_tanh_bf16(b["l_f"], S * F, stream)
            lin(lin.input(b["l_f"], S, F, stream), lw["ff2"], b["l_t"], S, stream)
            if i + 1 < len(layers):
                ln = layers[i + 1]["norm1"]
                g.groot_n15_residual_layer_norm_f32(b["l_h32"], b["l_t"], ln[0], ln[1],
                                                    b["l_xn"], S, D, eps, stream)
            else:
                g.groot_n15_residual_add_f32_to_bf16(b["l_h32"], b["l_t"], b["vl"], S * D,
                                                     stream)

    def cross_kv(self, S: int, stream: int) -> None:
        """K/V of every cross-attention DiT layer over the vl features. They do
        not change across denoising steps, so they are projected once."""
        lin, b = self._lin, self._b
        a = lin.input(b["vl"], S, self._d.vl_hidden, stream)
        for lw in self._w["dit_layers"]:
            if lw["cross"]:
                lin(a, lw["k"], lw["k_cache"], S, stream)
                lin(a, lw["v"], lw["v_cache"], S, stream)

    def state(self, stream: int) -> None:
        """``bufs['state']`` -> the state token, row 0 of ``bufs['sa']``."""
        g, d, b, w = self._g, self._d, self._b, self._w["head"]
        Hh, De = d.head_hidden, d.dit_hidden
        g.groot_n15_bmm_bf16(b["state"], w["state1"][0], b["s_t1"], 1, Hh, d.state_dim, stream)
        g.groot_n15_bias_act_bf16(b["s_t1"], w["state1"][1], b["s_h"], 1, Hh, BIAS_RELU, stream)
        g.groot_n15_bmm_bf16(b["s_h"], w["state2"][0], b["s_t2"], 1, De, Hh, stream)
        g.groot_n15_bias_act_bf16(b["s_t2"], w["state2"][1], b["sa"], 1, De, BIAS_ONLY, stream)

    def denoise(self, S: int, stream: int) -> None:
        """``bufs['noise']`` -> actions ``bufs['actions']`` (fp32 [H, action_dim])."""
        g, d, b = self._g, self._d, self._b
        H, A = d.action_horizon, d.action_dim
        g.groot_n15_actions_init(b["noise"], b["actions"], b["a_bf16"], H * A, stream)
        dt = 1.0 / d.denoising_steps
        for step in range(d.denoising_steps):
            self._action_tokens(step, stream)
            self._dit(step, S, stream)
            self._decode(stream)
            g.groot_n15_euler_step(b["pred"] + (d.dit_tokens - H) * A * 2, b["actions"],
                                   b["a_bf16"], dt, H * A, stream)

    def run(self, S: int, stream: int) -> None:
        """One policy call for a prompt of ``S`` tokens (image rows included)."""
        self.vision(stream)
        self.language(S, stream)
        self.vl_adapter(S, stream)
        self.cross_kv(S, stream)
        self.state(stream)
        self.denoise(S, stream)

    # ----------------------------------------------------------- denoising

    def _action_tokens(self, step: int, stream: int) -> None:
        """Actions -> action tokens, rows ``1 + future_tokens ..`` of ``sa``."""
        g, d, b, w = self._g, self._d, self._b, self._w["head"]
        H, A, De = d.action_horizon, d.action_dim, d.dit_hidden
        g.groot_n15_bmm_bf16(b["a_bf16"], w["act1"][0], b["a_t1"], H, De, A, stream)
        g.groot_n15_concat_bias_bf16(b["a_t1"], w["act1"][1], self._t["tau"][step], b["a_cat"],
                                     H, De, De, stream)
        g.groot_n15_bmm_bf16(b["a_cat"], w["act2"][0], b["a_t2"], H, De, 2 * De, stream)
        g.groot_n15_bias_act_bf16(b["a_t2"], w["act2"][1], b["a_h"], H, De, BIAS_SWISH, stream)
        g.groot_n15_bmm_bf16(b["a_h"], w["act3"][0], b["a_t3"], H, De, De, stream)
        rows = b["sa"] + (1 + d.future_tokens) * De * 2
        g.groot_n15_add_rows_bf16(b["a_t3"], w["act3"][1], self._t["position"], rows, H, De, H,
                                  stream)

    def _dit(self, step: int, S: int, stream: int) -> None:
        """The DiT over ``sa`` for one step; leaves ``bufs['d_out']``."""
        g, d, b, lin = self._g, self._d, self._b, self._lin
        T, D, F = d.dit_tokens, d.dit_hidden, d.dit_intermediate
        eps = d.dit_eps
        ada = self._t["ada"][step]
        cross, self_attn = self._attn["dit_cross"], self._attn["dit_self"]
        layers = self._w["dit_layers"]
        ada_eps = d.dit_ada_eps
        g.groot_n15_ada_layer_norm_bf16(b["sa"], 0, 0, ada[0][0], ada[0][1], b["d_xn"], T, D,
                                        ada_eps, stream)
        res_in = b["sa"]
        for i, lw in enumerate(layers):
            a = lin.input(b["d_xn"], T, D, stream)
            lin(a, lw["q"], b["d_q"], T, stream)
            if lw["cross"]:
                cross.run_kv(lw["k_cache"], lw["v_cache"], T, S, stream)
            else:
                lin(a, lw["k"], b["d_k"], T, stream)
                lin(a, lw["v"], b["d_v"], T, stream)
                self_attn.run(T, T, stream)
            lin(lin.input(b["d_o"], T, D, stream), lw["o"], b["d_t"], T, stream)
            g.groot_n15_layer_norm_bf16(res_in, b["d_t"], b["d_h"], 0, 0, b["d_xn"], T, D, eps,
                                        stream)
            res_in = b["d_h"]
            lin(lin.input(b["d_xn"], T, D, stream), lw["ff1"], b["d_f"], T, stream)
            g.groot_n15_gelu_tanh_bf16(b["d_f"], T * F, stream)
            lin(lin.input(b["d_f"], T, F, stream), lw["ff2"], b["d_t"], T, stream)
            if i + 1 < len(layers):
                a1, sh = ada[i + 1]
                g.groot_n15_ada_layer_norm_bf16(b["d_h"], b["d_t"], b["d_h"], a1, sh, b["d_xn"],
                                                T, D, ada_eps, stream)
        a1, sh = self._t["ada_out"][step]
        g.groot_n15_ada_layer_norm_bf16(b["d_h"], b["d_t"], b["d_h"], a1, sh, b["d_xn"], T, D,
                                        d.dit_out_eps, stream)
        lin(lin.input(b["d_xn"], T, D, stream), self._w["head"]["proj_out"], b["d_out"], T,
            stream)

    def _decode(self, stream: int) -> None:
        """DiT output -> velocity rows ``bufs['pred']`` (bf16 [T, action_dim])."""
        g, d, b, w = self._g, self._d, self._b, self._w["head"]
        T, Hh, A, Do = d.dit_tokens, d.head_hidden, d.action_dim, d.dit_output
        g.groot_n15_bmm_bf16(b["d_out"], w["dec1"][0], b["p_t1"], T, Hh, Do, stream)
        g.groot_n15_bias_act_bf16(b["p_t1"], w["dec1"][1], b["p_h"], T, Hh, BIAS_RELU, stream)
        g.groot_n15_bmm_bf16(b["p_h"], w["dec2"][0], b["p_t2"], T, A, Hh, stream)
        g.groot_n15_bias_act_bf16(b["p_t2"], w["dec2"][1], b["pred"], T, A, BIAS_ONLY, stream)
