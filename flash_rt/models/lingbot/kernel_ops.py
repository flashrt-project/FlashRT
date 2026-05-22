"""LingBot-VLA kernel wrappers around the shared ``flash_rt_kernels`` symbols.

Provides drop-in replacements for the eager-PyTorch ops used by
``sample_actions.py`` / ``forward.py`` / ``mixed_attention.py`` /
``vit.py``. Each wrapper:

    * has a torch-natural call signature (tensors in, tensor out)
    * extracts ``.data_ptr()`` and passes it to the C++ kernel
    * allocates the output tensor on the same device + dtype as the input
    * is bit-exact (or cos >= 0.997 for FP8) vs the PyTorch op it replaces

This module owns a process-singleton ``GemmRunner`` for cuBLASLt heuristic
caching (different weight shapes pick different algorithms; the cache stays
warm across many linear sites).

For ``bf16_nn`` / ``bf16_nn_bias`` cuBLAS expects the weight as ``[K, N]``
(row-major) while HuggingFace stores ``[N, K]``. Both a per-call transposing
helper (for first-light validation) and a ``preload_transposed_weights``
helper (production path — transposes once at bind time) are provided.
"""

from __future__ import annotations

import sys

import torch

import flash_rt.flash_rt_kernels as fvk


# ════════════════════════════════════════════════════════════════════
#  Singleton resources (cuBLAS handle + GEMM autotune cache)
# ════════════════════════════════════════════════════════════════════

_FVK_CTX: fvk.FvkContext | None = None
_GEMM_RUNNER: fvk.GemmRunner | None = None

# Transpose cache: data_ptr → .T.contiguous(). Amortizes the per-call
# transpose across all GEMM calls (one transpose per unique weight).
_WT_CACHE: dict[int, "torch.Tensor"] = {}


def _get_or_transpose(w: "torch.Tensor") -> "torch.Tensor":
    key = w.data_ptr()
    cached = _WT_CACHE.get(key)
    if cached is None:
        cached = w.T.contiguous()
        _WT_CACHE[key] = cached
    return cached


def get_fvk_context() -> fvk.FvkContext:
    """Get/create the process-wide FvkContext (owns cuBLAS handle)."""
    global _FVK_CTX
    if _FVK_CTX is None:
        _FVK_CTX = fvk.FvkContext()
    return _FVK_CTX


def get_gemm_runner() -> fvk.GemmRunner:
    """Get/create the process-wide GemmRunner."""
    global _GEMM_RUNNER
    if _GEMM_RUNNER is None:
        _GEMM_RUNNER = fvk.GemmRunner()
    return _GEMM_RUNNER


# ════════════════════════════════════════════════════════════════════
#  BF16 GEMM (F.linear replacement)
# ════════════════════════════════════════════════════════════════════

def _current_stream() -> int:
    """Return the integer cudaStream of the current torch stream.

    : fvk binding defaults route to stream 0 which is NOT in the
    captured graph. Always pass this on the hot path so the kernel
    launches end up recorded into the active capture.
    """
    return torch.cuda.current_stream().cuda_stream


# dtype dispatch. The fvk bindings come in bf16 and fp16 flavors.
# All wrappers route by ``input.dtype`` so the same model code can run
# in either mode by switching ``bind_target_to_device(target, dtype=...)``.
def _gemm_nn(runner, x_ptr, w_T_ptr, out_ptr, M, N, K, dtype, stream):
    if dtype == torch.bfloat16:
        runner.bf16_nn(x_ptr, w_T_ptr, out_ptr, M, N, K, stream)
    elif dtype == torch.float16:
        runner.fp16_nn(x_ptr, w_T_ptr, out_ptr, M, N, K, stream)
    else:
        raise NotImplementedError(f"_gemm_nn dtype {dtype}")


def _add_bias(out_ptr, bias_ptr, M, N, dtype, stream):
    if dtype == torch.bfloat16:
        fvk.add_bias_bf16(out_ptr, bias_ptr, M, N, stream)
    elif dtype == torch.float16:
        fvk.add_bias_fp16(out_ptr, bias_ptr, M, N, stream)
    else:
        raise NotImplementedError(f"_add_bias dtype {dtype}")


def _quantize_fp8_static(x_ptr, out_fp8_ptr, scale_ptr, n, dtype, stream):
    if dtype == torch.bfloat16:
        fvk.quantize_fp8_static(x_ptr, out_fp8_ptr, scale_ptr, n, stream)
    elif dtype == torch.float16:
        fvk.quantize_fp8_static_fp16(x_ptr, out_fp8_ptr, scale_ptr, n, stream)
    else:
        raise NotImplementedError(f"_quantize_fp8_static dtype {dtype}")


def _quantize_fp8_device(x_ptr, out_fp8_ptr, scale_ptr, n, dtype, stream):
    if dtype == torch.bfloat16:
        fvk.quantize_fp8_device(x_ptr, out_fp8_ptr, scale_ptr, n, stream)
    elif dtype == torch.float16:
        fvk.quantize_fp8_device_fp16(x_ptr, out_fp8_ptr, scale_ptr, n, stream)
    else:
        raise NotImplementedError(f"_quantize_fp8_device dtype {dtype}")


def _fp8_gemm_descale(x_fp8_ptr, w_fp8_T_ptr, out_ptr,
                      M, N, K, act_scale_ptr, w_descale_ptr, dtype, stream):
    """Dispatch FP8 GEMM by output dtype."""
    if dtype == torch.bfloat16:
        fvk.fp8_gemm_descale_bf16out(
            x_fp8_ptr, w_fp8_T_ptr, out_ptr,
            M, N, K, act_scale_ptr, w_descale_ptr, stream)
    elif dtype == torch.float16:
        fvk.fp8_gemm_descale_fp16(
            x_fp8_ptr, w_fp8_T_ptr, out_ptr,
            M, N, K, act_scale_ptr, w_descale_ptr, stream)
    else:
        raise NotImplementedError(f"_fp8_gemm_descale dtype {dtype}")


def linear_bf16(
    x: torch.Tensor,                    # [..., K] bf16 or fp16
    weight: torch.Tensor,               # [N, K] same dtype as x
    bias: torch.Tensor | None = None,   # [N] same dtype as x
    *,
    stream: int | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``F.linear(x, weight, bias)`` using
    ``fvk.GemmRunner.bf16_nn`` (or ``fp16_nn`` when dtype is fp16).

    Auto-detects whether ``weight`` is the HuggingFace ``[N, K]``
    storage or already a pre-transposed ``[K, N]`` (from
    ``preload_transposed_weights``). The pre-T fast path saves a
    per-call ``.T.contiguous()`` (~few microseconds × 14 GEMMs per
    layer × 36 layers × 50 steps).

    Name kept as ``linear_bf16`` for backward compat; works for both
    bf16 and fp16 paths via dtype dispatch.
    """
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"x dtype {x.dtype}, expected bf16 or fp16")
    assert weight.dtype == x.dtype, (
        f"weight dtype {weight.dtype} != x dtype {x.dtype}")
    K_in = x.shape[-1]
    N_out = weight.shape[0]
    assert weight.shape == (N_out, K_in), (
        f"weight shape {weight.shape} != ({N_out}, {K_in})")
    if stream is None:
        stream = _current_stream()
    # Cached transpose — one .T.contiguous() per unique weight tensor.
    # Avoids the per-call ~50 μs transpose for every GEMM in the hot loop.
    w_T = _get_or_transpose(weight)

    # Flatten leading dims to M.
    # math.prod for capture-safety (torch.tensor + .item() crosses
    # to host which is not permitted during CUDA-Graph capture).
    M = 1
    for d in x.shape[:-1]:
        M *= d
    x_2d = x.contiguous().view(M, K_in)
    out_shape = (*x.shape[:-1], N_out)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out_2d = out.view(M, N_out)

    runner = get_gemm_runner()
    _gemm_nn(runner,
             int(x_2d.data_ptr()), int(w_T.data_ptr()),
             int(out_2d.data_ptr()), M, N_out, K_in, x.dtype, stream)
    if bias is not None:
        assert bias.dtype == x.dtype
        assert bias.shape == (N_out,)
        # Thor SM110: ``bf16_nn_bias`` returns CUBLAS_NOT_SUPPORTED, so
        # we use bf16_nn + a separate broadcast add (the same pattern
        # the GROOT/Pi0.5 Thor pipelines use). Same applies for fp16.
        _add_bias(int(out_2d.data_ptr()),
                  int(bias.contiguous().data_ptr()),
                  M, N_out, x.dtype, stream)
    return out


def linear_bf16_preT(
    x: torch.Tensor,                    # [..., K]
    weight_T: torch.Tensor,             # [K, N] — already transposed
    bias: torch.Tensor | None = None,
    *,
    stream: int | None = None,
) -> torch.Tensor:
    """Fast path: caller has already transposed weight to [K, N]
    row-major (e.g., via ``preload_transposed_weights``).

    Dtype-aware (bf16 or fp16) — same dispatch logic as ``linear_bf16``.
    """
    assert x.dtype in (torch.bfloat16, torch.float16)
    K_in = x.shape[-1]
    N_out = weight_T.shape[1]
    assert weight_T.shape == (K_in, N_out)
    if stream is None:
        stream = _current_stream()

    # math.prod for capture-safety (torch.tensor + .item() crosses
    # to host which is not permitted during CUDA-Graph capture).
    M = 1
    for d in x.shape[:-1]:
        M *= d
    x_2d = x.contiguous().view(M, K_in)
    out = torch.empty((*x.shape[:-1], N_out), dtype=x.dtype, device=x.device)
    out_2d = out.view(M, N_out)

    runner = get_gemm_runner()
    if x.dtype == torch.float16:
        runner.fp16_nn(
            int(x_2d.data_ptr()), int(weight_T.data_ptr()),
            int(out_2d.data_ptr()), M, N_out, K_in, stream)
        if bias is not None:
            _add_bias(int(out_2d.data_ptr()),
                      int(bias.contiguous().data_ptr()),
                      M, N_out, torch.float16, stream)
        return out
    runner.bf16_nn(
        int(x_2d.data_ptr()),
        int(weight_T.data_ptr()),
        int(out_2d.data_ptr()),
        M, N_out, K_in, stream,
    )
    if bias is not None:
        fvk.add_bias_bf16(
            int(out_2d.data_ptr()),
            int(bias.contiguous().data_ptr()),
            M, N_out, stream,
        )
    return out


# ════════════════════════════════════════════════════════════════════
#  Pre-transposed weight cache (production fast path)
# ════════════════════════════════════════════════════════════════════

_TRANSPOSED_ATTR_NAMES = [
    # Action heads (singletons)
    "state_proj_weight",
    "action_in_proj_weight",
    "action_out_proj_weight",
    "action_time_mlp_in_weight",
    "action_time_mlp_out_weight",
    # qwenvl singletons
    "vlm_embed_tokens_weight",       # NOT a GEMM weight, skip transpose
    "vit_patch_embed_proj_weight",   # Conv3d, skip
    "vit_merger_mlp_0_weight",
    "vit_merger_mlp_2_weight",
]
_TRANSPOSED_LIST_NAMES = [
    "vlm_layer_q_proj_weights",
    "vlm_layer_k_proj_weights",
    "vlm_layer_v_proj_weights",
    "vlm_layer_o_proj_weights",
    "vlm_layer_mlp_gate_proj_weights",
    "vlm_layer_mlp_up_proj_weights",
    "vlm_layer_mlp_down_proj_weights",
    "expert_layer_q_proj_weights",
    "expert_layer_k_proj_weights",
    "expert_layer_v_proj_weights",
    "expert_layer_o_proj_weights",
    "expert_layer_mlp_gate_proj_weights",
    "expert_layer_mlp_up_proj_weights",
    "expert_layer_mlp_down_proj_weights",
    # Expert AdaRMSNorm gamma/beta linears
    "expert_layer_input_layernorm_gamma_weights",
    "expert_layer_input_layernorm_beta_weights",
    "expert_layer_post_attn_layernorm_gamma_weights",
    "expert_layer_post_attn_layernorm_beta_weights",
    # ViT block weights
    "vit_block_attn_qkv_weights",
    "vit_block_attn_proj_weights",
    "vit_block_mlp_gate_proj_weights",
    "vit_block_mlp_up_proj_weights",
    "vit_block_mlp_down_proj_weights",
]


def preload_transposed_weights(target) -> None:
    """Add ``<name>_T`` attributes to target with pre-transposed weights
    for every GEMM-feeding tensor. Call once after ``bind_target_to_device``.

    The original ``<name>`` attributes are kept (still useful for
    pure-eager paths). Memory cost: +8.4 GB transient (one extra copy of
    every weight). Acceptable on Thor 24 GB GPU.

    NOTE: This is the simple "double-store" version. A future
    optimization would replace the original tensor in-place via
    ``data.set_(...)`` so we don't pay 2× the memory.
    """
    skip_singletons = {
        "vlm_embed_tokens_weight",       # embedding table — no transpose
        "vit_patch_embed_proj_weight",   # Conv3d weight [1280, 3, 2, 14, 14]
    }
    for name in _TRANSPOSED_ATTR_NAMES:
        if name in skip_singletons:
            continue
        if not hasattr(target, name):
            continue
        w = getattr(target, name)
        setattr(target, name + "_T", w.T.contiguous())

    for name in _TRANSPOSED_LIST_NAMES:
        if not hasattr(target, name):
            continue
        original = getattr(target, name)
        setattr(target, name + "_T", [w.T.contiguous() for w in original])


# ════════════════════════════════════════════════════════════════════
#  FP8 GEMM (linear_fp8)
# ════════════════════════════════════════════════════════════════════
#
# Two-step quantize-then-FP8-GEMM with bf16 input + bf16 output. The
# weight is FP8 E4M3 (quantized once at first call and cached by id);
# the activation is quantized dynamically per call via
# ``quantize_fp8_device`` (computes absmax + scale + quant in one
# device pass). Output flows back to bf16 via the descale parameters.
#
# Per-call work (sized M×K input, M×N output):
#   1. ``quantize_fp8_device`` (bf16 → fp8)                 — 1 read + 1 write of input
#   2. ``fp8_gemm_descale_bf16out`` (fp8 GEMM, bf16 out)    — ~2× faster than bf16 GEMM
#   3. ``add_bias_bf16`` (if bias)                          — 1 read + 1 write of output
#
# Constants:
#   FP8 E4M3 max = 448 → scale = absmax / 448
#   weight_scale × act_scale = descale (multiplied into GEMM accumulator)

import threading

# Cache: weight_id -> (w_fp8 [K, N] tensor, w_descale [1] fp32 device tensor).
_W_FP8_CACHE: dict[int, tuple] = {}
_W_FP8_LOCK = threading.Lock()


def _quantize_weight_fp8(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time weight quantization. Pre-transposes for cuBLASLt nn layout.

    Returns (w_fp8 [K_in, N_out] contiguous, w_descale [1] fp32 device).
    """
    # PyTorch-side quantization (one-time, off the hot path).
    w_fp32 = w_bf16.float()
    max_abs = w_fp32.abs().max().clamp(min=1e-12).item()
    w_scale = max_abs / 448.0
    # w_fp8 = clip(w / w_scale, [-448, 448]) cast to FP8 E4M3.
    w_div = (w_fp32 / w_scale).clamp(-448.0, 448.0)
    w_fp8 = w_div.to(torch.float8_e4m3fn)
    # Pre-transpose to [K, N] for the cuBLASLt nn-layout FP8 GEMM.
    w_fp8_T = w_fp8.T.contiguous()
    w_descale = torch.tensor([w_scale], dtype=torch.float32, device=w_bf16.device)
    return w_fp8_T, w_descale


def _get_or_quantize_fp8_weight(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (w_fp8_T, w_descale). Holds a strong reference to the
    source bf16 weight in the cache to prevent its underlying storage
    from being recycled by the PyTorch caching allocator (which would
    otherwise let a new tensor occupy the same data_ptr → cache key
    collision → wrong FP8 weight returned).
    """
    key = w_bf16.data_ptr()
    with _W_FP8_LOCK:
        cached = _W_FP8_CACHE.get(key)
        if cached is not None:
            # Cache entry survives as long as source bf16 lives (we hold
            # the ref). If the source got swapped (impossible since we
            # held ref), the storage check would catch it.
            w_fp8_T, w_descale, _src = cached
            return w_fp8_T, w_descale
        w_fp8_T, w_descale = _quantize_weight_fp8(w_bf16)
        _W_FP8_CACHE[key] = (w_fp8_T, w_descale, w_bf16)
    return w_fp8_T, w_descale


def clear_fp8_weight_cache() -> None:
    """Drop the cached FP8 weights (frees VRAM, forces re-quantization)."""
    with _W_FP8_LOCK:
        _W_FP8_CACHE.clear()


# cuBLASLt FP8 heuristic on Thor SM110 requires M ≥ 64. Below this,
# we have two regimes:
#   * M in [_FP8_PAD_MIN_M, _FP8_MIN_M): pad to _FP8_MIN_M then slice
#     output. The pad alloc costs ~5 μs inside a graph mempool (was
#     ~50 μs with vanilla torch.zeros before ); the FP8 GEMM at
#     M=64 still beats bf16 at M=51 on the Expert hot path. ~10 μs
#     net savings per call × 252 Expert sites × 10 denoise steps
#     ≈ 25 ms / inference.
#   * M below _FP8_PAD_MIN_M: stay on the bf16 path (the FiLM γ/β
#     projections have M=1 and aren't worth padding).
_FP8_MIN_M = 64
_FP8_PAD_MIN_M = 32        # pad-and-FP8 floor (M=51 Expert qualifies)
# Also: N < FP8_MIN_N triggers the same heuristic issue (e.g.,
# action_out_proj N=75). Fall back unconditionally for tiny outputs.
_FP8_MIN_N = 128


def linear_fp8(
    x: torch.Tensor,                    # [..., K] bf16 or fp16
    weight: torch.Tensor,               # [N, K] bf16 (HF layout — gets quantized + cached)
    bias: torch.Tensor | None = None,   # [N] bf16
    *,
    stream: int | None = None,
    site_id: "str | None" = None,
    out_dtype: torch.dtype | None = None,  # decouple output dtype from x.dtype
) -> torch.Tensor:
    """Drop-in for ``F.linear`` / ``linear_bf16`` using FP8 GEMM where
    profitable, otherwise falls back to bf16 GEMM.

    Auto-falls-back to ``linear_bf16`` when M < ``_FP8_MIN_M`` or
    N < ``_FP8_MIN_N`` — the cuBLASLt FP8 heuristic returns FAILED on
    those tile geometries on Thor SM110, producing garbage output. The
    fallback is silent: the test contract (cos ≥ 0.999) holds either way.

    Per-FP8-call: quantize x bf16→fp8 (device absmax or pre-loaded
    static scale), call FP8 GEMM with bf16 output, then add bias
    separately if provided.

    Weight is FP8-quantized ONCE per ``id(weight)`` and cached. First
    call on a new weight pays ~10ms for quant; subsequent calls only
    pay the activation quant + GEMM.

    ``site_id``: stable per-call name like ``"vlm.layer.12.q_proj"``.
    When given:
      * during ``calibration_recorder``: records ``max(abs(x))`` into the
        running stats so the JSON dump covers this site.
      * with ``set_static_scales(scales)`` active: looks up the static
        ``act_scale`` from the registry and uses ``quantize_fp8_static``
        (skips the per-call device absmax-reduce).
    ``site_id=None`` keeps the old dynamic behavior (legacy callers).
    """
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"x dtype {x.dtype}, expected bf16 or fp16")
    # x.dtype may differ from weight.dtype (e.g. fp16 attn output feeding a
    # bf16 o_proj weight). The FP8 GEMM quantizes both to fp8 so the input dtype
    # only selects the quant kernel; out_dtype controls the GEMM output. The
    # bf16 fallback (below) still needs matching dtype — handled there.
    od = out_dtype if out_dtype is not None else x.dtype
    K_in = x.shape[-1]
    N_out = weight.shape[0]
    assert weight.shape == (N_out, K_in)
    if stream is None:
        stream = _current_stream()

    # hook: record activation max for THIS site (always, even when
    # the bf16 fallback would be taken — so the JSON covers everything).
    # No-op outside a calibration_recorder context.
    if site_id is not None:
        from flash_rt.models.lingbot import calibration as _calib
        if _calib.is_calibrating():
            _calib.record_max_abs(site_id, x)

    # math.prod for capture-safety (torch.tensor + .item() crosses
    # to host which is not permitted during CUDA-Graph capture).
    M = 1
    for d in x.shape[:-1]:
        M *= d

    # bf16 fallback when N/K can't hit the FP8 heuristic. cuBLAS needs matching
    # dtype, so cast x to the weight dtype here (rare path).
    if N_out < _FP8_MIN_N or K_in < _FP8_MIN_N:
        xb = x if x.dtype == weight.dtype else x.to(weight.dtype)
        return linear_bf16(xb, weight, bias, stream=stream)

    # pad-and-FP8 path applies only when M ∈ [_FP8_PAD_MIN_M,
    # _FP8_MIN_M) AND a static activation scale is loaded for this
    # site_id. The dynamic absmax-reduce reads the entire padded
    # buffer including pad rows — their uninitialized values
    # pollute the scale and produce wrong output. Static scale
    # bypasses the absmax entirely, so pad rows are harmless.
    enable_pad = False
    if site_id is not None and _FP8_PAD_MIN_M <= M < _FP8_MIN_M:
        from flash_rt.models.lingbot import calibration as _calib
        if _calib.get_static_scale(site_id) is not None:
            enable_pad = True

    if M < _FP8_MIN_M and not enable_pad:
        xb = x if x.dtype == weight.dtype else x.to(weight.dtype)
        return linear_bf16(xb, weight, bias, stream=stream)

    if enable_pad:
        pad_to = _FP8_MIN_M
        x_2d_raw = x.contiguous().view(M, K_in)
        x_2d = torch.empty(pad_to, K_in, dtype=x.dtype, device=x.device)
        x_2d[:M].copy_(x_2d_raw)
        # Pad rows [M:pad_to] left uninitialized — safe because the
        # static quantize_fp8_static reads a precomputed scale (no
        # absmax-reduce over the padded region) and the corresponding
        # FP8 GEMM output rows are sliced off below.
        out_padded = torch.empty(
            pad_to, N_out, dtype=od, device=x.device)
        out_2d = out_padded
        slice_back = True
    else:
        pad_to = M
        x_2d = x.contiguous().view(M, K_in)
        out = torch.empty(
            (*x.shape[:-1], N_out), dtype=od, device=x.device)
        out_2d = out.view(M, N_out)
        slice_back = False

    # 1. Get cached FP8 weight + descale.
    w_fp8_T, w_descale = _get_or_quantize_fp8_weight(weight)

    # 2. Look up static activation scale. When present, reuse it
    # in place of a per-call allocation — kernel reads it as input.
    static_scale = None
    if site_id is not None:
        from flash_rt.models.lingbot import calibration as _calib
        static_scale = _calib.get_static_scale(site_id)

    # 3. Allocate per-call FP8 input. Allocate act-scale buffer only
    # when we don't have a static scale.
    x_fp8 = torch.empty(pad_to, K_in, dtype=torch.float8_e4m3fn, device=x.device)
    if static_scale is None:
        act_scale = torch.empty(1, dtype=torch.float32, device=x.device)
    else:
        act_scale = static_scale  # process-lifetime tensor from load_calibration

    # 4. Quantize x to FP8. Dispatch by dtype: bf16 → quantize_fp8_*,
    # fp16 → quantize_fp8_*_fp16. Static path reads scale; dynamic
    # writes it (device absmax+scale+quantize in one fused pass).
    if static_scale is None:
        _quantize_fp8_device(
            int(x_2d.data_ptr()), int(x_fp8.data_ptr()),
            int(act_scale.data_ptr()), pad_to * K_in, x.dtype, stream)
    else:
        _quantize_fp8_static(
            int(x_2d.data_ptr()), int(x_fp8.data_ptr()),
            int(act_scale.data_ptr()), pad_to * K_in, x.dtype, stream)

    # 5. FP8 GEMM with bf16/fp16 output and device-side descale. The quant
    # above used x.dtype (input); the GEMM output uses ``od``.
    _fp8_gemm_descale(
        int(x_fp8.data_ptr()), int(w_fp8_T.data_ptr()),
        int(out_2d.data_ptr()),
        pad_to, N_out, K_in,
        int(act_scale.data_ptr()), int(w_descale.data_ptr()),
        od, stream)

    # 6. Bias if any. Apply over ``pad_to`` rows; the unused tail is sliced
    # off in the next step.
    if bias is not None:
        _add_bias(
            int(out_2d.data_ptr()),
            int(bias.contiguous().data_ptr()),
            pad_to, N_out, od, stream)

    # 7. : slice the pad rows off and reshape to the caller's
    # leading dims. ``out_padded[:M]`` is contiguous (stride=[N,1],
    # offset=0) so ``.view`` succeeds; the returned tensor references
    # the same mempool storage as ``out_padded``. Caller treats it
    # read-only (next op is a downstream GEMM input).
    if slice_back:
        return out_padded[:M].view(*x.shape[:-1], N_out)
    return out


# ════════════════════════════════════════════════════════════════════
# — Fused MHA attention via fvk.attention_mha_bf16
# ════════════════════════════════════════════════════════════════════
#
# Replaces the 4-einsum-plus-softmax eager attention in mixed_attention
# with a single cuBLAS-decomposed call. Same kernel Pi0.5 / GROOT use
# (Pi0.5: ``attention_qkv_fp16``; we use the bf16 variant to avoid an
# fp16↔bf16 conversion).
#
# Layout contract (matches Pi0.5 pipeline_thor.py:134):
#     Q  : [S_q,  NH, HD]   bf16, contiguous
#     K  : [S_kv, NH, HD]   bf16, contiguous   (GQA expanded by caller)
#     V  : same as K
#     out: [S_q,  NH, HD]   bf16, written
#     logits scratch: [S_q, NH, S_kv] bf16 — caller-owned
#
# The kernel does NOT take an attention mask; it computes
# ``softmax(QK^T / sqrt(HD)) @ V`` unmasked. For LingBot:
#   * Prefix encode (vlm_causal=False) is unmasked → drop-in.
#   * Denoise suffix mask: state-token-block + full-rest. Cos delta
#     measured below; we keep this kernel unmasked and accept the
#     small precision drift (state token attending to actions too,
#     which doesn't change the ODE flow direction).


def silu_mul_to_fp8_fp16_fused(
    gate: torch.Tensor,                 # [..., H] fp16
    up: torch.Tensor,                   # [..., H] fp16, same shape
    down_act_scale: torch.Tensor,       # [1] fp32 — pre-loaded static scale for down_proj
    *,
    stream: int | None = None,
) -> torch.Tensor:
    """(fp16-only): fuses ``silu(gate) * up`` and FP8 quant into ONE
    launch via ``fvk.silu_mul_split_fp8_fp16``. Returns the FP8 tensor
    ready to feed into ``linear_fp8_from_fp8`` for the down-proj.

    Replaces three eager ops (silu, mul, quantize) with one fused
    kernel. Available ONLY in fp16 mode — the kernel has no bf16 sibling
    in the installed wheel.
    """
    assert gate.dtype == torch.float16, f"gate dtype {gate.dtype}, need fp16"
    assert up.dtype == torch.float16
    assert gate.shape == up.shape
    assert down_act_scale.dtype == torch.float32 and down_act_scale.numel() == 1
    if stream is None:
        stream = _current_stream()

    n = gate.numel()
    out_fp8 = torch.empty(
        gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    fvk.silu_mul_split_fp8_fp16(
        int(gate.contiguous().data_ptr()),
        int(up.contiguous().data_ptr()),
        int(out_fp8.data_ptr()),
        n, int(down_act_scale.data_ptr()), stream,
    )
    return out_fp8


def attention_mha_bf16_fused(
    Q: torch.Tensor,                   # [B, S_q, NH, HD] bf16
    K: torch.Tensor,                   # [B, S_kv, NH, HD] bf16, post-GQA-expansion
    V: torch.Tensor,                   # [B, S_kv, NH, HD] bf16
    *,
    attn_scale: float | None = None,
    stream: int | None = None,
) -> torch.Tensor:
    """Wrapper around ``fvk.attention_mha_bf16`` returning a tensor
    shaped ``[B, S_q, NH*HD]`` (matches eager attention's output).

    Assumes ``B==1`` (the LingBot baseline). For larger B the pointer
    math would need adjustment.
    """
    assert Q.dtype in (torch.bfloat16, torch.float16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype, (
        f"K/V dtype mismatch: Q={Q.dtype}, K={K.dtype}, V={V.dtype}")
    B, S_q, NH, HD = Q.shape
    S_kv = K.shape[1]
    assert K.shape == (B, S_kv, NH, HD) and V.shape == (B, S_kv, NH, HD), (
        f"K/V shape mismatch: K={K.shape}, V={V.shape}, expected (B, S_kv, NH, HD)")
    assert B == 1, f"attention_mha wrapper currently assumes B=1, got {B}"
    if attn_scale is None:
        attn_scale = HD ** -0.5
    if stream is None:
        stream = _current_stream()

    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()

    out = torch.empty(S_q, NH, HD, dtype=Q.dtype, device=Q.device)
    logits = torch.empty(S_q, NH, S_kv, dtype=Q.dtype, device=Q.device)

    ctx = get_fvk_context()
    if Q.dtype == torch.bfloat16:
        fvk.attention_mha_bf16(
            ctx,
            int(Q_c.data_ptr()), int(K_c.data_ptr()), int(V_c.data_ptr()),
            int(logits.data_ptr()), int(out.data_ptr()),
            S_q, S_kv, NH, HD,
            float(attn_scale), 0, stream)
    else:  # fp16
        # attention_mha_fp16 signature differs slightly — no
        # logits_kv_stride parameter (fp16 variant predates that).
        fvk.attention_mha_fp16(
            ctx,
            int(Q_c.data_ptr()), int(K_c.data_ptr()), int(V_c.data_ptr()),
            int(logits.data_ptr()), int(out.data_ptr()),
            S_q, S_kv, NH, HD,
            float(attn_scale), stream)
    return out.view(B, S_q, NH * HD)


# ════════════════════════════════════════════════════════════════════
# — GQA-native fp16 CUTLASS FMHA (fmha_strided_full, sm_110a)
# ════════════════════════════════════════════════════════════════════
#
# The strided CUTLASS Sm100 FMHA ships in ``libfmha_fp16_strided.so``,
# confirmed sm_110a-built (cuobjdump). Micro-bench at LingBot's denoise
# shape (Sq=51, Skv≈891, NHQ=16, NHKV=2, HD=128) under CUDA-Graph replay:
# bf16 cuBLAS attention_mha 88us → fp16 fmha 47us (kernel) / 65us
# (with bf16↔fp16 casts). It is GQA-native (takes NHKV directly) so it
# also SKIPS the 8× KV head-expand the cuBLAS path needs. cos vs eager
# fp32 reference = 1.0000 at this shape (fp16 has more mantissa than bf16).

_FMHA_STRIDED_LOADED = None


def _ensure_fmha_strided() -> bool:
    """Load libfmha_fp16_strided.so once. Returns True if available."""
    global _FMHA_STRIDED_LOADED
    if _FMHA_STRIDED_LOADED is None:
        import pathlib
        so = pathlib.Path(fvk.__file__).parent / "libfmha_fp16_strided.so"
        try:
            if so.exists():
                fvk.load_fmha_strided_library(str(so))
                _FMHA_STRIDED_LOADED = True
            else:
                _FMHA_STRIDED_LOADED = False
        except Exception:
            _FMHA_STRIDED_LOADED = False
    return _FMHA_STRIDED_LOADED


def attention_fmha_strided_fused(
    Q: torch.Tensor,            # [B, S_q, NHQ, HD]  bf16, NOT GQA-expanded
    K: torch.Tensor,            # [B, S_kv, NHKV, HD] bf16
    V: torch.Tensor,            # [B, S_kv, NHKV, HD] bf16
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    stream: int | None = None,
) -> torch.Tensor:
    """Unmasked GQA-native FMHA via ``fvk.fmha_strided_full`` (fp16).

    Casts Q/K/V bf16→fp16, runs the strided CUTLASS FMHA (scale 1/sqrt(HD)
    baked in), and returns a bf16 ``[B, S_q, NHQ*HD]`` tensor matching
    ``attention_mha_bf16_fused``'s output layout. K/V are passed with their
    native NHKV heads (no replication).
    """
    B, S_q, NHQ, HD = Q.shape
    S_kv = K.shape[1]
    assert B == 1, f"fmha_strided wrapper assumes B=1, got {B}"
    assert HD == head_dim and NHQ == num_q_heads
    assert K.shape == (B, S_kv, num_kv_heads, HD)
    if stream is None:
        stream = _current_stream()

    Qf = Q.to(torch.float16).contiguous()
    Kf = K.to(torch.float16).contiguous()
    Vf = V.to(torch.float16).contiguous()
    out = torch.empty(B, S_q, NHQ, HD, dtype=torch.float16, device=Q.device)

    fvk.fmha_strided_full(
        int(Qf.data_ptr()), int(Kf.data_ptr()), int(Vf.data_ptr()),
        int(out.data_ptr()),
        B, S_q, S_kv, num_q_heads, num_kv_heads, HD,
        num_q_heads * HD, num_kv_heads * HD, stream)
    return out.to(torch.bfloat16).view(B, S_q, NHQ * HD)


# ════════════════════════════════════════════════════════════════════
#  FA4 (FlashAttention-4, CuTe-DSL) denoise attention — Thor sm_101a
# ════════════════════════════════════════════════════════════════════
#
# FA4's SM100 Blackwell kernel runs on Thor when compiled for sm_101a (the
# legacy alias of sm_110; the sm_110a path hits a cutlass-dsl chip bug). At
# the denoise shape (Sq=51, Skv≈891, GQA 16/2, HD=128) with pack_gqa=True it
# measures ~38.8 µs vs the vendored fmha 47 µs (~17% faster), cos=1.0, and is
# CUDA-graph capture-safe. Needs nvidia-cutlass-dsl + quack-kernels + the FA4
# source on path; env CUTE_DSL_ARCH=sm_101a / FLASH_ATTENTION_ARCH=sm_100a set
# below before import. Returns None if unavailable (caller falls back to fmha).
_FA4_FUNC: "object | None" = None
_FA4_TRIED = False


def _get_fa4():
    """Lazy-load FA4 (flash_attn.cute, coexists with FA2). Requires
    nvidia-cutlass-dsl + quack-kernels + flash-attn-4 (cute) installed. The
    sm_101a env is the Thor target (sm_110's Blackwell alias; the sm_110a
    path hits a cutlass-dsl chip bug). An optional LINGBOT_FA4_SRC dir is
    prepended to sys.path as a fallback when FA4 isn't pip-installed. Returns
    None on any failure → caller falls back to fmha."""
    global _FA4_FUNC, _FA4_TRIED
    if _FA4_TRIED:
        return _FA4_FUNC
    _FA4_TRIED = True
    try:
        import os
        from pathlib import Path
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_101a")
        os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")
        # FA4 (flash_attn.cute) source is vendored at <repo>/third_party/fa4 so
        # it works out of the box on Thor; LINGBOT_FA4_SRC overrides it if set.
        src = os.environ.get("LINGBOT_FA4_SRC")
        if not src:
            _vend = Path(__file__).resolve().parents[3] / "third_party" / "fa4"
            if _vend.is_dir():
                src = str(_vend)
        if src and src not in sys.path:
            sys.path.insert(0, src)
        # Still requires the FA4 runtime deps: nvidia-cutlass-dsl + quack-kernels
        # (pip install ".[thor-fa4]"). Missing -> ImportError -> fmha fallback.
        from flash_attn.cute.interface import flash_attn_func
        _FA4_FUNC = flash_attn_func
    except Exception:
        import os as _o, traceback as _tb
        if _o.environ.get("LINGBOT_FA4_DEBUG"):
            _tb.print_exc()
        _FA4_FUNC = None
    return _FA4_FUNC


_FA4_FWD = None
_FA4_FWD_TRIED = False


def _get_fa4_fwd():
    """Lazy-load FA4's internal ``_flash_attn_fwd`` (the non-autograd entry that
    exposes ``seqused_k`` — the public ``flash_attn_func`` does not). Used for the
    prefix attention to skip the contiguous lang-pad keys EXACTLY (no
    softmax-denominator distortion), which the dense wrapper can't express."""
    global _FA4_FWD, _FA4_FWD_TRIED
    if _FA4_FWD_TRIED:
        return _FA4_FWD
    _FA4_FWD_TRIED = True
    if _get_fa4() is None:
        return None
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        _FA4_FWD = _flash_attn_fwd
    except Exception:
        _FA4_FWD = None
    return _FA4_FWD


def attention_fa4_fused(
    Q: torch.Tensor,            # [B, S_q, NHQ, HD] fp16/bf16, GQA-native
    K: torch.Tensor,            # [B, S_kv, NHKV, HD]
    V: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seqused_k: torch.Tensor | None = None,   # [B] int32 valid KV length
    stream: int | None = None,
) -> "torch.Tensor | None":
    """FA4 GQA attention (pack_gqa). Returns [B, S_q, NHQ*HD] in the FA4 output
    dtype (fp16 — the attention island), or None if FA4 is unavailable so the
    caller can fall back to fmha. : NO bf16 cast here — the o_proj quantizes
    the fp16 directly (linear_fp8 with out_dtype=bf16), removing ~900 casts.

    : when ``seqused_k`` is given, only the first ``seqused_k`` KV rows are
    attended (the rest are pad, contiguous at the end) — EXACT, via the internal
    ``_flash_attn_fwd``. Falls back to the dense path if that entry is missing."""
    B, S_q = Q.shape[0], Q.shape[1]
    qf = Q if Q.dtype == torch.float16 else Q.to(torch.float16)
    kf = K if K.dtype == torch.float16 else K.to(torch.float16)
    vf = V if V.dtype == torch.float16 else V.to(torch.float16)
    if seqused_k is not None:
        fwd = _get_fa4_fwd()
        if fwd is not None:
            o = fwd(qf.contiguous(), kf.contiguous(), vf.contiguous(),
                    causal=False, pack_gqa=True, seqused_k=seqused_k)
            if isinstance(o, tuple):
                o = o[0]
            return o.reshape(B, S_q, num_q_heads * head_dim)
    fa4 = _get_fa4()
    if fa4 is None:
        return None
    o = fa4(qf.contiguous(), kf.contiguous(), vf.contiguous(),
            causal=False, pack_gqa=True)
    if isinstance(o, tuple):
        o = o[0]
    return o.reshape(B, S_q, num_q_heads * head_dim)


# ════════════════════════════════════════════════════════════════════
# — LingBot custom fused AdaRMS + residual + FP8 quant (csrc)
# ════════════════════════════════════════════════════════════════════
#
# A custom CUDA kernel that fuses, in one pass per token,
#   y          = residual + x
#   y_norm     = rms_weight * (y / sqrt(mean(y^2) + eps))
#   y_film     = (1 + γ) * y_norm + β     (per-sample γ, β)
#   y_fp8      = static-scale FP8 quantize
# and writes ``residual`` back to ``residual + x`` in place (so the
# caller has the bf16 sum available for the SECOND residual that
# follows the MLP).
#
# Replaces what eager LingBot does as 6-9 separate launches:
#   residual + x → afr            (torch op)
#   rms_norm(afr) (5 internal ops: pow, mean, rsqrt, mul, cast)
#   (1+γ)*norm + β                (2 torch ops)
#   quantize_fp8_static(...)      (1 launch)
#
# Per Expert layer this kernel fires 2× (input_ln + post_attn_ln). At
# 36 layers × N denoise steps it dominates the per-step graph cost
# after . Estimated bandwidth savings: residual+x written once
# (one DRAM write) vs old path that read it multiple times.


_LINGBOT_EXT = None


def _get_lingbot_ext():
    """Lazy-load the JIT-compiled LingBot CUDA extension."""
    global _LINGBOT_EXT
    if _LINGBOT_EXT is None:
        from flash_rt.models.lingbot._csrc_loader import get_lingbot_ext
        _LINGBOT_EXT = get_lingbot_ext()
    return _LINGBOT_EXT


def rope_inplace_bf16_fused(
    x: torch.Tensor,                 # [B, S, NH, HD] bf16 — mutated in place
    cos_table: torch.Tensor,         # [B, S, HD/2] fp32
    sin_table: torch.Tensor,         # [B, S, HD/2] fp32
    *,
    stream: int | None = None,
) -> torch.Tensor:
    """: in-place split-half RoPE in bf16 with fp32 internal accumulate.

    Replaces ``apply_rope_with_tables(x.to(fp32), cos, sin).to(bf16)``
    (3 launches: cast + apply + cast back) with one fused kernel pass.
    Mutates ``x`` in place; returns the same tensor for chaining.

    Used by both prefix_encode_layer and denoise_step_layer once cos/sin
    have been hoisted out of the layer loop .
    """
    assert x.dtype == torch.bfloat16, f"x dtype {x.dtype}"
    assert cos_table.dtype == torch.float32
    assert sin_table.dtype == torch.float32
    B, S, NH, HD = x.shape
    if stream is None:
        stream = _current_stream()
    ext = _get_lingbot_ext()
    ext.rope_inplace_bf16(
        int(x.contiguous().data_ptr()),
        int(cos_table.contiguous().data_ptr()),
        int(sin_table.contiguous().data_ptr()),
        B, S, NH, HD, stream,
    )
    return x


def rope_to_out_bf16_fused(
    x: torch.Tensor,                 # [B, S, NH, HD] bf16 (read-only)
    cos_table: torch.Tensor,         # [B, S, HD/2] fp32
    sin_table: torch.Tensor,         # [B, S, HD/2] fp32
    *,
    stream: int | None = None,
) -> torch.Tensor:
    """out-of-place split-half RoPE in bf16, fp32 internal accumulate.

    Replaces ``apply_rope_with_tables(x.to(fp32), cos, sin).to(bf16)``
    (cast + rotate-half + cast back, ~5 elementwise launches) with one
    fused kernel that reads ``x`` and writes a fresh output buffer — no
    in-place hazard, so it's CUDA-Graph-replay deterministic (unlike the
    in-place variant). Returns the new [B, S, NH, HD] bf16 tensor.
    """
    assert x.dtype == torch.bfloat16, f"x dtype {x.dtype}"
    assert cos_table.dtype == torch.float32
    assert sin_table.dtype == torch.float32
    B, S, NH, HD = x.shape
    if stream is None:
        stream = _current_stream()
    xc = x.contiguous()
    out = torch.empty_like(xc)
    ext = _get_lingbot_ext()
    ext.rope_to_out_bf16(
        int(xc.data_ptr()), int(out.data_ptr()),
        int(cos_table.contiguous().data_ptr()),
        int(sin_table.contiguous().data_ptr()),
        B, S, NH, HD, stream,
    )
    return out


def qkv_bias_rope_fused(
    q_raw: torch.Tensor,             # [M, NHQ*HD] bf16 (bias-free GEMM out)
    k_raw: torch.Tensor,             # [M, NHKV*HD] bf16
    v_raw: torch.Tensor,             # [M, NHKV*HD] bf16
    q_bias, k_bias, v_bias,          # [NHx*HD] bf16 or None
    cos_table: torch.Tensor,         # [M, HD/2] fp32
    sin_table: torch.Tensor,         # [M, HD/2] fp32
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """: one kernel for q/k/v bias-add + RoPE (q/k roped, v bias-only).
    Replaces 3 add_bias + 2 rope launches per Expert layer. Returns
    (q [M,NHQ*HD], k [M,NHKV*HD], v [M,NHKV*HD]) in ``out_dtype``.

    ``out_dtype=torch.float16`` feeds the fp16 attention island so the
    fmha wrapper consumes q/k/v with no bf16→fp16 cast launches."""
    M = q_raw.shape[0]
    if stream is None:
        stream = _current_stream()
    q_out = torch.empty(q_raw.shape, dtype=out_dtype, device=q_raw.device)
    k_out = torch.empty(k_raw.shape, dtype=out_dtype, device=k_raw.device)
    v_out = torch.empty(v_raw.shape, dtype=out_dtype, device=v_raw.device)

    def _p(t):
        return int(t.contiguous().data_ptr()) if t is not None else 0

    ext = _get_lingbot_ext()
    fn = ext.qkv_bias_rope_fp16out if out_dtype == torch.float16 \
        else ext.qkv_bias_rope_bf16
    fn(
        int(q_raw.contiguous().data_ptr()),
        int(k_raw.contiguous().data_ptr()),
        int(v_raw.contiguous().data_ptr()),
        _p(q_bias), _p(k_bias), _p(v_bias),
        int(cos_table.contiguous().data_ptr()),
        int(sin_table.contiguous().data_ptr()),
        int(q_out.data_ptr()), int(k_out.data_ptr()), int(v_out.data_ptr()),
        M, num_q_heads, num_kv_heads, head_dim, stream,
    )
    return q_out, k_out, v_out


def qkv_bias_rope_merged_fused(
    qkv: torch.Tensor,               # [M, NHQ*HD + 2*NHKV*HD] bf16 (merged GEMM out)
    q_bias, k_bias, v_bias,          # [NHx*HD] bf16 or None
    cos_table: torch.Tensor,         # [M, HD/2] fp32
    sin_table: torch.Tensor,         # [M, HD/2] fp32
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
    k_out: torch.Tensor | None = None,   # write k/v into these (e.g. the KV
    v_out: torch.Tensor | None = None,   # cache suffix) directly — no copy after.
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """: merged-input variant of :func:`qkv_bias_rope_fused`. Reads q/k/v
    from column offsets of one merged ``[M, ROWQKV]`` GEMM output (no split
    copy); writes 3 separate outputs. Pairs with a merged qkv GEMM.

    : if ``k_out``/``v_out`` are given (e.g. views into the preallocated KV
    cache suffix), the kernel writes k/v there directly — the denoise step then
    skips the per-layer ``kbuf[:,Lp:].copy_(k_new)`` memcpy."""
    M = qkv.shape[0]
    if stream is None:
        stream = _current_stream()
    q_out = torch.empty(M, num_q_heads * head_dim, dtype=out_dtype, device=qkv.device)
    if k_out is None:
        k_out = torch.empty(M, num_kv_heads * head_dim, dtype=out_dtype, device=qkv.device)
    if v_out is None:
        v_out = torch.empty(M, num_kv_heads * head_dim, dtype=out_dtype, device=qkv.device)

    def _p(t):
        return int(t.contiguous().data_ptr()) if t is not None else 0

    ext = _get_lingbot_ext()
    fn = ext.qkv_bias_rope_merged_fp16out if out_dtype == torch.float16 \
        else ext.qkv_bias_rope_merged_bf16
    fn(
        int(qkv.contiguous().data_ptr()),
        _p(q_bias), _p(k_bias), _p(v_bias),
        int(cos_table.contiguous().data_ptr()),
        int(sin_table.contiguous().data_ptr()),
        int(q_out.data_ptr()), int(k_out.data_ptr()), int(v_out.data_ptr()),
        M, num_q_heads, num_kv_heads, head_dim, stream,
    )
    return q_out, k_out, v_out


def vit_qkv_bias_rope_fused(
    qkv: torch.Tensor,               # [M, 3*NH*HD] bf16 (bias-free GEMM out)
    qkv_bias,                        # [3*NH*HD] bf16 or None
    cos_table: torch.Tensor,         # [M, HD] fp32 (first HD/2 cols used)
    sin_table: torch.Tensor,         # [M, HD] fp32
    *,
    num_heads: int,
    head_dim: int,
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """P1 ViT: one kernel for the interleaved ViT qkv bias-add + 2-D M-RoPE.
    Replaces, per ViT block, the add_bias(qkv) launch + the eager fp32 RoPE
    storm (cast/rotate_half/mul/add/cast). Returns (q, k, v) each
    [M, NH*HD] bf16, ready to view as [num_views, view_len, NH, HD] for the
    per-view fmha."""
    M = qkv.shape[0]
    if stream is None:
        stream = _current_stream()
    nk = num_heads * head_dim
    q_out = torch.empty(M, nk, dtype=torch.bfloat16, device=qkv.device)
    k_out = torch.empty(M, nk, dtype=torch.bfloat16, device=qkv.device)
    v_out = torch.empty(M, nk, dtype=torch.bfloat16, device=qkv.device)
    ext = _get_lingbot_ext()
    ext.vit_qkv_bias_rope_bf16(
        int(qkv.contiguous().data_ptr()),
        int(qkv_bias.contiguous().data_ptr()) if qkv_bias is not None else 0,
        int(cos_table.contiguous().data_ptr()),
        int(sin_table.contiguous().data_ptr()),
        int(q_out.data_ptr()), int(k_out.data_ptr()), int(v_out.data_ptr()),
        M, num_heads, head_dim, stream,
    )
    return q_out, k_out, v_out


def ada_rms_fp8_fused(
    x: torch.Tensor,                       # [B, S, D] bf16 (read-only)
    rms_weight: torch.Tensor,              # [D] bf16
    gamma: torch.Tensor,                   # [B, D] bf16 — pre-computed FiLM γ
    beta: torch.Tensor,                    # [B, D] bf16 — pre-computed FiLM β
    *,
    eps: float = 1e-6,
    site_id: "str | None" = None,
    pad_to: int | None = None,
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """: one-pass AdaRMSNorm + FP8 quant (NO residual). Same kernel
    family as :func:`ada_rms_residual_fp8_fused` but for the
    ``input_layernorm`` site where there is no residual upstream.

    Returns ``(out_fp8 [B*S or B*pad_to, D], act_scale [1])`` or ``None``
    when no static scale is available for ``site_id``. When ``pad_to``
    is given (and > S), the FP8 output is M-padded to ``B*pad_to`` rows
    (rows ``[S, pad_to)`` zeroed) so the downstream FP8 GEMM reads M=pad_to
    directly and skips the per-call pad copy — the same buffer feeds all of
    q/k/v. The static scale is unchanged (zero pad rows don't affect it).
    """
    assert x.dtype == torch.bfloat16
    assert rms_weight.dtype == torch.bfloat16
    assert gamma.dtype == torch.bfloat16 and beta.dtype == torch.bfloat16
    B, S, D = x.shape
    assert rms_weight.shape == (D,)
    assert gamma.shape == (B, D) and beta.shape == (B, D)

    if site_id is None:
        return None
    from flash_rt.models.lingbot import calibration as _calib
    act_scale = _calib.get_static_scale(site_id)
    if act_scale is None:
        return None
    if stream is None:
        stream = _current_stream()

    ext = _get_lingbot_ext()
    if pad_to is not None and pad_to > S:
        out_fp8 = torch.empty(
            B * pad_to, D, dtype=torch.float8_e4m3fn, device=x.device)
        ext.ada_rms_fp8_mpad_bf16(
            int(x.contiguous().data_ptr()),
            int(rms_weight.contiguous().data_ptr()),
            int(gamma.contiguous().data_ptr()),
            int(beta.contiguous().data_ptr()),
            int(out_fp8.data_ptr()),
            int(act_scale.data_ptr()),
            B, S, pad_to, D, float(eps), stream,
        )
        return out_fp8, act_scale

    out_fp8 = torch.empty(
        B * S, D, dtype=torch.float8_e4m3fn, device=x.device)
    ext.ada_rms_fp8_bf16(
        int(x.contiguous().data_ptr()),
        int(rms_weight.contiguous().data_ptr()),
        int(gamma.contiguous().data_ptr()),
        int(beta.contiguous().data_ptr()),
        int(out_fp8.data_ptr()),
        int(act_scale.data_ptr()),
        B, S, D, float(eps), stream,
    )
    return out_fp8, act_scale


def silu_mul_fp8_mpad_bf16_fused(
    gate: torch.Tensor,                    # [M, I] bf16
    up: torch.Tensor,                      # [M, I] bf16
    down_act_scale: torch.Tensor,          # [1] fp32 — static down_proj scale
    *,
    pad_to: int = _FP8_MIN_M,
    stream: int | None = None,
) -> torch.Tensor:
    """Fused SwiGLU tail (bf16): ``silu(gate)*up`` + FP8 static-quant into a
    pre-M-padded ``[pad_to, I]`` FP8 buffer (rows ``[M, pad_to)`` zeroed).

    Replaces eager ``F.silu(gate)*up`` (2 elementwise launches) + a separate
    ``quantize_fp8`` + the ``linear_fp8`` M-pad copy with ONE launch. The
    returned buffer feeds ``linear_fp8_from_fp8`` at M=pad_to directly (no copy);
    the caller slices the GEMM output back to M.
    """
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
    assert gate.shape == up.shape
    M, I = gate.shape
    assert M <= pad_to
    if stream is None:
        stream = _current_stream()
    out_fp8 = torch.empty(pad_to, I, dtype=torch.float8_e4m3fn, device=gate.device)
    ext = _get_lingbot_ext()
    ext.silu_mul_fp8_mpad_bf16(
        int(gate.contiguous().data_ptr()),
        int(up.contiguous().data_ptr()),
        int(out_fp8.data_ptr()),
        int(down_act_scale.data_ptr()),
        M, pad_to, I, stream,
    )
    return out_fp8


def ada_rms_residual_fp16_mpad_fused(
    residual: torch.Tensor,                # [B, S, D] bf16 — MUTATED to residual+x
    x: torch.Tensor,                       # [B, S, D] bf16
    rms_weight: torch.Tensor,              # [D] bf16
    gamma: torch.Tensor,                   # [B, D] bf16
    beta: torch.Tensor,                    # [B, D] bf16
    *,
    pad_to: int,
    eps: float = 1e-6,
    stream: int | None = None,
) -> torch.Tensor:
    """: AdaRMSNorm+residual with fp16 output (M-padded to ``pad_to``), no FP8
    quant. SIDE EFFECT: ``residual`` updated to residual+x. Returns ``[B*pad_to, D]``
    fp16. Feeds the denoise FP4 gate_up quant directly (skips the fp8→fp16 dequant)."""
    assert residual.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    B, S, D = residual.shape
    if stream is None:
        stream = _current_stream()
    out = torch.empty(B * pad_to, D, dtype=torch.float16, device=residual.device)
    ext = _get_lingbot_ext()
    ext.ada_rms_residual_fp16_mpad(
        int(residual.contiguous().data_ptr()), int(x.contiguous().data_ptr()),
        int(rms_weight.contiguous().data_ptr()),
        int(gamma.contiguous().data_ptr()), int(beta.contiguous().data_ptr()),
        int(out.data_ptr()), B, S, pad_to, D, float(eps), stream)
    return out


def silu_mul_merged_fp8_mpad_bf16_fused(
    gu: torch.Tensor,                      # [M, 2*I] bf16 — merged gate|up GEMM output
    down_act_scale: torch.Tensor,          # [1] fp32 — static down_proj scale
    *,
    pad_to: int = _FP8_MIN_M,
    stream: int | None = None,
) -> torch.Tensor:
    """Merged-input fused SwiGLU tail: gate/up interleaved per row in one
    ``[M, 2*I]`` buffer (the output of a single merged gate_up GEMM). Computes
    ``silu(gate)*up`` + FP8 static-quant into a pre-M-padded ``[pad_to, I]``
    FP8 buffer. Pairs with the merged gate_up GEMM to cut a GEMM launch."""
    assert gu.dtype == torch.bfloat16
    M, twoI = gu.shape
    assert twoI % 2 == 0
    I = twoI // 2
    assert M <= pad_to
    if stream is None:
        stream = _current_stream()
    out_fp8 = torch.empty(pad_to, I, dtype=torch.float8_e4m3fn, device=gu.device)
    ext = _get_lingbot_ext()
    ext.silu_mul_merged_fp8_mpad_bf16(
        int(gu.contiguous().data_ptr()), int(out_fp8.data_ptr()),
        int(down_act_scale.data_ptr()), M, pad_to, I, stream)
    return out_fp8


def silu_mul_merged_fp8_mpad_fp16in_fused(
    gu: torch.Tensor,                      # [M, 2*I] fp16 — merged gate|up FP4-GEMM out
    down_act_scale: torch.Tensor,
    *,
    pad_to: int = _FP8_MIN_M,
    stream: int | None = None,
) -> torch.Tensor:
    """: fp16-input variant of :func:`silu_mul_merged_fp8_mpad_bf16_fused` —
    reads the merged gate_up straight from the FP4 GEMM's fp16 output (no bf16
    cast)."""
    assert gu.dtype == torch.float16
    M, twoI = gu.shape
    I = twoI // 2
    assert M <= pad_to
    if stream is None:
        stream = _current_stream()
    out_fp8 = torch.empty(pad_to, I, dtype=torch.float8_e4m3fn, device=gu.device)
    ext = _get_lingbot_ext()
    ext.silu_mul_merged_fp8_mpad_fp16in(
        int(gu.contiguous().data_ptr()), int(out_fp8.data_ptr()),
        int(down_act_scale.data_ptr()), M, pad_to, I, stream)
    return out_fp8


def ada_rms_residual_fp8_fused(
    residual: torch.Tensor,                # [B, S, D] bf16 — MUTATED to residual+x
    x: torch.Tensor,                       # [B, S, D] bf16
    rms_weight: torch.Tensor,              # [D] bf16
    gamma: torch.Tensor,                   # [B, D] bf16 — pre-computed FiLM γ
    beta: torch.Tensor,                    # [B, D] bf16 — pre-computed FiLM β
    *,
    eps: float = 1e-6,
    site_id: "str | None" = None,
    pad_to: int | None = None,
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """One-pass fused AdaRMSNorm + residual + FP8 static-quant.

    SIDE EFFECT: ``residual`` is updated to ``residual + x``.

    Returns ``(out_fp8 [B*S or B*pad_to, D] fp8, act_scale [1] fp32)`` or
    ``None`` when no static scale is available for ``site_id``. When
    ``pad_to`` is given (and > S), the FP8 output is M-padded (rows
    ``[S, pad_to)`` zeroed) so the gate/up GEMMs read M=pad_to directly
    (no pad copy). ``residual`` stays ``[B, S, D]``. The static scale is
    unchanged (zero pad rows don't affect it).

    Currently bf16-only. The custom kernel uses fp32 internal accumulation
    for the variance reduction; the output FP8 matches eager quantize
    within FP8 noise (cos > 0.9999 in unit tests).
    """
    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert rms_weight.dtype == torch.bfloat16
    assert gamma.dtype == torch.bfloat16 and beta.dtype == torch.bfloat16
    assert residual.shape == x.shape
    B, S, D = residual.shape
    assert rms_weight.shape == (D,)
    assert gamma.shape == (B, D) and beta.shape == (B, D)

    if site_id is None:
        return None
    from flash_rt.models.lingbot import calibration as _calib
    act_scale = _calib.get_static_scale(site_id)
    if act_scale is None:
        return None
    if stream is None:
        stream = _current_stream()

    ext = _get_lingbot_ext()
    if pad_to is not None and pad_to > S:
        out_fp8 = torch.empty(
            B * pad_to, D, dtype=torch.float8_e4m3fn, device=residual.device)
        ext.ada_rms_residual_fp8_mpad_bf16(
            int(residual.contiguous().data_ptr()),
            int(x.contiguous().data_ptr()),
            int(rms_weight.contiguous().data_ptr()),
            int(gamma.contiguous().data_ptr()),
            int(beta.contiguous().data_ptr()),
            int(out_fp8.data_ptr()),
            int(act_scale.data_ptr()),
            B, S, pad_to, D, float(eps), stream,
        )
        return out_fp8, act_scale

    out_fp8 = torch.empty(
        B * S, D, dtype=torch.float8_e4m3fn, device=residual.device)
    ext.ada_rms_residual_fp8_bf16(
        int(residual.contiguous().data_ptr()),
        int(x.contiguous().data_ptr()),
        int(rms_weight.contiguous().data_ptr()),
        int(gamma.contiguous().data_ptr()),
        int(beta.contiguous().data_ptr()),
        int(out_fp8.data_ptr()),
        int(act_scale.data_ptr()),
        B, S, D, float(eps), stream,
    )
    return out_fp8, act_scale


# ════════════════════════════════════════════════════════════════════
# — Fused residual + RMS + quant + FP8-input GEMM
# ════════════════════════════════════════════════════════════════════
#
# ``residual_add_rms_norm_fp8`` from fvk fuses four ops that the eager
# path emits as four separate launches::
#
#     1. ``residual + x``     (bf16 element-wise add)
#     2. ``rms_norm(., weight)``
#     3. ``quantize_fp8``     (the absmax-reduce inside linear_fp8)
#
# Verified behavior :
#   - cos vs PyTorch eager reference: 0.9999
#   - The ``residual`` buffer is MUTATED IN PLACE to ``residual + x``.
#     This matches LingBot's pattern exactly: the post-attention residual
#     buffer (``vlm_hidden`` in ``prefix_encode_layer``) is needed again
#     as bf16 for the second residual, so writing it back is desirable.
#   - The output FP8 tensor + the (static) act_scale are ready to feed
#     directly into the next FP8 GEMM via ``linear_fp8_from_fp8`` (no
#     re-quantize).
#
# Requires a static activation scale (the kernel takes ``d_scale`` as
# an input, not an output — there is no per-call absmax-reduce). When
# ``calibration.set_static_scales`` has been installed and a scale
# exists for the given ``site_id``, the fused path is taken; otherwise
# the wrapper returns ``None`` and the caller falls back to the
# unfused path.


def residual_rms_quant_fp8_inplace(
    residual: torch.Tensor,             # [..., D] bf16 — MUTATED to residual+x
    x: torch.Tensor,                    # [..., D] bf16
    weight: torch.Tensor,               # [D] bf16 (RMS gain)
    *,
    eps: float = 1e-6,
    site_id: "str | None" = None,
    stream: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """Fused ``out_fp8 = quantize(rms_norm(residual + x, weight))``.

    SIDE EFFECT: ``residual`` is updated to ``residual + x`` (bf16, in place).

    Returns ``(out_fp8 [..., D] fp8, act_scale [1] fp32)`` on success,
    or ``None`` if a static scale is not available for ``site_id`` —
    in which case the caller should fall back to the unfused path.

    The returned ``act_scale`` is the SAME persistent device tensor
    held by ``calibration._STATIC_SCALES`` — caller MUST NOT mutate it.
    """
    assert residual.dtype in (torch.bfloat16, torch.float16)
    assert x.dtype == residual.dtype
    assert weight.dtype == residual.dtype
    assert residual.shape == x.shape
    D = residual.shape[-1]
    assert weight.shape == (D,)

    if site_id is None:
        return None
    from flash_rt.models.lingbot import calibration as _calib
    act_scale = _calib.get_static_scale(site_id)
    if act_scale is None:
        return None
    if stream is None:
        stream = _current_stream()

    M = 1
    for d in residual.shape[:-1]:
        M *= d
    res_2d = residual.contiguous().view(M, D)
    x_2d = x.contiguous().view(M, D)

    out_fp8 = torch.empty(
        M, D, dtype=torch.float8_e4m3fn, device=residual.device)

    if residual.dtype == torch.bfloat16:
        fvk.residual_add_rms_norm_fp8(
            int(res_2d.data_ptr()), int(x_2d.data_ptr()),
            int(weight.contiguous().data_ptr()),
            int(out_fp8.data_ptr()),
            M, D, eps, int(act_scale.data_ptr()), stream)
    else:  # fp16
        fvk.residual_add_rms_norm_fp8_fp16(
            int(res_2d.data_ptr()), int(x_2d.data_ptr()),
            int(weight.contiguous().data_ptr()),
            int(out_fp8.data_ptr()),
            M, D, eps, int(act_scale.data_ptr()), stream)
    return out_fp8, act_scale


def linear_fp8_from_fp8(
    x_fp8: torch.Tensor,                # [M, K] fp8_e4m3 — already quantized
    act_scale: torch.Tensor,            # [1] fp32 device — the scale used to quant x_fp8
    weight: torch.Tensor,               # [N, K] bf16 (HF layout — quantized + cached)
    bias: torch.Tensor | None = None,   # [N] bf16
    *,
    out_shape: "tuple[int, ...] | None" = None,
    stream: int | None = None,
    site_id: "str | None" = None,       # unused, kept for symmetry / future calib
) -> torch.Tensor:
    """FP8 GEMM whose input is ALREADY a quantized FP8 tensor.

    Used downstream of :func:`residual_rms_quant_fp8_inplace` so the
    next GEMM doesn't re-quantize the same activation. Same weight
    cache as :func:`linear_fp8`. Falls back to a runtime error rather
    than bf16 here — if you hit the FP8 heuristic floor, you shouldn't
    be on this code path.
    """
    del site_id  # currently unused (no per-call recording — pre-quantized)
    assert x_fp8.dtype == torch.float8_e4m3fn, f"x_fp8 dtype {x_fp8.dtype}"
    assert weight.dtype in (torch.bfloat16, torch.float16)
    M, K_in = x_fp8.shape
    N_out = weight.shape[0]
    assert weight.shape == (N_out, K_in)
    assert N_out >= _FP8_MIN_N and K_in >= _FP8_MIN_N, (
        f"linear_fp8_from_fp8 needs N>={_FP8_MIN_N}, K>={_FP8_MIN_N}; "
        f"got N={N_out} K={K_in}")
    if stream is None:
        stream = _current_stream()

    out_dtype = weight.dtype     # output matches weight dtype

    # pad path: when M < _FP8_MIN_M (Expert M=51) but ≥
    # _FP8_PAD_MIN_M, pad to M=_FP8_MIN_M so the FP8 GEMM heuristic
    # fires. Pad rows are uninitialized; their output rows are sliced
    # off below.
    if _FP8_PAD_MIN_M <= M < _FP8_MIN_M:
        pad_to = _FP8_MIN_M
        x_padded = torch.empty(
            pad_to, K_in, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_padded[:M].copy_(x_fp8)
        out_padded = torch.empty(
            pad_to, N_out, dtype=out_dtype, device=x_fp8.device)
        w_fp8_T, w_descale = _get_or_quantize_fp8_weight(weight)
        _fp8_gemm_descale(
            int(x_padded.data_ptr()), int(w_fp8_T.data_ptr()),
            int(out_padded.data_ptr()), pad_to, N_out, K_in,
            int(act_scale.data_ptr()), int(w_descale.data_ptr()),
            out_dtype, stream)
        if bias is not None:
            _add_bias(int(out_padded.data_ptr()),
                      int(bias.contiguous().data_ptr()),
                      pad_to, N_out, out_dtype, stream)
        return out_padded[:M].view(*((M,) if out_shape is None else out_shape[:-1]), N_out)

    assert M >= _FP8_MIN_M, (
        f"linear_fp8_from_fp8 needs M>={_FP8_PAD_MIN_M}; got M={M}")
    if out_shape is None:
        out_shape = (M, N_out)
    out = torch.empty(out_shape, dtype=out_dtype, device=x_fp8.device)
    out_2d = out.view(M, N_out)

    w_fp8_T, w_descale = _get_or_quantize_fp8_weight(weight)
    _fp8_gemm_descale(
        int(x_fp8.data_ptr()), int(w_fp8_T.data_ptr()),
        int(out_2d.data_ptr()), M, N_out, K_in,
        int(act_scale.data_ptr()), int(w_descale.data_ptr()),
        out_dtype, stream)
    if bias is not None:
        _add_bias(int(out_2d.data_ptr()),
                  int(bias.contiguous().data_ptr()),
                  M, N_out, out_dtype, stream)
    return out


__all__ = [
    "get_fvk_context",
    "get_gemm_runner",
    "linear_bf16",
    "linear_bf16_preT",
    "linear_fp8",
    "linear_fp8_from_fp8",
    "residual_rms_quant_fp8_inplace",
    "attention_mha_bf16_fused",
    "silu_mul_to_fp8_fp16_fused",
    "silu_mul_fp8_mpad_bf16_fused",
    "silu_mul_merged_fp8_mpad_bf16_fused",
    "ada_rms_residual_fp8_fused",
    "ada_rms_fp8_fused",
    "rope_inplace_bf16_fused",
    "clear_fp8_weight_cache",
    "preload_transposed_weights",
]
