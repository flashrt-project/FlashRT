"""GR00T N1.7 BF16 forward pipeline for Jetson Orin (SM87).

BF16 sibling of :mod:`flash_rt.models.groot_n17.pipeline_rtx_fp16`, with the
same stage decomposition and the same op order (which is the validated RTX
one), so a per-stage cosine comparison against the FP16 path or against HF
reference activations isolates dtype rather than algorithm.

Why BF16 and not a reuse of the FP16 pipeline:

* SM87 has no FP8 tensor cores, so the production spec's FP8 weights are
  dequantized anyway — see :mod:`flash_rt.models.groot_n17.weight_spec_orin`.
* Every INT8 kernel in the SM87 build is bf16-in / bf16-out
  (``quantize_int8_rowwise``, ``cutlass_int8_rowwise_bf16out``); there is no
  ``quantize_int8_rowwise_fp16`` in this build. INT8 is the one measured DiT
  lever on Orin (1.56x at Sa=41), so the surrounding math has to be bf16.
* The HF eager reference this port is gated against runs in bf16.

Kernel gaps on SM87 (bf16 exists for everything else):

===========================  ==========================================
missing bf16 kernel          what this module does instead
===========================  ==========================================
``gpu_repeat_interleave_     not needed — FA2 ``fwd_bf16_causal`` takes
heads`` (GQA expand)           ``num_heads_q`` / ``num_heads_kv`` separately
``attention_mha_causal``     FA2 ``fwd_bf16_causal``
``mul`` (elementwise a*b)    torch in-place ``mul_``
===========================  ==========================================

``rope_rotate_half`` used to be in that table. It is not a gap: the fused bf16
rotate-half kernel ``rope_neox_qk_bf16`` is built and exported — from the
*separate* ``flash_rt_qwen3_vl_kernels`` extension (``csrc/
qwen3_vl_bindings.cpp``), which is why a survey of ``flash_rt_kernels`` alone
missed it. Wiring it removed **11.902 ms** of GPU time per backbone (19.5% of
it), measured by monkeypatching the shim to a no-op and taking the delta rather
than by counting ops; the earlier "~2.7 ms" estimate was 4.4x low. The kernel is
**bit-identical** to the shim at every real shape (docs §6.22). The torch shim
(``_rope_rotate_half``) stays as the fallback for builds without that extension
and as the numerical reference the kernel is gated against.

Those shims need live tensors rather than raw pointers, so the ViT and LLM
forwards take an extra ``tbufs`` dict of tensors. Everything else stays
pointer-only: no allocation, no host/device traffic, CUDA-Graph capturable.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

#: One-shot flags so a fallback is announced once per process, not once per
#: layer (40 rope sites per backbone).
_warned_no_rope_kernel = False
_warned_full_width_table = False


#: Sentinel distinct from ``None`` so "resolved to absent" is cached too.
_VLK_UNRESOLVED = object()
_VLK_ROPE_QK = _VLK_UNRESOLVED


def _rope_neox_qk_kernel():
    """The fused bf16 rotate-half kernel, or ``None`` if this build lacks it.

    Resolved once and cached. It lives in ``flash_rt_qwen3_vl_kernels``, not in
    the ``flash_rt_kernels`` module every other call in this pipeline uses, so
    its absence means an older/partial build rather than a broken one — the
    torch shim below is correct, just ~12 ms/backbone slower. Announced once
    rather than silently (red line #4: a performance fallback still has to be
    observable).
    """
    global _VLK_ROPE_QK, _warned_no_rope_kernel
    if _VLK_ROPE_QK is _VLK_UNRESOLVED:
        try:
            from flash_rt import flash_rt_qwen3_vl_kernels as vlk
            _VLK_ROPE_QK = getattr(vlk, "rope_neox_qk_bf16", None)
        except ImportError:
            _VLK_ROPE_QK = None
        if _VLK_ROPE_QK is None and not _warned_no_rope_kernel:
            _warned_no_rope_kernel = True
            logger.warning(
                "flash_rt_qwen3_vl_kernels.rope_neox_qk_bf16 is unavailable; "
                "falling back to the torch rotate-half shim for the ViT and LLM "
                "rope. Results are bit-identical either way, but the shim costs "
                "~11.9 ms more GPU time per backbone (docs §6.22). Rebuild the "
                "qwen3_vl extension to get it back.")
    return _VLK_ROPE_QK


def _rope_qk(Q_t: torch.Tensor, K_t: torch.Tensor, tbufs: dict, rows: int,
             q_heads: int, k_heads: int, head_dim: int, stream) -> None:
    """Rotate Q and K in place — one fused launch instead of ten torch kernels.

    Replaces the two ``_rope_rotate_half`` calls this site used to make. The
    kernel reads bf16, accumulates in fp32 and writes bf16, i.e. the *same*
    rounding points as the shim, and it was verified **bit-identical** at all
    three real shapes (ViT MHA 512x16x64; LLM GQA 141x16x8x128 and
    148x16x8x128), with both arms equidistant from an fp64 reference. So this
    is purely a launch-count and memory-traffic win and carries no precision
    content — which is also why the gate for it is ``torch.equal``, not a
    cosine threshold.

    ``q_heads`` and ``k_heads`` are separate because the LLM site is GQA
    (16 vs 8). Swapping them does not raise; it rotates K with the wrong head
    stride, so callers pass them by name from the site's own dims.

    Falls back to the shim when the kernel is absent or when the caller could
    not supply half-width tables (see ``_rope_half_table``).
    """
    kern = _rope_neox_qk_kernel()
    cos_h = tbufs.get("cos_half")
    sin_h = tbufs.get("sin_half")
    if kern is not None and cos_h is not None and sin_h is not None:
        # In-place is explicitly supported: each thread owns one
        # (row, head, d < head_dim/2) triple and writes back exactly the two
        # elements it read, so there is no cross-thread hazard.
        kern(Q_t.data_ptr(), K_t.data_ptr(),
             cos_h.data_ptr(), sin_h.data_ptr(),
             Q_t.data_ptr(), K_t.data_ptr(),
             rows, q_heads, k_heads, head_dim, int(stream))
        return
    _rope_rotate_half(Q_t, tbufs["cos"], tbufs["sin"])
    _rope_rotate_half(K_t, tbufs["cos"], tbufs["sin"])


def _rope_rotate_half(x: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> None:
    """In-place Qwen3/LLaMA rotate-half RoPE, bf16 storage / fp32 math.

    **Fallback and numerical reference** — the serving path is
    :func:`_rope_qk`, whose fused bf16 kernel is bit-identical to this shim at
    every real shape. Kept because (a) builds without the
    ``flash_rt_qwen3_vl_kernels`` extension still have to run, and (b) it is
    what the kernel is gated against, so deleting it would leave nothing to
    compare to.

    Same math and same rounding points as both references:
    ``fvk.rope_rotate_half_fp16`` (``csrc/kernels/rope_qwen3.cu``, which does
    ``__half2float`` → fp32 → ``__float2half``) and HF's
    ``apply_rotary_pos_emb_vision`` (which upcasts q/k/cos/sin to float32 and
    casts back). Rotating in the storage dtype instead would lose ~3 decimal
    digits per layer and accumulate over 24 ViT + 16 LLM layers.

    Args:
        x:   ``(S, NH, HD)`` — modified in place.
        cos: ``(S, HD)`` whose two halves are identical (HF ``cat(emb, emb)``
             and ``build_vit_rope_tables`` both produce this), so indexing the
             full width is equivalent to the kernel's ``rope_idx = s*HD + d``
             over ``d < HD/2``.
        sin: same shape as ``cos``.
    """
    half = x.shape[-1] // 2
    xf = x.float()
    rot = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
    out = xf * cos.float().unsqueeze(1) + rot * sin.float().unsqueeze(1)
    x.copy_(out.to(x.dtype))


def _rope_half_table(t: torch.Tensor, what: str):
    """The unduplicated half of a ``cat(emb, emb)`` rope table, contiguous.

    ``rope_neox_qk_bf16`` indexes ``cos_tab[row * (head_dim/2) + d]`` and
    applies that one value to *both* halves of the head. That is equivalent to
    the shim's full-width indexing only when the table's two halves are
    identical — true for every table this port builds (HF's
    ``apply_interleaved_mrope`` and ``build_vit_rope_tables`` both end in
    ``cat(freqs, freqs)``), but assumed-is-not-checked here would mean a table
    from some other rope scheme silently rotates the second half of every head
    with the wrong frequencies. Nothing raises in that case; the cosine just
    drops. So check it.

    Returns ``None`` when the table is not half-slicable, which makes
    :func:`_rope_qk` fall back to the shim — correct but slower, so it is
    announced once rather than left silent.
    """
    global _warned_full_width_table
    if t.ndim != 2 or t.shape[1] % 2:
        reason = f"expected (rows, head_dim) with an even head_dim, got {tuple(t.shape)}"
    else:
        half = t.shape[1] // 2
        if torch.equal(t[:, :half], t[:, half:]):
            return t[:, :half].contiguous()
        reason = "its two halves are not identical"
    if not _warned_full_width_table:
        _warned_full_width_table = True
        logger.warning(
            "%s cannot feed the fused rotate-half kernel (%s); using the torch "
            "shim for this rope site. Bit-identical results, ~11.9 ms/backbone "
            "slower (docs §6.22).", what, reason)
    return None


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: VLLN — LayerNorm on backbone_features
# ─────────────────────────────────────────────────────────────────────────


def vlln_forward(gemm, fvk, bufs, weights, dims,
                 scales_dev=None, *, attn=None, stream: int = 0) -> None:
    """LayerNorm(2048) on backbone features ``(B, S, 2048)``.

    ``vlln`` is ``nn.LayerNorm(2048)`` with bias and PyTorch's default
    ``eps=1e-5`` (gr00t_n1d7.py:84). The kernel reads a flat ``S × D``
    row-major buffer regardless of leading batch dim, so ``S = B * seq_len``.

    Required:
        bufs["x"], bufs["out"]     — bf16 (S × D)
        weights["vlln_w"/"vlln_b"] — bf16 (D,)
        dims["S"], dims["D"]
    """
    fvk.layer_norm(
        int(bufs["x"]), int(weights["vlln_w"]), int(weights["vlln_b"]),
        int(bufs["out"]), int(dims["S"]), int(dims["D"]), 1e-5, int(stream),
    )


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Qwen3-VL ViT (24 layers) + DeepStack taps
# ─────────────────────────────────────────────────────────────────────────


def qwen3vl_vit_forward(gemm, fvk, bufs, weights, dims, tbufs,
                        *, attn, stream: int = 0,
                        layers_subset=None,
                        deepstack_taps=(5, 11, 17),
                        deepstack_capture=None) -> None:
    """24-layer Qwen3-VL ViT (bf16 GEMMs, multi-view batched FA2).

    Per layer, in-place residual updates:

        xn = LayerNorm(h, norm1_w/b, eps=1e-6)
        Q, K, V = xn @ {q,k,v}_w + {q,k,v}_b          — 3 split GEMMs
        Q, K = rope_rotate_half(Q|K, cos, sin)
        O = attn.run("vit", li, Sper_view, Sper_view)  — per-view FMHA
        h += O @ o_w + o_b
        xn = LayerNorm(h, norm2_w/b, eps=1e-6)
        h1 = gelu_tanh(xn @ fc1_w + fc1_b)
        h += h1 @ fc2_w + fc2_b

    The fused QKV is split into 3 GEMMs (vs HF's single fused Linear) so RoPE
    can be applied to contiguous Q/K with the split-half convention; attention
    runs in separated-Q/K/V mode. Q/K/V/O slots come from
    ``attn.get_slot_ptrs("vit", li)`` and are layer-shared.

    Args:
        bufs: ``h`` (S, D) in-place, ``xn`` (S, D), ``o_proj_out`` (S, D),
            ``fc1_out`` (S, ff_inner) — bf16 device pointers.
        tbufs: ``Q`` (S, NH, HD), ``K`` (S, NH, HD), ``cos`` (S, HD),
            ``sin`` (S, HD) — live bf16 tensors aliasing the same slots.
            ``cos_half``/``sin_half`` (S, HD/2) are the same tables with the
            duplicated half dropped, which is what the fused rope kernel
            indexes; when they are absent ``_rope_qk`` falls back to the shim
            and reads ``cos``/``sin`` instead.
        weights: per-layer lists ``norm1_w/b``, ``norm2_w/b``, ``q_w/b``,
            ``k_w/b``, ``v_w/b``, ``o_w/b``, ``fc1_w/b``, ``fc2_w/b``.
        dims: ``S``, ``D``, ``NH``, ``HD``, ``ff_inner``, ``Sper_view``.
        deepstack_taps: layer indices exposed to ``deepstack_capture``.
        deepstack_capture: ``list[Callable[[int], None]]`` receiving the ``h``
            pointer right after each tap layer's residual update.
    """
    S = int(dims["S"])
    D = int(dims["D"])
    NH = int(dims["NH"])
    HD = int(dims["HD"])
    FF = int(dims["ff_inner"])

    h_ptr = int(bufs["h"])
    xn_ptr = int(bufs["xn"])
    o_proj_out = int(bufs["o_proj_out"])
    fc1_out_ptr = int(bufs["fc1_out"])
    Q_t = tbufs["Q"].view(S, NH, HD)
    K_t = tbufs["K"].view(S, NH, HD)

    layer_iter = range(24) if layers_subset is None else list(layers_subset)
    Sper = int(dims.get("Sper_view", S))

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vit", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = (
            slots["Q"], slots["K"], slots["V"], slots["O"])

        fvk.layer_norm(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["q_w"][li]), Q_ptr, S, D, D, int(stream))
        fvk.add_bias_bf16(Q_ptr, int(weights["q_b"][li]), S, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["k_w"][li]), K_ptr, S, D, D, int(stream))
        fvk.add_bias_bf16(K_ptr, int(weights["k_b"][li]), S, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["v_w"][li]), V_ptr, S, D, D, int(stream))
        fvk.add_bias_bf16(V_ptr, int(weights["v_b"][li]), S, D, int(stream))

        _rope_qk(Q_t, K_t, tbufs, S, NH, NH, HD, stream)

        attn.run("vit", li, q_seq=Sper, kv_seq=Sper, stream=int(stream))

        gemm.bf16_nn(O_ptr, int(weights["o_w"][li]), o_proj_out, S, D, D, int(stream))
        fvk.add_bias_bf16(o_proj_out, int(weights["o_b"][li]), S, D, int(stream))
        fvk.residual_add(h_ptr, o_proj_out, S * D, int(stream))

        fvk.layer_norm(
            h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                     S, FF, D, int(stream))
        fvk.add_bias_bf16(fc1_out_ptr, int(weights["fc1_b"][li]), S, FF, int(stream))
        fvk.gelu_inplace(fc1_out_ptr, S * FF, int(stream))
        gemm.bf16_nn(fc1_out_ptr, int(weights["fc2_w"][li]), o_proj_out,
                     S, D, FF, int(stream))
        fvk.add_bias_bf16(o_proj_out, int(weights["fc2_b"][li]), S, D, int(stream))
        fvk.residual_add(h_ptr, o_proj_out, S * D, int(stream))

        if deepstack_capture is not None and li in deepstack_taps:
            deepstack_capture[deepstack_taps.index(li)](h_ptr)


def deepstack_merge_forward(gemm, fvk, bufs, weights, dims,
                            *, attn=None, stream: int = 0) -> None:
    """3 DeepStack mergers tapping ViT layers ``[5, 11, 17]``.

    Per HF ``Qwen3VLVisionPatchMerger`` with ``use_postshuffle_norm=True``,
    each merger ``j``:

        x = LayerNorm(tap_j.view(N//4, 4*D), norm_w/b, eps=1e-6)
        x = gelu_erf(x @ fc1_w + fc1_b)     # (Nout, Dmid) -> (Nout, Dmid)
        out_j = x @ fc2_w + fc2_b           # -> (Nout, Dout=2048)

    The spatial-merge reshape is a no-op pointer-wise: row-major
    ``(N, D)`` and ``(N//4, 4*D)`` share the same byte layout.

    Args:
        bufs: ``in`` list[3] bf16 ptrs (Nin, Din); ``ln_out``, ``fc1_out``
            shared bf16 ptrs (Nout, Dmid); ``out`` list[3] bf16 ptrs (Nout, Dout).
        weights: per-merger ``norm_w/b``, ``fc1_w/b``, ``fc2_w/b``.
        dims: ``Nin``, ``Din``, ``Nout``, ``Dmid``, ``Dout``.
    """
    Nout = int(dims["Nout"])
    Dmid = int(dims["Dmid"])
    Dout = int(dims["Dout"])

    ln_out = int(bufs["ln_out"])
    fc1_out = int(bufs["fc1_out"])

    for j in range(3):
        in_ptr = int(bufs["in"][j])
        out_ptr = int(bufs["out"][j])

        fvk.layer_norm(
            in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
            ln_out, Nout, Dmid, 1e-6, int(stream))

        gemm.bf16_nn(ln_out, int(weights["fc1_w"][j]), fc1_out,
                     Nout, Dmid, Dmid, int(stream))
        fvk.add_bias_bf16(fc1_out, int(weights["fc1_b"][j]), Nout, Dmid, int(stream))
        # nn.GELU() — exact erf. Qwen3VLVisionPatchMerger hardcodes it and does
        # NOT read vision_config.hidden_act ("gelu_pytorch_tanh"), which applies
        # only to the vision blocks. The tanh variant here costs ~1e-3 cosine
        # per merger and compounds into the LLM's image tokens.
        fvk.gelu_erf_bf16(fc1_out, Nout * Dmid, int(stream))

        gemm.bf16_nn(fc1_out, int(weights["fc2_w"][j]), out_ptr,
                     Nout, Dout, Dmid, int(stream))
        fvk.add_bias_bf16(out_ptr, int(weights["fc2_b"][j]), Nout, Dout, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# Stage 3: truncated Qwen3-VL LLM (16 layers, causal GQA, M-RoPE)
# ─────────────────────────────────────────────────────────────────────────


def qwen3vl_llm_forward(gemm, fvk, bufs, weights, dims, tbufs,
                        *, attn, stream: int = 0,
                        layers_subset=None) -> None:
    """16 truncated Qwen3-VL LLM decoder layers (bf16).

    Per layer, in-place residual updates:

        xn = RMSNorm(h, in_ln_w, eps=1e-6)
        Q = xn @ q_w          — (S, NHQ*HD)
        K = xn @ k_w          — (S, NHKV*HD)
        V = xn @ v_w
        Q = RMSNorm(Q, q_norm_w, HD)   # per-head, BEFORE rope (Qwen3)
        K = RMSNorm(K, k_norm_w, HD)
        Q, K = rope_rotate_half(Q|K, mrope_cos, mrope_sin)
        O = attn.run("llm", li, S, S)  # FA2 causal, native GQA (no expand)
        h += O @ o_w
        xn = RMSNorm(h, post_ln_w, eps=1e-6)
        gate = xn @ gate_w; up = xn @ up_w
        gate = silu(gate) * up
        h += gate @ down_w
        if li in deepstack layers: h += deepstack_inject[li]

    Per-head Q/K RMSNorm is expressed as one flat call over ``(S*NH, HD)``:
    the norm is over the last dim and the weight is shared across heads, so
    flat-and-norm is identical to a per-head loop.

    Unlike the FP16 RTX path there is no GQA pre-expansion: FA2's
    ``fwd_bf16_causal`` takes ``num_heads_q=16`` and ``num_heads_kv=8``
    directly, so K/V slots hold 8 heads and ``gpu_repeat_interleave_heads``
    (FP16-only in this build) is not called.

    An INT8 FFN tier was built and measured here, then removed: it saves 3.93 ms
    of GPU kernel time over the 16 layers but only 0.6-1.0 ms of eager wall,
    and it drops ``backbone_features`` to cos 0.992077 against the shipped
    ``THR_FUSED_CONSUMED`` floor of 0.995. See docs §6.11 before rebuilding it.

    Args:
        bufs: ``h``, ``xn`` (S, D); ``Q`` (S, NHQ*HD); ``K``, ``V``
            (S, NHKV*HD); ``o_proj_out`` (S, D); ``gate_out``, ``up_out``
            (S, FF) — bf16 device pointers.
        tbufs: ``Q`` (S, NHQ, HD), ``K`` (S, NHKV, HD), ``cos``/``sin``
            (S, HD), ``gate``/``up`` (S, FF) — live bf16 tensors.
            ``cos_half``/``sin_half`` (S, HD/2) as for the ViT forward.
        weights: per-layer ``in_ln_w``, ``post_ln_w``, ``q_norm_w``,
            ``k_norm_w``, ``q_w``, ``k_w``, ``v_w``, ``o_w``, ``gate_w``,
            ``up_w``, ``down_w``; plus ``deepstack_inject`` (length-16 list of
            int ptrs, 0 = no injection).
        dims: ``S``, ``D``, ``NHQ``, ``NHKV``, ``HD``, ``FF``.
    """
    S = int(dims["S"])
    D = int(dims["D"])
    NHQ = int(dims["NHQ"])
    NHKV = int(dims["NHKV"])
    HD = int(dims["HD"])
    FF = int(dims["FF"])

    h_ptr = int(bufs["h"])
    xn_ptr = int(bufs["xn"])
    Q_ptr = int(bufs["Q"])
    K_ptr = int(bufs["K"])
    V_ptr = int(bufs["V"])
    o_out_ptr = int(bufs["o_proj_out"])
    gate_ptr = int(bufs["gate_out"])
    up_ptr = int(bufs["up_out"])
    Q_t = tbufs["Q"].view(S, NHQ, HD)
    K_t = tbufs["K"].view(S, NHKV, HD)
    gate_t = tbufs["gate"]
    up_t = tbufs["up"]

    inject_ptrs = weights.get("deepstack_inject", [0] * 16)
    layer_iter = range(16) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        slots = attn.get_slot_ptrs("llm")

        fvk.rms_norm(h_ptr, int(weights["in_ln_w"][li]), xn_ptr,
                     S, D, 1e-6, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["q_w"][li]), Q_ptr,
                     S, NHQ * HD, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["k_w"][li]), K_ptr,
                     S, NHKV * HD, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["v_w"][li]), V_ptr,
                     S, NHKV * HD, D, int(stream))

        fvk.rms_norm(Q_ptr, int(weights["q_norm_w"][li]), Q_ptr,
                     S * NHQ, HD, 1e-6, int(stream))
        fvk.rms_norm(K_ptr, int(weights["k_norm_w"][li]), K_ptr,
                     S * NHKV, HD, 1e-6, int(stream))

        _rope_qk(Q_t, K_t, tbufs, S, NHQ, NHKV, HD, stream)

        attn.run("llm", li, q_seq=S, kv_seq=S, stream=int(stream))

        gemm.bf16_nn(int(slots["O"]), int(weights["o_w"][li]), o_out_ptr,
                     S, D, NHQ * HD, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, S * D, int(stream))

        fvk.rms_norm(h_ptr, int(weights["post_ln_w"][li]), xn_ptr,
                     S, D, 1e-6, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["gate_w"][li]), gate_ptr,
                     S, FF, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["up_w"][li]), up_ptr,
                     S, FF, D, int(stream))
        fvk.silu_bf16(gate_ptr, S * FF, int(stream))
        gate_t.mul_(up_t)
        gemm.bf16_nn(gate_ptr, int(weights["down_w"][li]), o_out_ptr,
                     S, D, FF, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, S * D, int(stream))

        inject_ptr = int(inject_ptrs[li]) if li < len(inject_ptrs) else 0
        if inject_ptr != 0:
            fvk.residual_add(h_ptr, inject_ptr, S * D, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# Stage 5: VL self-attention (4 layers)
# ─────────────────────────────────────────────────────────────────────────


def vl_self_attn_forward(gemm, fvk, bufs, weights, dims,
                         *, attn, stream: int = 0,
                         layers_subset=None) -> None:
    """4-layer ``SelfAttentionTransformer`` (diffusers BasicTransformerBlock,
    ``norm_type="layer_norm"``, ``activation_fn="gelu-approximate"``,
    ``positional_embeddings=None`` per the N1.7 config). No RoPE.

    Per layer, in-place residual updates:

        xn = LayerNorm(h, norm1_w/b, eps=1e-5)
        Q, K, V = xn @ {q,k,v}_w + {q,k,v}_b      — (T, 2048)
        O = attn.run("vl_self_attn", li, T, T)     — MHA 32x64
        h += O @ o_w + o_b
        xn = LayerNorm(h, norm3_w/b, eps=1e-5)
        h1 = gelu_tanh(xn @ fc1_w + fc1_b)         — (T, 8192)
        h += h1 @ fc2_w + fc2_b

    Q/K/V/O slots are shared across the 4 layers (layer-sequential).

    Args:
        bufs: ``h`` (T, D) in-place, ``xn`` (T, D), ``o_proj_out`` (T, D),
            ``fc1_out`` (T, ff_inner) — bf16 device pointers.
        weights: per-layer ``norm1_w/b``, ``norm3_w/b``, ``q_w/b``, ``k_w/b``,
            ``v_w/b``, ``o_w/b``, ``fc1_w/b``, ``fc2_w/b``.
        dims: ``T``, ``D``, ``NH``, ``HD``, ``ff_inner``.
    """
    T = int(dims["T"])
    D = int(dims["D"])
    FF = int(dims["ff_inner"])

    h_ptr = int(bufs["h"])
    xn_ptr = int(bufs["xn"])
    o_proj_out = int(bufs["o_proj_out"])
    fc1_out_ptr = int(bufs["fc1_out"])

    layer_iter = range(4) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vl_self_attn", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = (
            slots["Q"], slots["K"], slots["V"], slots["O"])

        fvk.layer_norm(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["q_w"][li]), Q_ptr, T, D, D, int(stream))
        fvk.add_bias_bf16(Q_ptr, int(weights["q_b"][li]), T, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["k_w"][li]), K_ptr, T, D, D, int(stream))
        fvk.add_bias_bf16(K_ptr, int(weights["k_b"][li]), T, D, int(stream))
        gemm.bf16_nn(xn_ptr, int(weights["v_w"][li]), V_ptr, T, D, D, int(stream))
        fvk.add_bias_bf16(V_ptr, int(weights["v_b"][li]), T, D, int(stream))

        attn.run("vl_self_attn", li, q_seq=T, kv_seq=T, stream=int(stream))

        gemm.bf16_nn(O_ptr, int(weights["o_w"][li]), o_proj_out, T, D, D, int(stream))
        fvk.add_bias_bf16(o_proj_out, int(weights["o_b"][li]), T, D, int(stream))
        fvk.residual_add(h_ptr, o_proj_out, T * D, int(stream))

        fvk.layer_norm(
            h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream))

        gemm.bf16_nn(xn_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                     T, FF, D, int(stream))
        fvk.add_bias_bf16(fc1_out_ptr, int(weights["fc1_b"][li]), T, FF, int(stream))
        fvk.gelu_inplace(fc1_out_ptr, T * FF, int(stream))
        gemm.bf16_nn(fc1_out_ptr, int(weights["fc2_w"][li]), o_proj_out,
                     T, D, FF, int(stream))
        fvk.add_bias_bf16(o_proj_out, int(weights["fc2_b"][li]), T, D, int(stream))
        fvk.residual_add(h_ptr, o_proj_out, T * D, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# Stage 7b: AlternateVLDiT (32 layers) — the per-inference hot path
# ─────────────────────────────────────────────────────────────────────────
#
# INT8 tier (SM87 rowwise). Two helpers, one contract:
#
# * The activation scale stays **device-side** end to end. ``quantize_int8_
#   rowwise`` writes it to a device pointer and ``cutlass_int8_rowwise_bf16out``
#   reads it back there; nothing is ever copied to the host. Reading it on the
#   host would break CUDA-graph capture, and the graph is not optional here:
#   measured, the INT8 DiT is 33.39 ms/step eager vs 9.37 ms/step captured,
#   i.e. **2.1x SLOWER than bf16 without the graph** (docs §6.3).
# * The quantize is a **separate pass**, not fused into the norms. That is a
#   kernel gap, not a choice: SM87 has no INT8-output ``ada_layer_norm`` and no
#   INT8-output ``layer_norm_no_affine``. ``gate_residual_ada_norm_int8`` looks
#   like the right kernel but computes ``rsqrt(mean(r*r) + eps)`` — RMS, no mean
#   subtraction (csrc/kernels/fusion.cu:190) — while this DiT uses a
#   mean-subtracting ``AdaLayerNorm`` at eps=1e-5. Using it would silently
#   change the math, so the gap is reported instead (AGENTS.md red line #8).
#   The extra traffic is ~42 MB/step against 2182 MB/step of bf16 weight
#   traffic, and the launches are absorbed by the graph.
#   Size of the gap, measured at M=141 on the LLM shapes: the pass costs
#   **~37 us whether cols is 2048 or 6144** — flat, i.e. latency-bound rather
#   than bandwidth-bound (docs §6.11.1). Fusing it into the preceding norm's
#   epilogue is what would recover it, at every site that has one.


def _int8_quantize(fvk, src_ptr, i8_ptr, scale_ptr, rows, cols, stream) -> None:
    """bf16 (rows, cols) -> int8 (rows, cols) + fp32 (rows,) per-token scale."""
    fvk.quantize_int8_rowwise(int(src_ptr), int(i8_ptr), int(scale_ptr),
                              int(rows), int(cols), int(stream))


def _int8_nn(fvk, a8_ptr, w8_ptr, act_scale_ptr, weight_scale_ptr, out_ptr,
             M, N, K, stream) -> None:
    """int8 A (M,K) @ int8 B (N,K)^T -> bf16 out (M,N), rowwise scales.

    B is the **untransposed** ``nn.Linear`` layout (N, K) — the opposite of the
    bf16 tier, where the spec's ``T()`` stores (K, N) for ``gemm.bf16_nn``.
    """
    status = fvk.cutlass_int8_rowwise_bf16out(
        int(a8_ptr), int(w8_ptr), int(act_scale_ptr), int(weight_scale_ptr),
        int(out_ptr), int(M), int(N), int(K), int(stream))
    if status != 0:
        raise RuntimeError(
            f"cutlass_int8_rowwise_bf16out failed: status={status} "
            f"shape=({M},{N},{K})")


def dit_forward(gemm, fvk, bufs, weights, dims,
                *, attn, stream: int = 0, layers_subset=None) -> None:
    """32-layer ``AlternateVLDiT`` (``interleave_self_attention=True``,
    ``attend_text_every_n_blocks=2``). Two tiers, same op order:

    * **bf16** (default) — ``gemm.bf16_nn`` for all six GEMM sites.
    * **rowwise INT8** — selected by the presence of ``weights["q_w8"]``;
      ``quantize_int8_rowwise`` + ``cutlass_int8_rowwise_bf16out``. Measured
      1.56x over bf16 **under CUDA graph** and 2.1x *slower* without it (§6.3).
      The precision gate is §6.8: per-row W8A8 on real activations gives
      cos >= 0.999999 and worst-case 0.255 deg on the decoded action.

    Per HF ``AlternateVLDiT.forward`` (dit.py:339):
      * odd ``li``  -> SELF-attn over ``(B, Sa, D=1536)``
      * even ``li`` -> CROSS-attn to the post-VLSA backbone features, whose
        target alternates every 2 cross blocks: ``li % 4 == 0`` -> text
        (non-visual) positions, else visual positions. Cross K/V are
        precomputed once per prompt by the frontend and live in the
        ``dit_cross`` site's per-block slots.

    Per BasicTransformerBlock with ``norm_type="ada_norm"`` and
    ``activation_fn="gelu-approximate"`` (tanh — ``fvk.gelu_inplace`` is the
    tanh variant, ``csrc/kernels/activation.cu:41``):

        xn = AdaLayerNorm(h, scale_msa[li], shift_msa[li], eps=1e-5)
        Q = xn @ q_w + q_b                       — (Sa, D)
        [self]  K, V = xn @ {k,v}_w + {k,v}_b ; O = attn("dit_self", ...)
        [cross] O = attn("dit_cross", ..., kv_seq=Skv_text|Skv_image)
        h += O @ o_w + o_b
        xn = LayerNorm_no_affine(h, eps=1e-5)
        ff = gelu_tanh(xn @ ff_proj_w + ff_proj_b)   — (Sa, 6144)
        h += ff @ ff_down_w + ff_down_b

    ``bf16_nn_bias`` / ``bf16_nn_bias_gelu`` epilogues hit
    ``CUBLAS_STATUS_NOT_SUPPORTED`` at M=Sa=41 (M not 16-aligned), so bias is
    a separate ``add_bias_bf16`` — same workaround the N1.6 calibrate path
    and the FP16 RTX path use. Bias stays bf16 in **both** tiers: the INT8
    kernel's epilogue applies the rowwise scales and emits bf16, so the bias
    add is unchanged.

    Args:
        bufs: ``h``, ``xn`` (Sa, D); ``o_proj_out`` (Sa, D);
            ``ff_proj_out`` (Sa, FF) — bf16 device pointers. The INT8 tier
            additionally requires eight pre-allocated scratch entries —
            ``xn1_i8``/``xn2_i8``/``o_i8`` (Sa, D) int8, ``ff_i8`` (Sa, FF)
            int8, and ``xn1_s``/``xn2_s``/``o_s``/``ff_s`` (Sa,) fp32. They
            must exist **before** graph capture; a missing one raises rather
            than falling back to bf16, because a silent fallback is
            numerically plausible and would invalidate every INT8 measurement.
        weights: length-32 lists ``scale_msa``, ``shift_msa`` (bf16 (D,),
            precomputed per denoise step by the frontend), ``q_w/b``,
            ``k_w/b``, ``v_w/b``, ``o_w/b``, ``ff_proj_w/b``, ``ff_down_w/b``.
            INT8 tier adds, per family ``F`` in {q,k,v,o,ff_proj,ff_down}:
            ``F_w8`` (int8 (N,K), the **untransposed** nn.Linear layout) and
            ``F_s`` (fp32 (N,) per-output-channel scale).
            Optional ``bf16_families`` (a subsequence of those six names)
            exempts families from INT8, so a site can stay bf16 where INT8 is
            measured not to pay; it is only meaningful alongside the INT8
            weights and raises otherwise. An exempt family must not ship
            ``_w8`` and a non-exempt one must — both directions raise rather
            than guessing a GEMM's tier.
        dims: ``Sa``, ``D``, ``FF``, ``Skv_text``, ``Skv_image``.
    """
    Sa = int(dims["Sa"])
    D = int(dims["D"])
    FF = int(dims["FF"])
    Skv_text = int(dims.get("Skv_text", 0))
    Skv_image = int(dims.get("Skv_image", 0))

    h_ptr = int(bufs["h"])
    xn_ptr = int(bufs["xn"])
    o_out_ptr = int(bufs["o_proj_out"])
    ff_out_ptr = int(bufs["ff_proj_out"])

    use_int8 = "q_w8" in weights

    # ── tier dispatch, resolved once (not per call site) ──
    # The INT8 tier can exempt individual weight families, which the frontend
    # uses to keep a projection in bf16 where INT8 is measured not to pay. An
    # exempt family must NOT ship ``_w8`` and a non-exempt one MUST; both
    # directions raise, because a tier mix-up is numerically plausible and
    # would silently invalidate every INT8 measurement.
    #
    # mmq() runs one GEMM site in whichever tier its family is configured for.
    # In the bf16 tier every family resolves to gemm.bf16_nn and no quantize
    # call is made, so the call sequence is identical to a tier-free pipeline.
    fams = ("q", "k", "v", "o", "ff_proj", "ff_down")
    keep_bf16 = frozenset(weights.get("bf16_families", ()))
    unknown = keep_bf16 - set(fams)
    if unknown:
        raise KeyError(
            f"bf16_families names {sorted(unknown)}, which are not DiT weight "
            f"families {list(fams)}")
    i8 = {fam: (use_int8 and fam not in keep_bf16) for fam in fams}

    if use_int8:
        missing = [k for k in ("xn1_i8", "xn1_s", "xn2_i8", "xn2_s",
                               "o_i8", "o_s", "ff_i8", "ff_s")
                   if k not in bufs]
        if missing:
            raise KeyError(
                f"INT8 DiT tier needs scratch buffer(s) {missing}, but the "
                f"caller allocated the bf16 set. Allocate them before graph "
                f"capture; refusing to fall back to bf16 silently.")
        no_w8 = [f for f in fams if i8[f] and f + "_w8" not in weights]
        has_w8 = [f for f in fams if not i8[f] and f + "_w8" in weights]
        if no_w8 or has_w8:
            raise KeyError(
                f"INT8 DiT tier with bf16_families={sorted(keep_bf16)}: "
                f"families {no_w8} are INT8 but ship no _w8/_s, and {has_w8} "
                "are exempt but ship one anyway. The family list and the "
                "weight dict must agree; refusing to guess a GEMM's tier.")

        xn1_i8, xn1_s = int(bufs["xn1_i8"]), int(bufs["xn1_s"])
        xn2_i8, xn2_s = int(bufs["xn2_i8"]), int(bufs["xn2_s"])
        o_i8, o_s = int(bufs["o_i8"]), int(bufs["o_s"])
        ff_i8, ff_s = int(bufs["ff_i8"]), int(bufs["ff_s"])
    else:
        if keep_bf16:
            raise KeyError(
                f"bf16_families={sorted(keep_bf16)} was passed to the bf16 "
                "tier, where every family is already bf16; it only means "
                "something alongside the INT8 weights")

        xn1_i8 = xn1_s = xn2_i8 = xn2_s = 0
        o_i8 = o_s = ff_i8 = ff_s = 0

    # One definition, keyed on the per-family map: defining this per tier
    # branch instead (the obvious reading) makes every site INT8 whenever the
    # tier is, which silently ignores bf16_families and then KeyErrors on the
    # exempt family's absent _w8.
    def mmq(fam, li, a_bf16, a_i8, a_s, out, M, N, K):
        if i8[fam]:
            _int8_nn(fvk, a_i8, int(weights[fam + "_w8"][li]), a_s,
                     int(weights[fam + "_s"][li]), out, M, N, K, stream)
        else:
            gemm.bf16_nn(a_bf16, int(weights[fam + "_w"][li]), out,
                         M, N, K, int(stream))

    # xn after adaLN feeds Q and (self layers only) K and V, so it is quantized
    # once and shared by all three -- but only if at least one of them is INT8.
    qz_xn1 = i8["q"] or i8["k"] or i8["v"]

    layer_iter = range(32) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        is_self = (li % 2 == 1)
        # The backend's dit_self / dit_cross sites are indexed within their
        # own kind (16 entries each), not by the 0..31 layer index.
        j_attn = (li - 1) // 2 if is_self else li // 2

        fvk.ada_layer_norm_bf16(
            h_ptr, int(weights["scale_msa"][li]), int(weights["shift_msa"][li]),
            xn_ptr, Sa, D, 1e-5, int(stream))

        slots = attn.get_slot_ptrs("dit_self" if is_self else "dit_cross", j_attn)
        Q_ptr, K_ptr, V_ptr, O_ptr = (
            slots["Q"], slots["K"], slots["V"], slots["O"])

        # xn after adaLN feeds Q and (self layers only) K and V, so the three
        # sites share one quantization of it (see qz_xn1 above).
        if qz_xn1:
            _int8_quantize(fvk, xn_ptr, xn1_i8, xn1_s, Sa, D, stream)

        mmq("q", li, xn_ptr, xn1_i8, xn1_s, Q_ptr, Sa, D, D)
        fvk.add_bias_bf16(Q_ptr, int(weights["q_b"][li]), Sa, D, int(stream))

        if is_self:
            mmq("k", li, xn_ptr, xn1_i8, xn1_s, K_ptr, Sa, D, D)
            fvk.add_bias_bf16(K_ptr, int(weights["k_b"][li]), Sa, D, int(stream))
            mmq("v", li, xn_ptr, xn1_i8, xn1_s, V_ptr, Sa, D, D)
            fvk.add_bias_bf16(V_ptr, int(weights["v_b"][li]), Sa, D, int(stream))
            attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa, stream=int(stream))
        else:
            kv_seq = Skv_text if (li % 4 == 0) else Skv_image
            attn.run("dit_cross", j_attn, q_seq=Sa, kv_seq=kv_seq,
                     stream=int(stream))

        if i8["o"]:
            _int8_quantize(fvk, O_ptr, o_i8, o_s, Sa, D, stream)
        mmq("o", li, O_ptr, o_i8, o_s, o_out_ptr, Sa, D, D)
        fvk.add_bias_bf16(o_out_ptr, int(weights["o_b"][li]), Sa, D, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, Sa * D, int(stream))

        fvk.layer_norm_no_affine_bf16(h_ptr, xn_ptr, Sa, D, 1e-5, int(stream))

        if i8["ff_proj"]:
            _int8_quantize(fvk, xn_ptr, xn2_i8, xn2_s, Sa, D, stream)
        mmq("ff_proj", li, xn_ptr, xn2_i8, xn2_s, ff_out_ptr, Sa, FF, D)
        fvk.add_bias_bf16(ff_out_ptr, int(weights["ff_proj_b"][li]),
                          Sa, FF, int(stream))
        fvk.gelu_inplace(ff_out_ptr, Sa * FF, int(stream))
        if i8["ff_down"]:
            _int8_quantize(fvk, ff_out_ptr, ff_i8, ff_s, Sa, FF, stream)
        mmq("ff_down", li, ff_out_ptr, ff_i8, ff_s, o_out_ptr, Sa, D, FF)
        fvk.add_bias_bf16(o_out_ptr, int(weights["ff_down_b"][li]),
                          Sa, D, int(stream))
        fvk.residual_add(h_ptr, o_out_ptr, Sa * D, int(stream))
