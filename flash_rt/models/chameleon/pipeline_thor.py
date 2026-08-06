"""FlashRT — standalone Chameleon-7B Thor SM110 pipeline forward functions.

Chameleon-7B LLM (32-layer MHA, attention_bias=false, mlp_bias=false,
per-head QK LayerNorm + RoPE). Used by the standalone Chameleon Thor
frontend.

All functions use raw-pointer interface (int pointers + Python primitives)
for CUDA Graph compatibility. No dynamic allocation, no torch ops, no sync.

Functions:
    chameleon_forward            — Chameleon-7B LLM inference (dynamic FP8 default)
    chameleon_forward_fp16       — pure-FP16 reference path
    chameleon_forward_calibrate  — FP8 static-scale calibration
"""

from __future__ import annotations

import math

import flash_rt.flash_rt_kernels as _fvk
import torch

from flash_rt.hardware.thor.shared_primitives import (
    _measure_scale_gpu,
    _gpu_copy,
    _gpu_sync,
    _gpu_zero,
)

try:
    import flash_rt.flash_rt_fp4 as _fvk_fp4
except Exception:
    _fvk_fp4 = None


def _parse_fp4_layer_policy() -> frozenset:
    """FP4 FFN layer policy from FLASHRT_CHAMELEON_FP4_LAYERS env var.

    Values:
      - unset / "": default = L0-L2 FP4, L3-L31 FP8 (safe default).
      - "0-7": FP4 for L0..L7 inclusive.
      - "0-14,20-31": FP4 for L0..L14 + L20..L31 (skip outlier L15-L19).
        This is the SM120-sweep-validated aggressive setting (13ms savings on
        RTX 5090; expect similar Thor gains).
      - comma-separated list: "0,1,2,5" → FP4 on those specific layers.

    Returns the frozenset of FP8 layer indices (complement of FP4 set).
    """
    import os as _os
    val = _os.environ.get("FLASHRT_CHAMELEON_FP4_LAYERS", "").strip()
    if not val:
        return frozenset(range(3, 32))  # default: L0-L2 FP4

    fp4_layers = set()
    for chunk in val.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            fp4_layers.update(range(int(a), int(b) + 1))
        else:
            fp4_layers.add(int(chunk))
    fp4_layers = {li for li in fp4_layers if 0 <= li < 32}
    return frozenset(li for li in range(32) if li not in fp4_layers)


# ── FP4 GEMM variant and layer policy ──
FP4_VARIANT = 9
_FFN_FP8_LAYERS = _parse_fp4_layer_policy()


# ══════════════════════════════════════════════════════════════════
# Chameleon-7B LLM (32 layers, mixed FP4/FP8 GEMMs)
# ══════════════════════════════════════════════════════════════════

def chameleon_forward(
    gemm, fvk, bufs, weights, dims, scales_dev,
    *, attn, stream: int = 0,
    alpha_host=None, awq_v_proj=None,
    ffn_down_clamp_value: float = 10000.0,
    ffn_clamp_layers=None,
    dynamic_fp8_layers: frozenset = frozenset(),
    fp4_ffn_layers: frozenset = frozenset(),
    probe=None,
) -> None:
    """Chameleon-7B LLM forward pass (32 layers).

    Production precision: dynamic per-tensor FP8 on all layers, with the FFN of
    ``fp4_ffn_layers`` optionally run in NVFP4 W4A16 (decoupled from attention).

    Output: hidden_all = RMSNorm(x) written to bufs['hidden_all'] as
    full [Se, D] FP16 tensor (consumed by action_head_forward).

    ``ffn_down_clamp_value``: Clamp down_out (o_proj_out) to ±V after
    the FFN before residual_2. Chameleon-7B L31 down_proj's FP32
    accumulator × alpha can push output beyond ±65504 producing inf.
    Applied after both FP4 and FP8 FFN paths. Set to <= 0 to disable.

    ``ffn_clamp_layers``: Optional set/frozenset of layer indices where the
    FFN clamps are applied. ``None`` preserves the historical behavior and
    clamps every layer; callers can pass e.g. ``frozenset({31})`` to clamp
    only the deep outlier layer and avoid redundant elementwise passes.

    ``dynamic_fp8_layers``: frozenset of layer indices to run with full FP8
    GEMMs but RUNTIME (per-forward) per-tensor activation scaling instead of
    the static calibrated scale. Uses quantize_fp8_device_fp16 (GPU amax) +
    fp8_nn_dev_fp16 (device-scale GEMM). Restores 512-res precision (static
    scale is wrong for long sequences) at full FP8 speed. CUDA-Graph safe.
    This is the default for all 32 layers; see docs/chameleon_thor_sm110.md.

    ``fp4_ffn_layers``: frozenset of layer indices whose FFN runs in NVFP4
    W4A16 while attention stays on the dynamic-FP8 path (decoupled). Requires
    those layers' FP4 weights to be packed. 512-res default = L0-7 (~1.10x E2E,
    cos >= 0.99 on 3/4 variants); empty at 256-res (FP4 breaks precision). See
    docs/chameleon_thor_sm110.md (§6 FP4).

    Optional ``probe`` dict for layer-wise precision debugging::

        {
            'layers':    tuple[int, ...],  # layer indices to snapshot post-residual-2
            'bufs':      tuple[int, ...],  # device pointers, one per layer index,
                                            # each must hold >= Se*D fp16 elements
            'final_buf': int,              # device pointer for post-final-RMSNorm
                                            # snapshot. Pass 0 to skip.
        }
    """
    Se = int(dims['Se'])
    D = int(dims['D'])
    Dff = int(dims['Dff'])
    L = int(dims['L'])
    H = int(dims['H'])
    Hd = int(dims['Hd'])

    if alpha_host is None:
        alpha_host = weights.get('alpha_host')
    if awq_v_proj is None:
        awq_v_proj = weights.get('awq_v_proj')

    x_ptr = int(bufs['x'])
    xn_ptr = int(bufs['xn'])
    xn_fp8_ptr = int(bufs['xn_fp8'])
    o_proj_out_ptr = int(bufs['o_proj_out'])
    hidden_all_ptr = int(bufs['hidden_all'])

    # FP4 FFN buffers
    act_fp4_ptr = int(bufs['act_fp4'])
    act_sfa_ptr = int(bufs['act_sfa'])
    ffn_act_fp4_ptr = int(bufs['ffn_act_fp4'])
    ffn_act_sfa_ptr = int(bufs['ffn_act_sfa'])
    gu_merged_ptr = int(bufs['gu_merged'])

    cos_ptr = int(weights['rope_cos'])
    sin_ptr = int(weights['rope_sin'])

    # Layer-0 pre-attention RMSNorm + FP8 quantize
    fvk.rms_norm_fp16(
        x_ptr, int(weights['input_ln_w'][0]), xn_ptr,
        Se, D, 1e-5, int(stream),
    )
    fvk.quantize_fp8_static_fp16(
        xn_ptr, xn_fp8_ptr, int(scales_dev["act_qkv"][0]), Se * D, int(stream),
    )

    for li in range(L):
        clamp_this_layer = (
            ffn_down_clamp_value > 0.0
            and (ffn_clamp_layers is None or li in ffn_clamp_layers)
        )
        slots = attn.get_slot_ptrs("chameleon", li)
        Q_ptr = int(slots["Q"])
        K_ptr = int(slots["K"])
        V_ptr = int(slots["V"])
        O_ptr = int(slots["O"])

        d_act_qkv = int(scales_dev["act_qkv"][li])
        d_act_o = int(scales_dev["act_o"][li])
        d_act_gu = int(scales_dev["act_gu"][li])
        d_act_d = int(scales_dev["act_down"][li])

        d_w_qkv = int(weights['d_w_qkv'][li])
        d_w_o = int(weights['d_w_o'][li])
        d_w_gu = int(weights['d_w_gu'][li])
        d_w_d = int(weights['d_w_d'][li])

        q_w_ptr = int(weights['q_w'][li])
        k_w_ptr = int(weights['k_w'][li])
        v_w_ptr = int(weights['v_w'][li])


        # ═══ Dynamic per-tensor FP8 branch ═══
        # Runtime amax scaling (quantize_fp8_device_fp16 -> fp8_nn_dev_fp16)
        # instead of static-calibrated scale. The static per-tensor scale is
        # wrong for long-sequence (512-res) variants where deep-layer
        # activations shift; recomputing amax each forward restores precision
        # at full FP8 speed (no FP16 GEMM). CUDA-Graph safe: the device scale
        # pointer is dereferenced at replay, so it adapts per input.
        if li in dynamic_fp8_layers:
            # Self-sufficient entry: re-derive FP16 xn from residual stream.
            # Fused RMSNorm + dynamic FP8 quantize: amax is measured inside
            # the norm's own output-write pass, skipping the separate
            # absmax_kernel read of xn (one fewer full pass over Se*D).
            dyn_qkv = int(scales_dev['dyn_act_qkv'][li])
            dyn_o = int(scales_dev['dyn_act_o'][li])
            dyn_gu = int(scales_dev['dyn_act_gu'][li])
            dyn_d = int(scales_dev['dyn_act_down'][li])

            fvk.rms_norm_quantize_dynamic_fp8_fp16(
                x_ptr, int(weights['input_ln_w'][li]), xn_ptr, xn_fp8_ptr,
                dyn_qkv, Se, D, 1e-5, int(stream))

            # QKV: device-scale GEMM (xn_fp8 already produced above).
            # All three GEMMs read xn_fp8; Q writes into xn (aliased Q_O)
            # AFTER the quantize, so the clobber is harmless.
            gemm.fp8_nn_dev_fp16(xn_fp8_ptr, v_w_ptr, V_ptr,
                                 Se, D, D, dyn_qkv, d_w_qkv, int(stream))
            gemm.fp8_nn_dev_fp16(xn_fp8_ptr, k_w_ptr, K_ptr,
                                 Se, D, D, dyn_qkv, d_w_qkv, int(stream))
            gemm.fp8_nn_dev_fp16(xn_fp8_ptr, q_w_ptr, Q_ptr,
                                 Se, D, D, dyn_qkv, d_w_qkv, int(stream))

            fvk.qk_norm_rope_fused_fp16(
                Q_ptr, K_ptr,
                int(weights['q_norm_w'][li]), int(weights['q_norm_b'][li]),
                int(weights['k_norm_w'][li]), int(weights['k_norm_b'][li]),
                cos_ptr, sin_ptr,
                Se, H, Hd, 1e-5, int(stream))
            attn.run("chameleon", li, q_seq=Se, kv_seq=Se, stream=int(stream))

            # O projection: dynamic.
            fvk.quantize_fp8_device_fp16(
                O_ptr, xn_fp8_ptr, dyn_o, Se * D, int(stream))
            gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['o_w'][li]),
                                 o_proj_out_ptr, Se, D, D, dyn_o, d_w_o,
                                 int(stream))

            # Residual 1.
            fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

            # ── FFN: decoupled NVFP4 (opt-in) or dynamic FP8 ──
            # DECOUPLED mode keeps attention on dynamic FP8 (above) but runs the
            # FFN in NVFP4 W4A16 for FP4-eligible layers — isolating FFN
            # precision from attention precision. Falls back to dynamic FP8 FFN.
            _use_fp4_ffn = (
                _fvk_fp4 is not None
                and li in fp4_ffn_layers
                and int(weights['gu_w_fp4'][li]) != 0
            )
            if _use_fp4_ffn:
                # Post-attn RMSNorm -> FP16 xn (no FP8 quantize needed here;
                # the FP4 path quantizes xn_ptr directly below).
                fvk.rms_norm_fp16(x_ptr, int(weights['post_ln_w'][li]), xn_ptr,
                                  Se, D, 1e-5, int(stream))
                # NOTE: intended for SHALLOW layers only. Unlike the FP8 branch
                # below, there is no intermediate clamp on the SwiGLU output
                # (gate_geglu fuses silu*mul + FP4 quantize). A deep layer whose
                # gu exceeds fp16 max (~65504, e.g. L31) could overflow to
                # inf/nan *before* the FP4 quantize; the final o_proj_out clamp
                # cannot recover that. The shipped 512 default (L0-7) is safe.
                # gate+up merged NVFP4 GEMM (dynamic per-block SFA).
                _fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
                    xn_ptr, act_fp4_ptr, act_sfa_ptr, Se, D, False, int(stream))
                _fvk_fp4.cutlass_fp4_gemm_variant(
                    FP4_VARIANT, act_fp4_ptr, act_sfa_ptr,
                    int(weights['gu_w_fp4'][li]), int(weights['gu_sfb'][li]),
                    gu_merged_ptr, Se, 2 * Dff, D, 1.0, 0.0, int(stream))
                # fused SwiGLU + quantize down-proj input to NVFP4.
                _fvk_fp4.gate_geglu_fp4_sfa_v2_fp16(
                    gu_merged_ptr, ffn_act_fp4_ptr, ffn_act_sfa_ptr,
                    Se, Dff, int(stream))
                _fvk_fp4.cutlass_fp4_gemm_variant(
                    FP4_VARIANT, ffn_act_fp4_ptr, ffn_act_sfa_ptr,
                    int(weights['d_w_fp4'][li]), int(weights['d_sfb'][li]),
                    o_proj_out_ptr, Se, D, Dff, 1.0, 0.0, int(stream))
            else:
                # Post-attn residual add + RMSNorm + dynamic FP8 quantize,
                # fused into one elementwise kernel: residual is fp16-rounded
                # (same as residual_add_fp16), ssq is over the rounded values
                # (same as rms_norm reading the fp16 residual), the amax for
                # dyn_gu is folded into the xn write pass, and the residual is
                # register-cached so xn is never re-read from global.
                fvk.residual_add_rms_norm_quantize_dynamic_fp8_fp16(
                    x_ptr, o_proj_out_ptr, int(weights['post_ln_w'][li]),
                    xn_ptr, xn_fp8_ptr, dyn_gu, Se, D, 1e-5, int(stream))
                gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['gate_w'][li]),
                                     int(bufs['gate_out']), Se, Dff, D,
                                     dyn_gu, d_w_gu, int(stream))
                gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['up_w'][li]),
                                     int(bufs['up_out']), Se, Dff, D,
                                     dyn_gu, d_w_gu, int(stream))

                # Down input gu = silu(gate)*up in FP16, fused with its own
                # dynamic FP8 quantize on layers with no outlier clamp (amax
                # measured inside the SwiGLU write pass, skipping a separate
                # absmax_kernel read of the Se*Dff intermediate). Layers that
                # need the outlier clamp (default: L31 only) keep the
                # unfused gate_geglu -> clamp -> quantize sequence since the
                # clamp must run BEFORE the amax/scale is computed.
                if clamp_this_layer:
                    fvk.gate_geglu_fp16(int(bufs['gate_out']), int(bufs['up_out']),
                                        int(bufs['gate_out']), Se * Dff, int(stream))
                    fvk.clamp_inplace_fp16(int(bufs['gate_out']),
                                           float(ffn_down_clamp_value),
                                           Se * Dff, int(stream))
                    fvk.quantize_fp8_device_fp16(
                        int(bufs['gate_out']), int(bufs['gu_fp8']), dyn_d,
                        Se * Dff, int(stream))
                else:
                    fvk.gate_geglu_quantize_dynamic_fp8_fp16(
                        int(bufs['gate_out']), int(bufs['up_out']),
                        int(bufs['gate_out']), int(bufs['gu_fp8']), dyn_d,
                        Se * Dff, int(stream))
                gemm.fp8_nn_dev_fp16(int(bufs['gu_fp8']), int(weights['d_w'][li]),
                                     o_proj_out_ptr, Se, D, Dff, dyn_d, d_w_d,
                                     int(stream))

            if clamp_this_layer:
                fvk.clamp_inplace_fp16(o_proj_out_ptr,
                                       float(ffn_down_clamp_value),
                                       Se * D, int(stream))

            # Residual 2 + next-layer prep.
            if li < L - 1:
                fvk.residual_add_rms_norm_fp16(
                    x_ptr, o_proj_out_ptr, int(weights['input_ln_w'][li + 1]),
                    xn_ptr, Se, D, 1e-5, int(stream))
                # A following STATIC FP8 layer consumes xn_fp8; produce it.
                if (li + 1) not in dynamic_fp8_layers:
                    fvk.quantize_fp8_static_fp16(
                        xn_ptr, xn_fp8_ptr,
                        int(scales_dev["act_qkv"][li + 1]), Se * D, int(stream))
            else:
                fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D,
                                      int(stream))

            if probe is not None:
                _pl = probe.get('layers') or ()
                if li in _pl:
                    _gpu_copy(int(probe['bufs'][_pl.index(li)]),
                              x_ptr, Se * D * 2, stream)
            continue
        # ═══ End dynamic per-tensor FP8 branch ═══

        # ── QKV FP8 GEMMs ──
        # Chameleon has no attention bias; pass zero-buffer for fp8_nn_bias epilogue
        if alpha_host is not None:
            alpha_qkv = float(alpha_host[li * 4 + 0])
            zero_bias_ptr = int(bufs['zero_bias_d'])

            # V projection: AWQ path OR shared-scale path.
            # IMPORTANT ORDER: The chameleon attention backend aliases
            # Q_ptr to xn_ptr (bufs['xn']). If Q is written first, xn is
            # clobbered, and the AWQ V path reads Q's output instead of
            # the RMSNormed xn. So run AWQ V-quantize + V-GEMM BEFORE Q
            # (V's destination is the KV cache, separate buffer, safe).
            if awq_v_proj is not None:
                fvk.awq_quant_fp8_static_fp16(
                    xn_ptr,
                    int(awq_v_proj['inv_s_ptrs'][li]),
                    int(awq_v_proj['xn_v_fp8']),
                    int(awq_v_proj['act_scale_ptrs'][li]),
                    Se, D, int(stream),
                )
                v_alpha = float(awq_v_proj['alpha_host'][li])
                gemm.fp8_nn_bias(
                    int(awq_v_proj['xn_v_fp8']),
                    int(awq_v_proj['w_ptrs'][li]),
                    V_ptr, zero_bias_ptr,
                    Se, D, D, v_alpha, int(stream),
                )
            else:
                gemm.fp8_nn_bias(
                    xn_fp8_ptr, v_w_ptr, V_ptr, zero_bias_ptr,
                    Se, D, D, alpha_qkv, int(stream),
                )

            # Q and K: also use AWQ smoothed path (same xn_v_fp8, per-proj weights)
            if awq_v_proj is not None and 'w_q_ptrs' in awq_v_proj:
                q_alpha = float(awq_v_proj['alpha_q_host'][li])
                k_alpha = float(awq_v_proj['alpha_k_host'][li])
                gemm.fp8_nn_bias(
                    int(awq_v_proj['xn_v_fp8']),
                    int(awq_v_proj['w_q_ptrs'][li]),
                    Q_ptr, zero_bias_ptr,
                    Se, D, D, q_alpha, int(stream),
                )
                gemm.fp8_nn_bias(
                    int(awq_v_proj['xn_v_fp8']),
                    int(awq_v_proj['w_k_ptrs'][li]),
                    K_ptr, zero_bias_ptr,
                    Se, D, D, k_alpha, int(stream),
                )
            else:
                gemm.fp8_nn_bias(
                    xn_fp8_ptr, q_w_ptr, Q_ptr, zero_bias_ptr,
                    Se, D, D, alpha_qkv, int(stream),
                )
                gemm.fp8_nn_bias(
                    xn_fp8_ptr, k_w_ptr, K_ptr, zero_bias_ptr,
                    Se, D, D, alpha_qkv, int(stream),
                )
        else:
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, q_w_ptr, Q_ptr,
                Se, D, D, d_act_qkv, d_w_qkv, int(stream),
            )
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, k_w_ptr, K_ptr,
                Se, D, D, d_act_qkv, d_w_qkv, int(stream),
            )
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, v_w_ptr, V_ptr,
                Se, D, D, d_act_qkv, d_w_qkv, int(stream),
            )

        # ── Fused per-head QK LayerNorm + RoPE ──
        fvk.qk_norm_rope_fused_fp16(
            Q_ptr, K_ptr,
            int(weights['q_norm_w'][li]), int(weights['q_norm_b'][li]),
            int(weights['k_norm_w'][li]), int(weights['k_norm_b'][li]),
            cos_ptr, sin_ptr,
            Se, H, Hd, 1e-5, int(stream),
        )

        # ── MHA via attention backend ──
        attn.run("chameleon", li, q_seq=Se, kv_seq=Se, stream=int(stream))

        # ── O projection ──
        fvk.quantize_fp8_static_fp16(
            O_ptr, xn_fp8_ptr, d_act_o, Se * D, int(stream),
        )

        if alpha_host is not None:
            alpha_o = float(alpha_host[li * 4 + 1])
            gemm.fp8_nn_bias(
                xn_fp8_ptr, int(weights['o_w'][li]), o_proj_out_ptr,
                int(bufs['zero_bias_d']),
                Se, D, D, alpha_o, int(stream),
            )
        else:
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, int(weights['o_w'][li]), o_proj_out_ptr,
                Se, D, D, d_act_o, d_w_o, int(stream),
            )

        # ── Post-attention: residual + RMSNorm (path depends on FFN precision) ──
        if li in _FFN_FP8_LAYERS:
            fvk.residual_add_rms_norm_fp8_fp16(
                x_ptr, o_proj_out_ptr, int(weights['post_ln_w'][li]),
                xn_fp8_ptr, Se, D, 1e-5,
                d_act_gu, int(stream),
            )
        else:
            fvk.residual_add_fp16(
                x_ptr, o_proj_out_ptr, Se * D, int(stream),
            )
            fvk.rms_norm_fp16(
                x_ptr, int(weights['post_ln_w'][li]), xn_ptr,
                Se, D, 1e-5, int(stream),
            )

        # ── FFN (legacy static path; only for layers excluded from
        # dynamic_fp8_layers): NVFP4 if li not in _FFN_FP8_LAYERS, else FP8 ──
        gu_w_fp4_li = int(weights['gu_w_fp4'][li]) if li not in _FFN_FP8_LAYERS else 0
        fp4_available = (
            _fvk_fp4 is not None
            and li not in _FFN_FP8_LAYERS
            and gu_w_fp4_li != 0
        )
        if fp4_available:
            # FP4 path
            _fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
                xn_ptr, act_fp4_ptr, act_sfa_ptr,
                Se, D, False, int(stream),
            )
            _fvk_fp4.cutlass_fp4_gemm_variant(
                FP4_VARIANT,
                act_fp4_ptr, act_sfa_ptr,
                int(weights['gu_w_fp4'][li]), int(weights['gu_sfb'][li]),
                gu_merged_ptr, Se, 2 * Dff, D,
                1.0, 0.0, int(stream),
            )
            _fvk_fp4.gate_geglu_fp4_sfa_v2_fp16(
                gu_merged_ptr, ffn_act_fp4_ptr, ffn_act_sfa_ptr,
                Se, Dff, int(stream),
            )
            _fvk_fp4.cutlass_fp4_gemm_variant(
                FP4_VARIANT,
                ffn_act_fp4_ptr, ffn_act_sfa_ptr,
                int(weights['d_w_fp4'][li]), int(weights['d_sfb'][li]),
                o_proj_out_ptr, Se, D, Dff,
                1.0, 0.0, int(stream),
            )
        else:
            # FP8 path (also used as fallback when FP4 is unavailable
            # for a layer in the L0-2 range).
            gate_w_ptr = int(weights['gate_w'][li])
            up_w_ptr = int(weights['up_w'][li])
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, gate_w_ptr, int(bufs['gate_out']),
                Se, Dff, D, d_act_gu, d_w_gu, int(stream),
            )
            gemm.fp8_nn_dev_fp16(
                xn_fp8_ptr, up_w_ptr, int(bufs['up_out']),
                Se, Dff, D, d_act_gu, d_w_gu, int(stream),
            )
            # Down-proj: 2-tier adaptive dispatch.
            # Tier 1 (standard FP8): no AWQ dict → fused silu_mul_split_fp8
            # Tier 2 (AWQ D smooth): li in awq_d_layers → per-K smooth + FP8
            _has_awq = (awq_v_proj is not None
                        and 'inv_s_D_ptrs' in awq_v_proj)
            _in_awq_d = (_has_awq and li in awq_v_proj.get(
                'awq_d_layers', frozenset()))

            if _in_awq_d:
                # Tier 2: AWQ D smoothed FP8 path (fused SwiGLU).
                fvk.gate_geglu_fp16(int(bufs['gate_out']), int(bufs['up_out']),
                                    int(bufs['gate_out']), Se * Dff, int(stream))
                fvk.awq_quant_fp8_static_fp16(
                    int(bufs['gate_out']),
                    int(awq_v_proj['inv_s_D_ptrs'][li]),
                    int(bufs['gu_fp8']),
                    int(awq_v_proj['act_scale_D_ptrs'][li]),
                    Se, Dff, int(stream))
                gemm.fp8_nn_bias(
                    int(bufs['gu_fp8']),
                    int(awq_v_proj['w_D_ptrs'][li]),
                    o_proj_out_ptr, int(bufs['zero_bias_d']),
                    Se, D, Dff,
                    float(awq_v_proj['alpha_D_host'][li]),
                    int(stream))
            else:
                # Tier 1: standard per-tensor FP8 (no AWQ)
                fvk.silu_mul_split_fp8_fp16(
                    int(bufs['gate_out']), int(bufs['up_out']),
                    int(bufs['gu_fp8']),
                    Se * Dff, d_act_d, int(stream),
                )
                gemm.fp8_nn_dev_fp16(
                    int(bufs['gu_fp8']), int(weights['d_w'][li]),
                    o_proj_out_ptr,
                    Se, D, Dff, d_act_d, d_w_d, int(stream),
                )

        # ── Clamp down_out to fp16 range ──
        # Chameleon-7B L31 down_proj's FP32 accumulator × alpha can push
        # the FP16 output beyond ±65504 producing inf, which propagates
        # through the final RMSNorm and destroys action precision. Mirror
        # the RTX FP8 path's clamp (pipeline_rtx.py:829-833).
        if clamp_this_layer:
            fvk.clamp_inplace_fp16(
                o_proj_out_ptr, float(ffn_down_clamp_value),
                Se * D, int(stream),
            )

        # ── Fused: residual_2 + next-layer input_ln + FP8 quantize ──
        # When AWQ V-proj is enabled we need the FP16 xn buffer populated
        # for the NEXT layer's V-proj activation smoothing kernel (which
        # reads FP16 xn, not FP8). Split the fused path into
        # residual_add_rms_norm_fp16 + quantize_fp8_static_fp16 (one
        # extra launch per layer) so xn_ptr stays valid.
        if li < L - 1:
            if awq_v_proj is not None:
                fvk.residual_add_rms_norm_fp16(
                    x_ptr, o_proj_out_ptr,
                    int(weights['input_ln_w'][li + 1]),
                    xn_ptr, Se, D, 1e-5, int(stream),
                )
                fvk.quantize_fp8_static_fp16(
                    xn_ptr, xn_fp8_ptr,
                    int(scales_dev["act_qkv"][li + 1]),
                    Se * D, int(stream),
                )
            else:
                fvk.residual_add_rms_norm_fp8_fp16(
                    x_ptr, o_proj_out_ptr,
                    int(weights['input_ln_w'][li + 1]),
                    xn_fp8_ptr, Se, D, 1e-5,
                    int(scales_dev["act_qkv"][li + 1]), int(stream),
                )
        else:
            fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

        # ── Optional layer probe: snapshot post-residual-2 hidden state ──
        if probe is not None:
            probe_layers = probe.get('layers') or ()
            if li in probe_layers:
                idx = probe_layers.index(li)
                snap_ptr = int(probe['bufs'][idx])
                if snap_ptr != 0:
                    _gpu_copy(snap_ptr, x_ptr, Se * D * 2, stream)

    # ── Final RMSNorm → hidden_all [Se, D] FP16 ──
    fvk.rms_norm_fp16(
        x_ptr, int(weights['final_norm_w']), hidden_all_ptr,
        Se, D, 1e-5, int(stream),
    )

    # ── Optional final probe ──
    if probe is not None:
        final_buf = int(probe.get('final_buf', 0))
        if final_buf != 0:
            _gpu_copy(final_buf, hidden_all_ptr, Se * D * 2, stream)


def chameleon_decode_step(
    gemm, fvk, bufs, weights, dims, scales_dev,
    *, attn, pos: int, stream: int = 0,
    ffn_down_clamp_value: float = 60000.0,
    ffn_clamp_layers=frozenset({31}),
) -> None:
    """Single-token (Se=1) incremental decode step at position ``pos``.

    Mirrors chameleon_forward's dynamic per-tensor FP8 branch op-for-op,
    with:
      * K/V GEMMs writing the KV cache row at ``pos``
        (``attn.kv_row_ptrs``) — history rows were filled by prefill;
      * RoPE cos/sin taken at row ``pos`` via byte offsets;
      * attention dispatched through ``attn.run_decode`` (bottom-right
        causal mask) over kv_len=pos+1 keys;
      * final RMSNorm written to ``hidden_all`` row 0.

    The token embedding must already be in ``bufs['x']`` row 0, and the
    residual stream in ``bufs['x']`` is updated in place across layers.
    CUDA-Graph safe (int pointers only, no allocations); ``pos`` is a host
    scalar, so callers run this eagerly, outside graph capture.
    """
    D = int(dims['D'])
    Dff = int(dims['Dff'])
    L = int(dims['L'])
    H = int(dims['H'])
    Hd = int(dims['Hd'])

    x_ptr = int(bufs['x'])
    xn_ptr = int(bufs['xn'])
    xn_fp8_ptr = int(bufs['xn_fp8'])
    o_proj_out_ptr = int(bufs['o_proj_out'])
    hidden_all_ptr = int(bufs['hidden_all'])

    cos_ptr = int(weights['rope_cos']) + pos * Hd * 2
    sin_ptr = int(weights['rope_sin']) + pos * Hd * 2

    for li in range(L):
        clamp_this_layer = (
            ffn_down_clamp_value > 0.0
            and (ffn_clamp_layers is None or li in ffn_clamp_layers)
        )
        dyn_qkv = int(scales_dev['dyn_act_qkv'][li])
        dyn_o = int(scales_dev['dyn_act_o'][li])
        dyn_gu = int(scales_dev['dyn_act_gu'][li])
        dyn_d = int(scales_dev['dyn_act_down'][li])

        d_w_o = int(weights['d_w_o'][li])
        d_w_gu = int(weights['d_w_gu'][li])
        d_w_d = int(weights['d_w_d'][li])

        K_row_ptr, V_row_ptr = attn.kv_row_ptrs("chameleon", li, pos)
        slots = attn.get_slot_ptrs("chameleon", li)
        Q_ptr = int(slots["Q"])
        O_ptr = int(slots["O"])

        # Fused RMSNorm + dynamic FP8 quantize from the residual stream.
        fvk.rms_norm_quantize_dynamic_fp8_fp16(
            x_ptr, int(weights['input_ln_w'][li]), xn_ptr, xn_fp8_ptr,
            dyn_qkv, 1, D, 1e-5, int(stream))

        # QKV GEMMs (M=1): K/V land in the cache row at pos, Q in Q_O.
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['v_w'][li]), V_row_ptr,
                             1, D, D, dyn_qkv, int(weights['d_w_qkv'][li]),
                             int(stream))
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['k_w'][li]), K_row_ptr,
                             1, D, D, dyn_qkv, int(weights['d_w_qkv'][li]),
                             int(stream))
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['q_w'][li]), Q_ptr,
                             1, D, D, dyn_qkv, int(weights['d_w_qkv'][li]),
                             int(stream))

        fvk.qk_norm_rope_fused_fp16(
            Q_ptr, K_row_ptr,
            int(weights['q_norm_w'][li]), int(weights['q_norm_b'][li]),
            int(weights['k_norm_w'][li]), int(weights['k_norm_b'][li]),
            cos_ptr, sin_ptr,
            1, H, Hd, 1e-5, int(stream))
        attn.run_decode("chameleon", li, kv_len=pos + 1, stream=int(stream))

        # O projection (M=1).
        fvk.quantize_fp8_device_fp16(
            O_ptr, xn_fp8_ptr, dyn_o, D, int(stream))
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['o_w'][li]),
                             o_proj_out_ptr, 1, D, D, dyn_o, d_w_o,
                             int(stream))

        # Residual 1.
        fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, D, int(stream))

        # FFN: residual add + RMSNorm + dynamic FP8 quantize (fused).
        fvk.residual_add_rms_norm_quantize_dynamic_fp8_fp16(
            x_ptr, o_proj_out_ptr, int(weights['post_ln_w'][li]),
            xn_ptr, xn_fp8_ptr, dyn_gu, 1, D, 1e-5, int(stream))
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['gate_w'][li]),
                             int(bufs['gate_out']), 1, Dff, D,
                             dyn_gu, d_w_gu, int(stream))
        gemm.fp8_nn_dev_fp16(xn_fp8_ptr, int(weights['up_w'][li]),
                             int(bufs['up_out']), 1, Dff, D,
                             dyn_gu, d_w_gu, int(stream))

        if clamp_this_layer:
            fvk.gate_geglu_fp16(int(bufs['gate_out']), int(bufs['up_out']),
                                int(bufs['gate_out']), Dff, int(stream))
            fvk.clamp_inplace_fp16(int(bufs['gate_out']),
                                   float(ffn_down_clamp_value),
                                   Dff, int(stream))
            fvk.quantize_fp8_device_fp16(
                int(bufs['gate_out']), int(bufs['gu_fp8']), dyn_d,
                Dff, int(stream))
        else:
            fvk.gate_geglu_quantize_dynamic_fp8_fp16(
                int(bufs['gate_out']), int(bufs['up_out']),
                int(bufs['gate_out']), int(bufs['gu_fp8']), dyn_d,
                Dff, int(stream))
        gemm.fp8_nn_dev_fp16(int(bufs['gu_fp8']), int(weights['d_w'][li]),
                             o_proj_out_ptr, 1, D, Dff, dyn_d, d_w_d,
                             int(stream))

        if clamp_this_layer:
            fvk.clamp_inplace_fp16(o_proj_out_ptr,
                                   float(ffn_down_clamp_value),
                                   D, int(stream))

        # Residual 2 (next layer re-derives xn from the residual stream).
        fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, D, int(stream))

    # Final RMSNorm → hidden_all row 0.
    fvk.rms_norm_fp16(
        x_ptr, int(weights['final_norm_w']), hidden_all_ptr,
        1, D, 1e-5, int(stream),
    )


def chameleon_forward_fp16(
    gemm, fvk, bufs, weights, dims,
    *, attn, stream: int = 0,
    ffn_gate_clamp_value: float = 10000.0,
    probe=None,
) -> None:
    """Chameleon-7B FP16-only forward (no FP8, no FP4, no AWQ).

    Ported from pipeline_rtx.chameleon_forward. All 32 layers run pure
    FP16 GEMMs via ``gemm.fp16_nn`` — same on Thor as RTX.

    Precision-optimal path (cosine target ≥ 0.99 vs HF bf16) at the
    cost of ~2× the FP8 path latency in the LLM. Recommended when
    downstream ActionHead is sensitive to accumulated FP8 error.

    Weights required (all FP16, KN row-major layout — spec built with
    ``use_fp8=False``):
        q_w[li], k_w[li], v_w[li], o_w[li]        : (D, D)
        gate_w[li], up_w[li]                       : (D, Dff)
        d_w[li] / down_w[li]                       : (Dff, D)
        input_ln_w[li], post_ln_w[li]              : (D,)
        q_norm_w/b[li], k_norm_w/b[li]             : (1, HD) or (HD,)
        final_norm_w                               : (D,)
        rope_cos, rope_sin                          : (max_pos, HD)

    Output: hidden_all = RMSNorm(x_post_res_2, final_norm_w) written to
    bufs['hidden_all'] as (Se, D) FP16 (consumed by action_head_forward).
    """
    Se = int(dims['Se'])
    D = int(dims['D'])
    Dff = int(dims['Dff'])
    L = int(dims['L'])
    H = int(dims['H'])
    Hd = int(dims['Hd'])

    x_ptr = int(bufs['x'])
    xn_ptr = int(bufs['xn'])
    o_proj_out_ptr = int(bufs['o_proj_out'])
    hidden_all_ptr = int(bufs['hidden_all'])
    # Reuse existing gate/up buffers (Se, Dff) fp16
    gate_ptr = int(bufs['gate_out'])
    up_ptr = int(bufs['up_out'])

    cos_ptr = int(weights['rope_cos'])
    sin_ptr = int(weights['rope_sin'])

    q_w = weights['q_w']
    k_w = weights['k_w']
    v_w = weights['v_w']
    o_w = weights['o_w']
    gate_w = weights['gate_w']
    up_w = weights['up_w']
    down_w = weights['d_w']  # frontend uses 'd_w' key for down projection

    input_ln_w = weights['input_ln_w']
    post_ln_w = weights['post_ln_w']
    q_norm_w = weights['q_norm_w']
    q_norm_b = weights['q_norm_b']
    k_norm_w = weights['k_norm_w']
    k_norm_b = weights['k_norm_b']
    final_norm_w = int(weights['final_norm_w'])

    for li in range(L):
        slots = attn.get_slot_ptrs("chameleon", li)
        Q_ptr = int(slots["Q"])
        K_ptr = int(slots["K"])
        V_ptr = int(slots["V"])

        # input_layernorm (RMSNorm eps=1e-5) → xn
        fvk.rms_norm_fp16(
            x_ptr, int(input_ln_w[li]), xn_ptr,
            Se, D, 1e-5, int(stream),
        )

        # Q / K / V GEMMs (no bias in Chameleon).
        # NOTE: Q_ptr aliases xn_ptr on Thor (chameleon slots["Q_O"] =
        # bufs['xn'].data_ptr()). Because gemm.fp16_nn reads A (xn) and
        # writes D (Q) at the SAME fp16 dtype and SAME buffer, cuBLAS
        # in-place semantics are undefined → output corruption.
        # Route Q through o_proj_out_ptr scratch, then copy into Q slot.
        # (V and K write to separate KV cache buffers; safe direct.)
        gemm.fp16_nn(xn_ptr, int(v_w[li]), V_ptr, Se, D, D, int(stream))
        gemm.fp16_nn(xn_ptr, int(k_w[li]), K_ptr, Se, D, D, int(stream))
        gemm.fp16_nn(xn_ptr, int(q_w[li]), o_proj_out_ptr,
                     Se, D, D, int(stream))
        _gpu_copy(Q_ptr, o_proj_out_ptr, Se * D * 2, stream)

        # Per-head QK LayerNorm + RoPE (in-place).
        fvk.qk_norm_rope_fused_fp16(
            Q_ptr, K_ptr,
            int(q_norm_w[li]), int(q_norm_b[li]),
            int(k_norm_w[li]), int(k_norm_b[li]),
            cos_ptr, sin_ptr,
            Se, H, Hd, 1e-5, int(stream),
        )

        # Causal MHA (CUTLASS SM110 FMHA / cuBLAS fallback).
        attn.run("chameleon", li, q_seq=Se, kv_seq=Se, stream=int(stream))

        # O projection.
        gemm.fp16_nn(Q_ptr, int(o_w[li]), o_proj_out_ptr,
                     Se, D, D, int(stream))

        # Residual 1.
        fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

        # post_attention_layernorm.
        fvk.rms_norm_fp16(
            x_ptr, int(post_ln_w[li]), xn_ptr,
            Se, D, 1e-5, int(stream),
        )

        # FFN gate / up (SwiGLU).
        gemm.fp16_nn(xn_ptr, int(gate_w[li]), gate_ptr,
                     Se, Dff, D, int(stream))
        gemm.fp16_nn(xn_ptr, int(up_w[li]), up_ptr,
                     Se, Dff, D, int(stream))
        fvk.gate_geglu_fp16(gate_ptr, up_ptr, gate_ptr, Se * Dff, int(stream))

        # L31 gate*up overflow guard (fp16 max ≈ 65504, Chameleon L31
        # gate*up amax observed ≈ 48000). See pipeline_rtx.py:243-256.
        if ffn_gate_clamp_value > 0.0:
            fvk.clamp_inplace_fp16(
                gate_ptr, float(ffn_gate_clamp_value),
                Se * Dff, int(stream),
            )

        # down projection.
        gemm.fp16_nn(gate_ptr, int(down_w[li]), o_proj_out_ptr,
                     Se, D, Dff, int(stream))

        # Residual 2.
        fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

        # Optional probe: snapshot post-residual-2 hidden state.
        if probe is not None:
            probe_layers = probe.get('layers') or ()
            if li in probe_layers:
                idx = probe_layers.index(li)
                snap_ptr = int(probe['bufs'][idx])
                if snap_ptr != 0:
                    _gpu_copy(snap_ptr, x_ptr, Se * D * 2, stream)

    # Final RMSNorm → hidden_all.
    fvk.rms_norm_fp16(
        x_ptr, final_norm_w, hidden_all_ptr,
        Se, D, 1e-5, int(stream),
    )

    if probe is not None:
        final_buf = int(probe.get('final_buf', 0))
        if final_buf != 0:
            _gpu_copy(final_buf, hidden_all_ptr, Se * D * 2, stream)


# ══════════════════════════════════════════════════════════════════
# Chameleon-7B LLM calibration (FP16 + amax measurement)
# ══════════════════════════════════════════════════════════════════

def _d2h_float(d_ptr: int) -> float:
    """Read a single float32 from device to host."""
    t = torch.empty(1, dtype=torch.float32, device='cuda')
    import ctypes
    ctypes.CDLL('libcudart.so').cudaMemcpy(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_void_p(d_ptr),
        4, 2,  # cudaMemcpyDeviceToDevice is 3, D2H is 2
    )
    return float(t.item())


def _d2h_floats(d_ptr: int, n: int) -> list:
    """Read n float32 values from device to host."""
    t = torch.empty(n, dtype=torch.float32, device='cuda')
    import ctypes
    ctypes.CDLL('libcudart.so').cudaMemcpy(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_void_p(d_ptr),
        n * 4, 2,
    )
    return t.cpu().tolist()


def chameleon_forward_calibrate(
    gemm, fvk_mod, bufs, weights, dims,
    calib_scales_ptr, stream: int = 0,
    attn_calib_scales_ptr: int = 0,
) -> None:
    """Calibrate Chameleon-7B FP8 scales.

    4 quantization points per layer × 32 layers = 128 scales.
    Points: act_qkv, act_o, act_gu, act_down.
    """
    Se = int(dims['Se'])
    D = int(dims['D'])
    Dff = int(dims['Dff'])
    L = int(dims['L'])
    H = int(dims['H'])
    Hd = int(dims['Hd'])

    import numpy as np

    x_ptr = int(bufs['x'])
    xn_ptr = int(bufs['xn'])
    o_proj_out_ptr = int(bufs['o_proj_out'])
    gate_out_ptr = int(bufs['gate_out'])
    up_out_ptr = int(bufs['up_out'])
    down_out_ptr = int(bufs['down_out'])
    Q_ptr_buf = int(bufs['Q'])
    K_ptr_buf = int(bufs['K'])
    V_ptr_buf = int(bufs['V'])
    O_ptr_buf = int(bufs['O'])

    calib_buf = int(bufs['calib_buf'])
    d_scale = int(bufs['d_scale'])
    fp8_scratch = int(bufs['fp8_scratch'])
    norm_scratch = int(bufs['norm_scratch'])

    cos_ptr = int(weights['rope_cos'])
    sin_ptr = int(weights['rope_sin'])

    w_scales_dev = int(weights['w_scales_flat'])
    ws_host = _d2h_floats(w_scales_dev, L * 4)

    _gpu_zero(calib_buf, L * 4 * 4, stream)

    for li in range(L):
        # ── 1. RMSNorm → measure amax (act_qkv scale) ──
        fvk_mod.rms_norm_fp16(
            x_ptr, int(weights['input_ln_w'][li]), norm_scratch,
            Se, D, 1e-5, int(stream),
        )
        _measure_scale_gpu(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        as_qkv = _d2h_float(d_scale)
        cs_qkv = calib_buf + (li * 4 + 0) * 4
        _gpu_copy(cs_qkv, d_scale, 4, stream)

        # ── 2. Quantize xn → FP8 ──
        fvk_mod.quantize_fp8_static_fp16(
            norm_scratch, int(bufs['xn_fp8']), cs_qkv,
            Se * D, int(stream),
        )

        # ── 3. Q/K/V FP8 GEMMs (no bias in Chameleon) ──
        q_w_ptr = int(weights['q_w'][li])
        k_w_ptr = int(weights['k_w'][li])
        v_w_ptr = int(weights['v_w'][li])

        alpha_qkv = float(np.float32(as_qkv) * np.float32(ws_host[li * 4 + 0]))
        zero_bias_ptr = int(bufs['zero_bias_d'])

        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), q_w_ptr, Q_ptr_buf, zero_bias_ptr,
            Se, D, D, alpha_qkv, int(stream),
        )
        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), k_w_ptr, K_ptr_buf, zero_bias_ptr,
            Se, D, D, alpha_qkv, int(stream),
        )
        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), v_w_ptr, V_ptr_buf, zero_bias_ptr,
            Se, D, D, alpha_qkv, int(stream),
        )

        # ── 4. Fused QK LayerNorm + RoPE ──
        fvk_mod.qk_norm_rope_fused_fp16(
            Q_ptr_buf, K_ptr_buf,
            int(weights['q_norm_w'][li]), int(weights['q_norm_b'][li]),
            int(weights['k_norm_w'][li]), int(weights['k_norm_b'][li]),
            cos_ptr, sin_ptr,
            Se, H, Hd, 1e-5, int(stream),
        )

        # ── 4b. Optional Q/K/V amax for FP8 attention ──
        if attn_calib_scales_ptr:
            n_qkv = Se * H * Hd
            for i, ptr in enumerate((Q_ptr_buf, K_ptr_buf, V_ptr_buf)):
                _measure_scale_gpu(
                    fvk_mod, ptr, n_qkv, d_scale, fp8_scratch, stream)
                _gpu_sync(stream)
                _gpu_copy(
                    attn_calib_scales_ptr + (li * 3 + i) * 4,
                    d_scale, 4, stream,
                )

        # ── 5. Attention (cuBLAS — no FMHA during calibration) ──
        attn_scale = 1.0 / math.sqrt(float(Hd))
        fvk_mod.attention_qkv_fp16(
            bufs['ctx'], Q_ptr_buf, K_ptr_buf, V_ptr_buf,
            int(bufs['logits']), O_ptr_buf,
            Se, Se, H, Hd, attn_scale, int(stream),
        )

        # ── 6. O proj — measure amax → quantize → GEMM ──
        _measure_scale_gpu(fvk_mod, O_ptr_buf, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        as_o = _d2h_float(d_scale)
        cs_o = calib_buf + (li * 4 + 1) * 4
        _gpu_copy(cs_o, d_scale, 4, stream)
        fvk_mod.quantize_fp8_static_fp16(
            O_ptr_buf, int(bufs['xn_fp8']), cs_o,
            Se * D, int(stream),
        )
        alpha_o = float(np.float32(as_o) * np.float32(ws_host[li * 4 + 1]))
        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), int(weights['o_w'][li]), o_proj_out_ptr,
            zero_bias_ptr,
            Se, D, D, alpha_o, int(stream),
        )

        # ── 7. Residual 1 ──
        fvk_mod.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

        # ── 8. Post-attn RMSNorm → measure amax (act_gu scale) ──
        fvk_mod.rms_norm_fp16(
            x_ptr, int(weights['post_ln_w'][li]), norm_scratch,
            Se, D, 1e-5, int(stream),
        )
        _measure_scale_gpu(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        as_gu = _d2h_float(d_scale)
        cs_gu = calib_buf + (li * 4 + 2) * 4
        _gpu_copy(cs_gu, d_scale, 4, stream)

        # ── 9. Quantize → FP8 ──
        fvk_mod.quantize_fp8_static_fp16(
            norm_scratch, int(bufs['xn_fp8']), cs_gu,
            Se * D, int(stream),
        )

        # ── 10. Gate + Up FP8 GEMMs (no bias) ──
        gate_w_ptr = int(weights['gate_w'][li])
        up_w_ptr = int(weights['up_w'][li])
        alpha_gu = float(np.float32(as_gu) * np.float32(ws_host[li * 4 + 2]))
        zero_bias_dff = int(bufs['zero_bias_dff'])
        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), gate_w_ptr, gate_out_ptr,
            zero_bias_dff,
            Se, Dff, D, alpha_gu, int(stream),
        )
        gemm.fp8_nn_bias(
            int(bufs['xn_fp8']), up_w_ptr, up_out_ptr,
            zero_bias_dff,
            Se, Dff, D, alpha_gu, int(stream),
        )

        # ── 11. SiLU(gate)*up → measure amax → FP8 ──
        silu_scr = int(bufs['silu_scratch'])
        _gpu_copy(silu_scr, gate_out_ptr, Se * Dff * 2, stream)
        fvk_mod.gate_geglu_fp16(silu_scr, up_out_ptr, down_out_ptr,
                                Se * Dff, int(stream))
        _measure_scale_gpu(fvk_mod, down_out_ptr, Se * Dff, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        as_d = _d2h_float(d_scale)
        cs_d = calib_buf + (li * 4 + 3) * 4
        _gpu_copy(cs_d, d_scale, 4, stream)
        fvk_mod.silu_mul_split_fp8_fp16(
            gate_out_ptr, up_out_ptr, int(bufs['gu_fp8']),
            Se * Dff, cs_d, int(stream),
        )

        # ── 12. Down FP8 GEMM ──
        alpha_d = float(np.float32(as_d) * np.float32(ws_host[li * 4 + 3]))
        gemm.fp8_nn_bias(
            int(bufs['gu_fp8']), int(weights['d_w'][li]), o_proj_out_ptr,
            zero_bias_ptr,
            Se, D, Dff, alpha_d, int(stream),
        )

        # ── 13. Residual 2 ──
        fvk_mod.residual_add_fp16(x_ptr, o_proj_out_ptr, Se * D, int(stream))

    # ── Final RMSNorm ──
    fvk_mod.rms_norm_fp16(
        x_ptr, int(weights['final_norm_w']), int(bufs['xn']),
        Se, D, 1e-5, int(stream),
    )

    # ── Copy calibrated scales to output ──
    _gpu_copy(calib_scales_ptr, calib_buf, L * 4 * 4, stream)
    _gpu_sync(stream)


__all__ = [
    "chameleon_forward",
    "chameleon_forward_fp16",
    "chameleon_forward_calibrate",
]
