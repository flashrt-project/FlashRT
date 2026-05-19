"""G6.5 — AdaLN-modulate + gated-residual fusion for Motus FFN paths.

After G6.4 the captured graph still spends ~26 ms on the upstream
``(1 + e[k+1].squeeze(2)) * norm(x) + e[k].squeeze(2)`` modulate
chain. Per layer per step Motus hits 4 such modulate sites (video
joint, video FFN, action joint, action FFN) and 4 gated residuals.

This gate fuses the FFN paths only (the joint-attention rewrite is a
larger refactor scoped for a later gate). Net replacement per call:

    ffn_input = ada_layer_norm_bf16(x, scale=e[4], shift=e[3], eps)
        replaces upstream's
            norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        cutting 5 launches → 1 (kernel internally adds 1 to scale, see
        csrc/kernels/dit_bf16.cu:122).

    out = gate_mul_residual(x, ffn_out, gate=e[5])
        replaces upstream's
            x + ffn_out * e[5].squeeze(2)
        cutting 2-3 launches → 1.

Coverage: 30 layers × 10 steps × 2 paths (video FFN + action FFN) =
600 modulate sites + 600 gated residuals per replay. With ~4-5 launches
saved per pair, expected total saving ~12-20 ms.

Numerical contract: ada_layer_norm_bf16 reads scale/shift as bf16
(reduction in fp32, output bf16). Upstream chain returned fp32 from
the multiply-add then implicit-cast to bf16 at next op. Effective
delta is the scale/shift quantization to bf16 (~7 mantissa bits less
than fp32) — drift expected O(1e-5) per layer, ~1e-3 cumulative; cos
floor 0.997 has plenty of headroom.

Toggle: FLASH_RT_MOTUS_NO_G6_5=1.
"""

from __future__ import annotations

import logging
import os

import torch

from flash_rt.models.motus._stream import cs

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_TRACE = os.environ.get('FLASH_RT_MOTUS_G6_5_TRACE', '0') == '1'
_JOINT_PROFILE = os.environ.get('FLASH_RT_MOTUS_JOINT_PROFILE', '0') == '1'
_JOINT_PROFILE_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
_SEQ_LENS_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_JOINT_CAT_BUF_CACHE: dict[tuple[int, int, int, int, torch.device],
                           tuple[torch.Tensor, ...]] = {}


def _jp_start() -> torch.cuda.Event | None:
    if not _JOINT_PROFILE:
        return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _jp_end(name: str, e0: torch.cuda.Event | None) -> None:
    if e0 is None:
        return
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    _JOINT_PROFILE_EVENTS.setdefault(name, []).append((e0, e1))


def reset_joint_profile_events() -> None:
    _JOINT_PROFILE_EVENTS.clear()


def joint_profile_totals() -> dict[str, tuple[int, float]]:
    torch.cuda.synchronize()
    out: dict[str, tuple[int, float]] = {}
    for name, evs in _JOINT_PROFILE_EVENTS.items():
        total = 0.0
        n = 0
        for e0, e1 in evs:
            try:
                total += e0.elapsed_time(e1)
                n += 1
            except RuntimeError:
                continue
        out[name] = (n, total)
    return out


def _get_static_seq_lens(batch: int, total_len: int,
                         device: torch.device) -> torch.Tensor:
    """Return a cached CUDA int64 [B] seq-lens tensor for FA2.

    Motus default shape is static across denoise replay, but the old joint
    attention path rebuilt ``torch.full((B,), total_len)`` 300 times per
    inference. Cache by (device, batch, total_len) so graph capture reuses a
    stable pointer and the hot path has no torch fill op.
    """
    dev_idx = device.index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    key = (int(dev_idx), int(batch), int(total_len))
    buf = _SEQ_LENS_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.empty((batch,), dtype=torch.long, device=device)
        buf.fill_(int(total_len))
        _SEQ_LENS_CACHE[key] = buf
    return buf


def _joint_cat_buffers(sa, b: int, total_l: int, n: int, d: int,
                       device: torch.device) -> tuple[torch.Tensor, ...]:
    shape = (int(b), int(total_l), int(n), int(d))
    key = (int(b), int(total_l), int(n), int(d), device)
    bufs = _JOINT_CAT_BUF_CACHE.get(key)
    if bufs is None or bufs[0].device != device:
        bufs = tuple(torch.empty(shape, dtype=torch.bfloat16, device=device)
                     for _ in range(3))
        _JOINT_CAT_BUF_CACHE[key] = bufs
    return bufs


def _flat_heads_no_copy(x: torch.Tensor) -> torch.Tensor:
    """View [B,L,H,D] attention output as [B*L,H*D] without material copy.

    Motus production uses B=1. After slicing a contiguous joint FA output on
    the sequence axis, the payload is still dense for B=1 even if PyTorch's
    generic contiguity check may force a physical copy. This helper keeps that
    path metadata-only and falls back to the old copy-safe path otherwise.
    """
    if x.dim() != 4:
        return x.contiguous().view(-1, x.shape[-1])
    b, l, h, d = x.shape
    if (b == 1 and x.stride(-1) == 1 and x.stride(-2) == d
            and x.stride(1) == h * d):
        return x.as_strided((l, h * d), (h * d, 1))
    return x.flatten(2).contiguous().reshape(-1, h * d)


def _dense_b1_or_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Return x if its B=1 payload is dense, otherwise materialize.

    Several Motus tensors are slices from a larger static workspace. PyTorch
    marks them non-contiguous because the batch stride reflects the parent
    allocation, but our kernels only need linear dense payload from data_ptr.
    """
    if x.is_contiguous():
        return x
    if x.dim() == 3 and x.shape[0] == 1 and x.stride(-1) == 1:
        if x.stride(1) == x.shape[2]:
            return x
    if x.dim() == 4 and x.shape[0] == 1 and x.stride(-1) == 1:
        if x.stride(2) == x.shape[3] and x.stride(1) == x.shape[2] * x.shape[3]:
            return x
    return x.contiguous()


def _bf16_dense_b1_or_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return _dense_b1_or_contiguous(x)


def _joint_self_attn_direct_qkv_cat(
    sa,
    norm_video: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes,
    freqs,
    a_packed: torch.Tensor,
    u_packed: torch.Tensor,
    action_block,
    und_block,
    L_a: int,
    L_u: int,
):
    """G7.44 prior path: producers write directly into joint Q/K/V workspace.

    This keeps the current FA2 body but bypasses the intermediate video
    q_rope/k_rope/v tensors and the physical concat3_qkv pass. It is
    intentionally gated until graph/cos prove it is profitable.
    """
    from importlib import import_module
    from flash_rt.models.motus import _rope_swap as _rs

    site = getattr(sa, '_fused_qkv_site', None)
    if site is None or not hasattr(sa, '_fused_qkv_fn'):
        return None
    if not (site.has_bias and hasattr(fvk, 'qkv_split_bias_norm_rope_v_cat_bf16')):
        return None
    if not hasattr(fvk, 'qkv_split_norm2_cat_bf16'):
        return None
    if _rs._FREQ_GRID_RE_FP32 is None or _rs._FREQ_GRID_IM_FP32 is None:
        return None

    b, L_v, c = norm_video.shape
    n = sa.num_heads
    d = sa.head_dim
    total_l = int(L_v + L_a + L_u)
    q_cat, k_cat, v_cat = _joint_cat_buffers(
        sa, b, total_l, n, d, norm_video.device)

    _e = _jp_start()
    site._qkv_bias_skip_once = True
    _ = sa._fused_qkv_fn(norm_video)
    packed = site._last_packed_qkv
    _jp_end('direct.video_qkv_gemm', _e)

    seq_len = int(_rs._FREQ_GRID_RE_FP32.shape[0])
    _e = _jp_start()
    if (hasattr(fvk, 'qkv_split_joint3_cat_bf16')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_47_JOINT_QKV3_CAT',
                               '0') != '1'
            and b == 1):
        fvk.qkv_split_joint3_cat_bf16(
            int(packed.data_ptr()), int(site.bias.data_ptr()),
            int(sa.norm_q.weight.data_ptr()), int(sa.norm_k.weight.data_ptr()),
            int(_rs._FREQ_GRID_RE_FP32.data_ptr()),
            int(_rs._FREQ_GRID_IM_FP32.data_ptr()),
            int(a_packed.data_ptr()),
            int(action_block.wan_action_norm_q.weight.data_ptr()),
            int(action_block.wan_action_norm_k.weight.data_ptr()),
            int(u_packed.data_ptr()),
            int(und_block.wan_und_norm_q.weight.data_ptr()),
            int(und_block.wan_und_norm_k.weight.data_ptr()),
            int(q_cat.data_ptr()), int(k_cat.data_ptr()), int(v_cat.data_ptr()),
            int(b), total_l, int(L_v), int(L_a), int(L_u), int(n), int(d),
            seq_len, float(sa.norm_q.eps),
            float(action_block.wan_action_norm_q.eps),
            float(und_block.wan_und_norm_q.eps), cs())
    else:
        fvk.qkv_split_bias_norm_rope_v_cat_bf16(
            int(packed.data_ptr()), int(site.bias.data_ptr()),
            int(sa.norm_q.weight.data_ptr()), int(sa.norm_k.weight.data_ptr()),
            int(_rs._FREQ_GRID_RE_FP32.data_ptr()),
            int(_rs._FREQ_GRID_IM_FP32.data_ptr()),
            int(q_cat.data_ptr()), int(k_cat.data_ptr()), int(v_cat.data_ptr()),
            int(b), total_l, 0, int(L_v), int(n), int(d), seq_len,
            float(sa.norm_q.eps), cs())
        fvk.qkv_split_norm2_cat_bf16(
            int(a_packed.data_ptr()),
            int(action_block.wan_action_norm_q.weight.data_ptr()),
            int(action_block.wan_action_norm_k.weight.data_ptr()),
            int(u_packed.data_ptr()),
            int(und_block.wan_und_norm_q.weight.data_ptr()),
            int(und_block.wan_und_norm_k.weight.data_ptr()),
            int(q_cat.data_ptr()), int(k_cat.data_ptr()), int(v_cat.data_ptr()),
            int(b), total_l, int(L_v), int(L_a), int(L_u), int(n), int(d),
            float(action_block.wan_action_norm_q.eps),
            float(und_block.wan_und_norm_q.eps), cs())
    _jp_end('direct.qkv_to_cat', _e)

    wan_model_mod = import_module(type(sa).__module__)
    flash_attention = wan_model_mod.flash_attention
    _e = _jp_start()
    attn_out = flash_attention(
        q=q_cat, k=k_cat, v=v_cat, k_lens=seq_lens,
        window_size=sa.window_size)
    _jp_end('direct.joint_fa2', _e)

    x_out = attn_out[:, :L_v, :, :]
    action_out_h = attn_out[:, L_v:L_v + L_a, :, :]
    und_out_h = attn_out[:, L_v + L_a:, :, :]
    _e = _jp_start()
    y = sa.o(x_out.flatten(2))
    _jp_end('direct.video_o_proj', _e)
    return y, action_out_h, und_out_h


def _is_static_mod_fp8(mod) -> bool:
    return hasattr(mod, 'q') and hasattr(mod, 'scale')


def _static_mod_tensor(mod, idx: int | None):
    if idx is not None and _is_static_mod_fp8(mod):
        return mod[idx]
    return mod


def _adaln_modulation6_bf16(
    adaln_params: torch.Tensor,
    layer_modulation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return six contiguous BF16 modulation components.

    Upstream computes ``layer.modulation.unsqueeze(0) + adaln_params``
    as one large FP32 broadcast add, then chunks along the 6-way axis.
    Those chunks are strided, so the downstream kernels pay repeated
    ``to(bf16).contiguous()`` copies. This kernel performs the add,
    split, and BF16 cast directly into six [B, S, D] contiguous tensors.
    """
    if adaln_params.dtype != torch.float32:
        adaln_params = adaln_params.float()
    if layer_modulation.dtype != torch.float32:
        layer_modulation = layer_modulation.float()
    p = adaln_params if adaln_params.is_contiguous() else adaln_params.contiguous()
    m = (layer_modulation if layer_modulation.is_contiguous()
         else layer_modulation.contiguous())
    B, S, K6, D = p.shape
    if K6 != 6:
        raise ValueError(f'expected adaln_params [B,S,6,D], got {tuple(p.shape)}')
    outs = tuple(torch.empty(B, S, D, dtype=torch.bfloat16, device=p.device)
                 for _ in range(6))
    fvk.adaln_modulation6_bf16(
        int(p.data_ptr()), int(m.data_ptr()),
        int(outs[0].data_ptr()), int(outs[1].data_ptr()),
        int(outs[2].data_ptr()), int(outs[3].data_ptr()),
        int(outs[4].data_ptr()), int(outs[5].data_ptr()),
        B, S, D, cs())
    return outs


def _ada_modulate_bf16(
    x: torch.Tensor,
    scale_fp32: torch.Tensor,
    shift_fp32: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run ``fvk.ada_layer_norm_bf16(x, bf16(scale), bf16(shift)) -> bf16``.

    The kernel internally adds 1 to scale before the modulation.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x_c = x if x.is_contiguous() else x.contiguous()
    B, L, C = x_c.shape
    scale_b = scale_fp32.to(torch.bfloat16).contiguous()
    shift_b = shift_fp32.to(torch.bfloat16).contiguous()
    if scale_b.dim() == 3:
        scale_b = scale_b.reshape(B * L, C).contiguous()
        shift_b = shift_b.reshape(B * L, C).contiguous()
    out = torch.empty_like(x_c)
    fvk.ada_layer_norm_bf16(
        int(x_c.data_ptr()), int(scale_b.data_ptr()), int(shift_b.data_ptr()),
        int(out.data_ptr()), int(B * L), int(C), float(eps), cs())
    return out


def _bias_gate_residual_bf16(
    residual: torch.Tensor,
    x_no_bias: torch.Tensor,
    bias: torch.Tensor,
    gate_fp32: torch.Tensor,
    gate_idx: int | None = None,
) -> torch.Tensor:
    """G6.7: residual + (x_no_bias + bias) * gate, single fvk launch.

    Used when the upstream FP8 GEMM is run with bias_skip=True (raw GEMM
    output). The new fvk.bias_gate_mul_residual_bf16 kernel folds
    add_bias + gate_mul_residual into one launch.
    """
    if residual.dtype != torch.bfloat16:
        residual = residual.to(torch.bfloat16)
    if x_no_bias.dtype != torch.bfloat16:
        x_no_bias = x_no_bias.to(torch.bfloat16)
    res_c = residual.contiguous()
    x_c = x_no_bias.contiguous()
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    bias_c = bias.contiguous()
    gate_b = None
    use_gate_fp8 = (
        gate_idx is not None
        and _is_static_mod_fp8(gate_fp32)
        and hasattr(fvk, 'bias_gate_mul_residual_out_bf16_gate_fp8')
        and os.environ.get('FLASH_RT_MOTUS_NO_G7_27_GATE_OUT', '0') != '1'
        and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_GATE',
                           '0') != '1')
    if not use_gate_fp8:
        gate_fp32 = _static_mod_tensor(gate_fp32, gate_idx)
        gate_b = gate_fp32.to(torch.bfloat16).contiguous()
    B = int(res_c.shape[0])
    L = int(res_c.shape[1])
    C = int(res_c.shape[2])
    # Stage4 (2026-05-17) g1d fast-path: Wan time-embedding produces gate
    # values that are constant across the L (seq) axis (verified maxdiff=0
    # across all wan video FFN gate calls). We slice gate_b[:, 0:1, :]
    # (zero-copy view) and pass to dedicated g1d kernel which broadcasts
    # internally — saves 2.2 MB BF16 HBM read per call at C=3072.
    use_gate_g1d = (
        not use_gate_fp8
        and gate_b is not None
        and gate_b.dim() == 3
        and gate_b.shape[0] == 1
        and gate_b.shape[2] == C
        and hasattr(fvk, 'bias_gate_mul_residual_out_bf16_g1d')
        and os.environ.get('FLASH_RT_MOTUS_GATE_G1D', '1') == '1')
    if not use_gate_g1d:
        if (gate_b is not None and gate_b.dim() == 3
                and gate_b.shape != res_c.shape):
            gate_b = gate_b.expand(B, L, C).contiguous()
    if (hasattr(fvk, 'bias_gate_mul_residual_out_bf16')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_27_GATE_OUT', '0') != '1'):
        out = torch.empty_like(res_c)
        if use_gate_fp8:
            fvk.bias_gate_mul_residual_out_bf16_gate_fp8(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(bias_c.data_ptr()),
                int(gate_fp32.q[gate_idx].data_ptr()),
                int(gate_fp32.scale[gate_idx].data_ptr()),
                int(out.data_ptr()), B * L, C, cs())
        elif use_gate_g1d:
            # gate_b is [1, L, C] contiguous with values identical across
            # L (Wan time-embedding property); first C elements at offset
            # 0 are the gate vector — pass the base pointer directly.
            fvk.bias_gate_mul_residual_out_bf16_g1d(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(bias_c.data_ptr()), int(gate_b.data_ptr()),
                int(out.data_ptr()), B * L, C, cs())
        else:
            fvk.bias_gate_mul_residual_out_bf16(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(bias_c.data_ptr()), int(gate_b.data_ptr()),
                int(out.data_ptr()), B * L, C, cs())
    else:
        out = res_c.clone()
        fvk.bias_gate_mul_residual_bf16(
            int(out.data_ptr()), int(x_c.data_ptr()),
            int(bias_c.data_ptr()), int(gate_b.data_ptr()),
            B * L, C, cs())
    return out


def _gate_residual_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate_fp32: torch.Tensor,
    gate_idx: int | None = None,
) -> torch.Tensor:
    """``residual + x * gate`` → bf16, fused with internal fp32 accum.

    Returns a fresh tensor (clone of residual then in-place add).
    """
    if residual.dtype != torch.bfloat16:
        residual = residual.to(torch.bfloat16)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    res_c = residual.contiguous()
    x_c = x.contiguous()
    gate_b = None
    use_gate_fp8 = (
        gate_idx is not None
        and _is_static_mod_fp8(gate_fp32)
        and hasattr(fvk, 'gate_mul_residual_out_bf16_gate_fp8')
        and os.environ.get('FLASH_RT_MOTUS_NO_G7_27_GATE_OUT', '0') != '1'
        and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_GATE',
                           '0') != '1')
    if not use_gate_fp8:
        gate_fp32 = _static_mod_tensor(gate_fp32, gate_idx)
        gate_b = gate_fp32.to(torch.bfloat16).contiguous()
    n = int(res_c.numel())
    C = int(res_c.shape[-1])
    # Stage4 (2026-05-17) g1d fast-path: gate is L-constant (Wan time
    # embedding broadcast property, verified maxdiff=0). First C elements
    # at offset 0 of contiguous gate_b are the gate vector.
    use_gate_g1d = (
        not use_gate_fp8
        and gate_b is not None
        and gate_b.dim() == 3
        and gate_b.shape[0] == 1
        and gate_b.shape[-1] == C
        and hasattr(fvk, 'gate_mul_residual_out_bf16_g1d')
        and os.environ.get('FLASH_RT_MOTUS_GATE_G1D', '1') == '1')
    if not use_gate_g1d:
        if (gate_b is not None and gate_b.dim() == 3
                and gate_b.shape != res_c.shape):
            B = res_c.shape[0]
            L = res_c.shape[1]
            gate_b = gate_b.expand(B, L, C).contiguous()
    if (hasattr(fvk, 'gate_mul_residual_out_bf16')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_27_GATE_OUT', '0') != '1'):
        out = torch.empty_like(res_c)
        if use_gate_fp8:
            fvk.gate_mul_residual_out_bf16_gate_fp8(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(gate_fp32.q[gate_idx].data_ptr()),
                int(gate_fp32.scale[gate_idx].data_ptr()),
                int(out.data_ptr()), n, cs())
        elif use_gate_g1d:
            fvk.gate_mul_residual_out_bf16_g1d(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(gate_b.data_ptr()), int(out.data_ptr()),
                n // C, C, cs())
        else:
            fvk.gate_mul_residual_out_bf16(
                int(res_c.data_ptr()), int(x_c.data_ptr()),
                int(gate_b.data_ptr()), int(out.data_ptr()), n, cs())
    else:
        out = res_c.clone()
        fvk.gate_mul_residual(
            int(out.data_ptr()), int(x_c.data_ptr()), int(gate_b.data_ptr()),
            n, cs())
    return out


def _add_bf16_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if (hasattr(fvk, 'add_bf16_out')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_28_ADD_OUT', '0') != '1'
            and a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16):
        a_c = a if a.is_contiguous() else a.contiguous()
        b_c = b if b.is_contiguous() else b.contiguous()
        out = torch.empty_like(a_c)
        fvk.add_bf16_out(
            int(a_c.data_ptr()), int(b_c.data_ptr()), int(out.data_ptr()),
            int(out.numel()), cs())
        return out
    return a + b


def _joint_residual3_bf16(
    video_residual: torch.Tensor,
    video_x: torch.Tensor,
    video_bias: torch.Tensor,
    video_gate: torch.Tensor,
    video_gate_idx: int | None,
    action_residual: torch.Tensor,
    action_x: torch.Tensor,
    action_bias: torch.Tensor,
    action_gate: torch.Tensor,
    und_residual: torch.Tensor,
    und_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint-attn tail fusion: video/action gated residual + und add."""
    v_res = _bf16_dense_b1_or_contiguous(video_residual)
    v_x = _bf16_dense_b1_or_contiguous(video_x)
    v_bias = _bf16_dense_b1_or_contiguous(video_bias)
    use_v_gate_fp8 = (
        video_gate_idx is not None
        and _is_static_mod_fp8(video_gate)
        and hasattr(fvk, 'motus_joint_residual3_out_bf16_vgate_fp8')
        and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_GATE',
                           '0') != '1')
    v_gate_b = None if use_v_gate_fp8 else _static_mod_tensor(
        video_gate, video_gate_idx)
    v_g = None if use_v_gate_fp8 else _bf16_dense_b1_or_contiguous(v_gate_b)
    a_res = _bf16_dense_b1_or_contiguous(action_residual)
    a_x = _bf16_dense_b1_or_contiguous(action_x)
    a_g = _bf16_dense_b1_or_contiguous(action_gate)
    a_bias = None if action_bias is None else _bf16_dense_b1_or_contiguous(action_bias)
    u_res = _bf16_dense_b1_or_contiguous(und_residual)
    u_x = _bf16_dense_b1_or_contiguous(und_x)
    v_out = torch.empty_like(v_res)
    a_out = torch.empty_like(a_res)
    u_out = torch.empty_like(u_res)
    # Stage4 (2026-05-17) g1d fast-path: both video and action gates are
    # L-constant (Wan/action time-embedding broadcast pattern, verified
    # maxdiff=0). Pass v_g/a_g base pointers to g1d kernel — first C
    # elements at offset 0 are the gate vector; kernel broadcasts.
    use_joint_g1d = (
        action_bias is None
        and not use_v_gate_fp8
        and v_g is not None and a_g is not None
        and v_g.dim() == 3 and a_g.dim() == 3
        and v_g.shape[0] == 1 and a_g.shape[0] == 1
        and hasattr(fvk, 'motus_joint_residual3_out_bf16_g1d_action_nobias')
        and os.environ.get('FLASH_RT_MOTUS_JOINT_RES3_G1D', '1') == '1')
    if action_bias is None:
        if use_v_gate_fp8:
            fvk.motus_joint_residual3_out_bf16_vgate_fp8_action_nobias(
                int(v_res.data_ptr()), int(v_x.data_ptr()),
                int(v_bias.data_ptr()),
                int(video_gate.q[video_gate_idx].data_ptr()),
                int(video_gate.scale[video_gate_idx].data_ptr()),
                int(v_out.data_ptr()), int(v_out.numel()), int(v_out.shape[-1]),
                int(a_res.data_ptr()), int(a_x.data_ptr()),
                int(a_g.data_ptr()), int(a_out.data_ptr()), int(a_out.numel()),
                int(a_out.shape[-1]),
                int(u_res.data_ptr()), int(u_x.data_ptr()), int(u_out.data_ptr()),
                int(u_out.numel()), int(u_out.shape[-1]), cs())
        elif use_joint_g1d:
            fvk.motus_joint_residual3_out_bf16_g1d_action_nobias(
                int(v_res.data_ptr()), int(v_x.data_ptr()),
                int(v_bias.data_ptr()), int(v_g.data_ptr()),
                int(v_out.data_ptr()), int(v_out.numel()), int(v_out.shape[-1]),
                int(a_res.data_ptr()), int(a_x.data_ptr()),
                int(a_g.data_ptr()), int(a_out.data_ptr()), int(a_out.numel()),
                int(a_out.shape[-1]),
                int(u_res.data_ptr()), int(u_x.data_ptr()), int(u_out.data_ptr()),
                int(u_out.numel()), int(u_out.shape[-1]), cs())
        else:
            fvk.motus_joint_residual3_out_bf16_action_nobias(
                int(v_res.data_ptr()), int(v_x.data_ptr()),
                int(v_bias.data_ptr()), int(v_g.data_ptr()),
                int(v_out.data_ptr()), int(v_out.numel()), int(v_out.shape[-1]),
                int(a_res.data_ptr()), int(a_x.data_ptr()),
                int(a_g.data_ptr()), int(a_out.data_ptr()), int(a_out.numel()),
                int(a_out.shape[-1]),
                int(u_res.data_ptr()), int(u_x.data_ptr()), int(u_out.data_ptr()),
                int(u_out.numel()), int(u_out.shape[-1]), cs())
    elif use_v_gate_fp8:
        fvk.motus_joint_residual3_out_bf16_vgate_fp8(
            int(v_res.data_ptr()), int(v_x.data_ptr()),
            int(v_bias.data_ptr()),
            int(video_gate.q[video_gate_idx].data_ptr()),
            int(video_gate.scale[video_gate_idx].data_ptr()),
            int(v_out.data_ptr()), int(v_out.numel()),
            int(v_out.shape[-1]),
            int(a_res.data_ptr()), int(a_x.data_ptr()),
            int(a_bias.data_ptr()),
            int(a_g.data_ptr()), int(a_out.data_ptr()), int(a_out.numel()),
            int(a_out.shape[-1]),
            int(u_res.data_ptr()), int(u_x.data_ptr()), int(u_out.data_ptr()),
            int(u_out.numel()), int(u_out.shape[-1]), cs())
    else:
        fvk.motus_joint_residual3_out_bf16(
            int(v_res.data_ptr()), int(v_x.data_ptr()),
            int(v_bias.data_ptr()),
            int(v_g.data_ptr()), int(v_out.data_ptr()), int(v_out.numel()),
            int(v_out.shape[-1]),
            int(a_res.data_ptr()), int(a_x.data_ptr()),
            int(a_bias.data_ptr()),
            int(a_g.data_ptr()), int(a_out.data_ptr()), int(a_out.numel()),
            int(a_out.shape[-1]),
            int(u_res.data_ptr()), int(u_x.data_ptr()), int(u_out.data_ptr()),
            int(u_out.numel()), int(u_out.shape[-1]), cs())
    return v_out, a_out, u_out


def install_modulate_fuse(model) -> dict:
    """Patch ``WanVideoModule.process_ffn`` and the analogous
    ``process_ffn`` on the ActionExpert wrapper. Joint-attention paths
    are NOT touched here.
    """
    counts = {'video_ffn': 0, 'action_ffn': 0, 'skipped': 0,
              'modulation6': 0}

    if (hasattr(fvk, 'adaln_modulation6_bf16')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_25_MOD6', '0') != '1'):
        video_mod0 = model.video_module
        action_mod0 = getattr(model, 'action_module', None)

        def fused_video_compute_adaln(self, video_adaln_params, layer_idx):
            wan_layer = self.video_model.wan_model.blocks[layer_idx]
            return _adaln_modulation6_bf16(
                video_adaln_params, wan_layer.modulation)

        video_mod0.compute_adaln_modulation = (
            fused_video_compute_adaln.__get__(video_mod0))
        counts['modulation6'] += 1

        if action_mod0 is not None and hasattr(action_mod0, 'action_expert'):
            def fused_action_compute_adaln(self, action_adaln_params, layer_idx):
                action_layer = self.action_expert.blocks[layer_idx]
                return _adaln_modulation6_bf16(
                    action_adaln_params, action_layer.modulation)

            action_mod0.compute_adaln_modulation = (
                fused_action_compute_adaln.__get__(action_mod0))
            counts['modulation6'] += 1

    video_mod = model.video_module

    # ── WanVideoModule.process_ffn (motus.py L186-198) ────────────
    def fused_video_process_ffn(self, video_tokens, video_adaln_modulation,
                                layer_idx):
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        eps = float(wan_layer.norm2.eps)
        # G7.18: same pattern as G7.17 (Wan video QKV pre-norm), now
        # for FFN pre-norm. Prefill the FP8 up-site x_fp8 buffer via
        # ada_layer_norm_fp8 directly; FFN forward (G3d) detects the
        # _x_fp8_prefilled flag and skips its internal quant.
        from flash_rt.models.motus._fp8_swap import _STATE as _FP8_STATE
        ffn_mod = wan_layer.ffn
        up_site = getattr(ffn_mod, '_fp8_up_site', None)
        if (up_site is not None
                and not _FP8_STATE.calibrating
                and hasattr(fvk, 'ada_layer_norm_fp8')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_17',
                                    '0') != '1'):
            B0, L_v0, C0 = video_tokens.shape
            M_v = B0 * L_v0
            x_fp8 = up_site.ensure_x_fp8(M_v, video_tokens.device)
            video_c = (video_tokens if video_tokens.is_contiguous()
                       else video_tokens.contiguous())
            if (_is_static_mod_fp8(v_mod)
                    and hasattr(fvk, 'ada_layer_norm_fp8_modfp8')
                    and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_ADALN',
                                       '0') != '1'):
                fvk.ada_layer_norm_fp8_modfp8(
                    int(video_c.data_ptr()),
                    int(v_mod.q[4].data_ptr()), int(v_mod.q[3].data_ptr()),
                    int(v_mod.scale[4].data_ptr()),
                    int(v_mod.scale[3].data_ptr()),
                    int(x_fp8.data_ptr()), int(up_site.act_scale.data_ptr()),
                    M_v, C0, eps, cs())
            else:
                scale_b = v_mod[4].squeeze(2).to(torch.bfloat16).contiguous()
                shift_b = v_mod[3].squeeze(2).to(torch.bfloat16).contiguous()
                if scale_b.dim() == 3:
                    scale_b = scale_b.reshape(M_v, C0).contiguous()
                    shift_b = shift_b.reshape(M_v, C0).contiguous()
                fvk.ada_layer_norm_fp8(
                    int(video_c.data_ptr()), int(scale_b.data_ptr()),
                    int(shift_b.data_ptr()), int(x_fp8.data_ptr()),
                    int(up_site.act_scale.data_ptr()),
                    M_v, C0, eps, cs())
            up_site._x_fp8_prefilled = True
            ffn_input = video_c   # passthrough; FFN will use prefilled fp8
        else:
            ffn_input = _ada_modulate_bf16(
                video_tokens, v_mod[4].squeeze(2),
                v_mod[3].squeeze(2), eps)
        # FFN (G3d already replaced this with direct fvk chain).
        # G7.9: when down_site.bias_skip=True the FFN forward returns
        # the raw GEMM output (no bias added); we fold bias into the
        # gated residual via bias_gate_mul_residual_bf16 below.
        ffn_out = wan_layer.ffn(ffn_input)
        down_site = getattr(wan_layer.ffn, '_fp8_down_site', None)
        if (down_site is not None and down_site.bias_skip
                and down_site.has_bias):
            return _bias_gate_residual_bf16(
                video_tokens, ffn_out, down_site.bias,
                v_mod if _is_static_mod_fp8(v_mod) else v_mod[5].squeeze(2),
                5 if _is_static_mod_fp8(v_mod) else None)
        return _gate_residual_bf16(
            video_tokens, ffn_out,
            v_mod if _is_static_mod_fp8(v_mod) else v_mod[5].squeeze(2),
            5 if _is_static_mod_fp8(v_mod) else None)

    video_mod.process_ffn = fused_video_process_ffn.__get__(video_mod)
    counts['video_ffn'] += 1

    # ── G6.6: WanVideoModule.process_joint_attention (motus.py L206-274) ──
    # Fuse two paths:
    #   (a) Pre-attn AdaLN modulate (L225-226): video and action.
    #       both norm1's are WanLayerNorm → ada_layer_norm_bf16 applies.
    #       und has no modulate.
    #   (b) Gated residuals at the end (L270-271): video and action.
    #       und uses plain residual (kept as-is).
    # The body in between (QKV einsum, RoPE, MoT joint attn, o-proj)
    # is replicated verbatim from upstream — that's a separate rewrite.
    def fused_video_process_joint(self, video_tokens, action_tokens,
                                  video_adaln_modulation,
                                  action_adaln_modulation, layer_idx,
                                  action_block, und_tokens, und_block):
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Fused pre-attn modulate (replaces L225, L226)
        eps_v = float(wan_layer.norm1.eps)
        # G7.17: prefill the fused QKV site's x_fp8 buffer directly via
        # the fused ada_layer_norm_fp8 kernel — skips the bf16 (B*L_v, C)
        # round-trip (~15 MB at T=2520 D=3072 wan video) PLUS the
        # downstream redundant quantize launch in fused_qkv_fn. Falls
        # back to the 2-launch chain during calibration (act_scale not
        # yet valid) and when the kernel binding is missing.
        from flash_rt.models.motus._fp8_swap import _STATE as _FP8_STATE
        sa = wan_layer.self_attn
        qkv_site = getattr(sa, '_fused_qkv_site', None)
        norm_video = None
        _e = _jp_start()
        if (qkv_site is not None
                and bool(getattr(qkv_site, 'nvfp4_ready', False))
                and getattr(qkv_site, 'nvfp4_inv_s', None) is None
                and hasattr(fvk, 'ada_layer_norm_nvfp4_swizzled')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_39_NVFP4_ADALN',
                                   '0') != '1'):
            B0, L_v0, C0 = video_tokens.shape
            M_v = B0 * L_v0
            from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                _swizzled_sf_bytes)
            if (getattr(qkv_site, 'nvfp4_in_packed', None) is None
                    or qkv_site.nvfp4_in_packed.shape[0] < M_v):
                qkv_site.nvfp4_in_packed = torch.empty(
                    M_v, C0 // 2, dtype=torch.uint8,
                    device=video_tokens.device)
                qkv_site.nvfp4_in_sf = torch.zeros(
                    _swizzled_sf_bytes(M_v, C0), dtype=torch.uint8,
                    device=video_tokens.device)
                qkv_site.nvfp4_out = torch.empty(
                    M_v, 3 * C0, dtype=torch.bfloat16,
                    device=video_tokens.device)
            video_c = (video_tokens if video_tokens.is_contiguous()
                       else video_tokens.contiguous())
            if (_is_static_mod_fp8(v_mod)
                    and hasattr(fvk, 'ada_layer_norm_nvfp4_swizzled_modfp8')
                    and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_ADALN',
                                       '0') != '1'):
                fvk.ada_layer_norm_nvfp4_swizzled_modfp8(
                    int(video_c.data_ptr()),
                    int(v_mod.q[1].data_ptr()), int(v_mod.q[0].data_ptr()),
                    int(v_mod.scale[1].data_ptr()),
                    int(v_mod.scale[0].data_ptr()),
                    int(qkv_site.nvfp4_in_packed.data_ptr()),
                    int(qkv_site.nvfp4_in_sf.data_ptr()),
                    M_v, C0, eps_v, cs())
            else:
                scale_b = v_mod[1].squeeze(2).to(torch.bfloat16).contiguous()
                shift_b = v_mod[0].squeeze(2).to(torch.bfloat16).contiguous()
                if scale_b.dim() == 3:
                    scale_b = scale_b.reshape(M_v, C0).contiguous()
                    shift_b = shift_b.reshape(M_v, C0).contiguous()
                fvk.ada_layer_norm_nvfp4_swizzled(
                    int(video_c.data_ptr()), int(scale_b.data_ptr()),
                    int(shift_b.data_ptr()),
                    int(qkv_site.nvfp4_in_packed.data_ptr()),
                    int(qkv_site.nvfp4_in_sf.data_ptr()),
                    M_v, C0, eps_v, cs())
            qkv_site._x_fp8_prefilled = True
            norm_video = video_c
        elif (qkv_site is not None
                and not _FP8_STATE.calibrating
                and not bool(getattr(qkv_site, 'nvfp4_ready', False))
                and hasattr(fvk, 'ada_layer_norm_fp8')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_17',
                                    '0') != '1'):
            B0, L_v0, C0 = video_tokens.shape
            M_v = B0 * L_v0
            x_fp8 = qkv_site.ensure_x_fp8(M_v, video_tokens.device)
            video_c = (video_tokens if video_tokens.is_contiguous()
                       else video_tokens.contiguous())
            if (_is_static_mod_fp8(v_mod)
                    and hasattr(fvk, 'ada_layer_norm_fp8_modfp8')
                    and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_ADALN',
                                       '0') != '1'):
                fvk.ada_layer_norm_fp8_modfp8(
                    int(video_c.data_ptr()),
                    int(v_mod.q[1].data_ptr()), int(v_mod.q[0].data_ptr()),
                    int(v_mod.scale[1].data_ptr()),
                    int(v_mod.scale[0].data_ptr()),
                    int(x_fp8.data_ptr()), int(qkv_site.act_scale.data_ptr()),
                    M_v, C0, eps_v, cs())
            else:
                scale_b = v_mod[1].squeeze(2).to(torch.bfloat16).contiguous()
                shift_b = v_mod[0].squeeze(2).to(torch.bfloat16).contiguous()
                if scale_b.dim() == 3:
                    scale_b = scale_b.reshape(M_v, C0).contiguous()
                    shift_b = shift_b.reshape(M_v, C0).contiguous()
                fvk.ada_layer_norm_fp8(
                    int(video_c.data_ptr()), int(scale_b.data_ptr()),
                    int(shift_b.data_ptr()), int(x_fp8.data_ptr()),
                    int(qkv_site.act_scale.data_ptr()),
                    M_v, C0, eps_v, cs())
            qkv_site._x_fp8_prefilled = True
            # Pass video_tokens directly — fused_qkv_fn ignores its
            # content when _x_fp8_prefilled is True; only B, S shape
            # matters for downstream.
            norm_video = video_c
        else:
            norm_video = _ada_modulate_bf16(
                video_tokens, v_mod[1].squeeze(2), v_mod[0].squeeze(2),
                eps_v)
        _jp_end('pre_video_norm', _e)
        eps_a = float(action_block.norm1.eps)
        _e = _jp_start()
        _g724_a_pre = getattr(action_block, '_g724_state', None)
        use_action_adaln_fp8 = (
            _g724_a_pre is not None
            and _g724_a_pre.mode == 'fp8_ready'
            and not _FP8_STATE.calibrating
            and hasattr(fvk, 'awq_ada_layer_norm_fp8')
            and (os.environ.get('FLASH_RT_MOTUS_USE_G7_42_AU_QKV_NORM_FP8',
                                '0') == '1'
                 or os.environ.get('FLASH_RT_MOTUS_USE_G7_42_ACTION_QKV_NORM_FP8',
                                   '0') == '1')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_42_AU_QKV_NORM_FP8',
                               '0') != '1')
        if use_action_adaln_fp8:
            norm_action = None
        else:
            norm_action = _ada_modulate_bf16(
                action_tokens, a_mod[1].squeeze(2), a_mod[0].squeeze(2),
                eps_a)
        _jp_end('pre_action_norm', _e)

        # ── Body verbatim from upstream (motus.py L228-269) ────────
        B, L_v, C = norm_video.shape
        L_a = action_tokens.shape[1] if norm_action is None else norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n
        direct_joint_outputs = None

        # G7.20: BAGEL-style — replace action/und einsum + 3-narrow + 2-norm
        # 6-launch chain with: bf16_nn (1) + qkv_split_norm_rope_bf16
        # seq_len=0 (1) + V slice copy (1) = 3 launches per modality.
        # Plus eliminate torch.cat by writing each modality's Q/K/V
        # directly into pre-allocated joint buffer slices.
        # Per joint_attn call: 6 launches saved (cat×3 + action norm×1 +
        # action narrow×0 + action 1 launch saved on einsum→bf16_nn+split
        # equivalent) and corresponding intermediate bf16 buffers.
        bagel_eligible = (
            hasattr(fvk, 'qkv_split_norm_rope_bf16')
            and not _FP8_STATE.calibrating
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_20', '0') != '1')

        # ── Action QKV via fvk.bf16_nn into pre-alloc'd packed buffer ──
        if bagel_eligible:
            # Lazy-init persistent action/und QKV flattened weights on
            # the block. action_qkv shape (K=3, N=n, D=dim_a, E=d).
            # Required layout for bf16_nn: (D, K*N*E) = (D, 3*n*d).
            # Permute order (D, K, N, E) flattened gives chunks
            # k=0 (Q), k=1 (K), k=2 (V) of size N*E each.
            if not hasattr(action_block, '_action_qkv_flat'):
                w = action_block.wan_action_qkv.detach()  # (3, N, D, E)
                w = w.permute(2, 0, 1, 3).contiguous().reshape(
                    w.shape[2], 3 * w.shape[1] * w.shape[3])
                action_block._action_qkv_flat = w.contiguous()
            if not hasattr(und_block, '_und_qkv_flat'):
                w = und_block.wan_und_qkv.detach()
                w = w.permute(2, 0, 1, 3).contiguous().reshape(
                    w.shape[2], 3 * w.shape[1] * w.shape[3])
                und_block._und_qkv_flat = w.contiguous()
            # Block-level buffers (per layer, may be reused across steps).
            if not hasattr(action_block, '_action_packed_qkv'):
                qkv_dev = (action_tokens.device if norm_action is None
                           else norm_action.device)
                action_block._action_packed_qkv = torch.empty(
                    B * L_a, 3 * n * d, dtype=torch.bfloat16,
                    device=qkv_dev)
            a_packed = action_block._action_packed_qkv

        if bagel_eligible:
            _e = _jp_start()
            gemm = getattr(self.video_model.wan_model, '_g3b_gemm', None)
            if gemm is None:
                gemm = fvk.GemmRunner()
                self.video_model.wan_model._g3b_gemm = gemm
            # bf16_nn: A (M=B*L_a, K=dim_a) × B (K, N=3*n*d) -> D (M, N)
            norm_action_flat = None
            # G7.24: route to FP8 path if installed; capture calib amax
            # otherwise (during AWQ BF16 pass).
            _g724_a = getattr(action_block, '_g724_state', None)
            if (use_action_adaln_fp8
                    and _g724_a is not None and _g724_a.mode == 'fp8_ready'):
                _e2 = _jp_start()
                x_fp8 = _g724_a.ensure_buf(B * L_a, action_tokens.device)
                scale_b = a_mod[1].squeeze(2).to(torch.bfloat16).contiguous()
                shift_b = a_mod[0].squeeze(2).to(torch.bfloat16).contiguous()
                if scale_b.dim() == 3:
                    scale_b = scale_b.reshape(B * L_a, _g724_a.K).contiguous()
                    shift_b = shift_b.reshape(B * L_a, _g724_a.K).contiguous()
                action_c = (action_tokens if action_tokens.is_contiguous()
                            else action_tokens.contiguous())
                fvk.awq_ada_layer_norm_fp8(
                    int(action_c.data_ptr()), int(scale_b.data_ptr()),
                    int(shift_b.data_ptr()), int(_g724_a.inv_s.data_ptr()),
                    int(x_fp8.data_ptr()), int(_g724_a.act_scale.data_ptr()),
                    B * L_a, int(_g724_a.K), eps_a, cs())
                gemm.fp8_nn_dev(
                    int(x_fp8.data_ptr()), int(_g724_a.w_fp8.data_ptr()),
                    int(a_packed.data_ptr()),
                    B * L_a, int(_g724_a.N), int(_g724_a.K),
                    int(_g724_a.act_scale.data_ptr()),
                    int(_g724_a.w_scale.data_ptr()), cs())
                _jp_end('action_qkv_gemm_or_quant', _e2)
            elif _g724_a is not None and _g724_a.mode == 'fp8_ready':
                norm_action_flat = norm_action.contiguous().reshape(B * L_a, -1)
                from flash_rt.models.motus._action_und_qkv_fp8_swap import (
                    g724_fp8_gemm as _g724_fp8_gemm)
                from flash_rt.models.motus._fp8_swap import _STATE as _G724_FP8_STATE
                _e2 = _jp_start()
                _g724_fp8_gemm(
                    _g724_a, norm_action_flat, a_packed, gemm, cs(),
                    calibrating=_G724_FP8_STATE.calibrating)
                _jp_end('action_qkv_gemm_or_quant', _e2)
            else:
                norm_action_flat = norm_action.contiguous().reshape(B * L_a, -1)
                if _g724_a is not None and _g724_a.mode == 'calib_amax':
                    from flash_rt.models.motus._action_und_qkv_fp8_swap import (
                        g724_capture_amax as _g724_cap)
                    _g724_cap(_g724_a, norm_action_flat)
                _e2 = _jp_start()
                gemm.bf16_nn(
                    int(norm_action_flat.data_ptr()),
                    int(action_block._action_qkv_flat.data_ptr()),
                    int(a_packed.data_ptr()),
                    int(B * L_a), int(3 * n * d), int(norm_action_flat.shape[1]),
                    cs())
                _jp_end('action_qkv_gemm_or_quant', _e2)
            a_q_n = None
            a_k_n = None
            a_v = None
            use_qkv_norm2 = (
                hasattr(fvk, 'qkv_split_norm2_bf16')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_31_QKV_NORM2',
                                   '0') != '1')
            # freqs ptrs unused when seq_len=0; pass any valid (cached
            # video freq grid) buffer to satisfy non-null assertion.
            from flash_rt.models.motus import _rope_swap as _rs
            fr_ptr = (int(_rs._FREQ_GRID_RE_FP32.data_ptr())
                      if _rs._FREQ_GRID_RE_FP32 is not None else 0)
            fi_ptr = (int(_rs._FREQ_GRID_IM_FP32.data_ptr())
                      if _rs._FREQ_GRID_IM_FP32 is not None else 0)
            if not use_qkv_norm2:
                a_v = a_packed[:, 2*n*d:3*n*d].view(B, L_a, n, d)
                # Fallback-only buffers. The direct joint path below writes
                # action/und QKV into the persistent joint workspace directly,
                # so allocating these on every successful direct call is pure
                # graph/Python overhead.
                a_q_n = torch.empty(B, L_a, n, d, dtype=torch.bfloat16,
                                     device=a_packed.device)
                a_k_n = torch.empty(B, L_a, n, d, dtype=torch.bfloat16,
                                     device=a_packed.device)
                _e2 = _jp_start()
                fvk.qkv_split_norm_rope_bf16(
                    int(a_packed.data_ptr()),
                    int(action_block.wan_action_norm_q.weight.data_ptr()),
                    int(action_block.wan_action_norm_k.weight.data_ptr()),
                    fr_ptr, fi_ptr,
                    int(a_q_n.data_ptr()), int(a_k_n.data_ptr()),
                    int(B), int(L_a), int(n), int(d), int(0),
                    float(action_block.wan_action_norm_q.eps), cs())
                _jp_end('action_qkv_split_norm', _e2)
                a_q = a_q_n
                a_k = a_k_n
            _jp_end('action_qkv_pack_norm', _e)
        else:
            _e = _jp_start()
            a_qkv = torch.einsum(
                "BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
            a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
            a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(
                B, L_a, n, d)
            a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(
                B, L_a, n, d)
            a_v = a_v_h.view(B, L_a, n, d)
            _jp_end('action_qkv_pack_norm', _e)

        _e = _jp_start()
        _g724_u_pre = getattr(und_block, '_g724_state', None)
        use_und_ln_fp8 = (
            _g724_u_pre is not None
            and _g724_u_pre.mode == 'fp8_ready'
            and not _FP8_STATE.calibrating
            and hasattr(fvk, 'awq_layer_norm_fp8_bf16')
            and getattr(und_block.norm1, 'weight', None) is not None
            and getattr(und_block.norm1, 'bias', None) is not None
            and (os.environ.get('FLASH_RT_MOTUS_USE_G7_42_AU_QKV_NORM_FP8',
                                '0') == '1'
                 or os.environ.get('FLASH_RT_MOTUS_USE_G7_42_UND_QKV_NORM_FP8',
                                   '0') == '1')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_42_AU_QKV_NORM_FP8',
                               '0') != '1')
        if use_und_ln_fp8:
            norm_und = None
        else:
            norm_und = und_block.norm1(und_tokens)
        _jp_end('pre_und_norm', _e)
        L_u = und_tokens.shape[1] if norm_und is None else norm_und.shape[1]
        if bagel_eligible:
            _e = _jp_start()
            if not hasattr(und_block, '_und_packed_qkv'):
                qkv_dev = (und_tokens.device if norm_und is None
                           else norm_und.device)
                und_block._und_packed_qkv = torch.empty(
                    B * L_u, 3 * n * d, dtype=torch.bfloat16,
                    device=qkv_dev)
            u_packed = und_block._und_packed_qkv
            norm_und_flat = None
            # G7.24: route to FP8 path if installed; capture calib amax
            # otherwise (during AWQ BF16 pass).
            _g724_u = getattr(und_block, '_g724_state', None)
            if (use_und_ln_fp8
                    and _g724_u is not None and _g724_u.mode == 'fp8_ready'):
                _e2 = _jp_start()
                x_fp8 = _g724_u.ensure_buf(B * L_u, und_tokens.device)
                und_c = und_tokens if und_tokens.is_contiguous() else und_tokens.contiguous()
                fvk.awq_layer_norm_fp8_bf16(
                    int(und_c.data_ptr()), int(x_fp8.data_ptr()),
                    int(und_block.norm1.weight.data_ptr()),
                    int(und_block.norm1.bias.data_ptr()),
                    int(_g724_u.inv_s.data_ptr()),
                    int(_g724_u.act_scale.data_ptr()),
                    B * L_u, int(_g724_u.K),
                    float(und_block.norm1.eps), cs())
                gemm.fp8_nn_dev(
                    int(x_fp8.data_ptr()), int(_g724_u.w_fp8.data_ptr()),
                    int(u_packed.data_ptr()),
                    B * L_u, int(_g724_u.N), int(_g724_u.K),
                    int(_g724_u.act_scale.data_ptr()),
                    int(_g724_u.w_scale.data_ptr()), cs())
                _jp_end('und_qkv_gemm_or_quant', _e2)
            elif _g724_u is not None and _g724_u.mode == 'fp8_ready':
                norm_und_flat = norm_und.contiguous().reshape(B * L_u, -1)
                from flash_rt.models.motus._action_und_qkv_fp8_swap import (
                    g724_fp8_gemm as _g724_fp8_gemm)
                from flash_rt.models.motus._fp8_swap import _STATE as _G724_FP8_STATE
                _e2 = _jp_start()
                _g724_fp8_gemm(
                    _g724_u, norm_und_flat, u_packed, gemm, cs(),
                    calibrating=_G724_FP8_STATE.calibrating)
                _jp_end('und_qkv_gemm_or_quant', _e2)
            else:
                norm_und_flat = norm_und.contiguous().reshape(B * L_u, -1)
                if _g724_u is not None and _g724_u.mode == 'calib_amax':
                    from flash_rt.models.motus._action_und_qkv_fp8_swap import (
                        g724_capture_amax as _g724_cap)
                    _g724_cap(_g724_u, norm_und_flat)
                _e2 = _jp_start()
                gemm.bf16_nn(
                    int(norm_und_flat.data_ptr()),
                    int(und_block._und_qkv_flat.data_ptr()),
                    int(u_packed.data_ptr()),
                    int(B * L_u), int(3 * n * d), int(norm_und_flat.shape[1]),
                    cs())
                _jp_end('und_qkv_gemm_or_quant', _e2)

            if (use_qkv_norm2
                    and os.environ.get('FLASH_RT_MOTUS_USE_G7_44_JOINT_DIRECT_QKV_CAT',
                                       '1') == '1'
                    and os.environ.get('FLASH_RT_MOTUS_NO_G7_44_JOINT_DIRECT_QKV_CAT',
                                       '0') != '1'):
                seq_lens_direct = _get_static_seq_lens(
                    B, L_v + L_a + L_u, self.device)
                freqs_direct = self.video_model.wan_model.freqs
                if freqs_direct.device != self.device:
                    freqs_direct = freqs_direct.to(self.device)
                direct_joint_outputs = _joint_self_attn_direct_qkv_cat(
                    wan_layer.self_attn, norm_video, seq_lens_direct,
                    self.grid_sizes, freqs_direct, a_packed, u_packed,
                    action_block, und_block, L_a, L_u)

            if direct_joint_outputs is None:
                u_v = u_packed[:, 2*n*d:3*n*d].view(B, L_u, n, d)
                u_q_n = torch.empty(B, L_u, n, d, dtype=torch.bfloat16,
                                     device=u_packed.device)
                u_k_n = torch.empty(B, L_u, n, d, dtype=torch.bfloat16,
                                     device=u_packed.device)
                if use_qkv_norm2:
                    if a_q_n is None or a_k_n is None:
                        a_q_n = torch.empty(
                            B, L_a, n, d, dtype=torch.bfloat16,
                            device=a_packed.device)
                        a_k_n = torch.empty(
                            B, L_a, n, d, dtype=torch.bfloat16,
                            device=a_packed.device)
                    if a_v is None:
                        a_v = a_packed[:, 2*n*d:3*n*d].view(B, L_a, n, d)
                    _e2 = _jp_start()
                    fvk.qkv_split_norm2_bf16(
                        int(a_packed.data_ptr()),
                        int(action_block.wan_action_norm_q.weight.data_ptr()),
                        int(action_block.wan_action_norm_k.weight.data_ptr()),
                        int(a_q_n.data_ptr()), int(a_k_n.data_ptr()),
                        int(B), int(L_a), int(n), int(d),
                        float(action_block.wan_action_norm_q.eps),
                        int(u_packed.data_ptr()),
                        int(und_block.wan_und_norm_q.weight.data_ptr()),
                        int(und_block.wan_und_norm_k.weight.data_ptr()),
                        int(u_q_n.data_ptr()), int(u_k_n.data_ptr()),
                        int(L_u), float(und_block.wan_und_norm_q.eps), cs())
                    _jp_end('action_und_qkv_split_norm2', _e2)
                    a_q = a_q_n
                    a_k = a_k_n
                else:
                    _e2 = _jp_start()
                    fvk.qkv_split_norm_rope_bf16(
                        int(u_packed.data_ptr()),
                        int(und_block.wan_und_norm_q.weight.data_ptr()),
                        int(und_block.wan_und_norm_k.weight.data_ptr()),
                        fr_ptr, fi_ptr,
                        int(u_q_n.data_ptr()), int(u_k_n.data_ptr()),
                        int(B), int(L_u), int(n), int(d), int(0),
                        float(und_block.wan_und_norm_q.eps), cs())
                    _jp_end('und_qkv_split_norm', _e2)
                u_q = u_q_n
                u_k = u_k_n
            _jp_end('und_qkv_pack_norm', _e)
        else:
            _e = _jp_start()
            u_qkv = torch.einsum(
                "BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
            u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
            u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(
                B, L_u, n, d)
            u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(
                B, L_u, n, d)
            u_v = u_v_h.view(B, L_u, n, d)
            _jp_end('und_qkv_pack_norm', _e)

        seq_lens = _get_static_seq_lens(B, L_v + L_a + L_u, self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        if direct_joint_outputs is None:
            _e = _jp_start()
            y, action_out_h, und_out_h = wan_layer.self_attn(
                norm_video, seq_lens, self.grid_sizes, freqs,
                action_q=a_q, action_k=a_k, action_v=a_v,
                und_q=u_q, und_k=u_k, und_v=u_v)
            _jp_end('video_self_attn', _e)
        else:
            y, action_out_h, und_out_h = direct_joint_outputs

        _e = _jp_start()
        _e2 = _jp_start()
        a_o_site = getattr(action_block.wan_action_o, '_awq_fp8_site', None)
        u_o_site = getattr(und_block.wan_und_o, '_awq_fp8_site', None)
        use_o_quant2 = (
            a_o_site is not None
            and u_o_site is not None
            and hasattr(fvk, 'awq_quant2_fp8_static_bf16')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_41_O_QUANT2',
                               '0') != '1')
        _post_attn_mega_done = False  # set True inside if use_o_quant2 and mega fires
        if use_o_quant2:
            from flash_rt.models.motus._fp8_swap import _STATE as _FP8_STATE
            if _FP8_STATE.calibrating:
                use_o_quant2 = False

        if use_o_quant2:
            gemm = getattr(self.video_model.wan_model, '_g3b_gemm', None)
            if gemm is None:
                gemm = fvk.GemmRunner()
                self.video_model.wan_model._g3b_gemm = gemm

            a_flat2 = _flat_heads_no_copy(action_out_h)
            u_flat2 = _flat_heads_no_copy(und_out_h)
            Ma = int(a_flat2.shape[0])
            Mu = int(u_flat2.shape[0])
            a_x_fp8, _ = a_o_site.ensure_buf(Ma, a_flat2.device)
            u_x_fp8, _ = u_o_site.ensure_buf(Mu, u_flat2.device)
            fvk.awq_quant2_fp8_static_bf16(
                int(a_flat2.data_ptr()), int(a_o_site.inv_s.data_ptr()),
                int(a_x_fp8.data_ptr()), int(a_o_site.act_scale.data_ptr()),
                Ma, int(a_o_site.K),
                int(u_flat2.data_ptr()), int(u_o_site.inv_s.data_ptr()),
                int(u_x_fp8.data_ptr()), int(u_o_site.act_scale.data_ptr()),
                Mu, int(u_o_site.K), cs())

            # ── Post-attn megakernel (env-gated, additive) ──
            # Fuses action+und O GEMM + bias + modulation + residual_add
            # into one launch. Bypasses the residual3 path for action+und;
            # video residual still uses existing fused path below.
            # Mega always applies bias in its epilogue + does residual_add.
            # v2 also handles wan post-attn (y -> +bias +gate +residual to video_tokens).
            _use_post_attn_mega = (
                os.environ.get('FLASH_RT_MOTUS_USE_POST_ATTN_MEGA', '0') == '1'
                and hasattr(fvk, 'motus_post_attn_megakernel')
                and a_o_site.has_bias and u_o_site.has_bias
                and (not _FP8_STATE.calibrating))
            # v2 enabled only when wan video_gate is BF16 (not static-fp8) for simplicity.
            _wan_gate_is_bf16 = not _is_static_mod_fp8(v_mod)
            _use_post_attn_mega_v2 = (_use_post_attn_mega
                                       and _wan_gate_is_bf16
                                       and hasattr(fvk, 'motus_post_attn_megakernel_v2'))
            _wan_mega_done = False
            if _use_post_attn_mega:
                if not hasattr(a_o_site, '_alpha_cached'):
                    a_o_site._alpha_cached = float(
                        a_o_site.act_scale[0].item() * a_o_site.w_scale[0].item())
                if not hasattr(u_o_site, '_alpha_cached'):
                    u_o_site._alpha_cached = float(
                        u_o_site.act_scale[0].item() * u_o_site.w_scale[0].item())
                a_mod_flat = a_mod[2].squeeze(2).contiguous().view(-1)
                a_res_flat = action_tokens.contiguous().view(Ma, -1)
                u_res_flat = und_tokens.contiguous().view(Mu, -1)
                a_o_bias = a_o_site.bias
                u_o_bias = u_o_site.bias
                _e2_mega = _jp_start()
                if _use_post_attn_mega_v2:
                    # wan: y is already wan_O projected. We apply (bias?) + gate + residual.
                    # Find wan O bias (deferred if v_bias_skip).
                    o_fp8_site_local = getattr(wan_layer.self_attn.o, '_fp8_site', None)
                    v_bias_skip_local = (o_fp8_site_local is not None
                                          and o_fp8_site_local.bias_skip
                                          and o_fp8_site_local.has_bias)
                    wan_bias_ptr = (int(o_fp8_site_local.bias.data_ptr())
                                     if v_bias_skip_local else 0)
                    # Gate: v_mod[2].squeeze(2) → [B, 1, N_w] → flat [N_w]
                    v_gate_flat = v_mod[2].squeeze(2).contiguous().view(-1)
                    L_v_int = int(video_tokens.shape[1])
                    N_w = int(video_tokens.shape[-1])
                    v_res_flat = video_tokens.contiguous().view(L_v_int, N_w)
                    y_flat = y.contiguous().view(L_v_int, N_w)
                    rc = fvk.motus_post_attn_megakernel_v2(
                        int(a_x_fp8.data_ptr()),
                        int(a_o_site.w_fp8.data_ptr()),
                        float(a_o_site._alpha_cached),
                        int(a_o_bias.data_ptr()),
                        int(a_mod_flat.data_ptr()),
                        int(a_res_flat.data_ptr()),
                        int(a_res_flat.data_ptr()),
                        Ma, int(a_o_site.N),
                        int(u_x_fp8.data_ptr()),
                        int(u_o_site.w_fp8.data_ptr()),
                        float(u_o_site._alpha_cached),
                        int(u_o_bias.data_ptr()),
                        int(u_res_flat.data_ptr()),
                        int(u_res_flat.data_ptr()),
                        Mu, int(u_o_site.N),
                        int(y_flat.data_ptr()),
                        wan_bias_ptr,
                        int(v_gate_flat.data_ptr()),
                        int(v_res_flat.data_ptr()),
                        int(v_res_flat.data_ptr()),
                        L_v_int, N_w,
                        int(a_o_site.K),
                        cs())
                    _wan_mega_done = True
                    video_tokens = v_res_flat.view(B, L_v_int, N_w)
                else:
                    rc = fvk.motus_post_attn_megakernel(
                        int(a_x_fp8.data_ptr()),
                        int(a_o_site.w_fp8.data_ptr()),
                        float(a_o_site._alpha_cached),
                        int(a_o_bias.data_ptr()),
                        int(a_mod_flat.data_ptr()),
                        int(a_res_flat.data_ptr()),
                        int(a_res_flat.data_ptr()),
                        Ma, int(a_o_site.N),
                        int(u_x_fp8.data_ptr()),
                        int(u_o_site.w_fp8.data_ptr()),
                        float(u_o_site._alpha_cached),
                        int(u_o_bias.data_ptr()),
                        int(u_res_flat.data_ptr()),
                        int(u_res_flat.data_ptr()),
                        Mu, int(u_o_site.N),
                        int(a_o_site.K),
                        cs())
                _jp_end('post_attn_megakernel', _e2_mega)
                if rc != 0:
                    raise RuntimeError(f'post_attn_megakernel rc={rc}')
                action_tokens = a_res_flat.view(B, L_a, -1)
                und_tokens = u_res_flat.view(B, L_u, -1)
                _jp_end('action_o_proj', _e2)
                _jp_end('und_o_proj', _e2)
                _jp_end('action_und_o_proj', _e)
                _post_attn_mega_done = True
                action_out = None
                und_out = None
            else:
                _post_attn_mega_done = False

            if not _post_attn_mega_done:
                action_out = getattr(action_block, '_action_o_out_buf', None)
                if (action_out is None
                        or action_out.shape[0] < Ma
                        or action_out.shape[1] != a_o_site.N
                        or action_out.device != a_flat2.device):
                    action_out = torch.empty(
                        Ma, a_o_site.N, dtype=torch.bfloat16,
                        device=a_flat2.device)
                    action_block._action_o_out_buf = action_out
                else:
                    action_out = action_out[:Ma, :a_o_site.N]
                und_out = getattr(und_block, '_und_o_out_buf', None)
                if (und_out is None
                        or und_out.shape[0] < Mu
                        or und_out.shape[1] != u_o_site.N
                        or und_out.device != u_flat2.device):
                    und_out = torch.empty(
                        Mu, u_o_site.N, dtype=torch.bfloat16,
                        device=u_flat2.device)
                    und_block._und_o_out_buf = und_out
                else:
                    und_out = und_out[:Mu, :u_o_site.N]
                gemm.fp8_nn_dev(
                    int(a_x_fp8.data_ptr()), int(a_o_site.w_fp8.data_ptr()),
                    int(action_out.data_ptr()),
                    Ma, int(a_o_site.N), int(a_o_site.K),
                    int(a_o_site.act_scale.data_ptr()),
                    int(a_o_site.w_scale.data_ptr()), cs())
                gemm.fp8_nn_dev(
                    int(u_x_fp8.data_ptr()), int(u_o_site.w_fp8.data_ptr()),
                    int(und_out.data_ptr()),
                    Mu, int(u_o_site.N), int(u_o_site.K),
                    int(u_o_site.act_scale.data_ptr()),
                    int(u_o_site.w_scale.data_ptr()), cs())
                if a_o_site.has_bias and not a_o_site.bias_skip:
                    fvk.add_bias_bf16(
                        int(action_out.data_ptr()), int(a_o_site.bias.data_ptr()),
                        Ma, int(a_o_site.N), cs())
                if u_o_site.has_bias and not u_o_site.bias_skip:
                    fvk.add_bias_bf16(
                        int(und_out.data_ptr()), int(u_o_site.bias.data_ptr()),
                        Mu, int(u_o_site.N), cs())
                action_out = action_out.view(B, L_a, a_o_site.N)
                und_out = und_out.view(B, L_u, u_o_site.N)
                _jp_end('action_o_proj', _e2)
                _jp_end('und_o_proj', _e2)
        else:
            und_out = und_block.wan_und_o(und_out_h.flatten(2))
            _jp_end('und_o_proj', _e2)
            _e2 = _jp_start()
            action_out = action_block.wan_action_o(action_out_h.flatten(2))
            _jp_end('action_o_proj', _e2)
        _jp_end('action_und_o_proj', _e)

        # G6.7: if wan_layer.self_attn.o is FP8 with bias_skip, the
        # `y` returned above is the raw GEMM output (no bias). Fold
        # the bias into the gated residual via bias_gate_mul_residual.
        o_fp8_site = getattr(wan_layer.self_attn.o, '_fp8_site', None)
        _e = _jp_start()
        # G7.12: fold action_o bias into bias_gate_mul_residual when
        # the BF16 Linear's bias_skip flag is on. After G7.13 AWQ-FP8
        # replaces wan_action_o.forward, preserve the same bias-defer
        # contract through the AWQ site.
        action_o_mod = action_block.wan_action_o
        a_o_awq_site = getattr(action_o_mod, '_awq_fp8_site', None)
        a_o_awq_skip = (
            a_o_awq_site is not None
            and a_o_awq_site.bias_skip
            and a_o_awq_site.has_bias)
        a_o_bf16_skip = (
            getattr(action_o_mod, '_bf16_bias_skip_flag', None) is not None
            and action_o_mod._bf16_bias_skip_flag[0]
            and getattr(action_o_mod, '_bf16_bias', None) is not None)
        a_o_no_bias = (
            a_o_awq_site is not None
            and not a_o_awq_site.has_bias
            and hasattr(fvk, 'motus_joint_residual3_out_bf16_action_nobias'))
        a_o_skip = a_o_awq_skip or a_o_bf16_skip
        action_o_bias = (
            a_o_awq_site.bias if a_o_awq_skip else
            getattr(action_o_mod, '_bf16_bias', None))
        v_bias_skip = (o_fp8_site is not None and o_fp8_site.bias_skip
                       and o_fp8_site.has_bias)
        if _post_attn_mega_done:
            # action_tokens / und_tokens already residual'd in mega.
            if _wan_mega_done:
                # video_tokens also done by v2 mega; nothing more for wan.
                pass
            else:
                # v1: handle wan separately (existing path).
                if v_bias_skip:
                    video_tokens = _bias_gate_residual_bf16(
                        video_tokens, y, o_fp8_site.bias,
                        v_mod if _is_static_mod_fp8(v_mod) else v_mod[2].squeeze(2),
                        2 if _is_static_mod_fp8(v_mod) else None)
                else:
                    video_tokens = _gate_residual_bf16(
                        video_tokens, y,
                        v_mod if _is_static_mod_fp8(v_mod) else v_mod[2].squeeze(2),
                        2 if _is_static_mod_fp8(v_mod) else None)
        elif (v_bias_skip and (a_o_skip or (
                    a_o_no_bias
                and os.environ.get('FLASH_RT_MOTUS_USE_G7_46_JOINT_RES3_ACTION_NOBIAS',
                                       '1') == '1'))
                and hasattr(fvk, 'motus_joint_residual3_out_bf16')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_30_JOINT_RES3',
                                   '0') != '1'
                and os.environ.get('FLASH_RT_MOTUS_USE_G7_30_JOINT_RES3',
                                   '1') == '1'):
            video_tokens, action_tokens, und_tokens = _joint_residual3_bf16(
                video_tokens, y, o_fp8_site.bias,
                v_mod if _is_static_mod_fp8(v_mod) else v_mod[2].squeeze(2),
                2 if _is_static_mod_fp8(v_mod) else None,
                action_tokens, action_out, action_o_bias,
                a_mod[2].squeeze(2),
                und_tokens, und_out)
        else:
            if v_bias_skip:
                video_tokens = _bias_gate_residual_bf16(
                    video_tokens, y, o_fp8_site.bias,
                    v_mod if _is_static_mod_fp8(v_mod) else v_mod[2].squeeze(2),
                    2 if _is_static_mod_fp8(v_mod) else None)
            else:
                video_tokens = _gate_residual_bf16(
                    video_tokens, y,
                    v_mod if _is_static_mod_fp8(v_mod) else v_mod[2].squeeze(2),
                    2 if _is_static_mod_fp8(v_mod) else None)
            if a_o_skip:
                action_tokens = _bias_gate_residual_bf16(
                    action_tokens, action_out,
                    action_o_bias,
                    a_mod[2].squeeze(2))
            else:
                action_tokens = _gate_residual_bf16(
                    action_tokens, action_out, a_mod[2].squeeze(2))
            # und uses plain residual (no gate). Keep this in fvk so the hot
            # graph does not contain a PyTorch elementwise add.
            und_tokens = _add_bf16_out(und_tokens, und_out)
        _jp_end('post_residuals', _e)

        return video_tokens, action_tokens, und_tokens

    video_mod.process_joint_attention = (
        fused_video_process_joint.__get__(video_mod))
    counts['joint_attention'] = 1

    # G6.7: enable bias-deferred FP8 path for self_attn.o on each Wan
    # block. The patched joint-attention forward folds the bias into
    # bias_gate_mul_residual_bf16 (1 launch saved per layer per step).
    g67_enabled = (
        os.environ.get('FLASH_RT_MOTUS_NO_G6_7', '0') != '1')
    wan_blocks = model.video_model.wan_model.blocks
    if g67_enabled:
        n_bias_skip = 0
        for blk in wan_blocks:
            o_mod = blk.self_attn.o
            site = getattr(o_mod, '_fp8_site', None)
            if site is not None and site.has_bias:
                site.bias_skip = True
                n_bias_skip += 1
        counts['bias_skip_self_attn_o'] = n_bias_skip
        logger.info(
            f"[g6.7] enabled bias_skip on {n_bias_skip} self_attn.o sites")
    else:
        counts['bias_skip_self_attn_o'] = 0
        logger.info('[g6.7] FLASH_RT_MOTUS_NO_G6_7=1, bias-fused path off')

    # G7.12: enable bias-deferred BF16 path for action_o on each
    # ActionExpert block. fused_video_process_joint folds the bias
    # into bias_gate_mul_residual_bf16 (saves 1 add launch per layer
    # per step). und_o is left as-is because the eq. fold there would
    # require a clone (no in-place 'dst = a + b + bias' kernel) which
    # cancels the saving.
    g712_enabled = (
        os.environ.get('FLASH_RT_MOTUS_NO_G7_12', '0') != '1')
    action_blocks = getattr(model.action_expert, 'blocks', None)
    if g712_enabled and action_blocks is not None:
        n_a_skip = 0
        for ablk in action_blocks:
            ao = getattr(ablk, 'wan_action_o', None)
            flag = getattr(ao, '_bf16_bias_skip_flag', None)
            ao_bias = getattr(ao, '_bf16_bias', None)
            if flag is not None and ao_bias is not None:
                flag[0] = True
                n_a_skip += 1
        counts['bias_skip_action_o'] = n_a_skip
        logger.info(
            f"[g7.12] enabled bias_skip on {n_a_skip} action_o sites")
    else:
        counts['bias_skip_action_o'] = 0

    # G7.9: enable bias-deferred FP8 path for video FFN down-proj on
    # each Wan block. process_ffn (above) folds the down bias into
    # bias_gate_mul_residual_bf16 (saves 1 add_bias launch per layer
    # per step).
    g79_enabled = (
        os.environ.get('FLASH_RT_MOTUS_NO_G7_9', '0') != '1')
    if g79_enabled:
        n_ffn_skip = 0
        for blk in wan_blocks:
            ffn_mod = getattr(blk, 'ffn', None)
            down_site = getattr(ffn_mod, '_fp8_down_site', None) if ffn_mod is not None else None
            if down_site is not None and down_site.has_bias:
                down_site.bias_skip = True
                n_ffn_skip += 1
        counts['bias_skip_ffn_down'] = n_ffn_skip
        logger.info(
            f"[g7.9] enabled bias_skip on {n_ffn_skip} FFN down-proj sites")
    else:
        counts['bias_skip_ffn_down'] = 0
        logger.info('[g7.9] FLASH_RT_MOTUS_NO_G7_9=1, FFN bias-fused path off')

    # ── ActionModule.process_ffn (motus.py L464-476) ──────────────
    # ActionModule is held at model.action_module (NOT model.action_expert
    # which is the raw ActionExpert nn.Module). The wrapper exposes the
    # process_ffn method that gets called from the per-step loop.
    action_mod = getattr(model, 'action_module', None)
    if action_mod is not None and hasattr(action_mod, 'process_ffn'):
        def fused_action_process_ffn(self, action_tokens,
                                     action_adaln_modulation, layer_idx):
            action_block = self.action_expert.blocks[layer_idx]
            a_mod = action_adaln_modulation
            eps = float(action_block.norm2.eps) if hasattr(
                action_block.norm2, 'eps') else 1e-5
            from flash_rt.models.motus._fp8_swap import _STATE as _FP8_STATE
            up_site = getattr(action_block.ffn, '_fp8_up_site', None)
            if (up_site is not None
                    and not _FP8_STATE.calibrating
                    and hasattr(fvk, 'ada_layer_norm_fp8')
                    and os.environ.get('FLASH_RT_MOTUS_USE_G7_50_ACTION_FFN_PREFILL',
                                       '0') == '1'
                    and os.environ.get('FLASH_RT_MOTUS_NO_G7_50_ACTION_FFN_PREFILL',
                                       '0') != '1'):
                B0, L_a0, C0 = action_tokens.shape
                M_a = B0 * L_a0
                x_fp8 = up_site.ensure_x_fp8(M_a, action_tokens.device)
                action_c = (action_tokens if action_tokens.is_contiguous()
                            else action_tokens.contiguous())
                if (_is_static_mod_fp8(a_mod)
                        and hasattr(fvk, 'ada_layer_norm_fp8_modfp8')
                        and os.environ.get('FLASH_RT_MOTUS_NO_STATIC_MOD_BAKED_ADALN',
                                           '0') != '1'):
                    fvk.ada_layer_norm_fp8_modfp8(
                        int(action_c.data_ptr()),
                        int(a_mod.q[4].data_ptr()), int(a_mod.q[3].data_ptr()),
                        int(a_mod.scale[4].data_ptr()),
                        int(a_mod.scale[3].data_ptr()),
                        int(x_fp8.data_ptr()), int(up_site.act_scale.data_ptr()),
                        M_a, C0, eps, cs())
                else:
                    scale_b = a_mod[4].squeeze(2).to(torch.bfloat16).contiguous()
                    shift_b = a_mod[3].squeeze(2).to(torch.bfloat16).contiguous()
                    if scale_b.dim() == 3:
                        scale_b = scale_b.reshape(M_a, C0).contiguous()
                        shift_b = shift_b.reshape(M_a, C0).contiguous()
                    fvk.ada_layer_norm_fp8(
                        int(action_c.data_ptr()), int(scale_b.data_ptr()),
                        int(shift_b.data_ptr()), int(x_fp8.data_ptr()),
                        int(up_site.act_scale.data_ptr()), M_a, C0, eps, cs())
                up_site._x_fp8_prefilled = True
                ffn_input = action_c
            else:
                ffn_input = _ada_modulate_bf16(
                    action_tokens, a_mod[4].squeeze(2),
                    a_mod[3].squeeze(2), eps)
            ffn_out = action_block.ffn(ffn_input)
            return _gate_residual_bf16(
                action_tokens, ffn_out, a_mod[5].squeeze(2))

        action_mod.process_ffn = fused_action_process_ffn.__get__(action_mod)
        counts['action_ffn'] += 1
    else:
        counts['skipped'] += 1
        logger.info(
            "[g6.5] action_module.process_ffn not found; skipping "
            "(may be merged into another path in this checkpoint)")

    und_mod = getattr(model, 'und_module', None)
    if (os.environ.get('FLASH_RT_MOTUS_NO_G7_51_UND_FFN_RESIDUAL', '0') != '1'
            and und_mod is not None and hasattr(und_mod, 'process_ffn')):
        def fused_und_process_ffn(self, und_tokens, layer_idx):
            block = self.und_expert.blocks[layer_idx]
            from flash_rt.models.motus._fp8_swap import _STATE as _FP8_STATE
            up_site = getattr(block.ffn, '_fp8_up_site', None)
            use_prefill = os.environ.get(
                'FLASH_RT_MOTUS_USE_G7_51_UND_FFN_PREFILL', '0') == '1'
            if (use_prefill
                    and up_site is not None
                    and not _FP8_STATE.calibrating
                    and hasattr(fvk, 'layer_norm_no_affine_fp8_static_bf16')
                    and os.environ.get('FLASH_RT_MOTUS_NO_G7_51_UND_FFN_PREFILL',
                                       '0') != '1'):
                B0, L_u0, C0 = und_tokens.shape
                M_u = B0 * L_u0
                x_fp8 = up_site.ensure_x_fp8(M_u, und_tokens.device)
                und_c = und_tokens if und_tokens.is_contiguous() else und_tokens.contiguous()
                fvk.layer_norm_no_affine_fp8_static_bf16(
                    int(und_c.data_ptr()), int(x_fp8.data_ptr()),
                    int(up_site.act_scale.data_ptr()),
                    M_u, C0, float(block.norm2.eps), cs())
                up_site._x_fp8_prefilled = True
                ffn_input = und_c
            else:
                ffn_input = block.norm2(und_tokens)
            ffn_output = block.ffn(ffn_input)
            return _add_bf16_out(und_tokens, ffn_output)

        und_mod.process_ffn = fused_und_process_ffn.__get__(und_mod)
        counts['und_ffn'] = 1
    elif und_mod is None or not hasattr(und_mod, 'process_ffn'):
        counts['skipped'] += 1
        logger.info(
            "[g7.51] und_module.process_ffn not found; skipping")

    logger.info(f"[g6.5] modulate fuse installed: {counts}")
    return counts
