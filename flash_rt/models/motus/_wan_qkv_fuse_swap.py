"""G7.8 — Fuse Wan video Q/K/V into a single FP8 GEMM per block.

Today each Wan ``WanSelfAttention`` block calls ``self.q(x)``,
``self.k(x)``, ``self.v(x)`` as three separate FP8 W8A8 sites:

    qkv_fn(x):
        q = norm_q(self.q(x)).view(b, s, n, d)
        k = norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

That is **3 quantize_fp8_static + 3 fp8_nn_dev + 3 add_bias** launches
per layer per step, 30 layers x 10 steps = 300 invocations, so 1800
extra launches. After fuse: 1 quantize + 1 GEMM + 1 bias per block.
Saves ~6-12 ms in graph wall (per phase-diag estimate).

Implementation:
  1. Reconstruct bf16 weights from existing FP8 sites:
        bf16 ~= w_fp8.to(bf16) * w_scale
  2. Concatenate along output dim (K, N) -> (K, 3*N).
  3. Re-quantize with a unified scale = max(q_scale, k_scale, v_scale)
     (per-tensor; cos(action) >= 0.996 floor has ~0.0018 headroom).
  4. Concat biases (3*N,) bf16.
  5. Patch each wan_block.self_attn.forward to compute QKV via the
     fused site, then chunk(3) into q/k/v and continue with the
     upstream forward body unchanged.

Toggle: ``FLASH_RT_MOTUS_NO_G7_8=1``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from flash_rt.models.motus._stream import cs
from flash_rt.models.motus._fp8_swap import _Fp8Site
import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_JOINT_PROFILE = os.environ.get('FLASH_RT_MOTUS_JOINT_PROFILE', '0') == '1'
_JOINT_PROFILE_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
_SAGE_JOINT_BUF_CACHE: dict[tuple[int, int, int, int, int, torch.device],
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


def _concat3_qkv_bf16(q0, q1, q2, k0, k1, k2, v0, v1, v2):
    B, L0, H, D = q0.shape
    L1 = q1.shape[1]
    L2 = q2.shape[1]
    q_out = torch.empty(B, L0 + L1 + L2, H, D,
                        dtype=torch.bfloat16, device=q0.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)

    def s3(x):
        st = x.stride()
        return int(st[0]), int(st[1]), int(st[2])

    def fast_ok(x):
        st = x.stride()
        return (
            x.dtype == torch.bfloat16
            and x.is_cuda
            and x.shape[-2:] == (H, D)
            and int(st[-1]) == 1
            and int(st[-2]) == D
        )

    if (hasattr(fvk, 'concat3_qkv_bf16_fast')
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_40_CAT3_FAST',
                               '0') != '1'
            and all(fast_ok(x) for x in (q0, q1, q2, k0, k1, k2,
                                         v0, v1, v2))):
        def s2(x):
            st = x.stride()
            return int(st[0]), int(st[1])

        fvk.concat3_qkv_bf16_fast(
            int(q0.data_ptr()), int(q1.data_ptr()), int(q2.data_ptr()),
            int(k0.data_ptr()), int(k1.data_ptr()), int(k2.data_ptr()),
            int(v0.data_ptr()), int(v1.data_ptr()), int(v2.data_ptr()),
            int(q_out.data_ptr()), int(k_out.data_ptr()),
            int(v_out.data_ptr()),
            B, L0, L1, L2, H, D,
            *s2(q0), *s2(q1), *s2(q2),
            *s2(k0), *s2(k1), *s2(k2),
            *s2(v0), *s2(v1), *s2(v2),
            cs())
        return q_out, k_out, v_out

    fvk.concat3_qkv_bf16(
        int(q0.data_ptr()), int(q1.data_ptr()), int(q2.data_ptr()),
        int(k0.data_ptr()), int(k1.data_ptr()), int(k2.data_ptr()),
        int(v0.data_ptr()), int(v1.data_ptr()), int(v2.data_ptr()),
        int(q_out.data_ptr()), int(k_out.data_ptr()), int(v_out.data_ptr()),
        B, L0, L1, L2, H, D,
        *s3(q0), *s3(q1), *s3(q2),
        *s3(k0), *s3(k1), *s3(k2),
        *s3(v0), *s3(v1), *s3(v2),
        cs())
    return q_out, k_out, v_out


def _sage_joint_buffers(device: torch.device, b: int, total_L: int,
                        n: int, d: int) -> tuple[torch.Tensor, ...]:
    padded_L = ((total_L + 63) // 64) * 64
    key = (b, total_L, n, d, padded_L, device)
    bufs = _SAGE_JOINT_BUF_CACHE.get(key)
    if bufs is None:
        q8 = torch.empty(b, total_L, n, d, dtype=torch.int8, device=device)
        k8 = torch.empty_like(q8)
        v8 = torch.empty(
            b, d, n, padded_L, dtype=torch.float8_e4m3fn, device=device)
        qs = torch.empty(
            b, n, (total_L + 31) // 32, dtype=torch.float32, device=device)
        ks = torch.empty(
            b, n, (total_L + 63) // 64, dtype=torch.float32, device=device)
        vs = torch.empty(b, n, d, dtype=torch.float32, device=device)
        out = torch.empty(
            b, total_L, n, d, dtype=torch.bfloat16, device=device)
        bufs = (q8, k8, v8, qs, ks, vs, out)
        _SAGE_JOINT_BUF_CACHE[key] = bufs
    return bufs


_SAGE_JOINT_F16_BUF_CACHE: dict[tuple[int, int, int, int, torch.device],
                                tuple[torch.Tensor, ...]] = {}


def _sage_joint_f16_buffers(device: torch.device, b: int, total_L: int,
                            n: int, d: int) -> tuple[torch.Tensor, ...]:
    """Buffers for sage2 sv_f16 variant: V stays FP16/BF16, no v_scale,
    no V transpose. Saves the per-channel V scale reduce + TPP layout work."""
    key = (b, total_L, n, d, device)
    bufs = _SAGE_JOINT_F16_BUF_CACHE.get(key)
    if bufs is None:
        q8 = torch.empty(b, total_L, n, d, dtype=torch.int8, device=device)
        k8 = torch.empty_like(q8)
        v_fp16 = torch.empty(b, total_L, n, d, dtype=torch.float16, device=device)
        qs = torch.empty(
            b, n, (total_L + 31) // 32, dtype=torch.float32, device=device)
        ks = torch.empty(
            b, n, (total_L + 63) // 64, dtype=torch.float32, device=device)
        out = torch.empty(
            b, total_L, n, d, dtype=torch.bfloat16, device=device)
        bufs = (q8, k8, v_fp16, qs, ks, out)
        _SAGE_JOINT_F16_BUF_CACHE[key] = bufs
    return bufs


def _build_fused_qkv_site(q_site: _Fp8Site, k_site: _Fp8Site,
                          v_site: _Fp8Site,
                          q_bias: Optional[torch.Tensor],
                          k_bias: Optional[torch.Tensor],
                          v_bias: Optional[torch.Tensor],
                          label: str) -> _Fp8Site:
    """Build a unified _Fp8Site for the concatenated Q|K|V weight.

    Each input site stores its weight as (K, N) FP8 with a per-tensor
    scale (``w_scale``); reconstruct bf16, concat, re-quantize.
    """
    assert q_site.K == k_site.K == v_site.K, \
        f"K mismatch: q={q_site.K}, k={k_site.K}, v={v_site.K}"
    K = q_site.K
    N_q, N_k, N_v = q_site.N, k_site.N, v_site.N
    N_fused = N_q + N_k + N_v

    dev = q_site.w_fp8.device
    q_scale = float(q_site.w_scale.item())
    k_scale = float(k_site.w_scale.item())
    v_scale = float(v_site.w_scale.item())
    fused_scale = max(q_scale, k_scale, v_scale)

    # Reconstruct bf16 from FP8 + scale, concat along N (output) dim.
    q_bf = q_site.w_fp8.to(torch.bfloat16) * q_scale
    k_bf = k_site.w_fp8.to(torch.bfloat16) * k_scale
    v_bf = v_site.w_fp8.to(torch.bfloat16) * v_scale
    fused_bf = torch.cat([q_bf, k_bf, v_bf], dim=1).contiguous()  # (K, 3N)
    del q_bf, k_bf, v_bf

    nvfp4_w_packed = None
    nvfp4_w_sf = None
    nvfp4_awq_act = None
    if os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV', '0') == '1':
        from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
            quantize_weight_bf16_to_nvfp4_swz)
        nvfp4_w_packed, nvfp4_w_sf = quantize_weight_bf16_to_nvfp4_swz(
            fused_bf.t().contiguous())
        if os.environ.get('FLASH_RT_MOTUS_NVFP4_VIDEO_QKV_AWQ', '1') == '1':
            nvfp4_awq_act = torch.zeros(K, dtype=torch.float32, device=dev)

    # Re-quantize with the fused scale.
    fused_w_fp8 = (fused_bf.float() / fused_scale).clamp(-448, 448).to(
        _FP8).contiguous()
    del fused_bf

    # Concat bias (always present on Wan q/k/v Linears).
    biases = []
    for b in (q_bias, k_bias, v_bias):
        if b is None:
            biases.append(torch.zeros(0, dtype=torch.bfloat16, device=dev))
        else:
            biases.append(b.to(torch.bfloat16).contiguous())
    fused_bias = torch.cat(biases, dim=0).contiguous()
    has_bias = bool(fused_bias.numel() > 0)

    # Hand-built _Fp8Site bypassing __init__'s in-place quantize.
    site = _Fp8Site.__new__(_Fp8Site)
    site.w_fp8 = fused_w_fp8
    site.w_scale = torch.tensor(
        [fused_scale], dtype=torch.float32, device=dev)
    site.act_scale = torch.tensor(
        [1.0], dtype=torch.float32, device=dev)  # filled at calibration
    site.x_fp8_buf = None
    site.K = K
    site.N = N_fused
    site.label = label
    site.has_bias = has_bias
    site.bias = fused_bias if has_bias else None
    site.bias_skip = False
    site._x_fp8_prefilled = False
    site._last_packed_qkv = None
    site._qkv_bias_skip_once = False
    site.nvfp4_w_packed = nvfp4_w_packed
    site.nvfp4_w_sf = nvfp4_w_sf
    site.nvfp4_awq_act_amax_K = nvfp4_awq_act
    site.nvfp4_ready = False
    site.nvfp4_inv_s = None
    site.nvfp4_in_packed = None
    site.nvfp4_in_sf = None
    site.nvfp4_out = None
    return site


def _make_fused_qkv_forward(site: _Fp8Site, gemm: fvk.GemmRunner,
                             N_q: int, N_k: int, N_v: int):
    K = site.K
    N = site.N
    w_ptr = int(site.w_fp8.data_ptr())
    w_scale_ptr = int(site.w_scale.data_ptr())
    act_scale_ptr = int(site.act_scale.data_ptr())
    bias_ptr = int(site.bias.data_ptr()) if site.has_bias else 0

    # Local import to avoid circular ref at module load.
    from flash_rt.models.motus._fp8_swap import _STATE

    def fused_qkv(x: torch.Tensor):
        """x: (B, S, K) bf16 -> (q,k,v) tuple of (B, S, N_*) bf16."""
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K)
        M = flat.shape[0]
        device = flat.device

        if bool(getattr(site, 'nvfp4_ready', False)):
            if site.nvfp4_in_packed is None or site.nvfp4_in_packed.shape[0] < M:
                from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                    _swizzled_sf_bytes)
                site.nvfp4_in_packed = torch.empty(
                    M, K // 2, dtype=torch.uint8, device=device)
                site.nvfp4_in_sf = torch.zeros(
                    _swizzled_sf_bytes(M, K), dtype=torch.uint8, device=device)
                site.nvfp4_out = torch.empty(
                    M, N, dtype=torch.bfloat16, device=device)
            from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                _quantize_act_to_nvfp4)
            x_packed = site.nvfp4_in_packed[:M]
            x_sf = site.nvfp4_in_sf
            out = site.nvfp4_out[:M]
            prefilled = bool(getattr(site, '_x_fp8_prefilled', False))
            if prefilled:
                site._x_fp8_prefilled = False
            else:
                _quantize_act_to_nvfp4(flat, site.nvfp4_inv_s,
                                       x_packed, x_sf, M, K)
            _e = _jp_start()
            fvk.fp4_w4a16_gemm_sm120_bf16out(
                int(x_packed.data_ptr()),
                int(site.nvfp4_w_packed.data_ptr()),
                int(out.data_ptr()),
                M, N, K,
                int(x_sf.data_ptr()),
                int(site.nvfp4_w_sf.data_ptr()),
                1.0, cs())
            _jp_end('video_qkv_gemm', _e)

            skip_bias = bool(getattr(site, '_qkv_bias_skip_once', False))
            if skip_bias:
                site._qkv_bias_skip_once = False
            if bias_ptr and not skip_bias:
                _e = _jp_start()
                fvk.add_bias_bf16(int(out.data_ptr()), bias_ptr, M, N, cs())
                _jp_end('video_qkv_bias', _e)
            site._last_packed_qkv = out
            out = out.view(*in_shape[:-1], N)
            return (out.narrow(-1, 0, N_q),
                    out.narrow(-1, N_q, N_k),
                    out.narrow(-1, N_q + N_k, N_v))

        x_fp8 = site.ensure_x_fp8(M, device)
        n_act = M * K

        # G7.17: when joint_attn body pre-fills x_fp8 via the fused
        # ada_layer_norm_fp8 kernel, skip the redundant quantize step
        # here. The flag is set by the caller AFTER the prefill kernel
        # writes to x_fp8 and is reset to False by us after consumption
        # (next layer must reset or fill fresh).
        prefilled = bool(getattr(site, '_x_fp8_prefilled', False))
        if prefilled:
            site._x_fp8_prefilled = False
        elif _STATE.calibrating:
            if site.nvfp4_awq_act_amax_K is not None:
                amax = flat.detach().abs().amax(dim=0).float()
                torch.maximum(site.nvfp4_awq_act_amax_K, amax,
                              out=site.nvfp4_awq_act_amax_K)
            _e = _jp_start()
            fvk.quantize_fp8_device(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())
            _jp_end('video_qkv_quant', _e)
        else:
            _e = _jp_start()
            fvk.quantize_fp8_static(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())
            _jp_end('video_qkv_quant', _e)

        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        _e = _jp_start()
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), w_ptr, int(out.data_ptr()),
            M, N, K, act_scale_ptr, w_scale_ptr, cs())
        _jp_end('video_qkv_gemm', _e)

        skip_bias = bool(getattr(site, '_qkv_bias_skip_once', False))
        if skip_bias:
            site._qkv_bias_skip_once = False
        if bias_ptr and not skip_bias:
            _e = _jp_start()
            fvk.add_bias_bf16(int(out.data_ptr()), bias_ptr, M, N, cs())
            _jp_end('video_qkv_bias', _e)

        # G7.19: stash the packed (M, 3*dim) buffer on the site so the
        # patched self_attn forward can call qkv_split_norm_rope_bf16
        # directly on it (skipping the .narrow split + norm_q/k + rope
        # 5-launch chain).
        site._last_packed_qkv = out
        out = out.view(*in_shape[:-1], N)
        # chunk into q, k, v
        q = out.narrow(-1, 0, N_q)
        k = out.narrow(-1, N_q, N_k)
        v = out.narrow(-1, N_q + N_k, N_v)
        return q, k, v

    return fused_qkv


def install_wan_qkv_fuse(model) -> dict:
    """Install fused Q/K/V FP8 GEMM on every Wan block's self_attn.

    Must run AFTER install_fp8_swap (G4) and (currently) AFTER
    install_modulate_fuse (G6.5/6/7) — order unaffected since this
    only patches wan_block.self_attn.forward, which the patched
    fused_video_process_joint already invokes via the original
    upstream forward signature.
    """
    counts = {'fused': 0, 'skipped_no_fp8': 0, 'skipped_dim': 0}
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_8', '0') == '1':
        logger.info('[g7.8] FLASH_RT_MOTUS_NO_G7_8=1 — fuse skipped')
        return counts

    gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()

    wan_blocks = model.video_module.video_model.wan_model.blocks
    for blk in wan_blocks:
        sa = blk.self_attn
        q_site = getattr(sa.q, '_fp8_site', None)
        k_site = getattr(sa.k, '_fp8_site', None)
        v_site = getattr(sa.v, '_fp8_site', None)
        if not (q_site and k_site and v_site):
            counts['skipped_no_fp8'] += 1
            continue
        if not (q_site.K == k_site.K == v_site.K
                and q_site.N == k_site.N == v_site.N):
            counts['skipped_dim'] += 1
            continue

        q_bias = sa.q.bias if hasattr(sa.q, 'bias') else None
        k_bias = sa.k.bias if hasattr(sa.k, 'bias') else None
        v_bias = sa.v.bias if hasattr(sa.v, 'bias') else None

        fused_site = _build_fused_qkv_site(
            q_site, k_site, v_site, q_bias, k_bias, v_bias,
            label=f"{type(sa).__name__}.qkv_fused")
        fused_fn = _make_fused_qkv_forward(
            fused_site, gemm, q_site.N, k_site.N, v_site.N)
        sa._fused_qkv_fn = fused_fn
        sa._fused_qkv_site = fused_site

        # Free the now-redundant individual q/k/v FP8 sites' weight
        # storage to reclaim VRAM (~3 * 3072*3072 = ~9 MB per layer FP8;
        # 30 layers ~= 270 MB saved).
        # Don't actually free yet — the patched forward still references
        # sa.q/k/v? No, the patch below replaces qkv computation and
        # never calls them. Safe to nullify their w_fp8 storage.
        # But careful: sa.q.weight.data still points at the FP8 storage;
        # we leave it alone to avoid breaking other code (e.g. param
        # iteration) and just don't call .forward on them.
        counts['fused'] += 1

    # Patch the WanSelfAttention.forward (class-level patch) so every
    # block uses the fused path. Replicates upstream forward body
    # verbatim except qkv_fn replaced by a call to self._fused_qkv_fn.
    if counts['fused'] == 0:
        logger.info('[g7.8] no blocks fused — skipping forward patch')
        return counts

    sample_sa = wan_blocks[0].self_attn
    SaCls = type(sample_sa)
    # Avoid re-patching across reloads.
    if getattr(SaCls.forward, '_g7_8_patched', False):
        logger.info('[g7.8] forward already patched on %s', SaCls.__name__)
        return counts

    # Import once at patch time.
    from importlib import import_module
    wan_model_mod = import_module(SaCls.__module__)
    flash_attention = wan_model_mod.flash_attention
    rope_apply = wan_model_mod.rope_apply

    def fused_forward(self, x, seq_lens, grid_sizes, freqs,
                      action_q=None, action_k=None, action_v=None,
                      und_q=None, und_k=None, und_v=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # G7.19: when the fused QKV site is installed AND the
        # qkv_split_norm_rope_bf16 kernel is available AND the freq
        # grid is precomputed, run the master 5-into-1 fused kernel:
        #   .narrow(0:dim) split + RMSNorm(weight=norm_q) + RoPE
        #   .narrow(dim:2*dim) split + RMSNorm(weight=norm_k) + RoPE
        #   v = .narrow(2*dim:3*dim) view
        # Skips self.norm_q / self.norm_k / rope_apply 5-launch chain
        # AND eliminates the bf16 (B, S, dim) round-trips between them.
        site = getattr(self, '_fused_qkv_site', None)
        from flash_rt.models.motus import _rope_swap as _rs
        g719_eligible = (
            site is not None
            and hasattr(fvk, 'qkv_split_norm_rope_bf16')
            and _rs._FREQ_GRID_RE_FP32 is not None
            and _rs._FREQ_GRID_IM_FP32 is not None
            and (action_q is not None or und_q is not None)
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_19', '0') != '1')
        if g719_eligible and hasattr(self, '_fused_qkv_fn'):
            # Run the FP8 QKV GEMM, which stashes packed (B, S, 3*dim)
            # on site._last_packed_qkv. We ignore the returned splits.
            _e = _jp_start()
            use_bias_split = (
                site is not None
                and site.has_bias
                and hasattr(fvk, 'qkv_split_bias_norm_rope_v_bf16')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_32_QKV_BIAS_SPLIT',
                                   '0') != '1')
            if use_bias_split:
                site._qkv_bias_skip_once = True
            _ = self._fused_qkv_fn(x)
            packed = site._last_packed_qkv
            dim = self.num_heads * self.head_dim
            B0, S0 = b, s
            # Pre-allocated outputs for q_rope, k_rope. New each call;
            # CUDA graph capture pools the allocation.
            q_rope = torch.empty(B0, S0, n, d, dtype=torch.bfloat16,
                                  device=packed.device)
            k_rope = torch.empty(B0, S0, n, d, dtype=torch.bfloat16,
                                  device=packed.device)
            seq_len = _rs._FREQ_GRID_RE_FP32.shape[0]
            eps_qk = float(self.norm_q.eps)
            _e2 = _jp_start()
            if use_bias_split:
                v_biased = torch.empty(B0, S0, n, d, dtype=torch.bfloat16,
                                       device=packed.device)
                fvk.qkv_split_bias_norm_rope_v_bf16(
                    int(packed.data_ptr()), int(site.bias.data_ptr()),
                    int(self.norm_q.weight.data_ptr()),
                    int(self.norm_k.weight.data_ptr()),
                    int(_rs._FREQ_GRID_RE_FP32.data_ptr()),
                    int(_rs._FREQ_GRID_IM_FP32.data_ptr()),
                    int(q_rope.data_ptr()), int(k_rope.data_ptr()),
                    int(v_biased.data_ptr()),
                    int(B0), int(S0), int(n), int(d), int(seq_len),
                    eps_qk,
                    torch.cuda.current_stream().cuda_stream)
            else:
                fvk.qkv_split_norm_rope_bf16(
                    int(packed.data_ptr()),
                    int(self.norm_q.weight.data_ptr()),
                    int(self.norm_k.weight.data_ptr()),
                    int(_rs._FREQ_GRID_RE_FP32.data_ptr()),
                    int(_rs._FREQ_GRID_IM_FP32.data_ptr()),
                    int(q_rope.data_ptr()),
                    int(k_rope.data_ptr()),
                    int(B0), int(S0), int(n), int(d), int(seq_len),
                    eps_qk,
                    torch.cuda.current_stream().cuda_stream)
            _jp_end('video_qkv_split_norm_rope_kernel', _e2)
            # V is a packed view in the legacy path; the bias-fused
            # split kernel writes a materialized biased V.
            if use_bias_split:
                v_view = v_biased
            else:
                v_view = packed.view(B0, S0, 3 * dim).narrow(
                    -1, 2 * dim, dim).view(B0, S0, n, d)
            # G7.19 skips standard RoPE on (q, k); branch directly
            # into the MoT body using q_rope/k_rope as the
            # already-rotated video Q, K.
            q = q_rope
            k = k_rope
            v = v_view
            q_video_rope = q_rope
            k_video_rope = k_rope
            L_x = q.size(1)
            _jp_end('video_qkv_split_norm_rope', _e)
        elif hasattr(self, '_fused_qkv_fn'):
            _e = _jp_start()
            q_lin, k_lin, v_lin = self._fused_qkv_fn(x)   # (B, S, N_*)
            q = self.norm_q(q_lin).view(b, s, n, d)
            k = self.norm_k(k_lin).view(b, s, n, d)
            v = v_lin.view(b, s, n, d)
            q_video_rope = None
            k_video_rope = None
            L_x = None
            _jp_end('video_qkv_split_norm_rope', _e)
        else:
            _e = _jp_start()
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            q_video_rope = None
            k_video_rope = None
            L_x = None
            _jp_end('video_qkv_split_norm_rope', _e)

        # MoT branch: replicate upstream lines 197-260 verbatim.
        if action_q is not None or und_q is not None:
            if L_x is None:
                L_x = q.size(1)
            if q_video_rope is None:
                q_video_rope = rope_apply(q, grid_sizes, freqs)
                k_video_rope = rope_apply(k, grid_sizes, freqs)

            q_parts = [q_video_rope]
            k_parts = [k_video_rope]
            v_parts = [v]
            L_action = 0
            if action_q is not None:
                q_parts.append(action_q); k_parts.append(action_k)
                v_parts.append(action_v); L_action = action_q.size(1)
            L_und = 0
            if und_q is not None:
                q_parts.append(und_q); k_parts.append(und_k)
                v_parts.append(und_v); L_und = und_q.size(1)

            use_sage2_joint_f16 = (
                os.environ.get('FLASH_RT_MOTUS_USE_SAGE2_JOINT_F16', '0') == '1'
                and self.window_size == (-1, -1)
                and q_parts[0].dtype == torch.bfloat16
                and q_parts[0].shape[-1] == 128
                and len(q_parts) == 3
                and hasattr(fvk, 'concat3_qk_int8_v_fp16_d128')
                and hasattr(fvk, 'sage2_qk_int8_sv_f16_bf16_nhd_d128'))

            use_sage2_joint = (not use_sage2_joint_f16) and (
                os.environ.get('FLASH_RT_MOTUS_USE_SAGE2_JOINT', '0') == '1'
                and self.window_size == (-1, -1)
                and q_parts[0].dtype == torch.bfloat16
                and q_parts[0].shape[-1] == 128
                and len(q_parts) == 3
                and hasattr(fvk, 'concat3_qk_int8_v_fp8_d128')
                and hasattr(fvk, 'sage2_qk_int8_sv_f8_bf16_nhd_d128'))

            if (not use_sage2_joint
                    and not use_sage2_joint_f16
                    and len(q_parts) == 3
                    and q_parts[0].dtype == torch.bfloat16
                    and hasattr(fvk, 'concat3_qkv_bf16')
                    and os.environ.get('FLASH_RT_MOTUS_NO_G7_26_CAT3',
                                       '0') != '1'):
                _e = _jp_start()
                q_cat, k_cat, v_cat = _concat3_qkv_bf16(
                    q_parts[0], q_parts[1], q_parts[2],
                    k_parts[0], k_parts[1], k_parts[2],
                    v_parts[0], v_parts[1], v_parts[2])
                _jp_end('joint_concat3_qkv', _e)
            else:
                q_cat = k_cat = v_cat = None
                if (not use_sage2_joint) and (not use_sage2_joint_f16):
                    _e = _jp_start()
                    q_cat = torch.cat(q_parts, dim=1)
                    k_cat = torch.cat(k_parts, dim=1)
                    v_cat = torch.cat(v_parts, dim=1)
                    _jp_end('joint_concat3_qkv', _e)

            _e = _jp_start()
            if use_sage2_joint_f16:
                total_L = int(L_x + L_action + L_und)
                (sage_q8, sage_k8, sage_v16, sage_qs, sage_ks,
                 sage_out) = _sage_joint_f16_buffers(
                     q_parts[0].device, b, total_L, n, d)
                _eq = _jp_start()
                fvk.concat3_qk_int8_v_fp16_d128(
                    int(q_parts[0].data_ptr()), int(q_parts[1].data_ptr()), int(q_parts[2].data_ptr()),
                    int(k_parts[0].data_ptr()), int(k_parts[1].data_ptr()), int(k_parts[2].data_ptr()),
                    int(v_parts[0].data_ptr()), int(v_parts[1].data_ptr()), int(v_parts[2].data_ptr()),
                    int(sage_q8.data_ptr()),
                    int(sage_k8.data_ptr()),
                    int(sage_v16.data_ptr()),
                    int(sage_qs.data_ptr()),
                    int(sage_ks.data_ptr()),
                    int(b), int(L_x), int(L_action), int(L_und), int(n),
                    int(q_parts[0].stride(0)), int(q_parts[0].stride(1)),
                    int(q_parts[1].stride(0)), int(q_parts[1].stride(1)),
                    int(q_parts[2].stride(0)), int(q_parts[2].stride(1)),
                    int(k_parts[0].stride(0)), int(k_parts[0].stride(1)),
                    int(k_parts[1].stride(0)), int(k_parts[1].stride(1)),
                    int(k_parts[2].stride(0)), int(k_parts[2].stride(1)),
                    int(v_parts[0].stride(0)), int(v_parts[0].stride(1)),
                    int(v_parts[1].stride(0)), int(v_parts[1].stride(1)),
                    int(v_parts[2].stride(0)), int(v_parts[2].stride(1)),
                    cs())
                _jp_end('joint_sage2_f16_quant_concat', _eq)
                rc = fvk.sage2_qk_int8_sv_f16_bf16_nhd_d128(
                    int(sage_q8.data_ptr()),
                    int(sage_k8.data_ptr()),
                    int(sage_v16.data_ptr()),
                    int(sage_out.data_ptr()),
                    int(sage_qs.data_ptr()),
                    int(sage_ks.data_ptr()),
                    int(b), int(total_L), int(total_L), int(n),
                    float(d ** -0.5), cs())
                if rc != 0:
                    raise RuntimeError(f'[g7.sage2.joint_f16] raw attention rc={rc}')
                attn_out = sage_out
                _jp_end('joint_sage2_f16', _e)
            elif use_sage2_joint:
                total_L = int(L_x + L_action + L_und)
                (sage_q8, sage_k8, sage_v8, sage_qs, sage_ks, sage_vs,
                 sage_out) = _sage_joint_buffers(
                     q_parts[0].device, b, total_L, n, d)
                _eq = _jp_start()
                fvk.concat3_qk_int8_v_fp8_d128(
                    int(q_parts[0].data_ptr()), int(q_parts[1].data_ptr()), int(q_parts[2].data_ptr()),
                    int(k_parts[0].data_ptr()), int(k_parts[1].data_ptr()), int(k_parts[2].data_ptr()),
                    int(v_parts[0].data_ptr()), int(v_parts[1].data_ptr()), int(v_parts[2].data_ptr()),
                    int(sage_q8.data_ptr()),
                    int(sage_k8.data_ptr()),
                    int(sage_v8.data_ptr()),
                    int(sage_qs.data_ptr()),
                    int(sage_ks.data_ptr()),
                    int(sage_vs.data_ptr()),
                    int(b), int(L_x), int(L_action), int(L_und), int(n),
                    int(q_parts[0].stride(0)), int(q_parts[0].stride(1)),
                    int(q_parts[1].stride(0)), int(q_parts[1].stride(1)),
                    int(q_parts[2].stride(0)), int(q_parts[2].stride(1)),
                    int(k_parts[0].stride(0)), int(k_parts[0].stride(1)),
                    int(k_parts[1].stride(0)), int(k_parts[1].stride(1)),
                    int(k_parts[2].stride(0)), int(k_parts[2].stride(1)),
                    int(v_parts[0].stride(0)), int(v_parts[0].stride(1)),
                    int(v_parts[1].stride(0)), int(v_parts[1].stride(1)),
                    int(v_parts[2].stride(0)), int(v_parts[2].stride(1)),
                    cs())
                _jp_end('joint_sage2_quant_concat', _eq)
                rc = fvk.sage2_qk_int8_sv_f8_bf16_nhd_d128(
                    int(sage_q8.data_ptr()),
                    int(sage_k8.data_ptr()),
                    int(sage_v8.data_ptr()),
                    int(sage_out.data_ptr()),
                    int(sage_qs.data_ptr()),
                    int(sage_ks.data_ptr()),
                    int(sage_vs.data_ptr()),
                    int(b), int(total_L), int(total_L), int(n),
                    float(d ** -0.5), cs())
                if rc != 0:
                    raise RuntimeError(f'[g7.sage2.joint] raw attention rc={rc}')
                attn_out = sage_out
                _jp_end('joint_sage2', _e)
            else:
                attn_out = flash_attention(
                    q=q_cat, k=k_cat, v=v_cat,
                    k_lens=seq_lens, window_size=self.window_size)
                _jp_end('joint_fa2', _e)

            x_out = attn_out[:, :L_x, :, :]
            outputs = [x_out]
            start_idx = L_x
            if action_q is not None:
                outputs.append(
                    attn_out[:, start_idx:start_idx + L_action, :, :])
                start_idx += L_action
            else:
                outputs.append(None)
            if und_q is not None:
                outputs.append(
                    attn_out[:, start_idx:start_idx + L_und, :, :])
            else:
                outputs.append(None)

            _e = _jp_start()
            x_out = self.o(outputs[0].flatten(2))
            _jp_end('video_o_proj', _e)
            return x_out, outputs[1], outputs[2]

        # Single-modal branch (no MoT) — keep upstream's RoPE+FA path.
        _e = _jp_start()
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        attn_out = flash_attention(
            q=q, k=k, v=v, k_lens=seq_lens,
            window_size=self.window_size)
        x_out = self.o(attn_out.flatten(2))
        _jp_end('single_modal_attn_o', _e)
        return x_out

    fused_forward._g7_8_patched = True
    SaCls.forward = fused_forward
    logger.info(
        f'[g7.8] patched {SaCls.__name__}.forward; '
        f'fused {counts["fused"]} blocks')
    return counts


def install_wan_qkv_nvfp4(model) -> dict:
    """Enable the NVFP4 runtime path for the already-fused video QKV sites.

    Must run after G4 calibration so optional AWQ activation amax has been
    collected by _make_fused_qkv_forward, and before CUDA Graph capture.
    """
    counts = {'installed': 0, 'o_installed': 0, 'skipped': 0, 'reasons': {}}
    if os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV', '0') != '1':
        return counts
    try:
        wan_blocks = model.video_module.video_model.wan_model.blocks
    except AttributeError:
        wan_blocks = model.video_model.wan_model.blocks
    alpha = float(os.environ.get('FLASH_RT_MOTUS_NVFP4_VIDEO_QKV_AWQ_ALPHA',
                                 '0.5'))
    use_awq = os.environ.get('FLASH_RT_MOTUS_NVFP4_VIDEO_QKV_AWQ', '1') == '1'
    for blk in wan_blocks:
        site = getattr(blk.self_attn, '_fused_qkv_site', None)
        if site is None:
            counts['skipped'] += 1
            counts['reasons']['missing_fused_qkv_site'] = (
                counts['reasons'].get('missing_fused_qkv_site', 0) + 1)
            continue
        if site.nvfp4_w_packed is None or site.nvfp4_w_sf is None:
            counts['skipped'] += 1
            counts['reasons']['missing_nvfp4_weight'] = (
                counts['reasons'].get('missing_nvfp4_weight', 0) + 1)
            continue
        if use_awq:
            from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                _smoothquant_nvfp4_from_fp8_site)
            packed, sf, inv_s = _smoothquant_nvfp4_from_fp8_site(site, alpha)
            if packed is not None:
                site.nvfp4_w_packed = packed
                site.nvfp4_w_sf = sf
                site.nvfp4_inv_s = inv_s
        site.nvfp4_ready = True
        if (os.environ.get('FLASH_RT_MOTUS_NO_NVFP4_VIDEO_O', '0') != '1'
                and os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_VIDEO_O', '1') == '1'):
            o_site = getattr(blk.self_attn.o, '_fp8_site', None)
            if o_site is None:
                counts['reasons']['missing_video_o_fp8_site'] = (
                    counts['reasons'].get('missing_video_o_fp8_site', 0) + 1)
            elif _install_video_o_nvfp4_forward(blk.self_attn.o, o_site):
                counts['o_installed'] += 1
        if os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1':
            try:
                empty_fused = torch.empty(0, dtype=site.w_fp8.dtype,
                                          device=site.w_fp8.device)
                site.w_fp8 = empty_fused
                for mod in (blk.self_attn.q, blk.self_attn.k, blk.self_attn.v):
                    s = getattr(mod, '_fp8_site', None)
                    if s is not None and getattr(s, 'w_fp8', None) is not None:
                        empty = torch.empty(0, dtype=s.w_fp8.dtype,
                                            device=s.w_fp8.device)
                        mod.weight.data = empty
                        s.w_fp8 = empty
            except Exception:
                pass
        counts['installed'] += 1
    logger.info('[g7.8.nvfp4] enabled video QKV NVFP4 on %d layers',
                counts['installed'])
    return counts


def _install_video_o_nvfp4_forward(module, site) -> bool:
    """Optional NVFP4 W4A16 path for Wan video self-attn O projection.

    Existing FP8 behavior is left untouched unless
    FLASH_RT_MOTUS_USE_NVFP4_VIDEO_O=1. Weight is reconstructed from the
    calibrated FP8 site and quantized once; activation is quantized per call.
    """
    if getattr(module, '_nvfp4_video_o_ready', False):
        return False
    if site.K % 16 != 0 or site.N % 16 != 0:
        return False
    from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
        _quantize_act_to_nvfp4,
        _swizzled_sf_bytes,
        quantize_weight_bf16_to_nvfp4_swz,
    )
    K, N = int(site.K), int(site.N)
    w_bf16_KN = (site.w_fp8.to(torch.bfloat16)
                 * float(site.w_scale.item())).contiguous()
    w_packed, w_sf = quantize_weight_bf16_to_nvfp4_swz(
        w_bf16_KN.t().contiguous())
    site.nvfp4_w_packed = w_packed
    site.nvfp4_w_sf = w_sf
    site.nvfp4_inv_s = None
    bias_ptr = int(site.bias.data_ptr()) if site.has_bias else 0
    w_p_ptr = int(w_packed.data_ptr())
    w_sf_ptr = int(w_sf.data_ptr())

    def forward(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K)
        M = int(flat.shape[0])
        dev = flat.device
        if site.nvfp4_in_packed is None or site.nvfp4_in_packed.shape[0] < M:
            site.nvfp4_in_packed = torch.empty(
                M, K // 2, dtype=torch.uint8, device=dev)
            site.nvfp4_in_sf = torch.zeros(
                _swizzled_sf_bytes(M, K), dtype=torch.uint8, device=dev)
            site.nvfp4_out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        x_p = site.nvfp4_in_packed[:M]
        x_sf = site.nvfp4_in_sf
        out = site.nvfp4_out[:M]
        _quantize_act_to_nvfp4(flat, None, x_p, x_sf, M, K)
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            int(x_p.data_ptr()), w_p_ptr, int(out.data_ptr()),
            M, N, K, int(x_sf.data_ptr()), w_sf_ptr, 1.0, cs())
        if bias_ptr and not site.bias_skip:
            fvk.add_bias_bf16(int(out.data_ptr()), bias_ptr, M, N, cs())
        if in_dtype != torch.bfloat16:
            out = out.to(in_dtype)
        return out.view(*in_shape[:-1], N)

    module.forward = forward
    module._nvfp4_video_o_ready = True
    if os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1':
        try:
            empty = torch.empty(0, dtype=site.w_fp8.dtype,
                                device=site.w_fp8.device)
            module.weight.data = empty
            site.w_fp8 = empty
        except Exception:
            pass
    return True
