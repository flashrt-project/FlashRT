"""G6.3 — Wan cross-attention T5 K/V cache.

Wan video↔T5 cross-attention recomputes K/V on the (fixed-after-
set_prompt) T5 context every layer every step:

    30 layers × 10 steps × 2 (k+v) = 600 redundant FP8 GEMMs per call

Plus 300 redundant ``norm_k`` ops. From the G6.2 profile each FP8
GEMM at this shape ≈ 35 µs, so the whole cluster is ≈ 22-30 ms of
graph time that does not depend on the per-step input.

Fix: at ``set_prompt`` time (after the FP8 calibration pass has
written valid act_scales for self.k / self.v), run
``norm_k(self.k(t5_ctx))`` and ``self.v(t5_ctx)`` once per layer and
store the results in pointer-stable buffers attached to each
WanCrossAttention. The patched ``forward`` then skips k_proj / v_proj
/ norm_k entirely and feeds the cached tensors straight to FA2.

Pointer stability matters because the captured CUDA Graph reads from
these buffers on every replay. Allocating them ONCE and ``.copy_``-ing
in new contents on each set_prompt keeps capture valid across
re-prompts (combined with the existing graph drop in motus_rtx.py).

Lifecycle:
    1. install_wan_t5_kv_cache(model)
        — allocate buffers, swap each module.forward
    2. populate_wan_t5_kv_cache(model, t5_ctx)
        — run k/v projections + norm_k once, copy_ into buffers
    3. (graph capture as usual)
    4. on set_prompt with new t5: repopulate (step 2) + drop graph

The original (q, o) projections and ``norm_q`` stay live — only k/v
are cached. ``context_lens`` is honoured (we pass it through to FA2).
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import torch

from flash_rt.models.motus._stream import cs

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)
_CROSS_PROFILE = os.environ.get('FLASH_RT_MOTUS_CROSS_PROFILE', '0') == '1'
_CROSS_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
_USE_CROSS_O_BIAS_RESIDUAL = (
    os.environ.get('FLASH_RT_MOTUS_NO_G7_33_CROSS_O_BIAS_RES', '0') != '1')
_USE_CROSS_Q_BIAS_RMS = (
    os.environ.get('FLASH_RT_MOTUS_NO_G7_34_CROSS_Q_BIAS_RMS', '0') != '1')
_USE_NVFP4_CROSS_Q = (
    os.environ.get('FLASH_RT_MOTUS_NO_NVFP4_CROSS_Q', '0') != '1'
    and os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_CROSS_Q', '1') == '1')
_USE_NVFP4_CROSS_O = (
    os.environ.get('FLASH_RT_MOTUS_NO_NVFP4_CROSS_O', '0') != '1'
    and os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_CROSS_O', '1') == '1')
_USE_SAGE2_CROSS = (
    os.environ.get('FLASH_RT_MOTUS_NO_SAGE2_CROSS', '0') != '1'
    and os.environ.get('FLASH_RT_MOTUS_USE_SAGE2_CROSS', '1') == '1')
_USE_CROSS_NORM3_NVFP4_Q = (
    os.environ.get('FLASH_RT_MOTUS_NO_G7_49_CROSS_NORM3_NVFP4_Q', '0') != '1'
    and os.environ.get('FLASH_RT_MOTUS_USE_G7_49_CROSS_NORM3_NVFP4_Q', '0') == '1')
_SAGE_CROSS_BUF_CACHE: dict[tuple[int, int, int, int, torch.device],
                            tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor]] = {}


def _cp_start():
    if not _CROSS_PROFILE:
        return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _cp_end(name, e0):
    if e0 is None:
        return
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    _CROSS_EVENTS.setdefault(name, []).append((e0, e1))


def reset_cross_profile_events() -> None:
    _CROSS_EVENTS.clear()


def cross_profile_totals() -> dict[str, tuple[int, float]]:
    torch.cuda.synchronize()
    out = {}
    for name, evs in _CROSS_EVENTS.items():
        total = 0.0
        for e0, e1 in evs:
            total += e0.elapsed_time(e1)
        out[name] = (len(evs), total)
    return out


def _add_bf16_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if (hasattr(fvk, 'add_bf16_out')
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


def _bias_residual_out(
    residual: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    res_c = residual if residual.is_contiguous() else residual.contiguous()
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(res_c)
    fvk.bias_residual_out_bf16(
        int(res_c.data_ptr()), int(x_c.data_ptr()), int(bias.data_ptr()),
        int(out.data_ptr()), int(out.numel() // out.shape[-1]),
        int(out.shape[-1]), cs())
    return out


def _wan_attn_module():
    """Resolve the active wan.modules.attention.flash_attention symbol.

    G3c (install_fa2_attention) monkey-patched the module-level symbol.
    Looking it up at every call ensures the patched function (FA2) is
    used, not whatever the closure captured at install time.
    """
    import wan.modules.attention as _wan_attn  # noqa: WPS433
    return _wan_attn


def _sage_cross_buffers(device: torch.device, b: int, Lq: int,
                        n_heads: int, head_dim: int
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (b, Lq, n_heads, head_dim, device)
    bufs = _SAGE_CROSS_BUF_CACHE.get(key)
    if bufs is None:
        q8 = torch.empty(
            b, Lq, n_heads, head_dim, dtype=torch.int8, device=device)
        q_scale = torch.empty(
            b, n_heads, (Lq + 31) // 32,
            dtype=torch.float32, device=device)
        out = torch.empty(
            b, Lq, n_heads, head_dim,
            dtype=torch.bfloat16, device=device)
        bufs = (q8, q_scale, out)
        _SAGE_CROSS_BUF_CACHE[key] = bufs
    return bufs


def _make_cached_forward(module):
    """Build a cross-attention forward that reads K/V from the cache."""
    n_heads = module.num_heads
    head_dim = module.head_dim
    norm_q = module.norm_q
    q_proj = module.q
    q_site = getattr(q_proj, '_fp8_site', None)
    o_proj = module.o
    k_cache = module._t5_k_cache
    v_cache = module._t5_v_cache
    wan_attn = _wan_attn_module()
    use_sage2 = (
        _USE_SAGE2_CROSS
        and hasattr(fvk, 'sage2_qk_int8_sv_f8_bf16_nhd_d128')
        and hasattr(fvk, 'quant_per_warp_int8_bf16_d128')
        and hasattr(module, '_t5_k_int8_cache')
        and hasattr(module, '_t5_v_fp8_cache'))

    def forward(x, context, context_lens):  # context is ignored (cached)
        b = x.size(0)
        e = _cp_start()
        q_lin = q_proj(x)
        if (q_site is not None and q_site.bias_skip and q_site.has_bias
                and hasattr(fvk, 'bias_rms_norm_bf16')):
            q_lin_c = q_lin if q_lin.is_contiguous() else q_lin.contiguous()
            flat = q_lin_c.reshape(-1, n_heads * head_dim)
            q_norm = torch.empty_like(flat)
            fvk.bias_rms_norm_bf16(
                int(flat.data_ptr()), int(q_site.bias.data_ptr()),
                int(norm_q.weight.data_ptr()), int(q_norm.data_ptr()),
                int(flat.shape[0]), int(flat.shape[1]), float(norm_q.eps),
                cs())
            q = q_norm.view(b, -1, n_heads, head_dim)
        else:
            q = norm_q(q_lin).view(b, -1, n_heads, head_dim)
        _cp_end('cross_q_proj_norm', e)
        e = _cp_start()
        if use_sage2 and context_lens is None:
            Lq = int(q.shape[1])
            q_c = q if q.is_contiguous() else q.contiguous()
            sage_q8, sage_q_scale, sage_out = _sage_cross_buffers(
                q_c.device, b, Lq, n_heads, head_dim)
            e_q = _cp_start()
            fvk.quant_per_warp_int8_bf16_d128(
                int(q_c.data_ptr()), int(sage_q8.data_ptr()),
                int(sage_q_scale.data_ptr()),
                int(b), int(Lq), int(n_heads), cs())
            _cp_end('cross_sage_q_quant', e_q)
            rc = fvk.sage2_qk_int8_sv_f8_bf16_nhd_d128(
                int(sage_q8.data_ptr()),
                int(module._t5_k_int8_cache.data_ptr()),
                int(module._t5_v_fp8_cache.data_ptr()),
                int(sage_out.data_ptr()),
                int(sage_q_scale.data_ptr()),
                int(module._t5_k_scale.data_ptr()),
                int(module._t5_v_scale.data_ptr()),
                int(b), int(Lq), int(module._t5_k_int8_cache.shape[1]),
                int(n_heads), float(head_dim ** -0.5), cs())
            if rc != 0:
                raise RuntimeError(f'[g7.sage2.cross] raw attention rc={rc}')
            out = sage_out
        else:
            out = wan_attn.flash_attention(q, k_cache, v_cache,
                                           k_lens=context_lens)
        _cp_end('cross_fa2', e)
        e = _cp_start()
        out = out.flatten(2)
        out = o_proj(out)
        _cp_end('cross_o_proj', e)
        return out

    return forward


def _install_cross_residual_fusion(model) -> int:
    if (not _USE_CROSS_O_BIAS_RESIDUAL
            or not hasattr(fvk, 'bias_residual_out_bf16')):
        return 0
    video_module = getattr(model, 'video_module', None)
    if video_module is None or getattr(video_module, '_g733_cross_residual', False):
        return 0

    n_skip = 0
    for block in video_module.video_model.wan_model.blocks:
        site = getattr(block.cross_attn.o, '_fp8_site', None)
        if site is not None and site.has_bias:
            site.bias_skip = True
            n_skip += 1

    orig = video_module.process_cross_attention

    def fused_process_cross_attention(
        self,
        video_tokens: torch.Tensor,
        video_adaln_params: torch.Tensor,
        layer_idx: int,
        processed_t5_context: torch.Tensor,
    ) -> torch.Tensor:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None
        cross_in = None
        if (_USE_CROSS_NORM3_NVFP4_Q
                and hasattr(fvk, 'layer_norm_to_nvfp4_swizzled_bf16')
                and getattr(wan_layer.cross_attn.q,
                            '_nvfp4_cross_q_ready', False)):
            q_site = getattr(wan_layer.cross_attn.q, '_fp8_site', None)
            norm3 = wan_layer.norm3
            if (q_site is not None
                    and getattr(q_site, 'nvfp4_w_packed', None) is not None
                    and getattr(norm3, 'weight', None) is not None
                    and getattr(norm3, 'bias', None) is not None):
                from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                    _swizzled_sf_bytes)
                x_c = (video_tokens if video_tokens.is_contiguous()
                       else video_tokens.contiguous())
                B, L, C = x_c.shape
                M = int(B * L)
                if (getattr(q_site, 'nvfp4_in_packed', None) is None
                        or q_site.nvfp4_in_packed.shape[0] < M):
                    q_site.nvfp4_in_packed = torch.empty(
                        M, C // 2, dtype=torch.uint8, device=x_c.device)
                    q_site.nvfp4_in_sf = torch.zeros(
                        _swizzled_sf_bytes(M, C), dtype=torch.uint8,
                        device=x_c.device)
                    q_site.nvfp4_out = torch.empty(
                        M, q_site.N, dtype=torch.bfloat16, device=x_c.device)
                fvk.layer_norm_to_nvfp4_swizzled_bf16(
                    int(x_c.data_ptr()), int(norm3.weight.data_ptr()),
                    int(norm3.bias.data_ptr()),
                    int(q_site.nvfp4_in_packed.data_ptr()),
                    int(q_site.nvfp4_in_sf.data_ptr()),
                    M, C, float(norm3.eps), cs())
                wan_layer.cross_attn.q._nvfp4_prefilled_rows = M
                cross_in = x_c
        if cross_in is None:
            cross_in = wan_layer.norm3(video_tokens)
        cross_out = wan_layer.cross_attn(
            cross_in, processed_t5_context, context_lens)
        site = getattr(wan_layer.cross_attn.o, '_fp8_site', None)
        if site is not None and site.bias_skip and site.has_bias:
            return _bias_residual_out(video_tokens, cross_out, site.bias)
        return _add_bf16_out(video_tokens, cross_out)

    video_module.process_cross_attention = (
        fused_process_cross_attention.__get__(video_module))
    video_module._g733_cross_residual = True
    video_module._g733_cross_residual_orig = orig
    logger.info(
        f"[g7.33] installed cross-attn o bias+residual fusion; "
        f"bias_skip={n_skip}")
    return n_skip


def _install_cross_q_bias_rms(model) -> int:
    if (not _USE_CROSS_Q_BIAS_RMS
            or not hasattr(fvk, 'bias_rms_norm_bf16')):
        return 0
    video_module = getattr(model, 'video_module', None)
    if video_module is None:
        return 0
    n_skip = 0
    for block in video_module.video_model.wan_model.blocks:
        site = getattr(block.cross_attn.q, '_fp8_site', None)
        if site is not None and site.has_bias:
            site.bias_skip = True
            n_skip += 1
    logger.info(
        f"[g7.34] enabled cross-attn q bias+rms fusion; bias_skip={n_skip}")
    return n_skip


def _install_cross_linear_nvfp4_forward(module, site, label: str) -> bool:
    """Optional NVFP4 W4A16 path for Wan cross-attn Q/O projections.

    The original FP8 site remains the default. This forward is installed only
    behind explicit env gates and preserves ``site.bias_skip`` so the existing
    q bias+RMS and o bias+residual fusions still own those epilogues.
    """
    attr = f'_nvfp4_cross_{label}_ready'
    if getattr(module, attr, False):
        return False
    if site is None or site.K % 16 != 0 or site.N % 16 != 0:
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
            site.nvfp4_out = torch.empty(M, N, dtype=torch.bfloat16,
                                         device=dev)
        x_p = site.nvfp4_in_packed[:M]
        x_sf = site.nvfp4_in_sf
        out = site.nvfp4_out[:M]
        if getattr(module, '_nvfp4_prefilled_rows', 0) == M:
            module._nvfp4_prefilled_rows = 0
        else:
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
    setattr(module, attr, True)
    if os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1':
        try:
            empty = torch.empty(0, dtype=site.w_fp8.dtype,
                                device=site.w_fp8.device)
            module.weight.data = empty
            site.w_fp8 = empty
        except Exception:
            pass
    return True


def _install_cross_nvfp4(model) -> dict:
    counts = {'q_installed': 0, 'o_installed': 0, 'skipped': 0,
              'reasons': {}}
    if not (_USE_NVFP4_CROSS_Q or _USE_NVFP4_CROSS_O):
        return counts
    video_module = getattr(model, 'video_module', None)
    if video_module is None:
        counts['skipped'] += 1
        counts['reasons']['missing_video_module'] = 1
        return counts
    for block in video_module.video_model.wan_model.blocks:
        if _USE_NVFP4_CROSS_Q:
            q_site = getattr(block.cross_attn.q, '_fp8_site', None)
            if _install_cross_linear_nvfp4_forward(
                    block.cross_attn.q, q_site, 'q'):
                counts['q_installed'] += 1
            else:
                counts['reasons']['cross_q'] = (
                    counts['reasons'].get('cross_q', 0) + 1)
        if _USE_NVFP4_CROSS_O:
            o_site = getattr(block.cross_attn.o, '_fp8_site', None)
            if _install_cross_linear_nvfp4_forward(
                    block.cross_attn.o, o_site, 'o'):
                counts['o_installed'] += 1
            else:
                counts['reasons']['cross_o'] = (
                    counts['reasons'].get('cross_o', 0) + 1)
    logger.info('[g7.35.nvfp4] cross-attn NVFP4 q=%d o=%d',
                counts['q_installed'], counts['o_installed'])
    return counts


def install_wan_t5_kv_cache(
    model,
    dtype: torch.dtype = torch.bfloat16,
    t5_seq_len: int = 512,
    batch: int = 1,
) -> List[Tuple[str, object]]:
    """Allocate persistent K/V buffers and swap forward on every
    WanCrossAttention. Buffers are zero-initialized; a populate pass
    must run before any replay reads them.
    """
    handles: List[Tuple[str, object]] = []
    for name, module in model.named_modules():
        if type(module).__name__ != 'WanCrossAttention':
            continue
        n_heads = int(module.num_heads)
        head_dim = int(module.head_dim)
        device = next(module.parameters()).device
        k_buf = torch.zeros(
            batch, t5_seq_len, n_heads, head_dim,
            dtype=dtype, device=device).contiguous()
        v_buf = torch.zeros_like(k_buf)
        module._t5_k_cache = k_buf
        module._t5_v_cache = v_buf
        if _USE_SAGE2_CROSS:
            padded_t5 = ((t5_seq_len + 63) // 64) * 64
            module._t5_k_int8_cache = torch.empty_like(k_buf, dtype=torch.int8)
            module._t5_k_scale = torch.empty(
                batch, n_heads, (t5_seq_len + 63) // 64,
                dtype=torch.float32, device=device)
            module._t5_v_tpp = torch.empty(
                batch, head_dim, n_heads, padded_t5,
                dtype=dtype, device=device)
            module._t5_v_fp8_cache = torch.empty(
                batch, head_dim, n_heads, padded_t5,
                dtype=torch.float8_e4m3fn, device=device)
            module._t5_v_scale = torch.empty(
                batch, n_heads, head_dim, dtype=torch.float32, device=device)
        module.forward = _make_cached_forward(module)
        handles.append((name, module))
    _install_cross_q_bias_rms(model)
    _install_cross_residual_fusion(model)
    _install_cross_nvfp4(model)
    logger.info(
        f"[g6.3] installed T5 K/V cache on {len(handles)} cross-attn "
        f"modules; per-buf shape=[{batch},{t5_seq_len},*,*] {dtype}")
    return handles


def populate_wan_t5_kv_cache(model, t5_ctx: torch.Tensor) -> int:
    """Run the original k/v projections + norm_k on the cached T5
    context once, then ``copy_`` results into the per-module buffers
    so the next graph replay reads them.

    ``t5_ctx`` is the output of ``video_module.preprocess_t5_embeddings``,
    shape [B, S_t5, D] in bf16.
    """
    n = 0
    b = t5_ctx.size(0)
    for module in model.modules():
        if type(module).__name__ != 'WanCrossAttention':
            continue
        if not hasattr(module, '_t5_k_cache'):
            continue
        n_heads = int(module.num_heads)
        head_dim = int(module.head_dim)
        with torch.no_grad():
            # module.k / module.v may be FP8-swapped Linears; calling
            # the module invokes the swapped forward.
            k = module.norm_k(module.k(t5_ctx)).view(b, -1, n_heads, head_dim)
            v = module.v(t5_ctx).view(b, -1, n_heads, head_dim)
        # Match dtype of the cache buffer (bf16). norm_k may upcast in
        # weird circumstances; cast defensively.
        if k.dtype != module._t5_k_cache.dtype:
            k = k.to(module._t5_k_cache.dtype)
            v = v.to(module._t5_v_cache.dtype)
        k = k if k.is_contiguous() else k.contiguous()
        v = v if v.is_contiguous() else v.contiguous()
        module._t5_k_cache.copy_(k)
        module._t5_v_cache.copy_(v)
        if _USE_SAGE2_CROSS and hasattr(module, '_t5_k_int8_cache'):
            fvk.quant_per_block_int8_bf16_d128(
                int(k.data_ptr()), int(module._t5_k_int8_cache.data_ptr()),
                int(module._t5_k_scale.data_ptr()),
                int(b), int(k.shape[1]), int(n_heads), cs())
            fvk.concat3_v_transpose_pad_permute_bf16_d128(
                int(v.data_ptr()), int(v.data_ptr()), int(v.data_ptr()),
                int(module._t5_v_tpp.data_ptr()),
                int(b), int(v.shape[1]), int(0), int(0), int(n_heads),
                int(v.stride(0)), int(v.stride(1)),
                int(v.stride(0)), int(v.stride(1)),
                int(v.stride(0)), int(v.stride(1)), cs())
            fvk.v_tpp_bf16_quant_fp8_d128(
                int(module._t5_v_tpp.data_ptr()),
                int(module._t5_v_fp8_cache.data_ptr()),
                int(module._t5_v_scale.data_ptr()),
                int(b), int(v.shape[1]), int(n_heads), cs())
        n += 1
    logger.info(
        f"[g6.3] populated T5 K/V cache for {n} layers, "
        f"t5_ctx shape={tuple(t5_ctx.shape)}")
    return n
