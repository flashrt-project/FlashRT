"""Install Pi0.5 action-expert FFN megakernel V6tuned (ku256_sd4_su3).

Replaces ``action_module.process_ffn`` with a closure that fuses the
pre-FFN AdaLN+modulate -> FP8 GEMM_up -> bias -> GELU -> intermediate
FP8 quant -> FP8 GEMM_dn -> bias -> gate * acc -> residual_add chain
into one megakernel launch (plus the standalone ada_layer_norm_fp8
call that quantises the action tokens).

Shape lock matches the action-expert FFN hidden dimensions in the released
motus checkpoints: K_up=1024, N_up=4096, K_dn=4096, N_dn=1024. The CUDA
tile handles up to 32 token rows by dispatching 16-row M tiles; larger
finetune action layouts fall back to the original process_ffn path.

Env-disable: ``FLASH_RT_MOTUS_USE_ACTION_FFN_V6T=0``.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def _cs() -> int:
    return torch.cuda.current_stream().cuda_stream


class _State:
    __slots__ = (
        'up_w_NK', 'up_inv_s', 'up_bias', 'up_alpha',
        'dn_w_NK', 'dn_inv_s', 'dn_bias', 'dn_alpha', 'dn_act_scale',
        'M_capacity', 'K_up', 'N_up', 'K_dn', 'N_dn',
        'x_fp8_scr', 'up_fp8_scr', 'y_out_buf',
        'block', 'act_scale_dev',
    )


def install(model: Any, M_capacity: int = 32) -> int:
    """Install on model.action_expert; returns number of FFN layers patched.

    No-op if FLASH_RT_MOTUS_USE_ACTION_FFN_V6T=0, if the action expert is
    absent, or if the required FFN sites are not present.
    """
    if os.environ.get('FLASH_RT_MOTUS_USE_ACTION_FFN_V6T', '1') == '0':
        return 0
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception:
        return 0
    if not hasattr(fvk, 'action_ffn_v6t_launch_sm120'):
        return 0

    expert = getattr(model, 'action_expert', None)
    action_module = getattr(model, 'action_module', None)
    if expert is None or action_module is None:
        return 0
    blocks = getattr(expert, 'blocks', None)
    if blocks is None:
        return 0

    states_by_layer: dict[int, _State] = {}
    for layer_idx, blk in enumerate(blocks):
        ffn = getattr(blk, 'ffn', None)
        if ffn is None:
            continue
        up_site = getattr(ffn, '_awq_up_site', None)
        dn_site = getattr(ffn, '_awq_dn_site', None)
        if up_site is None or dn_site is None:
            continue
        K_up, N_up = up_site.K, up_site.N
        K_dn, N_dn = dn_site.K, dn_site.N
        if (K_up, N_up, K_dn, N_dn) != (1024, 4096, 4096, 1024):
            continue
        st = _State()
        st.K_up, st.N_up, st.K_dn, st.N_dn = K_up, N_up, K_dn, N_dn
        st.M_capacity = M_capacity
        dev = up_site.w_fp8.device
        # Kernel expects weights in (N, K) layout. motus stores w_fp8 in
        # (K, N) (post-G3b transpose); pre-transpose once per layer.
        st.up_w_NK = up_site.w_fp8.t().contiguous()
        st.dn_w_NK = dn_site.w_fp8.t().contiguous()
        st.up_inv_s = up_site.inv_s
        st.dn_inv_s = dn_site.inv_s
        st.up_bias = (up_site.bias if up_site.bias is not None
                      else torch.zeros(N_up, dtype=torch.bfloat16, device=dev))
        st.dn_bias = (dn_site.bias if dn_site.bias is not None
                      else torch.zeros(N_dn, dtype=torch.bfloat16, device=dev))
        up_act_scale = float(up_site.act_scale.flatten()[0].item())
        up_w_scale = float(up_site.w_scale.flatten()[0].item())
        dn_act_scale = float(dn_site.act_scale.flatten()[0].item())
        dn_w_scale = float(dn_site.w_scale.flatten()[0].item())
        st.up_alpha = up_act_scale * up_w_scale
        st.dn_alpha = dn_act_scale * dn_w_scale
        st.dn_act_scale = dn_act_scale
        st.act_scale_dev = up_site.act_scale
        st.x_fp8_scr = torch.zeros(M_capacity, K_up,
                                    dtype=torch.float8_e4m3fn, device=dev)
        st.up_fp8_scr = torch.zeros(M_capacity, K_dn,
                                     dtype=torch.float8_e4m3fn, device=dev)
        st.y_out_buf = torch.zeros(M_capacity, N_dn,
                                    dtype=torch.bfloat16, device=dev)
        st.block = blk
        states_by_layer[layer_idx] = st
    if not states_by_layer:
        return 0

    orig_process_ffn = action_module.process_ffn
    EPS = 1e-6

    def process_ffn_v6t(action_tokens, action_adaln_modulation, layer_idx):
        st = states_by_layer.get(layer_idx)
        if st is None:
            return orig_process_ffn(action_tokens, action_adaln_modulation,
                                    layer_idx)
        a_mod = action_adaln_modulation
        eps = (float(st.block.norm2.eps)
               if hasattr(st.block.norm2, 'eps') else EPS)
        in_shape = action_tokens.shape
        K_up = st.K_up
        N_dn = st.N_dn
        action_c = (action_tokens if action_tokens.is_contiguous()
                    else action_tokens.contiguous())
        M = action_c.shape[0] * action_c.shape[1]
        if M > st.M_capacity:
            return orig_process_ffn(action_tokens, action_adaln_modulation,
                                    layer_idx)
        scale_b = a_mod[4].squeeze(2).to(torch.bfloat16).contiguous()
        shift_b = a_mod[3].squeeze(2).to(torch.bfloat16).contiguous()
        if scale_b.dim() == 3:
            scale_b = scale_b.reshape(M, K_up).contiguous()
            shift_b = shift_b.reshape(M, K_up).contiguous()
        fvk.awq_ada_layer_norm_fp8(
            int(action_c.data_ptr()),
            int(scale_b.data_ptr()), int(shift_b.data_ptr()),
            int(st.up_inv_s.data_ptr()),
            int(st.x_fp8_scr.data_ptr()),
            int(st.act_scale_dev.data_ptr()),
            M, K_up, eps, _cs())
        gate = a_mod[5].squeeze(2).to(torch.bfloat16).contiguous()
        gate_flat = gate.reshape(-1, N_dn)
        if action_tokens.dtype != torch.bfloat16:
            residual_bf16 = action_tokens.to(torch.bfloat16).contiguous()
        else:
            residual_bf16 = action_c
        residual_flat = residual_bf16.reshape(-1, N_dn)
        rc = fvk.action_ffn_v6t_launch_sm120(
            int(st.x_fp8_scr.data_ptr()),
            int(st.up_w_NK.data_ptr()), int(st.up_bias.data_ptr()),
            int(st.dn_inv_s.data_ptr()), int(st.dn_w_NK.data_ptr()),
            int(st.dn_bias.data_ptr()),
            int(gate_flat.data_ptr()), int(residual_flat.data_ptr()),
            int(st.y_out_buf.data_ptr()),
            int(st.up_fp8_scr.data_ptr()),
            M, K_up, st.N_up, st.K_dn, N_dn,
            st.up_alpha, st.dn_alpha, st.dn_act_scale, _cs())
        if rc != 0:
            raise RuntimeError(f'action_ffn_v6t_launch rc={rc}')
        y = st.y_out_buf[:M]
        return y.view(*in_shape[:-1], N_dn)

    action_module.process_ffn = process_ffn_v6t
    return len(states_by_layer)
