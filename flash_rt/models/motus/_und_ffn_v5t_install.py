"""Install Pi0.5 und-module FFN megakernel V5tuned.

Replaces ``und_module.process_ffn`` with a closure that runs the
3-phase fused FP8 W4A8 FFN (norm pre-staged + GEMM_up + bias + GELU +
intermediate FP8 quant + GEMM_dn + bias + residual_add) in one
megakernel launch.

Shape lock: M ≤ 144 (capacity covers 138 und tokens), K_up=512,
N_up=2048, K_dn=2048, N_dn=512. Sites whose AWQ FP8 state shapes do
not match are skipped and fall back to the original ``process_ffn``.

Env-disable: ``FLASH_RT_MOTUS_USE_UND_FFN_V5T=0``.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def _parse_layer_set(value: str | None):
    if value is None or value.strip() == "":
        return None
    out = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo_s, hi_s = part.split('-', 1)
            lo, hi = int(lo_s), int(hi_s)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _cs() -> int:
    return torch.cuda.current_stream().cuda_stream


class _State:
    __slots__ = (
        'up_w_NK', 'up_inv_s', 'up_bias', 'up_alpha', 'up_act_scale',
        'dn_w_NK', 'dn_inv_s', 'dn_bias', 'dn_alpha', 'dn_act_scale',
        'M_capacity', 'K_up', 'N_up', 'K_dn', 'N_dn',
        'x_fp8_scr', 'up_fp8_scr', 'y_out_buf', 'barrier', 'block',
    )


def install(model: Any, M_capacity: int = 144) -> int:
    """Install on model.und_expert; returns number of FFN layers patched."""
    if os.environ.get('FLASH_RT_MOTUS_USE_UND_FFN_V5T', '1') == '0':
        return 0
    enabled_layers = _parse_layer_set(
        os.environ.get('FLASH_RT_MOTUS_UND_FFN_V5T_LAYERS'))
    disabled_layers = _parse_layer_set(
        os.environ.get('FLASH_RT_MOTUS_UND_FFN_V5T_SKIP_LAYERS'))
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception:
        return 0
    use_split_stage3 = (
        os.environ.get('FLASH_RT_MOTUS_USE_UND_FFN_V5SPLIT_STAGE3', '0')
        == '1' and hasattr(fvk, 'und_ffn_v5split_stage3_launch_sm120'))
    if use_split_stage3:
        M_capacity = max(M_capacity, 192)
    elif not hasattr(fvk, 'und_ffn_v5t_launch_sm120'):
        return 0

    expert = getattr(model, 'und_expert', None)
    und_module = getattr(model, 'und_module', None)
    if expert is None or und_module is None:
        return 0
    blocks = getattr(expert, 'blocks', None)
    if blocks is None:
        return 0

    states_by_layer: dict[int, _State] = {}
    for layer_idx, blk in enumerate(blocks):
        if enabled_layers is not None and layer_idx not in enabled_layers:
            continue
        if disabled_layers is not None and layer_idx in disabled_layers:
            continue
        ffn = getattr(blk, 'ffn', None)
        if ffn is None:
            continue
        up_site = getattr(ffn, '_awq_up_site', None)
        dn_site = getattr(ffn, '_awq_dn_site', None)
        if up_site is None or dn_site is None:
            continue
        K_up, N_up = up_site.K, up_site.N
        K_dn, N_dn = dn_site.K, dn_site.N
        if (K_up, N_up, K_dn, N_dn) != (512, 2048, 2048, 512):
            continue
        st = _State()
        st.K_up, st.N_up, st.K_dn, st.N_dn = K_up, N_up, K_dn, N_dn
        st.M_capacity = M_capacity
        dev = up_site.w_fp8.device
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
        st.up_act_scale = up_act_scale
        st.dn_act_scale = dn_act_scale
        st.x_fp8_scr = torch.zeros(M_capacity, K_up,
                                    dtype=torch.float8_e4m3fn, device=dev)
        st.up_fp8_scr = torch.zeros(M_capacity, K_dn,
                                     dtype=torch.float8_e4m3fn, device=dev)
        st.y_out_buf = torch.zeros(M_capacity, N_dn,
                                    dtype=torch.bfloat16, device=dev)
        st.barrier = torch.zeros(2, dtype=torch.uint32, device=dev)
        st.block = blk
        states_by_layer[layer_idx] = st
    if not states_by_layer:
        return 0

    orig_process_ffn = und_module.process_ffn

    def process_ffn_v5t(und_tokens, layer_idx):
        st = states_by_layer.get(layer_idx)
        if st is None:
            return orig_process_ffn(und_tokens, layer_idx)
        ffn_input = st.block.norm2(und_tokens)
        K_up = st.K_up
        N_dn = st.N_dn
        in_shape = und_tokens.shape
        ffn_input_flat = ffn_input.reshape(-1, K_up).contiguous()
        M = ffn_input_flat.shape[0]
        if M > st.M_capacity:
            raise RuntimeError(f'und_ffn_v5t M={M} > capacity '
                                f'{st.M_capacity}')
        if und_tokens.dtype != torch.bfloat16:
            residual_bf16 = und_tokens.to(torch.bfloat16).contiguous()
        else:
            residual_bf16 = (und_tokens if und_tokens.is_contiguous()
                              else und_tokens.contiguous())
        residual_flat = residual_bf16.reshape(-1, N_dn)
        if use_split_stage3:
            rc = fvk.und_ffn_v5split_stage3_launch_sm120(
                int(ffn_input_flat.data_ptr()), int(st.up_inv_s.data_ptr()),
                int(st.up_w_NK.data_ptr()), int(st.up_bias.data_ptr()),
                int(st.dn_inv_s.data_ptr()), int(st.dn_w_NK.data_ptr()),
                int(st.dn_bias.data_ptr()),
                int(residual_flat.data_ptr()),
                int(st.y_out_buf.data_ptr()),
                int(st.x_fp8_scr.data_ptr()), int(st.up_fp8_scr.data_ptr()),
                M, K_up, st.N_up, st.K_dn, N_dn,
                st.up_alpha, st.dn_alpha, st.up_act_scale, st.dn_act_scale,
                int(st.barrier.data_ptr()), _cs())
        else:
            rc = fvk.und_ffn_v5t_launch_sm120(
                int(ffn_input_flat.data_ptr()), int(st.up_inv_s.data_ptr()),
                int(st.up_w_NK.data_ptr()), int(st.up_bias.data_ptr()),
                int(st.dn_inv_s.data_ptr()), int(st.dn_w_NK.data_ptr()),
                int(st.dn_bias.data_ptr()),
                int(residual_flat.data_ptr()),
                int(st.y_out_buf.data_ptr()),
                int(st.x_fp8_scr.data_ptr()), int(st.up_fp8_scr.data_ptr()),
                M, K_up, st.N_up, st.K_dn, N_dn,
                st.up_alpha, st.dn_alpha, st.up_act_scale, st.dn_act_scale,
                int(st.barrier.data_ptr()), _cs())
        if rc != 0:
            raise RuntimeError(f'und_ffn_v5t_launch rc={rc}')
        y = st.y_out_buf[:M]
        return y.view(*in_shape[:-1], N_dn)

    und_module.process_ffn = process_ffn_v5t
    return len(states_by_layer)
