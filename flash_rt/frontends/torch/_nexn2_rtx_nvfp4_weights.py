"""FlashRT — Nex-N2-mini (qwen3_5_moe) NVFP4 W4A16 raw safetensors loader.

Reads the BF16 ``qwen3_5_moe`` checkpoint and quantizes the large GEMM
weights to NVFP4 on load via ``bf16_weight_to_nvfp4_swizzled`` (no
pre-quantized ckpt required). The 35B-A3B model does not fit a 32 GB
RTX 5090 in BF16, so NVFP4 W4A16 is the baseline weight format; the load
streams shard by shard and frees each BF16 weight right after quantizing
it, keeping peak VRAM well under the BF16 footprint.

Modules kept BF16 (not quantized):
  * embed_tokens, final norm, lm_head
  * every layernorm (input / post_attention / linear_attn.norm / q/k_norm)
  * the GDN ``linear_attn`` path: in_proj_{qkv,z,a,b}, conv1d, A_log,
    dt_bias  (in_proj must stay BF16 — quantizing it collapses GDN)
  * the MoE router (``mlp.gate``) and ``shared_expert_gate``

Quantized to NVFP4 (per-tensor global scale + per-16 block UE4M3 SF,
swizzled at load):
  * full-attn q/k/v/o proj
  * GDN out_proj
  * MoE experts (gate_up + down, per expert) and the shared expert

``bf16_weight_to_nvfp4_swizzled`` emits inverse-scaled SF, so the GEMM
``alpha`` equals the returned per-tensor ``global_scale`` directly (not
its reciprocal — see the qwen3 lm_head load note).

Returns a :class:`WeightHandles` whose ``ptrs`` surface mirrors the
qwen36 NVFP4 loader for unsurprising reuse by the pipeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class WeightHandles:
    ptrs: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)


def _anchor(handles: WeightHandles, t: torch.Tensor) -> int:
    handles.anchors.append(t)
    return int(t.data_ptr())


def _bf16_to_dev(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(torch.bfloat16).to(device).contiguous()


def _open_shards(ckpt_dir: str):
    from safetensors import safe_open

    idx_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    if not os.path.isfile(idx_path):
        raise RuntimeError(
            f'Nex-N2 ckpt missing model.safetensors.index.json: {ckpt_dir!r}')
    wmap = json.load(open(idx_path))['weight_map']
    handles_d = {}
    for shard in set(wmap.values()):
        handles_d[shard] = safe_open(
            os.path.join(ckpt_dir, shard), framework='pt', device='cpu')
    return handles_d, wmap


def _get(handles_d, wmap, key: str) -> torch.Tensor:
    if key not in wmap:
        raise KeyError(f'tensor {key!r} not in weight_map')
    return handles_d[wmap[key]].get_tensor(key)


def _has(wmap, key: str) -> bool:
    return key in wmap


def _sf_swz_bytes(n: int, k: int) -> int:
    """Byte size of the swizzled UE4M3 scale-factor tensor for an (n,k) weight."""
    n_blocks = k // 16
    n_row_super = (n + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def _quant_nvfp4(handles, out_dict, prefix, w_bf16_dev, fvk, device,
                 *, packed_dst=None, sf_dst=None, stream: int = 0) -> float:
    """Quantize one BF16 (N,K) weight on device to NVFP4.

    Stores ``<prefix>_packed`` / ``<prefix>_sf`` ptrs and returns the GEMM
    alpha. If ``packed_dst`` / ``sf_dst`` are given (contiguous slices of a
    stacked buffer) the quant writes into them and no per-weight anchor is
    added (the caller anchors the stack once).
    """
    n, k = int(w_bf16_dev.shape[0]), int(w_bf16_dev.shape[1])
    if k % 16 != 0:
        raise ValueError(f'{prefix}: K={k} not a multiple of 16')
    packed = packed_dst if packed_dst is not None else torch.empty(
        n, k // 2, dtype=torch.uint8, device=device)
    sf_swz = sf_dst if sf_dst is not None else torch.zeros(
        _sf_swz_bytes(n, k), dtype=torch.uint8, device=device)
    scratch = torch.zeros(1, dtype=torch.float32, device=device)
    out_gs = torch.zeros(1, dtype=torch.float32, device=device)
    fvk.bf16_weight_to_nvfp4_swizzled(
        int(w_bf16_dev.data_ptr()),
        int(packed.data_ptr()), int(sf_swz.data_ptr()),
        int(scratch.data_ptr()), int(out_gs.data_ptr()),
        n, k, stream)
    torch.cuda.synchronize()
    alpha = float(out_gs.item())
    if packed_dst is None:
        out_dict[prefix + '_packed'] = _anchor(handles, packed)
        out_dict[prefix + '_sf'] = _anchor(handles, sf_swz)
        out_dict[prefix + '_alpha'] = alpha
    return alpha


def _quant_from_ckpt(handles, out_dict, prefix, key, handles_d, wmap,
                     fvk, device) -> None:
    """Read a BF16 weight from the ckpt and quantize it to NVFP4."""
    w = _get(handles_d, wmap, key).to(device, non_blocking=True).to(
        torch.bfloat16).contiguous()
    _quant_nvfp4(handles, out_dict, prefix, w, fvk, device)
    del w


def _bf16_from_ckpt(handles, out_dict, name, key, handles_d, wmap, device,
                    *, optional=False, fold_one=False) -> None:
    if optional and not _has(wmap, key):
        out_dict[name] = 0
        out_dict[name + '_t'] = None
        return
    w = _get(handles_d, wmap, key)
    if fold_one:
        # Qwen3_5MoeRMSNorm: out = norm(x) * (1 + weight). Precompute the
        # (1 + w) effective weight in fp32 so the kernel does a plain
        # weight*norm. (The GDN RMSNormGated uses plain weight -- no fold.)
        t = (w.float() + 1.0).to(torch.bfloat16).to(device).contiguous()
    else:
        t = _bf16_to_dev(w, device)
    out_dict[name] = _anchor(handles, t)
    out_dict[name + '_shape'] = tuple(t.shape)
    # Named tensor handle: BF16-kept weights (norms, GDN in_proj, conv1d,
    # A_log/dt_bias, router, shared gate, embed, lm_head) are consumed by
    # the prefill forward as batched cuBLAS matmuls / elementwise inputs.
    # The pointer in out_dict[name] stays for the M=1 matvec decode path.
    out_dict[name + '_t'] = t


def _load_moe(handles, ld, lp, handles_d, wmap, fvk, device,
              n_experts: int) -> None:
    """Load one layer's MoE block: router (BF16) + experts + shared expert."""
    # Router gate (BF16) and shared-expert sigmoid gate (BF16).
    _bf16_from_ckpt(handles, ld, 'router_w', lp + 'mlp.gate.weight',
                    handles_d, wmap, device)
    _bf16_from_ckpt(handles, ld, 'shared_gate_w',
                    lp + 'mlp.shared_expert_gate.weight',
                    handles_d, wmap, device)

    # Shared expert (NVFP4).
    sp = lp + 'mlp.shared_expert.'
    _quant_from_ckpt(handles, ld, 'shared_gate_proj', sp + 'gate_proj.weight',
                     handles_d, wmap, fvk, device)
    _quant_from_ckpt(handles, ld, 'shared_up_proj', sp + 'up_proj.weight',
                     handles_d, wmap, fvk, device)
    _quant_from_ckpt(handles, ld, 'shared_down_proj', sp + 'down_proj.weight',
                     handles_d, wmap, fvk, device)

    # Routed experts: packed 3D tensors (E, out, in). Quantize each expert
    # into a contiguous slice of a per-layer stacked NVFP4 buffer so the
    # downstream grouped GEMM sees one contiguous weight per projection.
    gate_up = _get(handles_d, wmap, lp + 'mlp.experts.gate_up_proj')
    down = _get(handles_d, wmap, lp + 'mlp.experts.down_proj')
    e_gu, n_gu, k_gu = gate_up.shape   # (E, 2*inter, hidden)
    e_dn, n_dn, k_dn = down.shape      # (E, hidden, inter)
    if e_gu != n_experts or e_dn != n_experts:
        raise ValueError(
            f'expert count mismatch: gate_up {e_gu}, down {e_dn}, '
            f'expected {n_experts}')

    gu_packed = torch.empty(n_experts, n_gu, k_gu // 2,
                            dtype=torch.uint8, device=device)
    gu_sf = torch.zeros(n_experts, _sf_swz_bytes(n_gu, k_gu),
                        dtype=torch.uint8, device=device)
    dn_packed = torch.empty(n_experts, n_dn, k_dn // 2,
                            dtype=torch.uint8, device=device)
    dn_sf = torch.zeros(n_experts, _sf_swz_bytes(n_dn, k_dn),
                        dtype=torch.uint8, device=device)
    gu_alpha = torch.empty(n_experts, dtype=torch.float32)
    dn_alpha = torch.empty(n_experts, dtype=torch.float32)

    gate_up_dev = gate_up.to(device, non_blocking=True).to(torch.bfloat16)
    down_dev = down.to(device, non_blocking=True).to(torch.bfloat16)
    for e in range(n_experts):
        gu_alpha[e] = _quant_nvfp4(
            handles, ld, '', gate_up_dev[e].contiguous(), fvk, device,
            packed_dst=gu_packed[e], sf_dst=gu_sf[e])
        dn_alpha[e] = _quant_nvfp4(
            handles, ld, '', down_dev[e].contiguous(), fvk, device,
            packed_dst=dn_packed[e], sf_dst=dn_sf[e])
    del gate_up_dev, down_dev, gate_up, down

    ld['experts_gate_up_packed'] = _anchor(handles, gu_packed)
    ld['experts_gate_up_sf'] = _anchor(handles, gu_sf)
    ld['experts_gate_up_alpha'] = _anchor(handles, gu_alpha)
    ld['experts_down_packed'] = _anchor(handles, dn_packed)
    ld['experts_down_sf'] = _anchor(handles, dn_sf)
    ld['experts_down_alpha'] = _anchor(handles, dn_alpha)
    # Named tensor handles for per-expert slicing in the routed-expert
    # grouped loop (stacked (E, N, K/2) packed / (E, sf_bytes) SF / (E,)
    # alpha). gu_alpha / dn_alpha stay on host for cheap .item() per call.
    ld['experts_gate_up_packed_t'] = gu_packed
    ld['experts_gate_up_sf_t'] = gu_sf
    ld['experts_gate_up_alpha_t'] = gu_alpha
    ld['experts_down_packed_t'] = dn_packed
    ld['experts_down_sf_t'] = dn_sf
    ld['experts_down_alpha_t'] = dn_alpha
    ld['moe_intermediate'] = k_dn          # inter
    ld['n_experts'] = n_experts


def extract_weights_nexn2_nvfp4(
    ckpt_dir: str,
    fvk,
    device: str = 'cuda:0',
) -> WeightHandles:
    """Build :class:`WeightHandles` from a Nex-N2-mini BF16 ckpt directory."""
    cfg = json.load(open(os.path.join(ckpt_dir, 'config.json')))
    tc = cfg.get('text_config', cfg)
    num_layers = int(tc['num_hidden_layers'])
    hidden = int(tc['hidden_size'])
    vocab = int(tc['vocab_size'])
    head_dim = int(tc['head_dim'])
    n_q = int(tc['num_attention_heads'])
    n_kv = int(tc['num_key_value_heads'])
    n_experts = int(tc['num_experts'])
    experts_per_tok = int(tc['num_experts_per_tok'])
    layer_types = list(tc['layer_types'])
    rms_eps = float(tc.get('rms_norm_eps', 1e-6))
    rope_params = tc.get('rope_parameters', {}) or {}
    rope_theta = float(rope_params.get('rope_theta', 1.0e7))
    partial_rotary = float(rope_params.get('partial_rotary_factor',
                                           tc.get('partial_rotary_factor', 0.25)))

    handles, handles_d, wmap = WeightHandles(), *_open_shards(ckpt_dir)

    # ── Top-level BF16 tensors ──
    _bf16_from_ckpt(handles, handles.ptrs, 'embed_w',
                    'model.language_model.embed_tokens.weight',
                    handles_d, wmap, device)
    _bf16_from_ckpt(handles, handles.ptrs, 'final_norm_w',
                    'model.language_model.norm.weight',
                    handles_d, wmap, device, fold_one=True)
    _bf16_from_ckpt(handles, handles.ptrs, 'lm_head_w', 'lm_head.weight',
                    handles_d, wmap, device)

    # ── Per-layer ──
    per_layer: list = [None] * num_layers
    for i in range(num_layers):
        lp = f'model.language_model.layers.{i}.'
        ltype = layer_types[i]
        ld: dict = {'type': ltype, 'quant_format': 'nvfp4'}

        _bf16_from_ckpt(handles, ld, 'input_norm_w', lp + 'input_layernorm.weight',
                        handles_d, wmap, device, fold_one=True)
        _bf16_from_ckpt(handles, ld, 'post_norm_w',
                        lp + 'post_attention_layernorm.weight',
                        handles_d, wmap, device, fold_one=True)

        if ltype == 'full_attention':
            ap = lp + 'self_attn.'
            _quant_from_ckpt(handles, ld, 'q_proj', ap + 'q_proj.weight',
                             handles_d, wmap, fvk, device)
            _quant_from_ckpt(handles, ld, 'k_proj', ap + 'k_proj.weight',
                             handles_d, wmap, fvk, device)
            _quant_from_ckpt(handles, ld, 'v_proj', ap + 'v_proj.weight',
                             handles_d, wmap, fvk, device)
            _quant_from_ckpt(handles, ld, 'o_proj', ap + 'o_proj.weight',
                             handles_d, wmap, fvk, device)
            _bf16_from_ckpt(handles, ld, 'q_norm_w', ap + 'q_norm.weight',
                            handles_d, wmap, device, fold_one=True)
            _bf16_from_ckpt(handles, ld, 'k_norm_w', ap + 'k_norm.weight',
                            handles_d, wmap, device, fold_one=True)
        elif ltype == 'linear_attention':
            gp = lp + 'linear_attn.'
            # GDN in_proj path stays BF16.
            for nm, key in (('in_proj_qkv_w', 'in_proj_qkv.weight'),
                            ('in_proj_z_w', 'in_proj_z.weight'),
                            ('in_proj_a_w', 'in_proj_a.weight'),
                            ('in_proj_b_w', 'in_proj_b.weight'),
                            ('conv1d_w', 'conv1d.weight'),
                            ('A_log', 'A_log'),
                            ('dt_bias', 'dt_bias'),
                            ('gdn_norm_w', 'norm.weight')):
                _bf16_from_ckpt(handles, ld, nm, gp + key,
                                handles_d, wmap, device, optional=True)
            # out_proj → NVFP4.
            _quant_from_ckpt(handles, ld, 'out_proj', gp + 'out_proj.weight',
                             handles_d, wmap, fvk, device)
        else:
            raise ValueError(f'layer {i}: unknown layer_type {ltype!r}')

        # Every layer has a MoE FFN (mlp_only_layers is empty).
        _load_moe(handles, ld, lp, handles_d, wmap, fvk, device, n_experts)
        per_layer[i] = ld

    handles.ptrs['layers'] = per_layer
    handles.ptrs['vocab_size'] = vocab
    handles.ptrs['hidden'] = hidden
    handles.ptrs['head_dim'] = head_dim
    handles.ptrs['num_q_heads'] = n_q
    handles.ptrs['num_kv_heads'] = n_kv
    handles.ptrs['num_experts'] = n_experts
    handles.ptrs['experts_per_tok'] = experts_per_tok
    handles.ptrs['num_layers'] = num_layers
    handles.ptrs['layer_types'] = layer_types
    handles.ptrs['rms_norm_eps'] = rms_eps
    handles.ptrs['rope_theta'] = rope_theta
    handles.ptrs['partial_rotary_factor'] = partial_rotary
    handles.ptrs['quant_format'] = 'nvfp4'
    handles.ptrs['ckpt_dir'] = ckpt_dir
    handles.ptrs['mtp'] = None       # MTP weights not in the base ckpt
    handles.ptrs['dflash'] = None
    return handles
