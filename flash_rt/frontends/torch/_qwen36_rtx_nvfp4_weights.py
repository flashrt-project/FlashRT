"""FlashRT — Qwen3.6 NVFP4 W4A16 raw safetensors loader.

Drop-in replacement for ``extract_weights`` (FP8 path) that reads a
``compressed-tensors`` ``nvfp4-pack-quantized`` ckpt **directly from
safetensors**, with no dependency on transformers ``AutoModel`` or
``compressed_tensors``. Stays light (the framework's stated principle).

Ckpt schema (verified for prithivMLmods/Qwen3.6-27B-NVFP4):

  Quantized linear (256 of them: 16 full-attn × 4 + 64 layer × 3 MLP):
    <prefix>.weight_packed         u8       (out, in/2)
    <prefix>.weight_scale          f8_e4m3  (out, in/16)
    <prefix>.weight_global_scale   f32      (1,)

  Linear-attn projections stay BF16 (NOT in this ckpt's quant scope):
    linear_attn.{in_proj_qkv, in_proj_z, out_proj}.weight    bf16
    linear_attn.{in_proj_a, in_proj_b}.weight                bf16
    linear_attn.{A_log, dt_bias}                             bf16
    linear_attn.conv1d.weight                                bf16
    linear_attn.norm.weight                                  bf16

  Norms, embed, lm_head: BF16. Vision tower: BF16, skipped (text path).

  ``model.language_model.layers.<i>.input_layernorm.weight``,
  ``...post_attention_layernorm.weight``,
  ``model.language_model.embed_tokens.weight``,
  ``model.language_model.norm.weight``,
  ``lm_head.weight``.

Layer types:
  ``config.text_config.layer_types`` is a list[str] of length
  num_hidden_layers, each ``'full_attention'`` or ``'linear_attention'``.

Returns a :class:`WeightHandles` with the same surface area as
:func:`extract_weights` so the frontend's per-layer forward can switch
on ``layer['type']`` and ``layer.get('quant_format')`` to dispatch
either the FP8 GEMM (existing path) or the NVFP4 W4A16 GEMM (new
``fp4_w4a16_gemm_sm120_bf16out``).

Per-layer keys produced (NVFP4 path):

  Common (every layer):
    * ``input_norm_eff_w``        — bf16 (5120,) (1+w) precompute
                                     (Qwen3.5 RMSNorm convention,
                                     same as FP8 path)
    * ``post_attn_norm_eff_w``    — bf16 (5120,) (1+w)
    * ``mlp_gate_packed/_sf/_global`` — NVFP4 weight + sf_swz + global
    * ``mlp_up_packed/_sf/_global``
    * ``mlp_down_packed/_sf/_global``
    * ``quant_format`` = 'nvfp4'

  Linear-attn layers (BF16, same as FP8 path's bf16 helper tensors):
    * ``in_proj_qkv_w``           — bf16 (10240, 5120)
    * ``in_proj_z_w``             — bf16 (6144, 5120)
    * ``in_proj_a_w``             — bf16 (48, 5120)
    * ``in_proj_b_w``             — bf16 (48, 5120)
    * ``out_proj_w``              — bf16 (5120, 6144)
    * ``conv1d_w``, ``conv1d_b``  — bf16
    * ``head_norm_w``, ``A_log``, ``dt_bias`` — bf16

  Full-attn layers:
    * ``q_proj_packed/_sf/_global`` — NVFP4
    * ``k_proj_packed/_sf/_global``
    * ``v_proj_packed/_sf/_global``
    * ``o_proj_packed/_sf/_global``
    * ``q_norm_eff_w``, ``k_norm_eff_w`` — bf16 (256,) (1+w)

  Top-level:
    * ``embed_w`` (248320, 5120) bf16
    * ``final_norm_eff_w`` (5120,) bf16 (1+w)
    * ``lm_head_w`` (248320, 5120) bf16
    * ``layer_types``, ``num_layers``, ``vocab_size``, ``hidden``

The on-device SF swizzle (linear → CUTLASS Sm1xx blockscaled) is
performed **once** per quantized weight at load time using
``fvk.nvfp4_sf_linear_to_swizzled``. The original linear-layout SF
tensor is then released.
"""
from __future__ import annotations

import json
import os

import torch

from flash_rt.frontends.torch._qwen36_rtx_weights import (
    WeightHandles,
    _eff_rmsnorm_weight,
    _ensure_anchored,
)


def _load_quantized_proj(
    handles: WeightHandles,
    out_dict: dict,
    prefix: str,
    safetensors_handle,
    base_key: str,
    fvk,
    device: str,
    stream: int = 0,
) -> None:
    """Read 1 NVFP4 quantized linear (3 tensors), produce 3 ptrs.

    Tensors come from safe_open(device='cpu') — we explicitly move
    just packed and sf_lin to GPU here, do the reshape on-device,
    and free sf_lin. Vision tower / unused tensors never touch GPU.
    """
    packed_cpu = safetensors_handle.get_tensor(base_key + '.weight_packed')
    sf_lin_cpu = safetensors_handle.get_tensor(base_key + '.weight_scale')
    glb_cpu = safetensors_handle.get_tensor(base_key + '.weight_global_scale')

    # Explicitly move to GPU (one allocation each).
    packed = packed_cpu.to(device, non_blocking=True).contiguous()
    sf_lin = sf_lin_cpu.to(device, non_blocking=True).contiguous()

    rows, cols_div16 = sf_lin.shape
    cols_in = cols_div16 * 16
    n_blocks = cols_in // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    sf_bytes = n_row_super * n_col_super * 512
    sf_swz = torch.zeros(sf_bytes, dtype=torch.uint8, device=device)

    fvk.nvfp4_sf_linear_to_swizzled(
        int(sf_lin.data_ptr()),
        int(sf_swz.data_ptr()),
        rows, cols_in, False, stream,
    )
    # sf_lin no longer needed after reshape; CUDA cache will reclaim.
    del sf_lin, sf_lin_cpu, packed_cpu

    out_dict[prefix + '_packed'] = _ensure_anchored(handles, packed)
    out_dict[prefix + '_sf'] = _ensure_anchored(handles, sf_swz)
    # compressed-tensors NVFP4 convention: w_dequant = fp4 * sf / global.
    # Store the inverse (alpha = 1/global) here so the GEMM hot-path
    # passes it straight through without per-call division.
    raw_global = float(glb_cpu.to(torch.float32).item())
    inv_global = 1.0 / raw_global if raw_global != 0.0 else 0.0
    glb_f32 = glb_cpu.to(torch.float32).to(device).contiguous()
    out_dict[prefix + '_global'] = _ensure_anchored(handles, glb_f32)
    # raw value (for ref / debug)
    out_dict[prefix + '_global_value'] = raw_global
    # alpha = 1/global, fed directly to GEMM (CT convention)
    out_dict[prefix + '_alpha'] = inv_global


def _load_quantized_mlp_gate_up(
    handles: WeightHandles,
    out_dict: dict,
    safetensors_handle,
    base_mlp_key: str,
    fvk,
    device: str,
    stream: int = 0,
) -> None:
    """Load MLP gate/up as one row-concatenated NVFP4 projection.

    Qwen3.6's gate_proj and up_proj share the same global scale across
    all observed layers. Keeping them as one physical tensor lets the
    prefill path issue one large GEMM while the legacy fields remain
    valid pointers into the fused tensor, without duplicating weights.
    """
    gate_key = base_mlp_key + '.gate_proj'
    up_key = base_mlp_key + '.up_proj'
    gate_glb_cpu = safetensors_handle.get_tensor(
        gate_key + '.weight_global_scale')
    up_glb_cpu = safetensors_handle.get_tensor(
        up_key + '.weight_global_scale')
    gate_global = float(gate_glb_cpu.to(torch.float32).item())
    up_global = float(up_glb_cpu.to(torch.float32).item())
    gate_alpha = 1.0 / gate_global if gate_global != 0.0 else 0.0
    up_alpha = 1.0 / up_global if up_global != 0.0 else 0.0

    if abs(gate_alpha - up_alpha) > 1e-12 * max(gate_alpha, 1e-12):
        _load_quantized_proj(
            handles, out_dict, 'mlp_gate',
            safetensors_handle, gate_key, fvk, device, stream)
        _load_quantized_proj(
            handles, out_dict, 'mlp_up',
            safetensors_handle, up_key, fvk, device, stream)
        out_dict['mlp_gate_up_homogeneous_alpha'] = False
        return

    gate_packed = safetensors_handle.get_tensor(
        gate_key + '.weight_packed').to(device, non_blocking=True).contiguous()
    up_packed = safetensors_handle.get_tensor(
        up_key + '.weight_packed').to(device, non_blocking=True).contiguous()
    gate_sf_lin = safetensors_handle.get_tensor(
        gate_key + '.weight_scale').to(device, non_blocking=True).contiguous()
    up_sf_lin = safetensors_handle.get_tensor(
        up_key + '.weight_scale').to(device, non_blocking=True).contiguous()

    rows_gate, cols_div16 = gate_sf_lin.shape
    rows_up, cols_div16_up = up_sf_lin.shape
    if rows_gate != rows_up or cols_div16 != cols_div16_up:
        raise RuntimeError(
            'Qwen3.6 MLP gate/up shape mismatch: '
            f'gate={tuple(gate_sf_lin.shape)} up={tuple(up_sf_lin.shape)}')

    packed = torch.cat([gate_packed, up_packed], dim=0).contiguous()
    sf_lin = torch.cat([gate_sf_lin, up_sf_lin], dim=0).contiguous()
    rows_fused = rows_gate + rows_up
    cols_in = cols_div16 * 16
    n_blocks = cols_in // 16
    n_col_super = (n_blocks + 3) // 4
    n_row_super_fused = (rows_fused + 127) // 128
    sf_bytes = n_row_super_fused * n_col_super * 512
    sf_swz = torch.zeros(sf_bytes, dtype=torch.uint8, device=device)

    fvk.nvfp4_sf_linear_to_swizzled(
        int(sf_lin.data_ptr()),
        int(sf_swz.data_ptr()),
        rows_fused, cols_in, False, stream,
    )

    packed_base = _ensure_anchored(handles, packed)
    sf_base = _ensure_anchored(handles, sf_swz)
    gate_packed_bytes = int(gate_packed.numel() * gate_packed.element_size())
    gate_row_super = (rows_gate + 127) // 128
    gate_sf_bytes = int(gate_row_super * n_col_super * 512)

    out_dict['mlp_gate_up_packed'] = packed_base
    out_dict['mlp_gate_up_sf'] = sf_base
    out_dict['mlp_gate_up_alpha'] = gate_alpha
    out_dict['mlp_gate_up_N'] = int(rows_fused)
    out_dict['mlp_gate_up_homogeneous_alpha'] = True

    out_dict['mlp_gate_packed'] = packed_base
    out_dict['mlp_up_packed'] = packed_base + gate_packed_bytes
    out_dict['mlp_gate_sf'] = sf_base
    out_dict['mlp_up_sf'] = sf_base + gate_sf_bytes
    glb_f32 = gate_glb_cpu.to(torch.float32).to(device).contiguous()
    glb_ptr = _ensure_anchored(handles, glb_f32)
    out_dict['mlp_gate_global'] = glb_ptr
    out_dict['mlp_up_global'] = glb_ptr
    out_dict['mlp_gate_global_value'] = gate_global
    out_dict['mlp_up_global_value'] = up_global
    out_dict['mlp_gate_alpha'] = gate_alpha
    out_dict['mlp_up_alpha'] = up_alpha

    del gate_packed, up_packed, gate_sf_lin, up_sf_lin, sf_lin


def _bf16_anchor(handles: WeightHandles, t: torch.Tensor,
                 device: str = 'cuda:0') -> int:
    """Move CPU tensor to GPU bf16 contiguous, anchor, return ptr."""
    return _ensure_anchored(
        handles, t.to(torch.bfloat16).to(device).contiguous())


def _quant_bf16_lin_proj(handles: WeightHandles, ld: dict, prefix: str,
                          w_cpu: torch.Tensor, fvk,
                          device: str = 'cuda:0') -> None:
    """Quantize a BF16 weight (N, K) to NVFP4 swizzled with proper
    per-tensor global_scale. Writes ``<prefix>_packed`` /
    ``<prefix>_sf`` / ``<prefix>_alpha`` to ``ld``.

    Used for the prithivMLmods Qwen3.6 NVFP4 ckpt's three lin-attn
    projections (in_proj_qkv / in_proj_z / out_proj) that the upstream
    quantizer left as BF16. Cuts per-forward weight BW by 2× on those
    projections (48 layers × ~200MB BF16 → ~50MB NVFP4 packed + SF).
    Quality: e2m1 4-bit codebook noise; HF token match preserved at
    the spec level (greedy spec accept-prefix is bounded by main's
    argmax, so per-element noise is filtered through main argmax).
    """
    w_bf16 = w_cpu.to(torch.bfloat16).to(device).contiguous()
    N, K = w_bf16.shape
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
    n_blocks = K // 16
    n_row_super = (N + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    sf_bytes = n_row_super * n_col_super * 512
    sf_swz = torch.zeros(sf_bytes, dtype=torch.uint8, device=device)
    scratch_amax = torch.zeros(1, dtype=torch.float32, device=device)
    out_gs = torch.zeros(1, dtype=torch.float32, device=device)
    fvk.bf16_weight_to_nvfp4_swizzled(
        int(w_bf16.data_ptr()),
        int(packed.data_ptr()), int(sf_swz.data_ptr()),
        int(scratch_amax.data_ptr()), int(out_gs.data_ptr()),
        N, K, 0)
    torch.cuda.synchronize()
    ld[prefix + '_packed'] = _ensure_anchored(handles, packed)
    ld[prefix + '_sf'] = _ensure_anchored(handles, sf_swz)
    ld[prefix + '_alpha'] = float(out_gs.item())
    # w_bf16 is freed by Python GC; we don't anchor it.


def extract_weights_nvfp4(
    ckpt_dir: str,
    fvk,
    device: str = 'cuda:0',
) -> WeightHandles:
    """Build a :class:`WeightHandles` from a Qwen3.6 NVFP4 ckpt directory.

    Args:
      ckpt_dir: path to dir containing ``model.safetensors`` and
        ``config.json``.
      fvk: the flash_rt_kernels pybind module
        (provides ``nvfp4_sf_linear_to_swizzled``).
      device: cuda device.

    Returns:
      WeightHandles whose ``ptrs`` mirrors the FP8 path's schema, with
      NVFP4-specific ``*_packed/_sf/_global`` keys for quantized
      linears and ``quant_format='nvfp4'`` per layer dict.
    """
    from safetensors import safe_open

    cfg_path = os.path.join(ckpt_dir, 'config.json')
    st_path = os.path.join(ckpt_dir, 'model.safetensors')
    if not (os.path.isfile(cfg_path) and os.path.isfile(st_path)):
        raise RuntimeError(
            f'NVFP4 ckpt dir missing config.json or model.safetensors: '
            f'{ckpt_dir!r}'
        )

    cfg = json.load(open(cfg_path))
    text_cfg = cfg.get('text_config') or cfg.get('language_model') or {}
    layer_types = list(text_cfg.get('layer_types') or [])
    num_layers = int(text_cfg.get('num_hidden_layers')
                     or len(layer_types))
    hidden = int(text_cfg.get('hidden_size') or 5120)
    vocab = int(text_cfg.get('vocab_size') or 248320)

    if len(layer_types) != num_layers:
        raise RuntimeError(
            f'config.text_config.layer_types length {len(layer_types)} '
            f'!= num_hidden_layers {num_layers}'
        )

    handles = WeightHandles()
    per_layer: list[dict] = [None] * num_layers   # type: ignore[list-item]

    # Open with CPU mmap so vision tower / unused tensors never touch
    # GPU memory. Each call to get_tensor() returns a CPU tensor; we
    # explicitly .to(device) only the language-model + lm_head + embed
    # tensors we need.
    debug = bool(int(os.environ.get('FLASHRT_NVFP4_LOAD_DEBUG', '0') or '0'))

    def _vram_used():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9

    with safe_open(st_path, framework='pt', device='cpu') as f:
        if debug:
            print(f'  [load] open, vram = {_vram_used():.2f} GB')
        # Top-level (BF16) tensors.
        embed_cpu = f.get_tensor('model.language_model.embed_tokens.weight')
        handles.ptrs['embed_w'] = _bf16_anchor(handles, embed_cpu, device)

        final_norm_cpu = f.get_tensor('model.language_model.norm.weight')
        # _eff_rmsnorm_weight runs on cpu fp32 then to bf16; move to GPU.
        final_norm_eff = _eff_rmsnorm_weight(final_norm_cpu).to(device)
        handles.ptrs['final_norm_eff_w'] = _ensure_anchored(
            handles, final_norm_eff)

        lm_head_cpu = f.get_tensor('lm_head.weight')
        if bool(int(os.environ.get(
                'FLASHRT_QWEN36_KEEP_BF16_LM_HEAD', '0') or '0')):
            handles.ptrs['lm_head_w_bf16'] = _bf16_anchor(
                handles, lm_head_cpu, device)
        # G8: quantize lm_head to NVFP4 at load time. lm_head is the
        # single largest weight in MTP-spec hot path (2.5 GB BF16 read
        # per call × K+1 calls per cycle). NVFP4 cuts it to 0.6 GB.
        # Quality: spec accept-prefix is bounded by main forward's
        # argmax, so MTP-side draft argmax noise just lowers AL
        # marginally without affecting output token correctness.
        # Main forward's own lm_head argmax (after the verify graph)
        # is the gate; we measure AL + HF token match to confirm.
        _quant_bf16_lin_proj(
            handles, handles.ptrs, 'lm_head',
            lm_head_cpu, fvk, device)
        # Tied check: compare CPU data_ptrs (post-mmap, both are mmap views)
        handles.ptrs['lm_head_tied'] = bool(
            int(lm_head_cpu.data_ptr()) == int(embed_cpu.data_ptr())
        )

        handles.ptrs['vocab_size'] = vocab
        handles.ptrs['hidden'] = hidden
        handles.ptrs['num_layers'] = num_layers
        handles.ptrs['layer_types'] = layer_types
        handles.ptrs['quant_format'] = 'nvfp4'

        # Per-layer.
        for L in range(num_layers):
            t = layer_types[L]
            base = f'model.language_model.layers.{L}.'
            ld: dict = {'type': t, 'quant_format': 'nvfp4'}

            # Pre/post layernorm (Qwen3.5 RMSNorm (1+w) precompute).
            in_eff = _eff_rmsnorm_weight(
                f.get_tensor(base + 'input_layernorm.weight')).to(device)
            post_eff = _eff_rmsnorm_weight(
                f.get_tensor(base + 'post_attention_layernorm.weight')
            ).to(device)
            ld['input_norm_eff_w'] = _ensure_anchored(handles, in_eff)
            ld['post_attn_norm_eff_w'] = _ensure_anchored(handles, post_eff)

            # MLP (NVFP4 quantized in every layer). Gate/up share one
            # fused physical tensor and expose legacy offset pointers.
            _load_quantized_mlp_gate_up(
                handles, ld, f, base + 'mlp', fvk, device)
            _load_quantized_proj(
                handles, ld, 'mlp_down',
                f, base + 'mlp.down_proj', fvk, device)

            if t == 'linear_attention':
                la = base + 'linear_attn.'
                # G7: ckpt leaves the three large lin projections as
                # BF16. We quantize them to NVFP4 at load time (with
                # per-tensor global_scale) so the verify-path GEMMs
                # read FP4 weights instead of BF16 — cuts BW by 2× on
                # 48 lin layers (the dominant FP8/NVFP4 verify-forward
                # delta). in_proj_a / in_proj_b are tiny (N=48) so we
                # keep them BF16; the matvec/matmul cost is negligible.
                _quant_bf16_lin_proj(
                    handles, ld, 'in_proj_qkv',
                    f.get_tensor(la + 'in_proj_qkv.weight'),
                    fvk, device)
                _quant_bf16_lin_proj(
                    handles, ld, 'in_proj_z',
                    f.get_tensor(la + 'in_proj_z.weight'),
                    fvk, device)
                a_w = f.get_tensor(la + 'in_proj_a.weight').to(
                    torch.bfloat16).contiguous().to(device)
                b_w = f.get_tensor(la + 'in_proj_b.weight').to(
                    torch.bfloat16).contiguous().to(device)
                ld['in_proj_a_w'] = _ensure_anchored(handles, a_w)
                ld['in_proj_b_w'] = _ensure_anchored(handles, b_w)
                # A2c-2: concatenated (96, 5120) buffer for the dual
                # bf16_matvec call in the NVFP4 lin layer (S=1 decode).
                # in_proj_a/b are tiny (48 rows each); the cat is
                # cheap and shared across all 48 lin layers' init.
                ab_w = torch.cat([a_w, b_w], dim=0).contiguous()
                ld['in_proj_ab_w'] = _ensure_anchored(handles, ab_w)
                ld['in_proj_ab_w_t'] = ab_w
                _quant_bf16_lin_proj(
                    handles, ld, 'out_proj',
                    f.get_tensor(la + 'out_proj.weight'),
                    fvk, device)
                conv = f.get_tensor(la + 'conv1d.weight').to(
                    torch.bfloat16).squeeze(1).contiguous().to(device)
                ld['conv1d_w'] = _ensure_anchored(handles, conv)
                bias_key = la + 'conv1d.bias'
                if bias_key in f.keys():
                    ld['conv1d_b'] = _bf16_anchor(
                        handles, f.get_tensor(bias_key), device)
                else:
                    ld['conv1d_b'] = 0
                ld['head_norm_w'] = _bf16_anchor(
                    handles, f.get_tensor(la + 'norm.weight'), device)
                A_log_cpu = f.get_tensor(la + 'A_log').detach()
                dt_bias_cpu = f.get_tensor(la + 'dt_bias').detach()
                ld['A_log'] = _bf16_anchor(handles, A_log_cpu, device)
                ld['dt_bias'] = _bf16_anchor(handles, dt_bias_cpu, device)
                neg_a_log = (-A_log_cpu.float().exp()).contiguous().to(device)
                dt_bias_fp32 = dt_bias_cpu.float().contiguous().to(device)
                ld['neg_A_log_exp_fp32'] = _ensure_anchored(
                    handles, neg_a_log)
                ld['dt_bias_fp32'] = _ensure_anchored(
                    handles, dt_bias_fp32)
                ld['neg_A_log_exp_fp32_t'] = neg_a_log
                ld['dt_bias_fp32_t'] = dt_bias_fp32

            elif t == 'full_attention':
                sa = base + 'self_attn.'
                _load_quantized_proj(
                    handles, ld, 'q_proj', f, sa + 'q_proj', fvk, device)
                _load_quantized_proj(
                    handles, ld, 'k_proj', f, sa + 'k_proj', fvk, device)
                _load_quantized_proj(
                    handles, ld, 'v_proj', f, sa + 'v_proj', fvk, device)
                _load_quantized_proj(
                    handles, ld, 'o_proj', f, sa + 'o_proj', fvk, device)
                q_eff = _eff_rmsnorm_weight(
                    f.get_tensor(sa + 'q_norm.weight')).to(device)
                k_eff = _eff_rmsnorm_weight(
                    f.get_tensor(sa + 'k_norm.weight')).to(device)
                ld['q_norm_eff_w'] = _ensure_anchored(handles, q_eff)
                ld['k_norm_eff_w'] = _ensure_anchored(handles, k_eff)
            else:
                raise ValueError(
                    f'unknown layer_type {t!r} at idx {L}'
                )

            per_layer[L] = ld
            if debug and (L < 4 or L % 10 == 0):
                torch.cuda.synchronize()
                print(f'  [load] layer {L:2d} done ({t}), '
                      f'vram = {_vram_used():.2f} GB')

    handles.ptrs['layers'] = per_layer
    if debug:
        torch.cuda.synchronize()
        print(f'  [load] DONE, vram = {_vram_used():.2f} GB')
    return handles


def extract_mtp_weights_nvfp4(mtp: dict, handles: WeightHandles, fvk,
                              device: str = 'cuda:0') -> dict:
    """Convert FP8 MTP head weights to NVFP4 and add to handles.

    The Qwen3.6 mtp.safetensors only ships in the FP8 ckpt
    (compressed-tensors NVFP4 ckpt has no MTP module). To stay on a
    pure-NVFP4 hot path we one-time convert each FP8 projection to
    NVFP4 *directly*, in a single kernel launch with proper per-tensor
    global_scale.

    Pipeline (per projection):
      ``fp8_block128_to_nvfp4_swizzled_bf16``:
        FP8 e4m3 (with per-128 fp32 block scales)
        -> dequant inline in FP32 (lossless; no BF16 mantissa floor)
        -> per-tensor amax via atomicMax over FP32
        -> global_scale = max|W| / 2688
        -> per-NVFP4-block SF in UE4M3 with proper division by global,
           so the byte stays in UE4M3's well-represented range
           (instead of the denormal region where amax/6 lands when
           global_scale = 1 — that was the AL-killing bug in v1).

    GEMM contract: alpha = act_global * w_global. For per-token
    activation quant (act_global = 1) we pass alpha = global_scale
    of the WEIGHT, not 1.0.

    Args:
      mtp: dict[str, Tensor] — raw mtp.safetensors with the leading
        'mtp.' prefix stripped. weight = e4m3_fn, weight_scale_inv =
        fp32 (after _load_mtp_weights cast).
      handles: WeightHandles to extend.
      fvk: flash_rt_kernels module (provides
        fp8_block128_to_nvfp4_swizzled_bf16).
      device: cuda device for intermediate buffers.

    Returns:
      dict mirroring extract_mtp_weights but with NVFP4 ptrs:
        q_proj_packed/_sf/_alpha (= global_scale)  (similar for k/v/o + mlp)
        fc_w  (BF16, unchanged)
        all RMSNorm precomputes (unchanged)
    """
    out: dict = {'type': 'mtp', 'quant_format': 'nvfp4'}

    in_eff = _eff_rmsnorm_weight(mtp['layers.0.input_layernorm.weight'])
    post_eff = _eff_rmsnorm_weight(
        mtp['layers.0.post_attention_layernorm.weight'])
    out['input_norm_eff_w'] = _ensure_anchored(handles, in_eff.to(device))
    out['post_attn_norm_eff_w'] = _ensure_anchored(
        handles, post_eff.to(device))

    # FP8 -> NVFP4 swizzled (with per-tensor global_scale), one
    # projection at a time. Single kernel per projection.
    fp8_pairs = (
        ('q_proj',   'layers.0.self_attn.q_proj',   12288, 5120),
        ('k_proj',   'layers.0.self_attn.k_proj',    1024, 5120),
        ('v_proj',   'layers.0.self_attn.v_proj',    1024, 5120),
        ('o_proj',   'layers.0.self_attn.o_proj',    5120, 6144),
        ('mlp_gate', 'layers.0.mlp.gate_proj',      17408, 5120),
        ('mlp_up',   'layers.0.mlp.up_proj',        17408, 5120),
        ('mlp_down', 'layers.0.mlp.down_proj',       5120, 17408),
    )

    # One reusable FP32 scratch (1 fp32) for the global amax. The kernel
    # zeros it on each call so we can share across projections.
    scratch_amax = torch.zeros(1, dtype=torch.float32, device=device)

    for prefix, base, N, K in fp8_pairs:
        w_fp8 = mtp[base + '.weight']
        s_fp32 = mtp[base + '.weight_scale_inv']
        if w_fp8.device.type != 'cuda':
            w_fp8 = w_fp8.to(device)
        if s_fp32.device.type != 'cuda':
            s_fp32 = s_fp32.to(device)

        packed = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
        n_blocks = K // 16
        n_row_super = (N + 127) // 128
        n_col_super = (n_blocks + 3) // 4
        sf_bytes = n_row_super * n_col_super * 512
        sf_swz = torch.zeros(sf_bytes, dtype=torch.uint8, device=device)
        # Per-projection global_scale (kept around so we can read the
        # value to host AFTER the pipeline drains; one fp32 each).
        out_gs = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.fp8_block128_to_nvfp4_swizzled_bf16(
            int(w_fp8.data_ptr()),
            int(s_fp32.data_ptr()),
            int(packed.data_ptr()),
            int(sf_swz.data_ptr()),
            int(scratch_amax.data_ptr()),
            int(out_gs.data_ptr()),
            N, K, 0,
        )
        out[prefix + '_packed'] = _ensure_anchored(handles, packed)
        out[prefix + '_sf'] = _ensure_anchored(handles, sf_swz)
        # Force one host sync after the last projection rather than per
        # tensor — but at load time this is a one-shot cost so we sync
        # here for clarity. global_scale is read by the GEMM site as a
        # plain float in alpha.
        torch.cuda.synchronize()
        out[prefix + '_alpha'] = float(out_gs.item())
        del w_fp8, s_fp32, out_gs

    # head-dim RMSNorms.
    q_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.q_norm.weight'])
    k_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.k_norm.weight'])
    out['q_norm_eff_w'] = _ensure_anchored(handles, q_eff.to(device))
    out['k_norm_eff_w'] = _ensure_anchored(handles, k_eff.to(device))

    pre_h_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_hidden.weight'])
    pre_e_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_embedding.weight'])
    final_eff = _eff_rmsnorm_weight(mtp['norm.weight'])
    out['pre_fc_norm_hidden_eff_w'] = _ensure_anchored(
        handles, pre_h_eff.to(device))
    out['pre_fc_norm_embedding_eff_w'] = _ensure_anchored(
        handles, pre_e_eff.to(device))
    out['final_norm_eff_w'] = _ensure_anchored(
        handles, final_eff.to(device))

    # fc stays BF16: K=10240, N=5120 — small enough that the BF16 GEMM
    # at M=1 isn't the bottleneck and we keep full fc precision.
    fc_w = mtp['fc.weight'].to(device).contiguous()
    out['fc_w'] = _ensure_anchored(handles, fc_w)

    return out


def extract_mtp_weights_bf16_nvfp4(mtp: dict, handles: WeightHandles, fvk,
                                   device: str = 'cuda:0') -> dict:
    """Quantize a BF16/native MTP head to NVFP4 and add to handles.

    Some community Qwen3.6 MTP-preserved checkpoints publish the MTP
    head as plain BF16 tensors rather than the official FP8
    ``weight + weight_scale_inv`` layout. The runtime hot path still
    expects the same NVFP4 MTP schema as :func:`extract_mtp_weights_nvfp4`,
    so this loader mirrors it but uses the BF16 -> NVFP4 converter used
    elsewhere in the NVFP4 frontend.
    """
    out: dict = {
        'type': 'mtp',
        'quant_format': 'nvfp4',
        'source_format': 'bf16',
    }

    in_eff = _eff_rmsnorm_weight(mtp['layers.0.input_layernorm.weight'])
    post_eff = _eff_rmsnorm_weight(
        mtp['layers.0.post_attention_layernorm.weight'])
    out['input_norm_eff_w'] = _ensure_anchored(handles, in_eff.to(device))
    out['post_attn_norm_eff_w'] = _ensure_anchored(
        handles, post_eff.to(device))

    bf16_pairs = (
        ('q_proj',   'layers.0.self_attn.q_proj'),
        ('k_proj',   'layers.0.self_attn.k_proj'),
        ('v_proj',   'layers.0.self_attn.v_proj'),
        ('o_proj',   'layers.0.self_attn.o_proj'),
        ('mlp_gate', 'layers.0.mlp.gate_proj'),
        ('mlp_up',   'layers.0.mlp.up_proj'),
        ('mlp_down', 'layers.0.mlp.down_proj'),
    )

    keep_bf16 = bool(int(os.environ.get(
        'FLASHRT_QWEN36_MTP_KEEP_BF16', '1') or '0'))

    for prefix, base in bf16_pairs:
        _quant_bf16_lin_proj(
            handles, out, prefix, mtp[base + '.weight'], fvk, device)
        if keep_bf16:
            w = mtp[base + '.weight'].to(torch.bfloat16).to(
                device).contiguous()
            out[prefix + '_w_bf16'] = _ensure_anchored(handles, w)

    q_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.q_norm.weight'])
    k_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.k_norm.weight'])
    out['q_norm_eff_w'] = _ensure_anchored(handles, q_eff.to(device))
    out['k_norm_eff_w'] = _ensure_anchored(handles, k_eff.to(device))

    pre_h_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_hidden.weight'])
    pre_e_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_embedding.weight'])
    final_eff = _eff_rmsnorm_weight(mtp['norm.weight'])
    out['pre_fc_norm_hidden_eff_w'] = _ensure_anchored(
        handles, pre_h_eff.to(device))
    out['pre_fc_norm_embedding_eff_w'] = _ensure_anchored(
        handles, pre_e_eff.to(device))
    out['final_norm_eff_w'] = _ensure_anchored(
        handles, final_eff.to(device))

    fc_w = mtp['fc.weight'].to(torch.bfloat16).to(device).contiguous()
    out['fc_w'] = _ensure_anchored(handles, fc_w)

    return out


def assert_extraction_invariants_nvfp4(handles: WeightHandles) -> None:
    """Verify all NVFP4 ptr fields populated. Run once at frontend init."""
    p = handles.ptrs
    assert p.get('quant_format') == 'nvfp4'
    assert isinstance(p.get('layers'), list)
    layers = p['layers']
    n_full = sum(1 for L in layers if L['type'] == 'full_attention')
    n_lin = sum(1 for L in layers if L['type'] == 'linear_attention')
    assert n_full == 16, f'expected 16 full-attn, got {n_full}'
    assert n_lin == 48, f'expected 48 linear-attn, got {n_lin}'

    common_keys = {
        'input_norm_eff_w', 'post_attn_norm_eff_w', 'quant_format',
        'mlp_gate_packed', 'mlp_gate_sf', 'mlp_gate_global',
        'mlp_up_packed', 'mlp_up_sf', 'mlp_up_global',
        'mlp_gate_up_packed', 'mlp_gate_up_sf', 'mlp_gate_up_alpha',
        'mlp_gate_up_N', 'mlp_gate_up_homogeneous_alpha',
        'mlp_down_packed', 'mlp_down_sf', 'mlp_down_global',
    }
    lin_keys = common_keys | {
        # G7: in_proj_qkv / in_proj_z / out_proj are now NVFP4 (loader
        # quantizes them at load time; ckpt has them as BF16). The
        # tiny in_proj_a / in_proj_b stay BF16 (N=48 only).
        'in_proj_qkv_packed', 'in_proj_qkv_sf', 'in_proj_qkv_alpha',
        'in_proj_z_packed', 'in_proj_z_sf', 'in_proj_z_alpha',
        'out_proj_packed', 'out_proj_sf', 'out_proj_alpha',
        'in_proj_a_w', 'in_proj_b_w', 'in_proj_ab_w',
        'conv1d_w', 'head_norm_w', 'A_log', 'dt_bias',
    }
    full_keys = common_keys | {
        'q_proj_packed', 'q_proj_sf', 'q_proj_global',
        'k_proj_packed', 'k_proj_sf', 'k_proj_global',
        'v_proj_packed', 'v_proj_sf', 'v_proj_global',
        'o_proj_packed', 'o_proj_sf', 'o_proj_global',
        'q_norm_eff_w', 'k_norm_eff_w',
    }
    for L, ld in enumerate(layers):
        if ld['type'] == 'linear_attention':
            missing = lin_keys - set(ld.keys())
            assert not missing, f'lin layer {L} missing {missing}'
        else:
            missing = full_keys - set(ld.keys())
            assert not missing, f'full layer {L} missing {missing}'

    for k in ('embed_w', 'final_norm_eff_w',
              'lm_head_packed', 'lm_head_sf', 'lm_head_alpha', 'vocab_size',
              'hidden', 'num_layers', 'layer_types'):
        assert k in p, f'top-level missing {k!r}'
