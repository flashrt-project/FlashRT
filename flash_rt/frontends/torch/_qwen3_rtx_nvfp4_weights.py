"""FlashRT — Qwen3 (plain dense, e.g. Qwen3-8B) NVFP4 W4A4 raw safetensors loader.

Drop-in for the Qwen3-NVFP4 path. Reads a ``compressed-tensors``
``nvfp4-pack-quantized`` ckpt directly from the (multi-shard)
safetensors files, without depending on ``transformers.AutoModel`` or
``compressed_tensors`` runtime — same lightweight contract as
``_qwen36_rtx_nvfp4_weights``.

**Differences vs the Qwen3.6-27B NVFP4 loader (siblings file):**

* All layers are ``full_attention`` (no Gated-DeltaNet linear-attn).
* No MTP head (no spec-decode for v1).
* Plain Qwen3 RMSNorm: ``out = w * normed`` — NO ``(1+w)`` precompute.
* Key prefixes are ``model.layers.<i>...`` (no ``language_model.``).
* Per-linear ckpt schema gains ``input_global_scale`` (fp32 scalar) —
  the precomputed activation global scale GS_a from
  ``input_activations.dynamic="local"`` calibration. Baked into the
  GEMM ``alpha`` here:

      alpha = 1 / (input_global_scale * weight_global_scale)

  So the FP4 GEMM at run-time reads packed FP4 + per-block UE4M3 SF
  (computed dynamically from the BF16 input by
  ``quantize_bf16_to_nvfp4_swizzled``) and a single host-scalar alpha
  — no per-call division.
* ``lm_head`` is in the ckpt's ``ignore`` list — kept BF16 here, no
  load-time quantization. The frontend will use cuBLAS BF16 mat-vec.
* embed_tokens: BF16, stays unquantized (same as qwen36 path).

Ckpt schema (verified for JunHowie/Qwen3-8B-Instruct-2512-SFT-NVFP4):

    Quantized linear (one per q/k/v/o/gate/up/down × 36 layers = 252):
      <prefix>.weight_packed         u8       (out, in/2)
      <prefix>.weight_scale          f8_e4m3  (out, in/16)
      <prefix>.weight_global_scale   f32      (1,)
      <prefix>.input_global_scale    f32      (1,)        ← NEW

    BF16 (per layer):
      input_layernorm.weight                  bf16  (hidden,)
      post_attention_layernorm.weight         bf16  (hidden,)
      self_attn.q_norm.weight                 bf16  (head_dim,)
      self_attn.k_norm.weight                 bf16  (head_dim,)

    BF16 (top-level):
      model.embed_tokens.weight               bf16  (vocab, hidden)
      model.norm.weight                       bf16  (hidden,)
      lm_head.weight                          bf16  (vocab, hidden)

Returns a :class:`WeightHandles` whose ``ptrs`` mirrors the qwen36
NVFP4 loader's surface for unsurprising reuse:

    Top-level:
      embed_w           (bf16, vocab × hidden)         int ptr
      final_norm_w      (bf16, hidden,)                int ptr
                        — note: ``_w`` not ``_eff_w``: NO (1+w) trick
      lm_head_w         (bf16, vocab × hidden)         int ptr
      vocab_size, hidden, num_layers                   ints
      layer_types                                       list[str] (all "full_attention")
      quant_format                                      "nvfp4"

    Per-layer (every layer is full-attn):
      type=full_attention, quant_format=nvfp4
      input_norm_w        bf16 (hidden,)                — plain w
      post_attn_norm_w    bf16 (hidden,)                — plain w
      q_proj_packed/_sf/_alpha   NVFP4 + alpha=1/(GSa*GSw)
      k_proj_packed/_sf/_alpha
      v_proj_packed/_sf/_alpha
      o_proj_packed/_sf/_alpha
      mlp_gate_packed/_sf/_alpha
      mlp_up_packed/_sf/_alpha
      mlp_down_packed/_sf/_alpha
      q_norm_w           bf16 (head_dim,)
      k_norm_w           bf16 (head_dim,)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class WeightHandles:
    """Same shape as the qwen36 sibling — pointers + anchor list."""

    ptrs: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)


def _anchor(handles: WeightHandles, t: torch.Tensor) -> int:
    handles.anchors.append(t)
    return int(t.data_ptr())


def _bf16_to_dev(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(torch.bfloat16).to(device).contiguous()


def _open_shards(ckpt_dir: str):
    """Open every safetensors shard and return (handles_dict, weight_map).

    weight_map: tensor_full_key -> shard_filename
    handles_dict: shard_filename -> safe_open handle (CPU mmap)
    """
    from safetensors import safe_open

    idx_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    if os.path.isfile(idx_path):
        idx = json.load(open(idx_path))
        wmap = idx['weight_map']
    else:
        wmap = None  # single-file fallback

    if wmap is None:
        single = os.path.join(ckpt_dir, 'model.safetensors')
        if not os.path.isfile(single):
            raise RuntimeError(
                f'Qwen3 NVFP4 ckpt missing index.json AND model.safetensors: '
                f'{ckpt_dir!r}'
            )
        wmap = {}
        with safe_open(single, framework='pt', device='cpu') as f:
            for k in f.keys():
                wmap[k] = 'model.safetensors'

    handles_d = {}
    for shard in set(wmap.values()):
        handles_d[shard] = safe_open(
            os.path.join(ckpt_dir, shard), framework='pt', device='cpu',
        )
    return handles_d, wmap


def _get_tensor(handles_d, wmap, key: str) -> torch.Tensor:
    if key not in wmap:
        raise KeyError(f'tensor {key!r} not in weight_map')
    return handles_d[wmap[key]].get_tensor(key)


def _has(wmap, key: str) -> bool:
    return key in wmap


def _load_quantized_linear(
    handles: WeightHandles,
    out_dict: dict,
    prefix_short: str,
    base_key: str,
    handles_d,
    wmap,
    fvk,
    device: str,
    stream: int = 0,
) -> None:
    """Load 1 NVFP4 quantized linear → 3 ptrs + alpha.

    Reads weight_packed / weight_scale / weight_global_scale /
    input_global_scale; performs the linear→swizzled SF reshape on
    device; bakes alpha = 1/(GSa·GSw).
    """
    packed_cpu = _get_tensor(handles_d, wmap, base_key + '.weight_packed')
    sf_lin_cpu = _get_tensor(handles_d, wmap, base_key + '.weight_scale')
    gsw_cpu = _get_tensor(handles_d, wmap, base_key + '.weight_global_scale')
    gsa_key = base_key + '.input_global_scale'
    if _has(wmap, gsa_key):
        gsa_cpu = _get_tensor(handles_d, wmap, gsa_key)
    else:
        # Some ckpts may omit input_global_scale (treat as 1.0).
        gsa_cpu = torch.tensor([1.0], dtype=torch.float32)

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
        int(sf_lin.data_ptr()), int(sf_swz.data_ptr()),
        rows, cols_in, False, stream,
    )
    del sf_lin, sf_lin_cpu, packed_cpu

    out_dict[prefix_short + '_packed'] = _anchor(handles, packed)
    out_dict[prefix_short + '_sf'] = _anchor(handles, sf_swz)

    gsw = float(gsw_cpu.to(torch.float32).item())
    gsa = float(gsa_cpu.to(torch.float32).item())
    # GEMM alpha convention.
    #
    # compressed-tensors NVFP4 dequant convention:
    #   w_dequant = fp4_w * sf_w / weight_global_scale
    #
    # At runtime the activation quantize kernel
    # (`quantize_bf16_to_nvfp4_swizzled`) reads BF16 input directly and
    # produces per-block UE4M3 SF whose effective decode is
    # `fp4_a * sf_a ≈ a` — i.e. the kernel handles its own normalization
    # without consuming `input_global_scale`. Empirically validated
    # against a torch FP32 dequant matmul on layer-0 q_proj (cos ≥
    # 0.995, magnitude rel-err ≈ 1.3%, which is the FP4 quant noise
    # floor; see commit message for the run).
    #
    # Therefore alpha = 1 / GSw, NOT 1 / (GSa·GSw). `input_global_scale`
    # is captured but unused in the GEMM hot path; we keep `_gsa` as
    # debug info in case a future kernel variant needs it.
    if gsw == 0.0:
        alpha = 0.0
    else:
        alpha = 1.0 / gsw
    out_dict[prefix_short + '_alpha'] = alpha
    out_dict[prefix_short + '_gsw'] = gsw
    out_dict[prefix_short + '_gsa'] = gsa  # captured but unused at runtime


def extract_weights_qwen3_nvfp4(
    ckpt_dir: str,
    fvk,
    device: str = 'cuda:0',
) -> WeightHandles:
    """Build a :class:`WeightHandles` from a Qwen3 NVFP4 ckpt directory.

    Args:
      ckpt_dir: dir with ``config.json`` + sharded ``model-*.safetensors``
        + ``model.safetensors.index.json``.
      fvk: ``flash_rt_kernels`` pybind module (provides
        ``nvfp4_sf_linear_to_swizzled``).
      device: cuda device.
    """
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise RuntimeError(
            f'Qwen3 NVFP4 ckpt missing config.json: {ckpt_dir!r}')
    cfg = json.load(open(cfg_path))

    num_layers = int(cfg['num_hidden_layers'])
    hidden = int(cfg['hidden_size'])
    vocab = int(cfg['vocab_size'])
    head_dim = int(cfg.get('head_dim') or (hidden // cfg['num_attention_heads']))
    n_q = int(cfg['num_attention_heads'])
    n_kv = int(cfg['num_key_value_heads'])
    inter = int(cfg['intermediate_size'])
    rms_eps = float(cfg.get('rms_norm_eps', 1e-6))
    # Plain Qwen3 dense is all full_attention; no layer_types in config.
    layer_types = ['full_attention'] * num_layers
    rope_params = cfg.get('rope_parameters') or cfg.get('rope_scaling') or {}
    rope_theta = float(
        rope_params.get('rope_theta')
        or cfg.get('rope_theta')
        or 1_000_000.0
    )

    handles = WeightHandles()
    per_layer: list[dict] = [None] * num_layers   # type: ignore[list-item]

    debug = bool(int(os.environ.get('FLASHVLA_QWEN3_LOAD_DEBUG', '0') or '0'))

    def _vram_used():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9

    handles_d, wmap = _open_shards(ckpt_dir)
    if debug:
        print(f'  [load] open {len(handles_d)} shards, '
              f'vram = {_vram_used():.2f} GB')

    # Top-level BF16 tensors.
    embed = _bf16_to_dev(
        _get_tensor(handles_d, wmap, 'model.embed_tokens.weight'), device,
    )
    handles.ptrs['embed_w'] = _anchor(handles, embed)

    final_norm = _bf16_to_dev(
        _get_tensor(handles_d, wmap, 'model.norm.weight'), device,
    )
    handles.ptrs['final_norm_w'] = _anchor(handles, final_norm)

    lm_head = _bf16_to_dev(
        _get_tensor(handles_d, wmap, 'lm_head.weight'), device,
    )
    handles.ptrs['lm_head_w'] = _anchor(handles, lm_head)
    handles.ptrs['lm_head_tied'] = bool(cfg.get('tie_word_embeddings', False))

    # Optional NVFP4-quantized lm_head: compute the swizzled
    # representation at load time so a future kernel switch can pick
    # it up without a conversion pass. The ckpt has lm_head in the
    # `ignore` list (BF16 only); the production decode path keeps
    # lm_head BF16 because the W4A4 noise on a 152K-class argmax
    # accumulates over greedy decode steps (see qwen3_rtx.py lm_head
    # call site for details). `bf16_weight_to_nvfp4_swizzled`
    # produces packed + sf_swz + per-tensor global_scale; the
    # corresponding GEMM alpha is `out_gs` itself (the kernel emits
    # the inverse-scaled SF, so the GEMM post-mul = global_scale).
    N_lm, K_lm = lm_head.shape  # (vocab, hidden)
    packed_lm = torch.empty(
        N_lm, K_lm // 2, dtype=torch.uint8, device=device)
    n_blocks = K_lm // 16
    n_row_super = (N_lm + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    sf_bytes_lm = n_row_super * n_col_super * 512
    sf_swz_lm = torch.zeros(
        sf_bytes_lm, dtype=torch.uint8, device=device)
    scratch_amax = torch.zeros(1, dtype=torch.float32, device=device)
    out_gs = torch.zeros(1, dtype=torch.float32, device=device)
    fvk.bf16_weight_to_nvfp4_swizzled(
        int(lm_head.data_ptr()),
        int(packed_lm.data_ptr()), int(sf_swz_lm.data_ptr()),
        int(scratch_amax.data_ptr()), int(out_gs.data_ptr()),
        N_lm, K_lm, 0,
    )
    torch.cuda.synchronize()
    lm_alpha = float(out_gs.item())
    handles.ptrs['lm_head_packed'] = _anchor(handles, packed_lm)
    handles.ptrs['lm_head_sf'] = _anchor(handles, sf_swz_lm)
    handles.ptrs['lm_head_alpha'] = lm_alpha

    handles.ptrs['vocab_size'] = vocab
    handles.ptrs['hidden'] = hidden
    handles.ptrs['head_dim'] = head_dim
    handles.ptrs['num_q_heads'] = n_q
    handles.ptrs['num_kv_heads'] = n_kv
    handles.ptrs['intermediate'] = inter
    handles.ptrs['num_layers'] = num_layers
    handles.ptrs['layer_types'] = layer_types
    handles.ptrs['rms_norm_eps'] = rms_eps
    handles.ptrs['rope_theta'] = rope_theta
    handles.ptrs['rope_parameters'] = rope_params
    handles.ptrs['quant_format'] = 'nvfp4'
    handles.ptrs['ckpt_dir'] = ckpt_dir
    # MTP / spec-decode: not supported on the plain Qwen3 path.
    handles.ptrs['mtp'] = None
    handles.ptrs['dflash'] = None

    # Per-layer.
    for L in range(num_layers):
        base = f'model.layers.{L}.'
        ld: dict = {'type': 'full_attention', 'quant_format': 'nvfp4'}

        # Pre/post layernorms — plain w (NO 1+w).
        in_w = _bf16_to_dev(
            _get_tensor(handles_d, wmap, base + 'input_layernorm.weight'),
            device,
        )
        post_w = _bf16_to_dev(
            _get_tensor(
                handles_d, wmap, base + 'post_attention_layernorm.weight',
            ),
            device,
        )
        ld['input_norm_w'] = _anchor(handles, in_w)
        ld['post_attn_norm_w'] = _anchor(handles, post_w)

        # Self-attn: q/k/v/o NVFP4 + per-head q_norm/k_norm BF16.
        sa = base + 'self_attn.'
        for short, key_short in (('q_proj', 'q_proj'),
                                  ('k_proj', 'k_proj'),
                                  ('v_proj', 'v_proj'),
                                  ('o_proj', 'o_proj')):
            _load_quantized_linear(
                handles, ld, short, sa + key_short,
                handles_d, wmap, fvk, device,
            )
        # ── Fused QKV (decode-path only, M=1 GEMM consolidation) ──
        # The ckpt's q/k/v_proj all share the same activation post-
        # rms_norm and (verified across all 36 layers) yield IDENTICAL
        # weight_global_scale → the GEMM alpha is shared. We therefore
        # concat the packed weights + swizzled SFs along the row axis
        # into one fused tensor and run a single (M=1, N=6144, K=hidden)
        # GEMM at decode time, slicing the output for q/k/v.
        # Saves 2 launches per layer × 36 = 72 kernel launches per
        # decoded token. The byte-concat works because:
        #   * packed weights are (N, K/2) row-major u8 — torch.cat dim=0.
        #   * swizzled SFs are organized in 128-row super-rows, each
        #     `n_col_super * 512` bytes. N_q=4096 = 32 super-rows,
        #     N_k = N_v = 1024 = 8 super-rows each → fused 48 super-rows
        #     = byte-concat of the three swizzled blobs (no re-swizzle).
        q_a = float(ld['q_proj_alpha'])
        k_a = float(ld['k_proj_alpha'])
        v_a = float(ld['v_proj_alpha'])
        qkv_alpha_homogeneous = (
            abs(q_a - k_a) < 1e-12 * max(q_a, 1e-12)
            and abs(q_a - v_a) < 1e-12 * max(q_a, 1e-12)
        )
        if qkv_alpha_homogeneous:
            q_packed_t = handles.anchors[ld['q_proj_packed_idx']] if False else None
            # We don't have anchor indices; re-derive by scanning anchors
            # tail (the last 6 entries appended for q/k/v are
            # packed_q, sf_q, packed_k, sf_k, packed_v, sf_v in order).
            # Safer: rebuild fused by reading raw shards once.
            qkv_packed = torch.cat([
                _get_tensor(handles_d, wmap, sa + 'q_proj.weight_packed').to(device).contiguous(),
                _get_tensor(handles_d, wmap, sa + 'k_proj.weight_packed').to(device).contiguous(),
                _get_tensor(handles_d, wmap, sa + 'v_proj.weight_packed').to(device).contiguous(),
            ], dim=0).contiguous()
            # SFs: read linear, swizzle each, byte-concat (already
            # super-row-aligned). Simpler: read each linear-format SF,
            # concat at SF level (still in linear), then swizzle once.
            qkv_sf_lin = torch.cat([
                _get_tensor(handles_d, wmap, sa + 'q_proj.weight_scale').to(device).contiguous(),
                _get_tensor(handles_d, wmap, sa + 'k_proj.weight_scale').to(device).contiguous(),
                _get_tensor(handles_d, wmap, sa + 'v_proj.weight_scale').to(device).contiguous(),
            ], dim=0).contiguous()
            qkv_N, qkv_K_div16 = qkv_sf_lin.shape
            qkv_K = qkv_K_div16 * 16
            n_blocks = qkv_K // 16
            n_row_super = (qkv_N + 127) // 128
            n_col_super = (n_blocks + 3) // 4
            qkv_sf_swz = torch.zeros(
                n_row_super * n_col_super * 512,
                dtype=torch.uint8, device=device)
            fvk.nvfp4_sf_linear_to_swizzled(
                int(qkv_sf_lin.data_ptr()), int(qkv_sf_swz.data_ptr()),
                qkv_N, qkv_K, False, 0,
            )
            del qkv_sf_lin
            ld['qkv_proj_packed'] = _anchor(handles, qkv_packed)
            ld['qkv_proj_sf'] = _anchor(handles, qkv_sf_swz)
            ld['qkv_proj_alpha'] = q_a
            ld['qkv_proj_N'] = qkv_N         # 6144 = 4096 + 1024 + 1024
            ld['qkv_homogeneous_alpha'] = True
        else:
            ld['qkv_homogeneous_alpha'] = False
        q_norm_w = _bf16_to_dev(
            _get_tensor(handles_d, wmap, sa + 'q_norm.weight'), device,
        )
        k_norm_w = _bf16_to_dev(
            _get_tensor(handles_d, wmap, sa + 'k_norm.weight'), device,
        )
        ld['q_norm_w'] = _anchor(handles, q_norm_w)
        ld['k_norm_w'] = _anchor(handles, k_norm_w)

        # MLP: gate/up/down NVFP4.
        mp = base + 'mlp.'
        for short, key_short in (('mlp_gate', 'gate_proj'),
                                  ('mlp_up', 'up_proj'),
                                  ('mlp_down', 'down_proj')):
            _load_quantized_linear(
                handles, ld, short, mp + key_short,
                handles_d, wmap, fvk, device,
            )
        # ── Fused gate+up MLP (decode-path only) ──
        # Same trick as fused QKV: gate/up share input + same alpha
        # (verified across all 36 layers). Fused N = 2 × intermediate
        # = 24576. One M=1 GEMM replaces two.
        g_a = float(ld['mlp_gate_alpha'])
        u_a = float(ld['mlp_up_alpha'])
        gu_homogeneous = abs(g_a - u_a) < 1e-12 * max(g_a, 1e-12)
        if gu_homogeneous:
            gu_packed = torch.cat([
                _get_tensor(handles_d, wmap,
                             mp + 'gate_proj.weight_packed').to(device).contiguous(),
                _get_tensor(handles_d, wmap,
                             mp + 'up_proj.weight_packed').to(device).contiguous(),
            ], dim=0).contiguous()
            gu_sf_lin = torch.cat([
                _get_tensor(handles_d, wmap,
                             mp + 'gate_proj.weight_scale').to(device).contiguous(),
                _get_tensor(handles_d, wmap,
                             mp + 'up_proj.weight_scale').to(device).contiguous(),
            ], dim=0).contiguous()
            gu_N, gu_K_div16 = gu_sf_lin.shape
            gu_K = gu_K_div16 * 16
            n_blocks = gu_K // 16
            n_row_super = (gu_N + 127) // 128
            n_col_super = (n_blocks + 3) // 4
            gu_sf_swz = torch.zeros(
                n_row_super * n_col_super * 512,
                dtype=torch.uint8, device=device)
            fvk.nvfp4_sf_linear_to_swizzled(
                int(gu_sf_lin.data_ptr()), int(gu_sf_swz.data_ptr()),
                gu_N, gu_K, False, 0,
            )
            del gu_sf_lin
            ld['gate_up_packed'] = _anchor(handles, gu_packed)
            ld['gate_up_sf'] = _anchor(handles, gu_sf_swz)
            ld['gate_up_alpha'] = g_a
            ld['gate_up_N'] = gu_N           # 24576 = 2 × 12288
            ld['gate_up_homogeneous_alpha'] = True
        else:
            ld['gate_up_homogeneous_alpha'] = False

        per_layer[L] = ld
        if debug and (L < 4 or L % 9 == 0):
            torch.cuda.synchronize()
            print(f'  [load] layer {L:2d} done, '
                  f'vram = {_vram_used():.2f} GB')

    handles.ptrs['layers'] = per_layer

    if debug:
        torch.cuda.synchronize()
        print(f'  [load] DONE, vram = {_vram_used():.2f} GB')

    return handles


def assert_extraction_invariants_qwen3(handles: WeightHandles) -> None:
    """Verify all Qwen3-NVFP4 ptr fields populated. Run once at frontend init."""
    p = handles.ptrs
    assert p.get('quant_format') == 'nvfp4', \
        f'unexpected quant_format {p.get("quant_format")!r}'
    layers = p.get('layers')
    assert isinstance(layers, list) and len(layers) == p['num_layers']

    full_keys = {
        'input_norm_w', 'post_attn_norm_w', 'quant_format', 'type',
        'q_proj_packed', 'q_proj_sf', 'q_proj_alpha',
        'k_proj_packed', 'k_proj_sf', 'k_proj_alpha',
        'v_proj_packed', 'v_proj_sf', 'v_proj_alpha',
        'o_proj_packed', 'o_proj_sf', 'o_proj_alpha',
        'q_norm_w', 'k_norm_w',
        'mlp_gate_packed', 'mlp_gate_sf', 'mlp_gate_alpha',
        'mlp_up_packed', 'mlp_up_sf', 'mlp_up_alpha',
        'mlp_down_packed', 'mlp_down_sf', 'mlp_down_alpha',
    }
    for L, ld in enumerate(layers):
        assert ld['type'] == 'full_attention', \
            f'layer {L}: expected full_attention, got {ld["type"]!r}'
        missing = full_keys - set(ld.keys())
        assert not missing, f'layer {L} missing keys: {sorted(missing)}'

    for k in (
        'embed_w', 'final_norm_w', 'lm_head_w',
        'vocab_size', 'hidden', 'head_dim',
        'num_q_heads', 'num_kv_heads', 'intermediate',
        'num_layers', 'layer_types', 'rms_norm_eps', 'rope_theta',
    ):
        assert k in p, f'top-level missing {k!r}'
