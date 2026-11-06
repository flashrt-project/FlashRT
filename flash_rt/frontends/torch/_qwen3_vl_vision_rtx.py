"""FlashRT — Qwen3-VL ViT tower forward on RTX.

Kernelized SigLIP-style vision tower for ``Qwen3-VL``: patch embed →
learned position embedding → ``depth`` transformer blocks (rotate_half
2D-RoPE, full bidirectional FA2 attention, GELU-tanh MLP) → 2×2 patch
merger, with DeepStack feature taps at the configured layers.

All heavy compute runs through ``flash_rt_kernels`` / ``flash_rt_fa2`` /
``flash_rt_qwen3_vl_kernels``; tensor reshapes are metadata-only views.
On SM120 the linears use the CUTLASS block-128 FP8 path; on SM89 eligible
linears use the native Ada block-128 FP8 GEMM. BF16-protected linears use the
architecture-specific BF16 path. In all cases the residual stream (which
carries the late-layer massive-activation outlier) stays bf16.
The tower is prefill-once per image and the sequence length is
image-resolution dependent, so scratch is sized per forward (CUDA-Graph
bucketing by patch count is layered on top via ``forward_graph``).

Sibling of the language path in ``qwen3_rtx`` / ``qwen3_vl_rtx``.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any


class Qwen3VlVisionRtx:
    """Qwen3-VL ViT tower. Loads the vision weights from a checkpoint
    directory and runs the forward against the FlashRT kernel modules.
    Eligible linears pack as FP8 block-128 on SM89/SM120; sensitive
    early-layer paths can stay BF16 via explicit config overrides.

    Public surface:
      __init__(checkpoint_path, *, device='cuda:0', attention_backend='fa2')
      forward(pixel_values, pos_embeds, rope_cos, rope_sin)
        -> (image_embeds, deepstack_features)
    """

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 config: dict | None = None, fp8: bool | None = None,
                 attention_backend: str = 'fa2',
                 bf16_backend: str = 'w16a16',
                 bf16_first_blocks: int | None = None,
                 vla_jepa: bool = False) -> None:
        import torch

        from safetensors import safe_open

        self.device = device
        self._fvk = None
        self._fa2 = None
        self._vlk = None
        self._vit_sdpa_ctx = None  # set by _kernels() for explicit SDPA
        # patch-count -> (graph, static_inputs, captured_outputs)
        self._graphs: dict = {}
        self._graph_stream = None
        self._use_fp8_gemm = False
        if attention_backend not in ('fa2', 'sdpa'):
            raise ValueError(
                f'attention_backend must be "fa2" or "sdpa", got '
                f'{attention_backend!r}')
        self._attention_backend = attention_backend
        if bf16_backend not in ('w16a16', 'cublaslt'):
            raise ValueError(
                f"bf16_backend must be 'w16a16' or 'cublaslt', got {bf16_backend!r}")
        self._bf16_backend = bf16_backend
        # VLA-JEPA opts into the model-specific NVFP4/fusion policy below;
        # every other Qwen3-VL caller keeps the pre-existing tower path.
        self._vla_jepa = bool(vla_jepa)

        cfg = config if config is not None else _read_vision_config(
            checkpoint_path)
        self.depth = int(cfg['depth'])
        self.hidden = int(cfg['hidden_size'])
        self.num_heads = int(cfg['num_heads'])
        self.head_dim = self.hidden // self.num_heads
        self.spatial_merge_size = int(cfg['spatial_merge_size'])
        self.out_hidden = int(cfg['out_hidden_size'])
        self.intermediate = int(cfg['intermediate_size'])
        self.deepstack_indexes = tuple(cfg['deepstack_visual_indexes'])
        self.ln_eps = float(cfg.get('layer_norm_eps', 1e-6))
        self._device_sm = torch.cuda.get_device_capability(
            torch.device(device))[0] * 10 + torch.cuda.get_device_capability(
                torch.device(device))[1]
        self._use_fp8_gemm = self._device_sm >= 89
        if fp8 is not None:
            # Explicit override. Thor (sm_110) passes the sm>=89 test but its
            # Qwen3-VL module omits the FP8 block-128 ViT kernels, so it has to
            # ask for the BF16 tower.
            self._use_fp8_gemm = bool(fp8)

        import json
        import os
        index_path = os.path.join(
            checkpoint_path, 'model.safetensors.index.json')
        shards: dict[str, Any] = {}
        if os.path.isfile(index_path):
            with open(index_path) as f:
                wmap = json.load(f)['weight_map']
        else:
            single = os.path.join(checkpoint_path, 'model.safetensors')
            if not os.path.isfile(single):
                raise RuntimeError(
                    f'Qwen3-VL ckpt missing both index and model.safetensors: '
                    f'{checkpoint_path}')
            h = safe_open(single, framework='pt', device='cpu')
            shards['model.safetensors'] = h
            wmap = {key: 'model.safetensors' for key in h.keys()}

        def _w(key: str):
            shard = wmap[key]
            if shard not in shards:
                shards[shard] = safe_open(
                    os.path.join(checkpoint_path, shard),
                    framework='pt', device='cpu')
            return shards[shard].get_tensor(key).to(
                torch.bfloat16).to(device).contiguous()

        def _lin(w, bias, fp8: bool = True):
            """Pack a linear. fp8=True → (fp8 weight, block scale, bias);
            fp8=False → (None, bf16 weight, bias) for the bf16 path. The
            residual-writing GEMMs (attn proj, FFN fc2) stay bf16: their
            output feeds the late-layer massive-activation channel, where
            accumulated FP8 noise would be amplified."""
            if not fp8 or not self._use_fp8_gemm:
                return (None, w, bias)
            w8, ws = _quant_fp8_block128(w)
            return (w8, ws, bias)

        # FP8 block-128 GEMMs need K % 128 == 0; the ViT intermediate
        # (4304) is not aligned, so fc1 (rows) / fc2 (cols) are zero-padded
        # to the next multiple of 128. Padded fc1 rows/bias and fc2 columns
        # are zero, so GELU(0)=0 keeps the pad inert and the result is
        # unchanged.
        self.intermediate_padded = (
            (self.intermediate + 127) // 128) * 128

        p = 'model.visual.'
        # patch_embed (runs once, at layer 0) and the mergers (the output
        # projection, on the massive-activation hidden) stay bf16: they are
        # cheap and FP8 noise there is amplified through the whole tower.
        # The per-block GEMMs (the compute bulk, 27x) run FP8.
        self.patch_embed = _lin(
            _w(p + 'patch_embed.proj.weight').reshape(self.hidden, -1),
            _w(p + 'patch_embed.proj.bias'), fp8=False)
        self.pos_embed = _w(p + 'pos_embed.weight')

        # FP8 the bulk of the tower, but keep the first few blocks in bf16.
        # The late layers grow a massive-activation channel that amplifies
        # perturbations from the earliest blocks the most; protecting the
        # first 3 blocks recovers image_embeds cosine from 0.86 (all-FP8)
        # to 0.97 for ~+2 ms, vs 0.984 / +21 ms for the all-bf16 tower.
        self.bf16_first_blocks = max(0, int(
            bf16_first_blocks if bf16_first_blocks is not None
            else cfg.get('_bf16_first_blocks', 3)))
        raw_bf16_linears = cfg.get('_bf16_block_linears', {})
        self._bf16_block_linears: dict[int, frozenset[str]] = {}
        allowed_bf16_linears = frozenset(('qkv', 'proj', 'fc1', 'fc2'))
        for raw_idx, raw_names in raw_bf16_linears.items():
            idx = int(raw_idx)
            names = frozenset(str(name) for name in raw_names)
            bad = names - allowed_bf16_linears
            if bad:
                raise ValueError(
                    f'unsupported vision bf16 override(s) for block {idx}: '
                    + ', '.join(sorted(bad)))
            self._bf16_block_linears[idx] = names

        # NVFP4 (W4A4, SM120 block-scaled) policy for the tower linears:
        # ``fp4_sites`` selects the sites (comma list of qkv/proj/fc1/fc2) and
        # ``fp4_blocks`` the blocks ("all" or a comma list). Empty sites keep
        # the delivered FP8/bf16 tower. Weights are packed from the checkpoint's
        # bf16 values at load time, so this is a static, calibrated-in-advance
        # format choice (the per-16 activation scales stay dynamic, as in the
        # decoder's NVFP4 sites).
        if self._vla_jepa:
            # Delivered VLA-JEPA tower policy. Every value below was selected
            # by real-data calibration on the delivery set; the non-VLA branch
            # keeps the pre-existing tower behavior.
            # fc2's GELU(+bias) is fused into the NVFP4 activation producer
            # (byte-identical to the bias_gelu -> quantize pair).
            self._vit_gelu_fuse = True
            # fc1 GEMM + bias + GELU + NVFP4-out as one CUTLASS launch (the
            # repository's up-projection fold).
            self._vit_fc1_gelu_fuse = True
            # Fused qkv split(+bias) + RoPE + V write (one launch instead of two,
            # one pass over packed_qkv; bit-identical to the two-kernel chain).
            self._vit_merger_ln_fuse = True
            self._vit_qkv_post_fuse = True
            # The proj / fc2 residual(+bias) add rides the next LayerNorm->NVFP4
            # producer (one launch and one pass over the residual fewer per site;
            # bit-identical to the residual_add_bias -> LayerNorm chain).
            self._vit_res_ln_fuse = True
            # DeepStack merger MLPs on a forked graph branch: their outputs are
            # first read by the decoder, so they overlap the remaining tower.
            self._vit_ds_side = True
            self._ds_stream = None
            self._ds_hold = []
            self._vit_attn_tile = True
            # Production-target path: the NVFP4 epilogue fuses the proj
            # quantizer into the existing smallq attention tile (byte-
            # identical to bf16 O + standalone quantizer).
            self._vit_attn_fp4out = True
            # Every block takes the fused fc1 epilogue: it drops the bf16
            # rounding of the GELU result (min cosine 0.996608 on the VLA-JEPA
            # delivery set, -0.05 ms).
            self._vit_fc1_fuse_blocks = frozenset(range(self.depth))
            # Merger linears (3 DeepStack + the final merger) on the SM120 NVFP4
            # block-scaled GEMM. The norm output is bounded post-LayerNorm and the
            # merger shapes (M=128, K=4096) measure ~2.6x the FP8 block throughput.
            self._vit_merger_fp4 = True
            # fc1 only: its input is the post-LayerNorm merger activation, whose
            # range the per-16 block scales represent well. fc2 consumes the GELU
            # output, whose large per-channel tails cost accuracy in 4 bits, and
            # the decoder SwiGLU fold buys -0.17 ms for the same margin.
            self._vit_merger_fp4_sites = frozenset(("fc1",))
            self._vit_merger_fp4_scope = frozenset(("deepstack", "final"))
            # Calibrated default (real LIBERO frames, delivered boundary):
            # every tower linear on the SM120 block-scaled NVFP4 GEMM. The
            # full-NVFP4 tower passes both real-data sets (0.9910 min cosine
            # on the 16-frame delivery set, 0.9982 on the disjoint 8-frame
            # extended set) and removes 0.20 ms versus the FP8 residual path.
            if self._use_fp8_gemm and self._device_sm >= 120:
                self.fp4_sites = frozenset(("qkv", "proj", "fc1", "fc2"))
                self._fp4_block_set = frozenset(range(self.depth))
            else:
                self.fp4_sites = frozenset()
                self._fp4_block_set = frozenset()
        else:
            # Non-VLA callers keep the pre-existing Qwen3-VL tower behaviour;
            # the VLA-JEPA fusions and precision defaults are opt-in and must
            # not change another model's output or launch order.
            self._vit_gelu_fuse = False
            self._vit_fc1_gelu_fuse = False
            self._vit_merger_ln_fuse = False
            self._vit_qkv_post_fuse = False
            self._vit_res_ln_fuse = False
            self._vit_ds_side = False
            self._ds_stream = None
            self._ds_hold = []
            self._vit_attn_tile = False
            self._vit_attn_fp4out = False
            self._vit_fc1_fuse_blocks = frozenset()
            self._vit_merger_fp4 = False
            self._vit_merger_fp4_sites = frozenset()
            self._vit_merger_fp4_scope = frozenset()
            self.fp4_sites = frozenset()
            self._fp4_block_set = frozenset()

        self.blocks: list[dict] = []
        for i in range(self.depth):
            bp = f'{p}blocks.{i}.'
            f8 = self._use_fp8_gemm and i >= self.bf16_first_blocks
            bf16_linears = self._bf16_block_linears.get(i, frozenset())
            w_qkv, b_qkv = _w(bp + 'attn.qkv.weight'), _w(bp + 'attn.qkv.bias')
            w_proj, b_proj = _w(bp + 'attn.proj.weight'), _w(bp + 'attn.proj.bias')
            w_fc1 = _pad_rows(_w(bp + 'mlp.linear_fc1.weight'),
                              self.intermediate_padded)
            b_fc1 = _pad_rows(_w(bp + 'mlp.linear_fc1.bias'),
                              self.intermediate_padded)
            w_fc2 = _pad_cols(_w(bp + 'mlp.linear_fc2.weight'),
                              self.intermediate_padded)
            b_fc2 = _w(bp + 'mlp.linear_fc2.bias')
            blk = {
                'norm1_w': _w(bp + 'norm1.weight'),
                'norm1_b': _w(bp + 'norm1.bias'),
                'norm2_w': _w(bp + 'norm2.weight'),
                'norm2_b': _w(bp + 'norm2.bias'),
                'qkv': _lin(w_qkv, b_qkv, fp8=f8 and 'qkv' not in bf16_linears),
                'proj': _lin(w_proj, b_proj, fp8=f8 and 'proj' not in bf16_linears),
                'fc1': _lin(w_fc1, b_fc1, fp8=f8 and 'fc1' not in bf16_linears),
                'fc2': _lin(w_fc2, b_fc2, fp8=f8 and 'fc2' not in bf16_linears),
            }
            if i in self._fp4_block_set:
                self._kernels()          # binds self._fvk
                sites_here = self.fp4_sites
                for site, w_l in (('qkv', w_qkv), ('proj', w_proj),
                                  ('fc1', w_fc1), ('fc2', w_fc2)):
                    if site in sites_here:
                        blk[site + '_fp4'] = _pack_nvfp4_weight(self._fvk, w_l)
            self.blocks.append(blk)
        # Only the VLA-JEPA policy needs the kernel handles here (to pack its
        # NVFP4 weights); every other caller keeps the pre-existing lazy bind
        # in forward().
        if self._vla_jepa:
            self._kernels()
        _fp4_sites = (self._vit_merger_fp4_sites
                      if self._vit_merger_fp4 else frozenset())
        self.merger = _load_merger(
            _w, _lin, p + 'merger.', fvk=self._fvk,
            fp4_sites=(_fp4_sites if 'final' in self._vit_merger_fp4_scope
                       else frozenset()))
        self.deepstack_mergers = [
            _load_merger(
                _w, _lin, f'{p}deepstack_merger_list.{k}.', fvk=self._fvk,
                fp4_sites=(_fp4_sites
                           if 'deepstack' in self._vit_merger_fp4_scope
                           else frozenset()))
            for k in range(len(self.deepstack_indexes))
        ]
        self.intermediate = self.intermediate_padded

    # ── kernel handles (lazy) ──

    def _kernels(self):
        if self._fvk is None:
            if self._attention_backend == 'fa2':
                from flash_rt import flash_rt_fa2 as fa2
            else:
                fa2 = None
            from flash_rt import flash_rt_kernels as fvk
            from flash_rt import flash_rt_qwen3_vl_kernels as vlk
            self._fvk, self._fa2, self._vlk = fvk, fa2, vlk
            import torch
            self._num_sms = torch.cuda.get_device_properties(
                torch.device(self.device)).multi_processor_count
            # Model-local producer selection: VLA-JEPA uses the vectorized
            # entries when the build has them; every other caller keeps the
            # original symbols and therefore the original upstream kernels.
            if self._vla_jepa:
                self._ln_fp8_fn = getattr(
                    vlk, "layer_norm_to_fp8_block128_bf16_v2",
                    vlk.layer_norm_to_fp8_block128_bf16)
                self._gelu_fp8_fn = getattr(
                    vlk, "gelu_tanh_to_fp8_block128_bf16_v2",
                    vlk.gelu_tanh_to_fp8_block128_bf16)
                self._gelu_bias_fp8_fn = getattr(
                    vlk, "gelu_tanh_bias_to_fp8_block128_bf16_v2",
                    vlk.gelu_tanh_bias_to_fp8_block128_bf16)
            else:
                self._ln_fp8_fn = vlk.layer_norm_to_fp8_block128_bf16
                self._gelu_fp8_fn = vlk.gelu_tanh_to_fp8_block128_bf16
                self._gelu_bias_fp8_fn = (
                    vlk.gelu_tanh_bias_to_fp8_block128_bf16)
            # Capability check on the module and symbol only. Do NOT also gate
            # on head_dim: FA2 does instantiate the ViT's head_dim 64 (see
            # FA2_HDIMS), and excluding it would silently reroute the working
            # SM89/SM120 ViT attention to SDPA.
            self._vit_use_fa2 = self._attention_backend == 'fa2'
            if self._vit_use_fa2 and not hasattr(fa2, 'fwd_bf16'):
                raise RuntimeError(
                    'attention_backend="fa2" requires flash_rt_fa2.fwd_bf16')
            # On Thor's explicit SDPA path, prefer the cuDNN backend:
            # at the ViT shape (head_dim 64, non-causal) it measures ~2.3x
            # faster than torch's flash backend there. Probe once here, not on
            # the first forward — the probe launches an SDPA, which must not
            # happen inside a CUDA Graph capture.
            self._vit_sdpa_ctx = None
            if not self._vit_use_fa2:
                try:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    probe = torch.zeros(
                        1, self.num_heads, 8, self.head_dim,
                        device=self.device, dtype=torch.bfloat16)
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        torch.nn.functional.scaled_dot_product_attention(
                            probe, probe, probe)
                    self._vit_sdpa_ctx = (
                        lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION))
                except Exception:
                    self._vit_sdpa_ctx = None
        return self._fvk, self._fa2, self._vlk

    # ── primitives ──

    def _gemm(self, x, lin, M, N, K, stream, add_bias=True):
        """y = x @ W.T + bias via FP8 block-128 (2× tensor-core vs bf16).

        ``lin`` is a (weight_fp8, weight_block_scale, bias) tuple. The
        bf16 activation is dynamically quantized to FP8 with per-token,
        per-128-K-block scales; the GEMM accumulates in FP32 and emits
        bf16. The massive-activation outlier lives in the bf16 residual
        stream (added after the GEMM), so it is unaffected.
        """
        import torch

        fvk = self._fvk
        w8, ws, bias = lin
        y = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
        if w8 is None:                                  # bf16 path (w16a16)
            if self._bf16_backend == 'cublaslt' and hasattr(
                    fvk, 'bf16_matmul_cublaslt_bf16'):
                fvk.bf16_matmul_cublaslt_bf16(
                    x.data_ptr(), ws.data_ptr(), y.data_ptr(), M, N, K, stream)
            elif self._device_sm >= 120:
                fvk.w16a16_gemm_sm120_bf16(
                    x.data_ptr(), ws.data_ptr(), y.data_ptr(), M, N, K, 1.0,
                    stream)
            elif hasattr(self._vlk, 'bf16_matmul_cublaslt_bf16'):
                self._vlk.bf16_matmul_cublaslt_bf16(
                    x.data_ptr(), ws.data_ptr(), y.data_ptr(), M, N, K,
                    stream)
            else:
                fvk.bf16_matmul_bf16(
                    x.data_ptr(), ws.data_ptr(), y.data_ptr(), M, N, K,
                    stream)
        else:                                           # FP8 block-128 W8A8
            x_fp8 = torch.empty(
                M, K, dtype=torch.float8_e4m3fn, device=self.device)
            x_scale = torch.empty(
                M, K // 128, dtype=torch.float32, device=self.device)
            fvk.fp8_per_token_block128_quant_bf16(
                x.data_ptr(), x_fp8.data_ptr(), x_scale.data_ptr(),
                M, K, stream)
            if self._device_sm >= 120:
                fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                    x_fp8.data_ptr(), w8.data_ptr(), y.data_ptr(), M, N, K,
                    x_scale.data_ptr(), ws.data_ptr(), stream)
            else:
                self._vlk.fp8_block128_gemm_blockscaled_sm89_bf16out(
                    x_fp8.data_ptr(), w8.data_ptr(), y.data_ptr(), M, N, K,
                    x_scale.data_ptr(), ws.data_ptr(), stream)
        if bias is not None and add_bias:
            fvk.add_bias_bf16(y.data_ptr(), bias.data_ptr(), M, N, stream)
        return y

    def _layer_norm(self, x, w, b, M, D, stream):
        import torch
        y = torch.empty(M, D, dtype=torch.bfloat16, device=self.device)
        self._fvk.layer_norm(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(),
            M, D, self.ln_eps, stream)
        return y

    def _gemm_fp8(self, x_fp8, x_scale, lin, M, N, K, stream, add_bias=True,
                  schedule=-1):
        """FP8 block-128 GEMM from an already-quantized activation.

        ``x_fp8`` / ``x_scale`` are the FP8 e4m3 values and their per-token,
        per-128-K-block descale, as produced by the fused norm/gelu kernels.
        Same epilogue (bf16 out, optional bias) as the FP8 branch of
        ``_gemm``; it just skips the internal quantization step. ``add_bias``
        False leaves the bias for a downstream fused kernel.
        """
        import torch
        w8, ws, bias = lin
        y = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
        if self._device_sm >= 120:
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                x_fp8.data_ptr(), w8.data_ptr(), y.data_ptr(), M, N, K,
                x_scale.data_ptr(), ws.data_ptr(), stream, schedule)
        else:
            self._vlk.fp8_block128_gemm_blockscaled_sm89_bf16out(
                x_fp8.data_ptr(), w8.data_ptr(), y.data_ptr(), M, N, K,
                x_scale.data_ptr(), ws.data_ptr(), stream)
        if bias is not None and add_bias:
            self._fvk.add_bias_bf16(
                y.data_ptr(), bias.data_ptr(), M, N, stream)
        return y

    # ── NVFP4 (block-scaled W4A4) site path ─────────────────────────────
    def _layer_norm_to_fp4(self, x, w, b, M, D, stream):
        """LayerNorm fused with the NVFP4 (e2m1 + swizzled ue4m3) activation
        quantizer: one launch, the operand the block-scaled NVFP4 GEMM reads."""
        import torch
        xq = torch.empty(M, D // 2, dtype=torch.uint8, device=self.device)
        # ``empty``, not ``zeros``: a fill would be an extra kernel inside the
        # captured graph, and the producer writes every scale byte (M is a
        # multiple of the 128-row swizzle block and D a multiple of 64), so no
        # padding is ever read.
        xsf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(M, D),
                          dtype=torch.uint8, device=self.device)
        self._fvk.layer_norm_to_nvfp4_swizzled_bf16(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), xq.data_ptr(),
            xsf.data_ptr(), M, D, self.ln_eps, stream)
        return xq, xsf

    def _res_layer_norm_to_fp4(self, h, pend, w, b, M, D, stream):
        """``_layer_norm_to_fp4`` of ``h`` after applying the deferred residual
        add ``pend = (x, bias)`` (``h += x + bias``), fused when possible."""
        import torch
        if pend is None or not self._vit_res_ln_fuse or D != 1024:
            self._flush_residual(h, pend, M, D, stream)
            return self._layer_norm_to_fp4(h, w, b, M, D, stream)
        xq = torch.empty(M, D // 2, dtype=torch.uint8, device=self.device)
        xsf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(M, D),
                          dtype=torch.uint8, device=self.device)
        r = self._fvk.residual_add_bias_layer_norm_to_nvfp4_swizzled_bf16(
            h.data_ptr(), pend[0].data_ptr(), pend[1].data_ptr(), w.data_ptr(),
            b.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, D, self.ln_eps,
            stream)
        if r != 0:
            raise RuntimeError(f"residual_add_bias_layer_norm_to_nvfp4 failed: {r}")
        return xq, xsf

    def _flush_residual(self, h, pend, M, D, stream):
        if pend is not None:
            self._vlk.residual_add_bias_bf16(
                h.data_ptr(), pend[0].data_ptr(), pend[1].data_ptr(), M, D, stream)

    def _quant_to_fp4(self, x, M, D, stream):
        """Plain bf16 -> NVFP4 (e2m1 + swizzled ue4m3) activation quantizer."""
        import torch
        xq = torch.empty(M, D // 2, dtype=torch.uint8, device=self.device)
        # ``empty``, not ``zeros``: a fill would be an extra kernel inside the
        # captured graph, and the producer writes every scale byte (M is a
        # multiple of the 128-row swizzle block and D a multiple of 64), so no
        # padding is ever read.
        xsf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(M, D),
                          dtype=torch.uint8, device=self.device)
        # v2 is bit-identical to the v1 kernel (same math, input read once,
        # pipelined loads); it falls back to v1 internally above 4096 columns.
        self._fvk.quantize_bf16_to_nvfp4_swizzled_v2(
            x.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, D, stream)
        return xq, xsf

    def _gelu_bias_to_fp4(self, x, bias, M, D, stream):
        """GELU(+bias) fused into the NVFP4 activation quantizer.

        ``bias_gelu_quant_bf16_to_nvfp4_swizzled_v2_bf16round`` is the
        repository v2 producer with the GELU result rounded through bf16
        before the block amax/scale search, so its output is byte-identical to
        the [bias_gelu -> quantize] pair. It makes one pass over x instead,
        removing a launch and two bf16-sized memory transactions per block.
        """
        if not getattr(self, "_vit_gelu_fuse", True):
            self._fvk.bias_gelu_inplace_bf16(
                x.data_ptr(), bias.data_ptr(), M, D, stream)
            return self._quant_to_fp4(x, M, D, stream)

        import torch

        xq = torch.empty(M, D // 2, dtype=torch.uint8, device=self.device)
        xsf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(M, D),
                          dtype=torch.uint8, device=self.device)
        self._fvk.bias_gelu_quant_bf16_to_nvfp4_swizzled_v2_bf16round(
            x.data_ptr(), bias.data_ptr(), xq.data_ptr(), xsf.data_ptr(),
            M, D, stream)
        return xq, xsf

    def _gemm_bias_gelu_fp4(self, x4, xsf, lin_fp4, bias, M, N, K, stream):
        """NVFP4 GEMM + per-col bias + tanh-GELU + per-block NVFP4 quant in one
        launch (the repository motus/hyvla/GROOT up-projection fold): emits the
        next GEMM's packed operand and its swizzled scales directly, so the
        bf16 fc1 intermediate and its producer launch disappear."""
        import torch
        w4, wsf, alpha = lin_fp4
        y = torch.empty(M, N // 2, dtype=torch.uint8, device=self.device)
        ysf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(M, N),
                          dtype=torch.uint8, device=self.device)
        # VLA-JEPA uses the 128-deep K tile; every other model keeps the
        # upstream 256-deep entry (see the bindings).
        fn = self._fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120
        if self._vla_jepa:
            fn = getattr(
                self._fvk,
                "fp4_w4a16_gemm_bias_gelu_fp4out_sm120_k128", fn)
        fn(
            x4.data_ptr(), w4.data_ptr(), xsf.data_ptr(), wsf.data_ptr(),
            bias.data_ptr(), y.data_ptr(), ysf.data_ptr(), M, N, K, alpha,
            stream)
        return y, ysf

    def _gemm_fp4(self, x4, xsf, lin_fp4, M, N, K, stream, add_bias=True,
                  variant='base'):
        """SM120 block-scaled NVFP4 (W4A4) GEMM: bf16 out, same epilogue as the
        FP8 branch (optional bias).

        ``variant='pingpong'`` selects the repository's pingpong-scheduled
        instantiation of the same tile/kernel family (the qkv shape measures
        ~1.8 us faster per call at M=512)."""
        import torch
        w4, wsf, alpha = lin_fp4
        y = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
        if variant == 'base':
            # VLA-JEPA selects the model-scoped 128-deep K twin; every other
            # caller keeps the shared upstream <128,128,256> entry. The two
            # produce bit-identical output (tests/test_vla_jepa_kernels.py).
            fn = self._fvk.fp4_w4a16_gemm_sm120_bf16out
            if self._vla_jepa:
                fn = getattr(
                    self._fvk, 'fp4_w4a16_gemm_sm120_bf16out_k128', fn)
        else:
            fn = self._fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong
        fn(x4.data_ptr(), w4.data_ptr(), y.data_ptr(), M, N, K,
           xsf.data_ptr(), wsf.data_ptr(), alpha, stream)
        return y

    def _layer_norm_to_fp8(self, x, w, b, M, D, stream):
        """LayerNorm fused with the FP8 block-128 activation quant (one
        kernel, no bf16 round-trip). Returns (fp8 values, per-block scale)
        ready for ``_gemm_fp8``."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._ln_fp8_fn(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), xf.data_ptr(),
            xs.data_ptr(), M, D, self.ln_eps, stream)
        return xf, xs

    def _gelu_to_fp8(self, x, M, D, stream):
        """GELU-tanh fused with the FP8 block-128 activation quant."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._gelu_fp8_fn(
            x.data_ptr(), xf.data_ptr(), xs.data_ptr(), M, D, stream)
        return xf, xs

    def _gelu_bias_to_fp8(self, x, bias, M, D, stream):
        """fc1 bias + GELU-tanh + FP8 block-128 quant in one kernel (the
        bias rides this op instead of a standalone add_bias)."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._gelu_bias_fp8_fn(
            x.data_ptr(), bias.data_ptr(), xf.data_ptr(), xs.data_ptr(),
            M, D, stream)
        return xf, xs

    def _merger_forward(self, h, mg, stream, side=None):
        merge = self.spatial_merge_size * self.spatial_merge_size
        S = h.shape[0]
        norm_dim = mg['norm_w'].shape[0]
        fc1_fp4 = mg.get('fc1_fp4') is not None
        fc2_fp4 = mg.get('fc2_fp4') is not None
        fc1_fp8 = mg['fc1'][0] is not None
        fc2_fp8 = mg['fc2'][0] is not None
        M_merged = S // merge
        xn = xn_s = xn4 = xn4_s = None
        if norm_dim == self.hidden:                    # pre-shuffle norm
            if fc1_fp4:
                # The NVFP4 block scales are swizzled as a function of the
                # GEMM's [M, K], so the merged row cannot reuse the per-row
                # FP8 view: norm on the patch rows, then quantize the merged
                # view as the GEMM's operand.
                xn = self._layer_norm(h, mg['norm_w'], mg['norm_b'], S,
                                      self.hidden, stream)
                xn4, xn4_s = self._quant_to_fp4(
                    xn.view(M_merged, self.hidden * merge), M_merged,
                    self.hidden * merge, stream)
            elif fc1_fp8 and self._vit_merger_ln_fuse:
                # Norm and the FP8 block quant in one launch (the same fold the
                # tower blocks use); the merged row's 128-wide K blocks map
                # exactly onto the per-row blocks, so a view suffices.
                # Model-scoped: non-VLA callers keep the upstream
                # norm-to-bf16 + quantize-two-step chain below.
                xf, xs = self._layer_norm_to_fp8(
                    h, mg['norm_w'], mg['norm_b'], S, self.hidden, stream)
                xn = xf.view(M_merged, self.hidden * merge)
                xn_s = xs.view(M_merged, self.hidden * merge // 128)
            else:
                xn = self._layer_norm(
                    h, mg['norm_w'], mg['norm_b'], S, self.hidden, stream)
                xn = xn.view(M_merged, self.hidden * merge).contiguous()
                xn_s = None
        else:                                          # post-shuffle norm
            xs_in = h.view(M_merged, norm_dim)
            if fc1_fp4:
                xn4, xn4_s = self._layer_norm_to_fp4(
                    xs_in, mg['norm_w'], mg['norm_b'], M_merged, norm_dim,
                    stream)
            elif fc1_fp8 and self._vit_merger_ln_fuse:
                xn, xn_s = self._layer_norm_to_fp8(
                    xs_in, mg['norm_w'], mg['norm_b'], M_merged, norm_dim, stream)
            else:
                xn = self._layer_norm(
                    xs_in, mg['norm_w'], mg['norm_b'], M_merged, norm_dim, stream)
                xn_s = None
        M = M_merged
        if side is not None:
            # The norm read ``h`` on the tower stream; the MLP only reads the
            # norm output, so it runs on a forked branch that overlaps the
            # following tower blocks (joined in ``forward``). fc2 takes the
            # Cooperative FP8 tile there: the 64-row Pingpong tile, run as
            # low-priority work beside the PDL-chained tower, produced
            # replay-to-replay differences in its output (same inputs).
            # Off the critical path the Cooperative tile costs ~0.01 ms.
            import torch
            ev = torch.cuda.Event()
            ev.record(torch.cuda.ExternalStream(stream))
            side.wait_event(ev)
            self._ds_hold.extend(t for t in (xn, xn_s, xn4, xn4_s) if t is not None)
            with torch.cuda.stream(side):
                return self._merger_mlp(xn, xn_s, xn4, xn4_s, mg, M,
                                        side.cuda_stream, fp8_schedule=0)
        return self._merger_mlp(xn, xn_s, xn4, xn4_s, mg, M, stream)

    def _merger_mlp(self, xn, xn_s, xn4, xn4_s, mg, M, stream, fp8_schedule=-1):
        merge = self.spatial_merge_size * self.spatial_merge_size
        fc1_fp4 = mg.get('fc1_fp4') is not None
        fc2_fp4 = mg.get('fc2_fp4') is not None
        fc1_fp8 = mg['fc1'][0] is not None
        fc2_fp8 = mg['fc2'][0] is not None
        din = self.hidden * merge
        b1 = mg['fc1'][2]
        gq = gsf = None
        # fc1 bias rides the bias+GELU producer (no standalone add_bias).
        if fc1_fp4:
            if fc2_fp4 and b1 is not None:
                gq, gsf = self._gemm_bias_gelu_fp4(
                    xn4, xn4_s, mg['fc1_fp4'], b1, M, din, din, stream)
            else:
                f1 = self._gemm_fp4(xn4, xn4_s, mg['fc1_fp4'], M, din, din,
                                    stream, add_bias=False)
        elif fc1_fp8 and xn_s is not None:
            f1 = self._gemm_fp8(xn, xn_s, mg['fc1'], M, din, din, stream,
                                add_bias=False)
        else:
            f1 = self._gemm(xn, mg['fc1'], M, din, din, stream, add_bias=False)
        if fc2_fp4:
            if gq is None:
                if b1 is not None:
                    self._fvk.bias_gelu_inplace_bf16(
                        f1.data_ptr(), b1.data_ptr(), M, din, stream)
                else:
                    self._fvk.gelu_inplace(f1.data_ptr(), M * din, stream)
                gq, gsf = self._quant_to_fp4(f1, M, din, stream)
            out = self._gemm_fp4(gq, gsf, mg['fc2_fp4'], M, self.out_hidden,
                                 din, stream, add_bias=False)
            if mg['fc2'][2] is not None:
                self._fvk.add_bias_bf16(out.data_ptr(), mg['fc2'][2].data_ptr(),
                                        M, self.out_hidden, stream)
            return out
        if fc2_fp8 and self._vit_gelu_fuse:
            # GELU (+bias) and the fc2 operand's FP8 quant in one launch
            # (model-scoped; non-VLA callers keep the two-step chain below).
            if b1 is not None:
                gf, gs = self._gelu_bias_to_fp8(f1, b1, M, din, stream)
            else:
                gf, gs = self._gelu_to_fp8(f1, M, din, stream)
            return self._gemm_fp8(gf, gs, mg['fc2'], M, self.out_hidden, din,
                                  stream, schedule=fp8_schedule)
        if b1 is not None:
            self._fvk.bias_gelu_inplace_bf16(
                f1.data_ptr(), b1.data_ptr(), M, din, stream)
        else:
            self._fvk.gelu_inplace(f1.data_ptr(), M * din, stream)
        return self._gemm(f1, mg['fc2'], M, self.out_hidden, din, stream)

    # ── forward ──

    def forward(self, pixel_values, pos_embeds, rope_cos, rope_sin,
                nviews: int = 1):
        """Run the ViT tower.

        Args:
          nviews: number of independent image views packed along the patch
            axis (row-major, one view per contiguous S/nviews block). The
            tower is per-patch except for attention, which runs as a batched
            FA2 call so views never attend to each other.

          pixel_values: (S, patch_in) bf16 cuda — pre-patchified pixels,
            S = number of patches.
          pos_embeds: (S, hidden) bf16 cuda — grid-interpolated learned
            position embedding (built once per image by the caller).
          rope_cos / rope_sin: (S, head_dim/2) bf16 cuda — the 2D-RoPE
            tables for the patch grid (rotate_half convention).

        Returns:
          (image_embeds (S/merge^2, out_hidden), [deepstack features]).
        """
        import torch

        fvk, fa2, vlk = self._kernels()
        stream = torch.cuda.current_stream().cuda_stream
        S, K = pixel_values.shape
        H = self.hidden
        nh, hd = self.num_heads, self.head_dim
        scale = 1.0 / math.sqrt(hd)
        s_round = ((S + 127) // 128) * 128

        bf16 = torch.bfloat16
        dev = self.device

        # patch embed + grid-interpolated learned pos embed.
        if (self._vla_jepa and self.patch_embed[0] is None
                and self.patch_embed[2] is not None
                and hasattr(fvk, 'add_bias_add_bf16')):
            # bias and the position embedding in one pass (same roundings)
            h = self._gemm(pixel_values, self.patch_embed, S, H, K, stream, add_bias=False)
            r = fvk.add_bias_add_bf16(h.data_ptr(), self.patch_embed[2].data_ptr(),
                                      pos_embeds.data_ptr(), S, H, stream)
            if r != 0:
                raise RuntimeError(f'add_bias_add_bf16 failed: {r}')
        else:
            h = self._gemm(pixel_values, self.patch_embed, S, H, K, stream)
            fvk.residual_add(h.data_ptr(), pos_embeds.data_ptr(), S * H, stream)

        if S % nviews:
            raise ValueError(f'patch count {S} is not divisible by nviews {nviews}')
        P = S // nviews
        s_round_p = ((P + 127) // 128) * 128
        q = torch.empty(S, H, dtype=bf16, device=dev)
        k = torch.empty(S, H, dtype=bf16, device=dev)
        v = torch.empty(S, H, dtype=bf16, device=dev)
        o = torch.empty(nviews, P, nh, hd, dtype=bf16, device=dev)
        lse = torch.empty(nviews, nh, s_round_p, dtype=torch.float32, device=dev)

        deepstack: list = [None] * len(self.deepstack_indexes)
        side = None
        if self._vit_ds_side:
            if self._ds_stream is None:
                self._ds_stream = torch.cuda.Stream(device=dev)
            side = self._ds_stream
            self._ds_hold = []
        # residual add (h += x + bias) deferred into the next NVFP4 LayerNorm
        pend = None
        for i, blk in enumerate(self.blocks):
            # Activation-side fusion follows the linear that consumes it:
            # qkv/fc1 use fused norm->FP8 only when that GEMM is FP8; fc2
            # uses fused GELU->FP8 only when fc2 itself is FP8. This keeps
            # mixed early blocks legal for bring-up experiments.
            qkv_fp4 = blk.get('qkv_fp4') is not None
            qkv_fp8 = blk['qkv'][0] is not None
            if qkv_fp4:
                xq, xsf = self._res_layer_norm_to_fp4(
                    h, pend, blk['norm1_w'], blk['norm1_b'], S, H, stream)
                pend = None
                qkv = self._gemm_fp4(xq, xsf, blk['qkv_fp4'], S, 3 * H, H,
                                     stream, add_bias=False, variant='pingpong')
            elif qkv_fp8:
                self._flush_residual(h, pend, S, H, stream)
                pend = None
                xf, xs = self._layer_norm_to_fp8(
                    h, blk['norm1_w'], blk['norm1_b'], S, H, stream)
                qkv = self._gemm_fp8(xf, xs, blk['qkv'], S, 3 * H, H, stream,
                                     add_bias=False)
            else:
                self._flush_residual(h, pend, S, H, stream)
                pend = None
                xn = self._layer_norm(
                    h, blk['norm1_w'], blk['norm1_b'], S, H, stream)
                qkv = self._gemm(xn, blk['qkv'], S, 3 * H, H, stream,
                                 add_bias=False)
            # qkv bias rides the split (no standalone add_bias); the fused
            # form streams packed_qkv once for split + bias + RoPE + V write.
            if self._vit_qkv_post_fuse:
                vlk.qkv_split_bias_rope_bf16(
                    qkv.data_ptr(), blk['qkv'][2].data_ptr(),
                    rope_cos.data_ptr(), rope_sin.data_ptr(),
                    q.data_ptr(), k.data_ptr(), v.data_ptr(),
                    S, nh, nh, nh, hd, stream)
            else:
                vlk.qkv_split_bias_bf16(
                    qkv.data_ptr(), blk['qkv'][2].data_ptr(), q.data_ptr(),
                    k.data_ptr(), v.data_ptr(), S, H, H, H, stream)
                vlk.rope_neox_qk_bf16(
                    q.data_ptr(), k.data_ptr(), rope_cos.data_ptr(),
                    rope_sin.data_ptr(), q.data_ptr(), k.data_ptr(),
                    S, nh, nh, hd, stream)
            qv = q.view(nviews, P, nh, hd)
            kv = k.view(nviews, P, nh, hd)
            vv = v.view(nviews, P, nh, hd)
            fp4o = (self._vit_attn_fp4out and blk.get('proj_fp4') is not None
                    and self._vit_use_fa2 and self._vit_attn_tile
                    and P % 128 == 0
                    and (hasattr(fa2, "fwd_bf16_tile_fp4out")
                         or getattr(fa2, "has_xvla_tile64",
                                    lambda: False)()))
            if fp4o:
                # Prefer the production-target smallq fp4out tile; older
                # builds fall back to the upstream tile64 entry (batched when
                # available, otherwise one batch=1 call per view).
                ao_q = torch.empty(S, H // 2, dtype=torch.uint8, device=dev)
                ao_sf = torch.empty(self._fvk.nvfp4_sf_swizzled_bytes(S, H),
                                    dtype=torch.uint8, device=dev)
                n_col_blocks = (H // 16 + 3) // 4
                row_q = P * H
                if hasattr(fa2, "fwd_bf16_tile_fp4out"):
                    # Production-target path: the same smallq tile as the
                    # default ViT attention, with the NVFP4 epilogue fused
                    # in. Byte-identical to [bf16 O -> standalone quantizer].
                    fa2.fwd_bf16_tile_fp4out(
                        Q=qv.data_ptr(), K=kv.data_ptr(), V=vv.data_ptr(),
                        O_fp4=ao_q.data_ptr(), O_sf=ao_sf.data_ptr(),
                        softmax_lse=lse.data_ptr(),
                        batch=nviews, seqlen_q=P, seqlen_k=P,
                        num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                        q_strides=(row_q, H, hd),
                        k_strides=(row_q, H, hd),
                        v_strides=(row_q, H, hd),
                        o_strides=(row_q, H, hd),
                        softmax_scale=scale, stream=stream)
                elif (batched := getattr(
                        fa2, "fwd_bf16_tile64_qnvfp4_batched", None)) is not None:
                    # One call for all views; the batched entry writes each
                    # view's packed/SF slice from the same epilogue. Byte-
                    # identical to the per-view calls below.
                    batched(
                        Q=qv.data_ptr(), K=kv.data_ptr(), V=vv.data_ptr(),
                        O=o.data_ptr(), softmax_lse=lse.data_ptr(),
                        seqused_k=0,
                        o_packed=ao_q.data_ptr(), o_sfa=ao_sf.data_ptr(),
                        o_packed_row_bytes=H // 2,
                        o_sfa_n_col_blocks=n_col_blocks,
                        o_packed_batch_stride=P * (H // 2),
                        o_sfa_batch_stride=(P // 128) * n_col_blocks * 512,
                        batch=nviews, seqlen_q=P, seqlen_k=P,
                        num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                        q_strides=(row_q, H, hd),
                        k_strides=(row_q, H, hd),
                        v_strides=(row_q, H, hd),
                        o_strides=(row_q, H, hd),
                        softmax_scale=scale, num_sms=0, stream=stream,
                        enable_pdl=False)
                else:
                    # Older FA2 builds: one batch=1 call per view (the
                    # upstream epilogue historically assumed a single batch).
                    for vi in range(nviews):
                        fa2.fwd_bf16_tile64_qnvfp4(
                            Q=qv.data_ptr() + vi * row_q * 2,
                            K=kv.data_ptr() + vi * row_q * 2,
                            V=vv.data_ptr() + vi * row_q * 2,
                            O=o.data_ptr() + vi * row_q * 2,
                            softmax_lse=lse.data_ptr() + vi * nh * s_round_p * 4,
                            seqused_k=0,
                            o_packed=ao_q.data_ptr() + vi * P * (H // 2),
                            o_sfa=ao_sf.data_ptr()
                                  + vi * (P // 128) * n_col_blocks * 512,
                            o_packed_row_bytes=H // 2,
                            o_sfa_n_col_blocks=n_col_blocks,
                            batch=1, seqlen_q=P, seqlen_k=P,
                            num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                            q_strides=(row_q, H, hd),
                            k_strides=(row_q, H, hd),
                            v_strides=(row_q, H, hd),
                            o_strides=(row_q, H, hd),
                            softmax_scale=scale, num_sms=0, stream=stream,
                            enable_pdl=False)
            elif self._vit_use_fa2 and self._vit_attn_tile:
                # repository short-query FA2 tile entry: 64-row query tile
                # with the vendored 128-wide key tile (bit-identical, ~6%
                # faster at 2 x 256 patches x 16 heads, hd64). The VLA model
                # uses its hd64-enabled entry when available.
                tile_fn = (getattr(fa2, "fwd_bf16_tile_vla", None)
                           if self._vla_jepa else None)
                if tile_fn is None:
                    tile_fn = fa2.fwd_bf16_tile
                tile_fn(
                    Q=qv.data_ptr(), K=kv.data_ptr(), V=vv.data_ptr(),
                    O=o.data_ptr(), softmax_lse=lse.data_ptr(), seqused_k=0,
                    batch=nviews, seqlen_q=P, seqlen_k=P,
                    num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                    q_strides=(qv.stride(0), qv.stride(1), qv.stride(2)),
                    k_strides=(kv.stride(0), kv.stride(1), kv.stride(2)),
                    v_strides=(vv.stride(0), vv.stride(1), vv.stride(2)),
                    o_strides=(o.stride(0), o.stride(1), o.stride(2)),
                    softmax_scale=scale, stream=stream)
            elif self._vit_use_fa2:
                fa2.fwd_bf16(
                    Q=qv.data_ptr(), K=kv.data_ptr(), V=vv.data_ptr(),
                    O=o.data_ptr(), softmax_lse=lse.data_ptr(),
                    softmax_lse_accum=0, o_accum=0, batch=nviews,
                    seqlen_q=P, seqlen_k=P,
                    num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                    q_strides=(qv.stride(0), qv.stride(1), qv.stride(2)),
                    k_strides=(kv.stride(0), kv.stride(1), kv.stride(2)),
                    v_strides=(vv.stride(0), vv.stride(1), vv.stride(2)),
                    o_strides=(o.stride(0), o.stride(1), o.stride(2)),
                    softmax_scale=scale, num_sms=self._num_sms, stream=stream)
            else:
                # flash_rt_fa2 is not built (Thor sm_110). Patch attention is
                # bidirectional, so plain non-causal SDPA is equivalent. The
                # probed cuDNN backend is ~2.3x the flash backend at this
                # shape; fall back to the default dispatcher when the probe
                # failed.
                ctx = (self._vit_sdpa_ctx() if self._vit_sdpa_ctx is not None
                       else contextlib.nullcontext())
                with ctx:
                    oh = torch.nn.functional.scaled_dot_product_attention(
                        qv.transpose(1, 2), kv.transpose(1, 2),
                        vv.transpose(1, 2),
                        attn_mask=None, is_causal=False, scale=scale)
                o.copy_(oh.transpose(1, 2))
            # proj input is the attention output (not a norm/gelu), so its
            # quant cannot be fused upstream; the proj bias rides the residual.
            if blk.get('proj_fp4') is not None:
                if not fp4o:
                    ao = o.view(S, H)
                    ao_q, ao_sf = self._quant_to_fp4(ao, S, H, stream)
                attn = self._gemm_fp4(
                    ao_q, ao_sf, blk['proj_fp4'], S, H, H, stream,
                    add_bias=False)
            elif blk['proj'][0] is not None:
                ao = o.view(S, H).contiguous()
                ao_fp8 = torch.empty(
                    S, H, dtype=torch.float8_e4m3fn, device=dev)
                ao_scale = torch.empty(
                    S, H // 128, dtype=torch.float32, device=dev)
                fvk.fp8_per_token_block128_quant_bf16(
                    ao.data_ptr(), ao_fp8.data_ptr(), ao_scale.data_ptr(),
                    S, H, stream)
                attn = self._gemm_fp8(
                    ao_fp8, ao_scale, blk['proj'], S, H, H, stream,
                    add_bias=False)
            else:
                attn = self._gemm(
                    o.view(S, H), blk['proj'], S, H, H, stream,
                    add_bias=False)
            pend = (attn, blk['proj'][2])

            inter = self.intermediate
            fc1_fp4 = blk.get('fc1_fp4') is not None
            fc2_fp4 = blk.get('fc2_fp4') is not None
            fc1_fp8 = blk['fc1'][0] is not None
            fc2_fp8 = blk['fc2'][0] is not None
            if fc1_fp4:
                xq2, xsf2 = self._res_layer_norm_to_fp4(
                    h, pend, blk['norm2_w'], blk['norm2_b'], S, H, stream)
                pend = None
                if (fc2_fp4 and self._vit_fc1_gelu_fuse
                        and i in self._vit_fc1_fuse_blocks
                        and blk['fc1'][2] is not None):
                    # Fused NVFP4 GEMM + bias + tanh-GELU + per-block NVFP4
                    # quant epilogue (the repository fold motus/hyvla/GROOT use
                    # on this hardware): the bf16 fc1 intermediate and the
                    # separate producer launch both disappear.
                    gq, gsf = self._gemm_bias_gelu_fp4(
                        xq2, xsf2, blk['fc1_fp4'], blk['fc1'][2], S, inter, H,
                        stream)
                    f1 = None
                else:
                    f1 = self._gemm_fp4(xq2, xsf2, blk['fc1_fp4'], S, inter, H,
                                        stream, add_bias=False)
            elif fc1_fp8:
                self._flush_residual(h, pend, S, H, stream)
                pend = None
                xf2, xs2 = self._layer_norm_to_fp8(
                    h, blk['norm2_w'], blk['norm2_b'], S, H, stream)
                f1 = self._gemm_fp8(xf2, xs2, blk['fc1'], S, inter, H, stream,
                                    add_bias=False)
            else:
                self._flush_residual(h, pend, S, H, stream)
                pend = None
                xn2 = self._layer_norm(
                    h, blk['norm2_w'], blk['norm2_b'], S, H, stream)
                f1 = self._gemm(xn2, blk['fc1'], S, inter, H, stream,
                                add_bias=False)
            if fc2_fp4:
                if f1 is not None:
                    gq, gsf = self._gelu_bias_to_fp4(
                        f1, blk['fc1'][2], S, inter, stream)
                f2 = self._gemm_fp4(gq, gsf, blk['fc2_fp4'], S, H, inter,
                                    stream, add_bias=False)
            elif fc2_fp8:
                gf, gs = self._gelu_bias_to_fp8(
                    f1, blk['fc1'][2], S, inter, stream)
                f2 = self._gemm_fp8(gf, gs, blk['fc2'], S, H, inter, stream,
                                    add_bias=False)
            else:
                fvk.bias_gelu_inplace_bf16(
                    f1.data_ptr(), blk['fc1'][2].data_ptr(), S, inter, stream)
                f2 = self._gemm(f1, blk['fc2'], S, H, inter, stream,
                                add_bias=False)
            # fc2 bias rides the residual.
            pend = (f2, blk['fc2'][2])

            if i in self.deepstack_indexes:
                self._flush_residual(h, pend, S, H, stream)
                pend = None
                j = self.deepstack_indexes.index(i)
                deepstack[j] = self._merger_forward(
                    h, self.deepstack_mergers[j], stream, side=side)

        mg = self.merger
        merge = self.spatial_merge_size * self.spatial_merge_size
        # The last residual add, the merger's pre-shuffle LayerNorm and the
        # NVFP4 quantization of the merged fc1 operand in one launch: the
        # merged [S/4, 4H] operand has the patch rows' bytes, only its block
        # scales are laid out for the merged shape.
        fuse_ln = (self._vit_merger_ln_fuse and pend is not None and merge == 4
                   and mg['norm_w'].shape[0] == H == 1024 and S % 512 == 0
                   and mg.get('fc1_fp4') is not None and self._vit_res_ln_fuse)
        if not fuse_ln:
            self._flush_residual(h, pend, S, H, stream)
        if side is not None:
            # join before the final merger: the GEMM entries keep one cached
            # workspace per shape, and the final merger's shapes equal the
            # DeepStack mergers', so the two must not overlap
            ev = torch.cuda.Event()
            ev.record(side)
            torch.cuda.ExternalStream(stream).wait_event(ev)
        if fuse_ln:
            xq = torch.empty(S, H // 2, dtype=torch.uint8, device=dev)
            xsf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(S // merge, H * merge),
                              dtype=torch.uint8, device=dev)
            r = fvk.residual_add_bias_layer_norm_to_nvfp4_swizzled_bf16(
                h.data_ptr(), pend[0].data_ptr(), pend[1].data_ptr(),
                mg['norm_w'].data_ptr(), mg['norm_b'].data_ptr(), xq.data_ptr(),
                xsf.data_ptr(), S, H, self.ln_eps, stream, merge)
            if r != 0:
                raise RuntimeError(f"merged residual LayerNorm->NVFP4 failed: {r}")
            image_embeds = self._merger_mlp(None, None, xq.view(S // merge, H * merge // 2),
                                            xsf, mg, S // merge, stream)
        else:
            image_embeds = self._merger_forward(h, mg, stream)
        return image_embeds, deepstack

    def forward_graph(self, pixel_values, pos_embeds, rope_cos, rope_sin):
        """CUDA-Graph replay of ``forward`` (one graph per patch count).

        Inputs are staged into fixed buffers; returns the captured output
        tensors (valid until the next replay at the same patch count).
        """
        import torch

        self._kernels()
        if self._graph_stream is None:
            self._graph_stream = torch.cuda.Stream(device=self.device)
        S = pixel_values.shape[0]
        g = self._graphs.get(S)
        if g is None:
            si = {
                'pixel': pixel_values.clone(),
                'pos': pos_embeds.clone(),
                'cos': rope_cos.clone(),
                'sin': rope_sin.clone(),
            }
            gs = self._graph_stream
            gs.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(gs), torch.no_grad():
                for _ in range(2):
                    self.forward(si['pixel'], si['pos'], si['cos'], si['sin'])
            gs.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=gs), torch.no_grad():
                out = self.forward(
                    si['pixel'], si['pos'], si['cos'], si['sin'])
            gs.synchronize()
            torch.cuda.current_stream().wait_stream(gs)
            g = (graph, si, out)
            self._graphs[S] = g
        graph, si, out = g
        si['pixel'].copy_(pixel_values)
        si['pos'].copy_(pos_embeds)
        si['cos'].copy_(rope_cos)
        si['sin'].copy_(rope_sin)
        graph.replay()
        return out


def _pack_nvfp4_weight(fvk, w):
    """Pack a (N, K) bf16 weight for the SM120 block-scaled NVFP4 GEMM.

    Returns (packed e2m1 [N, K/2] u8, swizzled ue4m3 SF, alpha) with the same
    layouts the decoder's NVFP4 sites use: the repository's per-16 weight
    quantizer with its 4-step MSE scale search, then the SF swizzle.
    """
    import torch
    N, K = w.shape
    gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
    q = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    sf_lin = torch.empty(N, K // 16, dtype=torch.uint8, device=w.device)
    rc = fvk.nvfp4_skinny_quantize_weight(w.data_ptr(), gs.data_ptr(),
                                          q.data_ptr(), sf_lin.data_ptr(), N, K, 0, 4)
    if rc != 0:
        raise RuntimeError(f"NVFP4 weight quantization failed for [{N},{K}]")
    swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, K), dtype=torch.uint8,
                      device=w.device)
    fvk.nvfp4_sf_linear_to_swizzled(sf_lin.data_ptr(), swz.data_ptr(), N, K, True, 0)
    return (q, swz, float(gs.item()))


def _quant_fp8_block128(w):
    """Quantize a (N, K) bf16 weight to FP8 e4m3 with per-128x128-block
    descales. N and K must be multiples of 128. Returns (w_fp8 (N, K),
    block_scale (N/128, K/128) fp32), matching the convention of
    ``fp8_block128_gemm_cutlass_sm120_bf16out``."""
    import torch
    N, K = w.shape
    wv = w.float().view(N // 128, 128, K // 128, 128)
    amax = wv.abs().amax(dim=(1, 3))
    scale = (amax / 448.0).clamp_min(1e-12)
    wq = (wv / scale[:, None, :, None]).clamp(-448.0, 448.0)
    wq = wq.to(torch.float8_e4m3fn).view(N, K).contiguous()
    return wq, scale.float().contiguous()


def _pad_rows(t, n_rows: int):
    import torch
    if t.shape[0] >= n_rows:
        return t
    pad = torch.zeros(n_rows - t.shape[0], *t.shape[1:],
                      dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0).contiguous()


def _pad_cols(t, n_cols: int):
    import torch
    if t.shape[-1] >= n_cols:
        return t
    pad = torch.zeros(*t.shape[:-1], n_cols - t.shape[-1],
                      dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1).contiguous()


def _load_merger(load_fn, lin_fn, prefix: str, fvk=None,
                 fp4_sites: frozenset = frozenset()) -> dict:
    # The merger linears run FP8 block-128 when 128-aligned (the Qwen3-VL
    # dims are): preflight showed image_embeds cosine 0.9703 -> 0.9698
    # (argmax-safe) for ~-2.6 ms across the four mergers. The norm output is
    # bounded post-LayerNorm, so FP8 here does not touch the massive-
    # activation residual stream. ``fp4_sites`` additionally packs the named
    # linears for the SM120 block-scaled NVFP4 GEMM (measured at ~2.6x the FP8
    # block throughput at the merger shapes).
    def _merger_lin(name):
        w = load_fn(prefix + name + '.weight')
        bias = load_fn(prefix + name + '.bias')
        aligned = w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0
        return lin_fn(w, bias, fp8=aligned)

    d = {
        'norm_w': load_fn(prefix + 'norm.weight'),
        'norm_b': load_fn(prefix + 'norm.bias'),
        'fc1': _merger_lin('linear_fc1'),
        'fc2': _merger_lin('linear_fc2'),
    }
    if fvk is not None:
        for name, key in (('fc1', 'linear_fc1'), ('fc2', 'linear_fc2')):
            if name not in fp4_sites:
                continue
            w = load_fn(prefix + key + '.weight')
            if w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0:
                d[name + '_fp4'] = _pack_nvfp4_weight(fvk, w)
    return d


def _read_vision_config(checkpoint_path: str) -> dict:
    import json
    import os
    with open(os.path.join(checkpoint_path, 'config.json')) as f:
        cfg = json.load(f)
    return cfg['vision_config']
