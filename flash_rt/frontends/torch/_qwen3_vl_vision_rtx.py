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

import math
from typing import Any


class Qwen3VlVisionRtx:
    """Qwen3-VL ViT tower. Loads the vision weights from a checkpoint
    directory and runs the forward against the FlashRT kernel modules.
    Eligible linears pack as FP8 block-128 on SM89/SM120; sensitive
    early-layer paths can stay BF16 via explicit config overrides.

    Public surface:
      __init__(checkpoint_path, *, device='cuda:0')
      forward(pixel_values, pos_embeds, rope_cos, rope_sin)
        -> (image_embeds, deepstack_features)
    """

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 config: dict | None = None) -> None:
        import torch

        from safetensors import safe_open

        self.device = device
        self._fvk = None
        self._fa2 = None
        self._vlk = None
        # patch-count -> (graph, static_inputs, captured_outputs)
        self._graphs: dict = {}
        self._graph_stream = None
        self._use_fp8_gemm = False

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
        self.bf16_first_blocks = max(0, int(cfg.get('_bf16_first_blocks', 3)))
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

        self.blocks: list[dict] = []
        for i in range(self.depth):
            bp = f'{p}blocks.{i}.'
            f8 = self._use_fp8_gemm and i >= self.bf16_first_blocks
            bf16_linears = self._bf16_block_linears.get(i, frozenset())
            self.blocks.append({
                'norm1_w': _w(bp + 'norm1.weight'),
                'norm1_b': _w(bp + 'norm1.bias'),
                'norm2_w': _w(bp + 'norm2.weight'),
                'norm2_b': _w(bp + 'norm2.bias'),
                'qkv': _lin(_w(bp + 'attn.qkv.weight'),
                            _w(bp + 'attn.qkv.bias'),
                            fp8=f8 and 'qkv' not in bf16_linears),
                'proj': _lin(_w(bp + 'attn.proj.weight'),
                             _w(bp + 'attn.proj.bias'),
                             fp8=f8 and 'proj' not in bf16_linears),
                'fc1': _lin(
                    _pad_rows(_w(bp + 'mlp.linear_fc1.weight'),
                              self.intermediate_padded),
                    _pad_rows(_w(bp + 'mlp.linear_fc1.bias'),
                              self.intermediate_padded),
                    fp8=f8 and 'fc1' not in bf16_linears),
                'fc2': _lin(
                    _pad_cols(_w(bp + 'mlp.linear_fc2.weight'),
                              self.intermediate_padded),
                    _w(bp + 'mlp.linear_fc2.bias'),
                    fp8=f8 and 'fc2' not in bf16_linears),
            })
        self.merger = _load_merger(_w, _lin, p + 'merger.')
        self.deepstack_mergers = [
            _load_merger(_w, _lin, f'{p}deepstack_merger_list.{k}.')
            for k in range(len(self.deepstack_indexes))
        ]
        self.intermediate = self.intermediate_padded

    # ── kernel handles (lazy) ──

    def _kernels(self):
        if self._fvk is None:
            from flash_rt import flash_rt_fa2 as fa2
            from flash_rt import flash_rt_kernels as fvk
            from flash_rt import flash_rt_qwen3_vl_kernels as vlk
            self._fvk, self._fa2, self._vlk = fvk, fa2, vlk
            import torch
            self._num_sms = torch.cuda.get_device_properties(
                torch.device(self.device)).multi_processor_count
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
            if self._device_sm >= 120:
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

    def _gemm_fp8(self, x_fp8, x_scale, lin, M, N, K, stream, add_bias=True):
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
                x_scale.data_ptr(), ws.data_ptr(), stream)
        else:
            self._vlk.fp8_block128_gemm_blockscaled_sm89_bf16out(
                x_fp8.data_ptr(), w8.data_ptr(), y.data_ptr(), M, N, K,
                x_scale.data_ptr(), ws.data_ptr(), stream)
        if bias is not None and add_bias:
            self._fvk.add_bias_bf16(
                y.data_ptr(), bias.data_ptr(), M, N, stream)
        return y

    def _layer_norm_to_fp8(self, x, w, b, M, D, stream):
        """LayerNorm fused with the FP8 block-128 activation quant (one
        kernel, no bf16 round-trip). Returns (fp8 values, per-block scale)
        ready for ``_gemm_fp8``."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._vlk.layer_norm_to_fp8_block128_bf16(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), xf.data_ptr(),
            xs.data_ptr(), M, D, self.ln_eps, stream)
        return xf, xs

    def _gelu_to_fp8(self, x, M, D, stream):
        """GELU-tanh fused with the FP8 block-128 activation quant."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._vlk.gelu_tanh_to_fp8_block128_bf16(
            x.data_ptr(), xf.data_ptr(), xs.data_ptr(), M, D, stream)
        return xf, xs

    def _gelu_bias_to_fp8(self, x, bias, M, D, stream):
        """fc1 bias + GELU-tanh + FP8 block-128 quant in one kernel (the
        bias rides this op instead of a standalone add_bias)."""
        import torch
        xf = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=self.device)
        xs = torch.empty(M, D // 128, dtype=torch.float32, device=self.device)
        self._vlk.gelu_tanh_bias_to_fp8_block128_bf16(
            x.data_ptr(), bias.data_ptr(), xf.data_ptr(), xs.data_ptr(),
            M, D, stream)
        return xf, xs

    def _merger_forward(self, h, mg, stream):
        merge = self.spatial_merge_size * self.spatial_merge_size
        S = h.shape[0]
        norm_dim = mg['norm_w'].shape[0]
        if norm_dim == self.hidden:                    # pre-shuffle norm
            xn = self._layer_norm(
                h, mg['norm_w'], mg['norm_b'], S, self.hidden, stream)
            xn = xn.view(S // merge, self.hidden * merge).contiguous()
        else:                                          # post-shuffle norm
            xs = h.view(S // merge, norm_dim).contiguous()
            xn = self._layer_norm(
                xs, mg['norm_w'], mg['norm_b'], xs.shape[0], norm_dim, stream)
        M = xn.shape[0]
        din = self.hidden * merge
        # fc1 bias rides the bias+GELU kernel (no standalone add_bias).
        f1 = self._gemm(xn, mg['fc1'], M, din, din, stream, add_bias=False)
        b1 = mg['fc1'][2]
        if b1 is not None:
            self._fvk.bias_gelu_inplace_bf16(
                f1.data_ptr(), b1.data_ptr(), M, din, stream)
        else:
            self._fvk.gelu_inplace(f1.data_ptr(), M * din, stream)
        return self._gemm(f1, mg['fc2'], M, self.out_hidden, din, stream)

    # ── forward ──

    def forward(self, pixel_values, pos_embeds, rope_cos, rope_sin):
        """Run the ViT tower.

        Args:
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
        h = self._gemm(pixel_values, self.patch_embed, S, H, K, stream)
        fvk.residual_add(h.data_ptr(), pos_embeds.data_ptr(), S * H, stream)

        q = torch.empty(S, H, dtype=bf16, device=dev)
        k = torch.empty(S, H, dtype=bf16, device=dev)
        v = torch.empty(S, H, dtype=bf16, device=dev)
        o = torch.empty(1, S, nh, hd, dtype=bf16, device=dev)
        lse = torch.empty(1, nh, s_round, dtype=torch.float32, device=dev)

        deepstack: list = [None] * len(self.deepstack_indexes)
        for i, blk in enumerate(self.blocks):
            # Activation-side fusion follows the linear that consumes it:
            # qkv/fc1 use fused norm->FP8 only when that GEMM is FP8; fc2
            # uses fused GELU->FP8 only when fc2 itself is FP8. This keeps
            # mixed early blocks legal for bring-up experiments.
            qkv_fp8 = blk['qkv'][0] is not None
            if qkv_fp8:
                xf, xs = self._layer_norm_to_fp8(
                    h, blk['norm1_w'], blk['norm1_b'], S, H, stream)
                qkv = self._gemm_fp8(xf, xs, blk['qkv'], S, 3 * H, H, stream,
                                     add_bias=False)
            else:
                xn = self._layer_norm(
                    h, blk['norm1_w'], blk['norm1_b'], S, H, stream)
                qkv = self._gemm(xn, blk['qkv'], S, 3 * H, H, stream,
                                 add_bias=False)
            # qkv bias rides the split (no standalone add_bias).
            vlk.qkv_split_bias_bf16(
                qkv.data_ptr(), blk['qkv'][2].data_ptr(), q.data_ptr(),
                k.data_ptr(), v.data_ptr(), S, H, H, H, stream)
            vlk.rope_neox_qk_bf16(
                q.data_ptr(), k.data_ptr(), rope_cos.data_ptr(),
                rope_sin.data_ptr(), q.data_ptr(), k.data_ptr(),
                S, nh, nh, hd, stream)
            qv = q.view(1, S, nh, hd)
            kv = k.view(1, S, nh, hd)
            vv = v.view(1, S, nh, hd)
            fa2.fwd_bf16(
                Q=qv.data_ptr(), K=kv.data_ptr(), V=vv.data_ptr(),
                O=o.data_ptr(), softmax_lse=lse.data_ptr(),
                softmax_lse_accum=0, o_accum=0, batch=1,
                seqlen_q=S, seqlen_k=S,
                num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                q_strides=(qv.stride(0), qv.stride(1), qv.stride(2)),
                k_strides=(kv.stride(0), kv.stride(1), kv.stride(2)),
                v_strides=(vv.stride(0), vv.stride(1), vv.stride(2)),
                o_strides=(o.stride(0), o.stride(1), o.stride(2)),
                softmax_scale=scale, num_sms=self._num_sms, stream=stream)
            # proj input is the attention output (not a norm/gelu), so its
            # quant cannot be fused upstream; the proj bias rides the residual.
            if blk['proj'][0] is not None:
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
            vlk.residual_add_bias_bf16(
                h.data_ptr(), attn.data_ptr(), blk['proj'][2].data_ptr(),
                S, H, stream)

            inter = self.intermediate
            fc1_fp8 = blk['fc1'][0] is not None
            fc2_fp8 = blk['fc2'][0] is not None
            if fc1_fp8:
                xf2, xs2 = self._layer_norm_to_fp8(
                    h, blk['norm2_w'], blk['norm2_b'], S, H, stream)
                f1 = self._gemm_fp8(xf2, xs2, blk['fc1'], S, inter, H, stream,
                                    add_bias=False)
            else:
                xn2 = self._layer_norm(
                    h, blk['norm2_w'], blk['norm2_b'], S, H, stream)
                f1 = self._gemm(xn2, blk['fc1'], S, inter, H, stream,
                                add_bias=False)
            if fc2_fp8:
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
            vlk.residual_add_bias_bf16(
                h.data_ptr(), f2.data_ptr(), blk['fc2'][2].data_ptr(),
                S, H, stream)

            if i in self.deepstack_indexes:
                j = self.deepstack_indexes.index(i)
                deepstack[j] = self._merger_forward(
                    h, self.deepstack_mergers[j], stream)

        image_embeds = self._merger_forward(h, self.merger, stream)
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


def _load_merger(load_fn, lin_fn, prefix: str) -> dict:
    # The merger linears run FP8 block-128 when 128-aligned (the Qwen3-VL
    # dims are): preflight showed image_embeds cosine 0.9703 -> 0.9698
    # (argmax-safe) for ~-2.6 ms across the four mergers. The norm output is
    # bounded post-LayerNorm, so FP8 here does not touch the massive-
    # activation residual stream.
    def _merger_lin(name):
        w = load_fn(prefix + name + '.weight')
        bias = load_fn(prefix + name + '.bias')
        aligned = w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0
        return lin_fn(w, bias, fp8=aligned)

    return {
        'norm_w': load_fn(prefix + 'norm.weight'),
        'norm_b': load_fn(prefix + 'norm.bias'),
        'fc1': _merger_lin('linear_fc1'),
        'fc2': _merger_lin('linear_fc2'),
    }


def _read_vision_config(checkpoint_path: str) -> dict:
    import json
    import os
    with open(os.path.join(checkpoint_path, 'config.json')) as f:
        cfg = json.load(f)
    return cfg['vision_config']
