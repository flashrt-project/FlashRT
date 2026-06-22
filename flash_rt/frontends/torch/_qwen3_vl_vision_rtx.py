"""FlashRT — Qwen3-VL ViT tower forward (RTX SM120, BF16).

Kernelized SigLIP-style vision tower for ``Qwen3-VL``: patch embed →
learned position embedding → ``depth`` transformer blocks (rotate_half
2D-RoPE, full bidirectional FA2 attention, GELU-tanh MLP) → 2×2 patch
merger, with DeepStack feature taps at the configured layers.

All heavy compute runs through ``flash_rt_kernels`` / ``flash_rt_fa2`` /
``flash_rt_qwen3_vl_kernels``; tensor reshapes are metadata-only views.
The tower is prefill-once per image and the sequence length is
image-resolution dependent, so scratch is sized per forward (CUDA-Graph
bucketing by patch count is layered on top separately).

Sibling of the language path in ``qwen3_rtx`` / ``qwen3_vl_rtx``.
"""
from __future__ import annotations

import math
from typing import Any


class Qwen3VlVisionRtx:
    """Qwen3-VL ViT tower. Loads BF16 vision weights from a checkpoint
    directory and runs the forward against the FlashRT kernel modules.

    Public surface:
      __init__(checkpoint_path, *, device='cuda:0')
      forward(pixel_values, rope_cos, rope_sin)
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

        cfg = config if config is not None else _read_vision_config(
            checkpoint_path)
        self.depth = int(cfg['depth'])
        self.hidden = int(cfg['hidden_size'])
        self.num_heads = int(cfg['num_heads'])
        self.head_dim = self.hidden // self.num_heads
        self.spatial_merge_size = int(cfg['spatial_merge_size'])
        self.out_hidden = int(cfg['out_hidden_size'])
        self.deepstack_indexes = tuple(cfg['deepstack_visual_indexes'])
        self.ln_eps = float(cfg.get('layer_norm_eps', 1e-6))

        import json
        import os
        index_path = os.path.join(
            checkpoint_path, 'model.safetensors.index.json')
        wmap = json.load(open(index_path))['weight_map']
        shards: dict[str, Any] = {}

        def _w(key: str):
            shard = wmap[key]
            if shard not in shards:
                shards[shard] = safe_open(
                    os.path.join(checkpoint_path, shard),
                    framework='pt', device='cpu')
            return shards[shard].get_tensor(key).to(
                torch.bfloat16).to(device).contiguous()

        p = 'model.visual.'
        # patch_embed.proj.weight is (hidden, in_ch, T, ph, pw) → (hidden, K)
        self.patch_proj_w = _w(p + 'patch_embed.proj.weight').reshape(
            self.hidden, -1).contiguous()
        self.patch_proj_b = _w(p + 'patch_embed.proj.bias')
        self.pos_embed = _w(p + 'pos_embed.weight')

        self.blocks: list[dict] = []
        for i in range(self.depth):
            bp = f'{p}blocks.{i}.'
            self.blocks.append({
                'norm1_w': _w(bp + 'norm1.weight'),
                'norm1_b': _w(bp + 'norm1.bias'),
                'norm2_w': _w(bp + 'norm2.weight'),
                'norm2_b': _w(bp + 'norm2.bias'),
                'qkv_w': _w(bp + 'attn.qkv.weight'),
                'qkv_b': _w(bp + 'attn.qkv.bias'),
                'proj_w': _w(bp + 'attn.proj.weight'),
                'proj_b': _w(bp + 'attn.proj.bias'),
                'fc1_w': _w(bp + 'mlp.linear_fc1.weight'),
                'fc1_b': _w(bp + 'mlp.linear_fc1.bias'),
                'fc2_w': _w(bp + 'mlp.linear_fc2.weight'),
                'fc2_b': _w(bp + 'mlp.linear_fc2.bias'),
            })
        self.merger = _load_merger(_w, p + 'merger.')
        self.deepstack_mergers = [
            _load_merger(_w, f'{p}deepstack_merger_list.{k}.')
            for k in range(len(self.deepstack_indexes))
        ]

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

    def _gemm(self, x, w, bias, M, N, K, stream):
        import torch
        fvk = self._fvk
        y = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
        if K % 128 == 0:
            fvk.w16a16_gemm_sm120_bf16(
                x.data_ptr(), w.data_ptr(), y.data_ptr(), M, N, K, 1.0, stream)
        else:
            # w16a16 requires K % 128 == 0; the arbitrary-K bf16 matmul
            # also preserves accumulation precision on the ViT FFN
            # (K=intermediate) massive-activation channels.
            fvk.bf16_matmul_qwen36_bf16(
                x.data_ptr(), w.data_ptr(), y.data_ptr(), M, N, K, stream)
        if bias is not None:
            fvk.add_bias_bf16(y.data_ptr(), bias.data_ptr(), M, N, stream)
        return y

    def _layer_norm(self, x, w, b, M, D, stream):
        import torch
        y = torch.empty(M, D, dtype=torch.bfloat16, device=self.device)
        self._fvk.layer_norm(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(),
            M, D, self.ln_eps, stream)
        return y

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
        f1 = self._gemm(xn, mg['fc1_w'], mg['fc1_b'], M, din, din, stream)
        self._fvk.gelu_inplace(f1.data_ptr(), M * din, stream)
        return self._gemm(
            f1, mg['fc2_w'], mg['fc2_b'], M, self.out_hidden, din, stream)

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
        h = self._gemm(pixel_values, self.patch_proj_w, self.patch_proj_b,
                       S, H, K, stream)
        fvk.residual_add(h.data_ptr(), pos_embeds.data_ptr(), S * H, stream)

        q = torch.empty(S, H, dtype=bf16, device=dev)
        k = torch.empty(S, H, dtype=bf16, device=dev)
        v = torch.empty(S, H, dtype=bf16, device=dev)
        o = torch.empty(1, S, nh, hd, dtype=bf16, device=dev)
        lse = torch.empty(1, nh, s_round, dtype=torch.float32, device=dev)

        deepstack: list = [None] * len(self.deepstack_indexes)
        for i, blk in enumerate(self.blocks):
            xn = self._layer_norm(
                h, blk['norm1_w'], blk['norm1_b'], S, H, stream)
            qkv = self._gemm(
                xn, blk['qkv_w'], blk['qkv_b'], S, 3 * H, H, stream)
            fvk.qkv_split(qkv.data_ptr(), q.data_ptr(), k.data_ptr(),
                          v.data_ptr(), S, H, H, H, stream)
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
            attn = self._gemm(o.view(S, H), blk['proj_w'], blk['proj_b'],
                              S, H, H, stream)
            fvk.residual_add(h.data_ptr(), attn.data_ptr(), S * H, stream)

            xn2 = self._layer_norm(
                h, blk['norm2_w'], blk['norm2_b'], S, H, stream)
            inter = blk['fc1_w'].shape[0]
            f1 = self._gemm(
                xn2, blk['fc1_w'], blk['fc1_b'], S, inter, H, stream)
            fvk.gelu_inplace(f1.data_ptr(), S * inter, stream)
            f2 = self._gemm(
                f1, blk['fc2_w'], blk['fc2_b'], S, H, inter, stream)
            fvk.residual_add(h.data_ptr(), f2.data_ptr(), S * H, stream)

            if i in self.deepstack_indexes:
                j = self.deepstack_indexes.index(i)
                deepstack[j] = self._merger_forward(
                    h, self.deepstack_mergers[j], stream)

        image_embeds = self._merger_forward(h, self.merger, stream)
        return image_embeds, deepstack


def _load_merger(load_fn, prefix: str) -> dict:
    return {
        'norm_w': load_fn(prefix + 'norm.weight'),
        'norm_b': load_fn(prefix + 'norm.bias'),
        'fc1_w': load_fn(prefix + 'linear_fc1.weight'),
        'fc1_b': load_fn(prefix + 'linear_fc1.bias'),
        'fc2_w': load_fn(prefix + 'linear_fc2.weight'),
        'fc2_b': load_fn(prefix + 'linear_fc2.bias'),
    }


def _read_vision_config(checkpoint_path: str) -> dict:
    import json
    import os
    cfg = json.load(open(os.path.join(checkpoint_path, 'config.json')))
    return cfg['vision_config']
