"""Qwen3-VL official-BF16 multimodal frontend for the RTX-family backend.

First bring-up target:
  * official BF16 Qwen3-VL-2B-Instruct checkpoint
  * single-image chat prompt
  * greedy generation
  * CUDA Graph replay for fixed-shape prefill/decode buckets
  * no server/load_model registration yet

The vision tower reuses ``Qwen3VlVisionRtx`` in its BF16 mode. The language
stack is a direct BF16 Qwen3 decoder over generic FlashRT kernels.
This mirrors the project's precision-variant pattern: the optimized
NVFP4/FP8 frontends remain separate, while this file keeps an official BF16
checkpoint path for correctness bring-up and unsupported quantized targets.
"""
from __future__ import annotations

import collections
import json
import os
from typing import Any


_QWEN3_VL_RTX_BF16_VISION_FNS = (
    'rope_neox_qk_bf16',
    'residual_add_bias_bf16',
    'qkv_split_bias_bf16',
    'bf16_matmul_cublaslt_bf16',
    'qwen3_vl_bf16_gemv_m1',
)

_QWEN3_VL_RTX_BF16_CORE_FNS = (
    'embedding_lookup_bf16',
    'bf16_matmul_bf16',
    'rms_norm',
    'residual_add',
    'residual_add_rms_norm',
    'silu_mul_qwen36_bf16',
    'qwen3_q_norm_rope_qstage_bf16',
    'qwen3_k_norm_rope_kvwrite_bf16',
    'qwen3_q_norm_rope_qstage_prefill_bf16',
    'qwen3_k_norm_rope_kvwrite_prefill_bf16',
)


def _require_qwen3_vl_rtx_bf16_kernels():
    try:
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_qwen3_vl_kernels as vlk
    except ImportError as e:
        raise RuntimeError(
            'Qwen3VlTorchFrontendRtxBF16 requires flash_rt_kernels and '
            'flash_rt_qwen3_vl_kernels. Configure with '
            '-DGPU_ARCH=87 -DFLASHRT_BUILD_QWEN3_VL=ON and build '
            'flash_rt_kernels flash_rt_fa2 flash_rt_qwen3_vl_kernels.'
        ) from e
    missing_core = [name for name in _QWEN3_VL_RTX_BF16_CORE_FNS
                    if not hasattr(fvk, name)]
    missing_vl = [name for name in _QWEN3_VL_RTX_BF16_VISION_FNS
                  if not hasattr(vlk, name)]
    if missing_core or missing_vl:
        pieces = []
        if missing_core:
            pieces.append('flash_rt_kernels: ' + ', '.join(missing_core))
        if missing_vl:
            pieces.append('flash_rt_qwen3_vl_kernels: ' + ', '.join(missing_vl))
        raise RuntimeError(
            'Qwen3-VL RTX BF16 kernels are incomplete: ' + '; '.join(pieces))
    return fvk, vlk


class Qwen3VlTorchFrontendRtxBF16:
    """Batch-1 Qwen3-VL image+text inference with official BF16 weights.

    This is the BF16 precision variant for the RTX-family backend. The first
    validated target is Jetson Orin SM87, where the FP8/NVFP4 Qwen3-VL paths
    are not available.
    """

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 max_seq: int = 2048, max_pixels: int | None = None,
                 max_prefill_graphs: int | None = None,
                 max_decode_graphs: int | None = None) -> None:
        import torch
        from transformers import AutoProcessor

        from flash_rt.frontends.torch._qwen3_vl_bf16_weights import (
            assert_extraction_invariants_qwen3_vl_bf16,
            extract_weights_qwen3_vl_bf16,
        )
        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
            Qwen3VlVisionRtx,
        )
        from flash_rt.hardware.rtx.attn_backend_qwen3 import (
            RtxFlashAttnBackendQwen3,
        )

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        self.max_pixels = max_pixels
        self._prompt: dict[str, Any] | None = None
        if max_prefill_graphs is None:
            max_prefill_graphs = int(os.environ.get(
                'FLASHRT_QWEN3_VL_PREFILL_GRAPH_CACHE_MAX', '256'))
        if max_decode_graphs is None:
            max_decode_graphs = int(os.environ.get(
                'FLASHRT_QWEN3_VL_DECODE_GRAPH_CACHE_MAX', '256'))
        self.max_prefill_graphs = int(max_prefill_graphs)
        self.max_decode_graphs = int(max_decode_graphs)
        self._decode_graphs: collections.OrderedDict[tuple[int, int], Any] = (
            collections.OrderedDict())
        self._prefill_graphs: collections.OrderedDict[
            tuple[int, int, int, int], Any] = collections.OrderedDict()
        self._pg_buffers: collections.OrderedDict[
            tuple[int, int, int, int], dict[str, Any]] = (
                collections.OrderedDict())

        device_obj = torch.device(device)
        if device_obj.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Qwen3VlTorchFrontendRtxBF16 requires CUDA')
        cap = torch.cuda.get_device_capability(device_obj)
        if cap != (8, 7):
            raise RuntimeError(
                'Qwen3VlTorchFrontendRtxBF16 targets SM87 currently; '
                f'got sm_{cap[0]}{cap[1]} on {device}')

        self._fvk, self._vlk = _require_qwen3_vl_rtx_bf16_kernels()
        with open(os.path.join(self.checkpoint_path, 'config.json')) as f:
            self._cfg_raw = json.load(f)
        cfg = self._cfg_raw
        text_cfg = cfg['text_config']
        self._cfg = {
            'rms_norm_eps': float(text_cfg.get('rms_norm_eps', 1e-6)),
            'head_dim': int(text_cfg.get('head_dim') or
                            text_cfg['hidden_size'] //
                            text_cfg['num_attention_heads']),
            'hidden_size': int(text_cfg['hidden_size']),
            'vocab_size': int(text_cfg['vocab_size']),
            'num_hidden_layers': int(text_cfg['num_hidden_layers']),
            'num_q_heads': int(text_cfg['num_attention_heads']),
            'num_kv_heads': int(text_cfg['num_key_value_heads']),
            'intermediate': int(text_cfg['intermediate_size']),
            'rope_theta': float(text_cfg.get('rope_theta')
                                or cfg.get('rope_theta') or 1_000_000.0),
        }
        self._head_dim = self._cfg['head_dim']
        if self._head_dim != 128:
            raise RuntimeError(
                f'BF16 path requires head_dim=128, got {self._head_dim}')
        self._image_token_id = int(cfg['image_token_id'])
        self._video_token_id = int(cfg['video_token_id'])
        self._vision_start_token_id = int(cfg['vision_start_token_id'])
        vc = cfg['vision_config']
        self._merge = int(vc['spatial_merge_size'])
        self._vis_head_dim = int(vc['hidden_size']) // int(vc['num_heads'])
        self._num_grid_per_side = int(vc['num_position_embeddings'] ** 0.5)
        self._deepstack_layers = len(vc['deepstack_visual_indexes'])
        rope_scaling = text_cfg.get('rope_scaling') or cfg.get('rope_scaling')
        if not rope_scaling or 'mrope_section' not in rope_scaling:
            raise RuntimeError('Qwen3-VL config missing rope_scaling.mrope_section')
        self._mrope_section = tuple(rope_scaling['mrope_section'])
        eos = cfg.get('eos_token_id', text_cfg.get('eos_token_id'))
        if eos is None:
            self._eos_token_ids: set[int] = set()
        else:
            self._eos_token_ids = set(eos if isinstance(eos, list) else [eos])

        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        self._processor_kwargs = {'device': self.device}
        if max_pixels is not None:
            size = getattr(getattr(self.processor, 'image_processor', None),
                           'size', {})
            shortest = int(size.get('shortest_edge', 65536))
            self._processor_kwargs['size'] = {
                'shortest_edge': shortest,
                'longest_edge': int(max_pixels),
            }

        self._weights = extract_weights_qwen3_vl_bf16(
            self.checkpoint_path, device=self.device)
        assert_extraction_invariants_qwen3_vl_bf16(self._weights)
        self.vision = Qwen3VlVisionRtx(
            self.checkpoint_path, device=device, config=vc)
        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=self.max_seq, max_q_seq=self.max_seq, dtype=torch.bfloat16,
            num_layers=self._cfg['num_hidden_layers'],
            num_q_heads=self._cfg['num_q_heads'],
            num_kv_heads=self._cfg['num_kv_heads'],
            head_dim=self._cfg['head_dim'],
            device=self.device)
        self._alloc_buffers()
        self._build_mrope_caches()

    _PG_KEYS = ('input_ids', 'pixel_values', 'pos_embeds',
                'vcos', 'vsin', 'mcos', 'msin')

    def _graph_cache_get(self, cache, key):
        graph = cache.get(key)
        if graph is not None and isinstance(cache, collections.OrderedDict):
            cache.move_to_end(key)
        return graph

    def _trim_lru_graph_cache(self, cache, max_entries: int,
                              on_evict=None) -> None:
        if max_entries <= 0 or not isinstance(cache, collections.OrderedDict):
            return
        while len(cache) > max_entries:
            old_key, _ = cache.popitem(last=False)
            if on_evict is not None:
                on_evict(old_key)

    def _alloc_buffers(self) -> None:
        import torch

        cfg = self._cfg
        d = torch.device(self.device)
        bf16 = torch.bfloat16
        S = self.max_seq
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        qkv_n = (NQ + 2 * NKV) * HD
        self._qkv_N = qkv_n
        self._h_a = torch.empty(1, S, H, device=d, dtype=bf16)
        self._h_b = torch.empty(1, S, H, device=d, dtype=bf16)
        self._qkv_out = torch.empty(S, qkv_n, device=d, dtype=bf16)
        self._gate_up = torch.empty(S, 2 * I, device=d, dtype=bf16)
        self._gate_tmp = torch.empty(S, I, device=d, dtype=bf16)
        self._up_tmp = torch.empty(S, I, device=d, dtype=bf16)
        self._mlp_act = torch.empty(S, I, device=d, dtype=bf16)
        self._tmp_hidden = torch.empty(S, H, device=d, dtype=bf16)
        self._norm_buf = torch.empty(S, H, device=d, dtype=bf16)
        self._logits = torch.empty(1, cfg['vocab_size'], device=d, dtype=bf16)
        self._static_token_id = torch.zeros(1, 1, device=d, dtype=torch.long)
        self._graph_stream = torch.cuda.Stream(device=d)

    def _build_mrope_caches(self) -> None:
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        self._mrope_cos_cache, self._mrope_sin_cache = geo.build_mrope_cache(
            max_pos=self.max_seq + self._num_grid_per_side,
            head_dim=self._head_dim, rope_theta=self._cfg['rope_theta'],
            device=self.device)
        self._vision_rope_cos_cache, self._vision_rope_sin_cache = (
            geo.build_vision_rope_cache(
                max_hw=self.max_seq * self._merge,
                head_dim=self._vis_head_dim, device=self.device))

    def reset_state(self) -> None:
        self._attn.reset_cache()

    def _stage_prefill_inputs(self, P: int, S: int, span):
        p = self._prompt
        assert p is not None
        key = (int(P), int(S), int(span[0]), int(span[1]))
        if not isinstance(self._pg_buffers, collections.OrderedDict):
            self._pg_buffers = collections.OrderedDict(self._pg_buffers)
        if not isinstance(self._prefill_graphs, collections.OrderedDict):
            self._prefill_graphs = collections.OrderedDict(
                self._prefill_graphs)
        bufs = self._pg_buffers.get(key)
        if bufs is None:
            cap = self.max_prefill_graphs
            while cap > 0 and len(self._pg_buffers) >= cap:
                old_key, _ = self._pg_buffers.popitem(last=False)
                self._prefill_graphs.pop(old_key, None)
            bufs = {k: p[k].clone() for k in self._PG_KEYS}
            self._pg_buffers[key] = bufs
        else:
            self._pg_buffers.move_to_end(key)
            for k in self._PG_KEYS:
                bufs[k].copy_(p[k])
        return key

    def clear_graphs(self) -> None:
        """Drop captured Qwen3-VL BF16 prefill/decode CUDA Graphs."""
        for attr in ('_prefill_graphs', '_decode_graphs', '_pg_buffers'):
            cache = getattr(self, attr, None)
            if cache:
                cache.clear()

    def graph_cache_stats(self) -> dict[str, Any]:
        """Return lightweight BF16 CUDA Graph cache diagnostics."""
        return {
            'prefill': {
                'max_graphs': self.max_prefill_graphs,
                'graph_count': len(self._prefill_graphs),
                'buffer_count': len(self._pg_buffers),
                'graph_keys': list(self._prefill_graphs.keys()),
                'buffer_keys': list(self._pg_buffers.keys()),
            },
            'decode': {
                'max_graphs': self.max_decode_graphs,
                'graph_count': len(self._decode_graphs),
                'graph_keys': list(self._decode_graphs.keys()),
            },
        }

    def set_prompt(self, messages: list) -> None:
        """Preprocess a single-image Qwen3-VL chat prompt."""
        import torch
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt',
            processor_kwargs=self._processor_kwargs).to(self.device)
        input_ids = inputs['input_ids'][0]
        S = int(input_ids.shape[0])
        if S > self.max_seq:
            raise ValueError(
                f'prompt length {S} exceeds max_seq {self.max_seq}')

        image_grid = inputs.get('image_grid_thw')
        video_grid = inputs.get('video_grid_thw')
        if video_grid is not None and len(video_grid):
            raise ValueError('Qwen3VlTorchFrontendRtxBF16 v1 supports images, not video')
        pix_img = inputs.get('pixel_values')
        if pix_img is None:
            raise ValueError('Qwen3VlTorchFrontendRtxBF16 requires one image')
        pix_img = pix_img.to(torch.bfloat16)

        segs = geo.vision_segments(
            input_ids.cpu(), image_grid, None,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            spatial_merge_size=self._merge)
        if len(segs) != 1:
            raise ValueError(
                f'Qwen3VlTorchFrontendRtxBF16 v1 supports exactly one image; '
                f'got {len(segs)} vision segments')
        sg = segs[0]
        n_patch = int(sg['patches'])
        pixel_values = pix_img[:n_patch].contiguous()
        seg_grid = torch.tensor([sg['grid']], dtype=torch.long)
        pos_ids = geo.mrope_position_ids(
            input_ids.cpu(), image_grid.cpu(), None,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            vision_start_token_id=self._vision_start_token_id,
            spatial_merge_size=self._merge)
        mcos, msin = geo.mrope_cos_sin_cached(
            pos_ids, self._mrope_cos_cache, self._mrope_sin_cache,
            mrope_section=self._mrope_section)
        vcos, vsin = geo.vision_rope_cos_sin_cached(
            seg_grid, self._vision_rope_cos_cache, self._vision_rope_sin_cache,
            spatial_merge_size=self._merge)
        pos_embeds = geo.vision_pos_embeds(
            seg_grid, self.vision.pos_embed,
            num_grid_per_side=self._num_grid_per_side,
            spatial_merge_size=self._merge, device=self.device)

        self._prompt = {
            'input_ids': input_ids.contiguous(),
            'pixel_values': pixel_values,
            'span': sg['span'],
            'patches': n_patch,
            'mcos': mcos,
            'msin': msin,
            'vcos': vcos,
            'vsin': vsin,
            'pos_embeds': pos_embeds,
            'S': S,
            'mrope_max': int(pos_ids.max()),
        }
        self._prompt['pg_key'] = self._stage_prefill_inputs(n_patch, S, sg['span'])

    def _bf16_gemm(self, x, weight_ptr: int, out, M: int, N: int, K: int,
                   stream: int) -> None:
        if (M == 1 and K in (2048, 6144)
                and hasattr(self._vlk, 'qwen3_vl_bf16_gemv_m1')):
            self._vlk.qwen3_vl_bf16_gemv_m1(
                x.data_ptr(), int(weight_ptr), out.data_ptr(), N, K, stream)
            return
        if M > 1 and hasattr(self._vlk, 'bf16_matmul_cublaslt_bf16'):
            self._vlk.bf16_matmul_cublaslt_bf16(
                x.data_ptr(), int(weight_ptr), out.data_ptr(), M, N, K, stream)
            return
        self._fvk.bf16_matmul_bf16(
            x.data_ptr(), int(weight_ptr), out.data_ptr(), M, N, K, stream)

    def _layer_forward_prefill(self, L: int, h, cos_S, sin_S,
                               start_pos: int, S: int):
        import torch

        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        Nq = NQ * HD
        Nk = NKV * HD
        eps = cfg['rms_norm_eps']
        stream = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        h2 = h.view(S, H).contiguous()
        xn = self._norm_buf[:S]
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_w']), xn.data_ptr(),
            S, H, eps, stream)

        qkv = self._qkv_out[:S]
        self._bf16_gemm(
            xn, int(lw['qkv_proj_w']), qkv, S, int(lw['qkv_proj_N']), H,
            stream)
        fvk.qwen3_q_norm_rope_qstage_prefill_bf16(
            qkv[:, :Nq].data_ptr(), int(lw['q_norm_w']),
            cos_S.data_ptr(), sin_S.data_ptr(),
            self._attn.Q_buf[:, :S].data_ptr(),
            NQ, S, int(qkv.stride(0)), int(self._attn.Q_buf[:, :S].stride(1)),
            eps, stream)
        fvk.qwen3_k_norm_rope_kvwrite_prefill_bf16(
            qkv[:, Nq:Nq + Nk].data_ptr(), qkv[:, Nq + Nk:].data_ptr(),
            int(lw['k_norm_w']), cos_S.data_ptr(), sin_S.data_ptr(),
            self._attn.K_cache[L, start_pos:start_pos + S].data_ptr(),
            self._attn.V_cache[L, start_pos:start_pos + S].data_ptr(),
            NKV, S, int(qkv.stride(0)),
            int(self._attn.K_cache[L, start_pos:start_pos + S].stride(0)),
            eps, stream)

        self._attn.run(
            'full', layer_idx=L, q_seq=S, kv_seq=start_pos + S,
            stream=stream, causal=True)
        attn_2d = self._attn.O_buf[:, :S].reshape(S, H).contiguous()
        attn_out = self._tmp_hidden[:S]
        self._bf16_gemm(
            attn_2d, int(lw['o_proj_w']), attn_out, S, H, H, stream)

        xn2 = self._norm_buf[:S]
        fvk.residual_add_rms_norm(
            h.data_ptr(), attn_out.view(1, S, H).data_ptr(),
            int(lw['post_attn_norm_w']), xn2.data_ptr(), S, H, eps, stream)

        gate_up = self._gate_up[:S]
        self._bf16_gemm(
            xn2, int(lw['gate_up_w']), gate_up, S, int(lw['gate_up_N']), H,
            stream)
        gate = self._gate_tmp[:S]
        up = self._up_tmp[:S]
        gate.copy_(gate_up[:, :I])
        up.copy_(gate_up[:, I:])
        fvk.silu_mul_qwen36_bf16(
            gate.data_ptr(), up.data_ptr(),
            self._mlp_act[:S].data_ptr(), S * I, stream)
        down = self._tmp_hidden[:S]
        self._bf16_gemm(
            self._mlp_act[:S], int(lw['mlp_down_w']), down, S, H, I, stream)

        h_out = (self._h_a if (L % 2 == 0) else self._h_b)[:, :S]
        torch.add(h, down.view(1, S, H), out=h_out)
        return h_out

    def _layer_forward_decode(self, L: int, h, cos, sin, cur_pos: int):
        import torch

        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        I = cfg['intermediate']
        NQ = cfg['num_q_heads']
        NKV = cfg['num_kv_heads']
        HD = cfg['head_dim']
        Nq = NQ * HD
        Nk = NKV * HD
        eps = cfg['rms_norm_eps']
        stream = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        h2 = h.view(1, H).contiguous()
        xn = self._norm_buf[:1]
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_w']), xn.data_ptr(),
            1, H, eps, stream)
        qkv = self._qkv_out[:1]
        self._bf16_gemm(
            xn, int(lw['qkv_proj_w']), qkv, 1, int(lw['qkv_proj_N']), H,
            stream)
        kv_layer_stride = self._attn.kv_layer_stride_bytes
        kv_row_stride = self._attn.kv_row_stride_bytes
        kv_slot_off = L * kv_layer_stride + cur_pos * kv_row_stride
        fvk.qwen3_q_norm_rope_qstage_bf16(
            qkv[:, :Nq].data_ptr(), int(lw['q_norm_w']),
            cos.data_ptr(), sin.data_ptr(), self._attn.Q_buf[:, :1].data_ptr(),
            NQ, eps, stream)
        fvk.qwen3_k_norm_rope_kvwrite_bf16(
            qkv[:, Nq:Nq + Nk].data_ptr(), qkv[:, Nq + Nk:].data_ptr(),
            int(lw['k_norm_w']), cos.data_ptr(), sin.data_ptr(),
            self._attn.K_cache.data_ptr() + kv_slot_off,
            self._attn.V_cache.data_ptr() + kv_slot_off,
            NKV, eps, stream)
        self._attn.run(
            'full', layer_idx=L, q_seq=1, kv_seq=cur_pos + 1,
            stream=stream, causal=True)
        attn_2d = self._attn.O_buf[:, :1].reshape(1, H).contiguous()
        attn_out = self._tmp_hidden[:1]
        self._bf16_gemm(
            attn_2d, int(lw['o_proj_w']), attn_out, 1, H, H, stream)

        xn2 = self._norm_buf[:1]
        fvk.residual_add_rms_norm(
            h.data_ptr(), attn_out.view(1, 1, H).data_ptr(),
            int(lw['post_attn_norm_w']), xn2.data_ptr(), 1, H, eps, stream)
        gate_up = self._gate_up[:1]
        self._bf16_gemm(
            xn2, int(lw['gate_up_w']), gate_up, 1, int(lw['gate_up_N']), H,
            stream)
        gate = self._gate_tmp[:1]
        up = self._up_tmp[:1]
        gate.copy_(gate_up[:, :I])
        up.copy_(gate_up[:, I:])
        fvk.silu_mul_qwen36_bf16(
            gate.data_ptr(), up.data_ptr(), self._mlp_act[:1].data_ptr(),
            I, stream)
        down = self._tmp_hidden[:1]
        self._bf16_gemm(
            self._mlp_act[:1], int(lw['mlp_down_w']), down, 1, H, I, stream)
        h_out = (self._h_a if (L % 2 == 0) else self._h_b)[:, :1]
        torch.add(h, down.view(1, 1, H), out=h_out)
        return h_out

    def _prefill_body(self, p: dict[str, Any], *, use_vision_graph: bool):
        import torch

        self.reset_state()
        cfg = self._cfg
        fvk = self._fvk
        H = cfg['hidden_size']
        S = p['S']
        if S > self.max_seq:
            raise ValueError(f'prompt length {S} exceeds max_seq {self.max_seq}')
        stream = torch.cuda.current_stream().cuda_stream
        h = self._h_a[:, :S]
        fvk.embedding_lookup_bf16(
            p['input_ids'].view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']),
            h.data_ptr(), S, H, stream)
        a, b = p['span']
        if use_vision_graph:
            emb, deepstack = self.vision.forward_graph(
                p['pixel_values'], p['pos_embeds'], p['vcos'], p['vsin'])
        else:
            emb, deepstack = self.vision.forward(
                p['pixel_values'], p['pos_embeds'], p['vcos'], p['vsin'])
        h[0, a:b].copy_(emb.to(torch.bfloat16))

        cur = h
        for L in range(cfg['num_hidden_layers']):
            cur = self._layer_forward_prefill(
                L, cur, p['mcos'], p['msin'], 0, S)
            if L < self._deepstack_layers:
                fvk.residual_add(
                    cur[0, a:b].data_ptr(),
                    deepstack[L].to(torch.bfloat16).data_ptr(),
                    (b - a) * H, stream)

        x = cur.view(S, H)[S - 1:S].contiguous()
        xn = self._norm_buf[:1]
        fvk.rms_norm(
            x.data_ptr(), int(self._weights.ptrs['final_norm_w']),
            xn.data_ptr(), 1, H, cfg['rms_norm_eps'], stream)
        self._bf16_gemm(
            xn, int(self._weights.ptrs['lm_head_w']), self._logits,
            1, cfg['vocab_size'], H, stream)
        self._cur_pos = S
        return self._logits

    def prefill(self):
        """Run multimodal prefill eagerly and return next-token logits."""
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill()')
        return self._prefill_body(self._prompt, use_vision_graph=True)

    def _capture_prefill_graph(self, st: dict[str, Any], key):
        import torch

        P, S, a, b = key
        st = dict(st)
        st['S'] = int(S)
        st['span'] = (int(a), int(b))
        gs = self._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.inference_mode():
            for _ in range(2):
                self._prefill_body(st, use_vision_graph=False)
        gs.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gs), torch.inference_mode():
            self._prefill_body(st, use_vision_graph=False)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        return graph

    def prefill_graph(self):
        """Run multimodal prefill through a fixed-shape CUDA Graph bucket."""
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill_graph()')
        key = self._prompt.get('pg_key')
        if key is None:
            return self.prefill()
        if not isinstance(self._pg_buffers, collections.OrderedDict):
            self._pg_buffers = collections.OrderedDict(self._pg_buffers)
        if not isinstance(self._prefill_graphs, collections.OrderedDict):
            self._prefill_graphs = collections.OrderedDict(
                self._prefill_graphs)
        st = self._pg_buffers.get(key)
        if st is None:
            self._prefill_graphs.pop(key, None)
            P, S, a, b = key
            self._stage_prefill_inputs(P, S, (a, b))
            st = self._pg_buffers[key]
        graph = self._graph_cache_get(self._prefill_graphs, key)
        if graph is None:
            graph = self._capture_prefill_graph(st, key)
            self._prefill_graphs[key] = graph
            if isinstance(self._prefill_graphs, collections.OrderedDict):
                self._prefill_graphs.move_to_end(key)
            if isinstance(self._pg_buffers, collections.OrderedDict):
                self._pg_buffers.move_to_end(key)
            self._trim_lru_graph_cache(
                self._prefill_graphs, self.max_prefill_graphs,
                lambda old_key: self._pg_buffers.pop(old_key, None))
        elif isinstance(self._pg_buffers, collections.OrderedDict):
            self._pg_buffers.move_to_end(key)
        graph.replay()
        self._cur_pos = int(key[1])
        return self._logits

    def _decode_token_tensor(self, token, *, cache_pos: int, rope_pos: int):
        import torch

        cfg = self._cfg
        H = cfg['hidden_size']
        stream = torch.cuda.current_stream().cuda_stream
        h = self._h_a[:, :1]
        self._fvk.embedding_lookup_bf16(
            token.view(-1).data_ptr(), int(self._weights.ptrs['embed_w']),
            h.data_ptr(), 1, H, stream)
        cos = self._mrope_cos_cache[rope_pos:rope_pos + 1]
        sin = self._mrope_sin_cache[rope_pos:rope_pos + 1]
        cur = h
        for L in range(cfg['num_hidden_layers']):
            cur = self._layer_forward_decode(L, cur, cos, sin, cache_pos)
        xn = self._norm_buf[:1]
        self._fvk.rms_norm(
            cur.view(1, H).contiguous().data_ptr(),
            int(self._weights.ptrs['final_norm_w']),
            xn.data_ptr(), 1, H, cfg['rms_norm_eps'], stream)
        self._bf16_gemm(
            xn, int(self._weights.ptrs['lm_head_w']), self._logits,
            1, cfg['vocab_size'], H, stream)
        return self._logits

    def decode_step(self, token_id: int, *, cache_pos: int, rope_pos: int):
        import torch

        token = torch.tensor([[int(token_id)]], dtype=torch.long,
                             device=self.device)
        return self._decode_token_tensor(
            token, cache_pos=cache_pos, rope_pos=rope_pos)

    def _ensure_decode_graph(self, cache_pos: int, rope_pos: int):
        import torch

        key = (int(cache_pos), int(rope_pos))
        if not isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs = collections.OrderedDict(
                self._decode_graphs)
        graph = self._graph_cache_get(self._decode_graphs, key)
        if graph is not None:
            return graph
        gs = self._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.inference_mode():
            for _ in range(2):
                self._decode_token_tensor(
                    self._static_token_id,
                    cache_pos=cache_pos, rope_pos=rope_pos)
        gs.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gs), torch.inference_mode():
            self._decode_token_tensor(
                self._static_token_id, cache_pos=cache_pos, rope_pos=rope_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        self._decode_graphs[key] = graph
        if isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs.move_to_end(key)
        self._trim_lru_graph_cache(
            self._decode_graphs, self.max_decode_graphs)
        return graph

    def decode_step_with_graph(self, token_id: int, *, cache_pos: int,
                               rope_pos: int):
        self._static_token_id.fill_(int(token_id))
        self._ensure_decode_graph(cache_pos, rope_pos).replay()
        return self._logits

    def warmup_decode_graphs(self, n_tokens: int) -> None:
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before warmup_decode_graphs()')
        base_slot = int(self._prompt['S'])
        base_rope = int(self._prompt['mrope_max']) + 1
        for i in range(int(n_tokens)):
            self._ensure_decode_graph(base_slot + i, base_rope + i)

    def generate(self, messages: list, *, max_new_tokens: int = 64,
                 use_graph: bool = True) -> str:
        """Greedy single-image generation."""
        self.set_prompt(messages)
        logits = self.prefill_graph() if use_graph else self.prefill()
        p = self._prompt
        assert p is not None
        base_slot = int(p['S'])
        base_rope = int(p['mrope_max']) + 1
        tok = int(logits[0].float().argmax())
        out_ids = [tok]
        for i in range(max_new_tokens - 1):
            if tok in self._eos_token_ids:
                break
            if use_graph:
                logits = self.decode_step_with_graph(
                    tok, cache_pos=base_slot + i, rope_pos=base_rope + i)
            else:
                logits = self.decode_step(
                    tok, cache_pos=base_slot + i, rope_pos=base_rope + i)
            tok = int(logits[0].float().argmax())
            out_ids.append(tok)

        eos = [t for t in out_ids if t in self._eos_token_ids]
        if eos:
            out_ids = out_ids[:out_ids.index(eos[0])]
        return self.processor.tokenizer.decode(
            out_ids, skip_special_tokens=True)
