"""Qwen3-VL official-FP8 multimodal path for RTX SM89.

This wrapper combines the SM89 official-FP8 language frontend with the
Qwen3-VL vision tower. The single-image prefill path is CUDA-Graph captured:
processor/geometry run once, static buffers stage the prompt tensors, then
vision + S>1 FP8 blockscaled language prefill + final norm replay as one graph.
The lm_head runs eager after replay, matching the SM120 RTX frontend shape.
Decode remains owned by ``Qwen3VlFp8Sm89TextFrontend``.
"""
from __future__ import annotations

import collections
import os
from typing import Any


class Qwen3VlFp8Sm89Frontend:
    """Batch-1 Qwen3-VL prefill/decode frontend for SM89 official FP8."""

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 max_seq: int = 4096, max_prefill_seq: int | None = None,
                 max_pixels: int | None = None,
                 fuse_gate_up: bool = False,
                 fuse_qk_postproc: bool = True,
                 use_fp8_lm_head: bool = False,
                 vision_bf16_first_blocks: int = 3,
                 vision_bf16_block_linears: dict[int, tuple[str, ...]]
                 | None = None,
                 max_prefill_graphs: int | None = None,
                 max_decode_graphs: int | None = None) -> None:
        import json

        from transformers import AutoProcessor

        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
            Qwen3VlVisionRtx,
        )
        from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
            Qwen3VlFp8Sm89TextFrontend,
            _resolve_max_prefill_seq,
        )
        from flash_rt.frontends.torch.qwen3_vl_rtx import (
            _require_qwen3_vl_kernels,
        )

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)
        _require_qwen3_vl_kernels()
        self.max_prefill_seq = _resolve_max_prefill_seq(
            self.max_seq, max_prefill_seq)
        if max_prefill_graphs is None:
            max_prefill_graphs = int(os.environ.get(
                'FLASHRT_QWEN3_VL_PREFILL_GRAPH_CACHE_MAX', '256'))
        if max_decode_graphs is None:
            max_decode_graphs = int(os.environ.get(
                'FLASHRT_QWEN3_VL_DECODE_GRAPH_CACHE_MAX', '256'))
        self.max_prefill_graphs = int(max_prefill_graphs)
        self.max_decode_graphs = int(max_decode_graphs)
        self.llm = Qwen3VlFp8Sm89TextFrontend(
            checkpoint_path, device=device, max_seq=max_seq,
            max_prefill_seq=self.max_prefill_seq,
            fuse_gate_up=fuse_gate_up,
            fuse_qk_postproc=fuse_qk_postproc,
            use_fp8_lm_head=use_fp8_lm_head,
            max_decode_graphs=self.max_decode_graphs)
        self.arch = 'sm89'
        cfg = json.load(open(os.path.join(checkpoint_path, 'config.json')))
        vcfg = dict(cfg['vision_config'])
        vcfg['_bf16_first_blocks'] = int(vision_bf16_first_blocks)
        if vision_bf16_block_linears is not None:
            vcfg['_bf16_block_linears'] = {
                int(k): [str(name) for name in names]
                for k, names in vision_bf16_block_linears.items()
            }
        self.vision = Qwen3VlVisionRtx(
            checkpoint_path, device=device, config=vcfg)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self._image_token_id = int(cfg['image_token_id'])
        self._video_token_id = int(cfg['video_token_id'])
        self._vision_start_token_id = int(cfg['vision_start_token_id'])
        vc = cfg['vision_config']
        self._merge = int(vc['spatial_merge_size'])
        self._vis_head_dim = int(vc['hidden_size']) // int(vc['num_heads'])
        self._num_grid_per_side = int(vc['num_position_embeddings'] ** 0.5)
        self._deepstack_layers = len(vc['deepstack_visual_indexes'])
        self._rope_theta = float(cfg['text_config']['rope_theta'])
        self._head_dim = int(cfg['text_config']['head_dim'])
        self._mrope_section = tuple(
            cfg['text_config']['rope_scaling']['mrope_section'])
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
        self._mrope_cos_cache, self._mrope_sin_cache = geo.build_mrope_cache(
            max_pos=self.max_seq + self._num_grid_per_side,
            head_dim=self._head_dim, rope_theta=self._rope_theta,
            device=self.device)
        self._vision_rope_cos_cache, self._vision_rope_sin_cache = (
            geo.build_vision_rope_cache(
                max_hw=self.max_seq * self._merge,
                head_dim=self._vis_head_dim, device=self.device))
        eos = cfg.get('eos_token_id')
        if eos is None:
            eos = cfg['text_config'].get('eos_token_id')
        self._eos_token_ids = set(
            [] if eos is None else (eos if isinstance(eos, list) else [eos]))
        self.max_pixels = max_pixels
        if max_pixels is not None:
            for proc in (getattr(self.processor, 'image_processor', None),
                         getattr(self.processor, 'video_processor', None)):
                size = getattr(proc, 'size', None)
                if isinstance(size, dict) and 'longest_edge' in size:
                    size['longest_edge'] = int(max_pixels)
        self._prompt: dict[str, Any] | None = None
        self._decode_graphs: collections.OrderedDict[tuple[int, int], Any] = (
            collections.OrderedDict())
        self._prefill_graphs: collections.OrderedDict = (
            collections.OrderedDict())
        self._pg_buffers: collections.OrderedDict = collections.OrderedDict()

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

    def _stage_prefill_inputs(self, P: int, S: int, span):
        p = self._prompt
        key = (P, S, span[0], span[1])
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
        for attr in ('_prefill_graphs', '_decode_graphs', '_pg_buffers'):
            cache = getattr(self, attr, None)
            if cache:
                cache.clear()
        if hasattr(self.llm, 'clear_graphs'):
            self.llm.clear_graphs()

    def graph_cache_stats(self) -> dict[str, Any]:
        stats = {
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
        if hasattr(self.llm, 'graph_cache_stats'):
            stats['text'] = self.llm.graph_cache_stats()
        return stats

    def set_prompt(self, messages: list) -> None:
        import torch

        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        self._decode_graphs.clear()
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt',
            device=self.device).to(self.device)
        input_ids = inputs['input_ids'][0]
        S = int(input_ids.shape[0])
        if S > self.max_seq:
            raise ValueError(
                f'prompt length {S} exceeds max_seq {self.max_seq}')

        image_grid = inputs.get('image_grid_thw')
        video_grid = inputs.get('video_grid_thw')
        pix_img = inputs.get('pixel_values')
        pix_vid = inputs.get('pixel_values_videos')
        if pix_img is not None:
            pix_img = pix_img.to(torch.bfloat16)
        if pix_vid is not None:
            pix_vid = pix_vid.to(torch.bfloat16)

        segs = geo.vision_segments(
            input_ids.cpu(), image_grid, video_grid,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            spatial_merge_size=self._merge)
        seg_pix = []
        seg_grids = []
        spans = []
        seg_patches = []
        off_img = off_vid = 0
        for sg in segs:
            npp = sg['patches']
            if sg['kind'] == 'image':
                seg_pix.append(pix_img[off_img:off_img + npp])
                off_img += npp
            else:
                seg_pix.append(pix_vid[off_vid:off_vid + npp])
                off_vid += npp
            seg_grids.append(sg['grid'])
            spans.append(sg['span'])
            seg_patches.append(npp)

        seg_grid = torch.tensor(seg_grids, dtype=torch.long)
        pos_ids = geo.mrope_position_ids(
            input_ids.cpu(),
            image_grid.cpu() if image_grid is not None else None,
            video_grid.cpu() if video_grid is not None else None,
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
            'input_ids': input_ids,
            'pixel_values': torch.cat(seg_pix, dim=0).contiguous(),
            'spans': spans,
            'seg_patches': seg_patches,
            'mcos': mcos,
            'msin': msin,
            'vcos': vcos,
            'vsin': vsin,
            'pos_embeds': pos_embeds,
            'S': S,
            'mrope_max': int(pos_ids.max()),
        }
        if len(spans) == 1:
            self._prompt['pg_key'] = self._stage_prefill_inputs(
                seg_patches[0], S, spans[0])

    def prefill(self):
        import torch

        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill()')
        p = self._prompt
        llm = self.llm
        llm.reset_state()
        hidden = llm._cfg['hidden_size']
        S = int(p['S'])
        embed = llm._weights.anchors[0]
        h = embed[p['input_ids']].to(torch.bfloat16).view(S, hidden)
        seg_deepstacks = []
        off = 0
        for (a, b), n_patch in zip(p['spans'], p['seg_patches']):
            sl = slice(off, off + n_patch)
            off += n_patch
            emb, ds = self.vision.forward(
                p['pixel_values'][sl], p['pos_embeds'][sl],
                p['vcos'][sl], p['vsin'][sl])
            h[a:b].copy_(emb.to(torch.bfloat16))
            seg_deepstacks.append(ds)

        deep = {}
        for layer in range(self._deepstack_layers):
            rows = []
            for (a, b), ds in zip(p['spans'], seg_deepstacks):
                rows.append((a, b, ds[layer].to(torch.bfloat16).contiguous()))
            deep[layer] = rows
        logits = llm.forward_hidden_prefill_fp8_blockscaled(
            h, p['mcos'], p['msin'], 0, deepstack_by_layer=deep)
        torch.cuda.synchronize()
        return logits

    def _prefill_graph_body(self, st, P, S, a, b):
        import torch

        llm = self.llm
        hidden = llm._cfg['hidden_size']
        embed = llm._weights.anchors[0]
        h = embed[st['input_ids']].to(torch.bfloat16).view(S, hidden)
        emb, ds = self.vision.forward(
            st['pixel_values'], st['pos_embeds'], st['vcos'], st['vsin'])
        h[a:b].copy_(emb.to(torch.bfloat16))

        deep = {}
        for layer in range(self._deepstack_layers):
            deep[layer] = [(a, b, ds[layer].to(torch.bfloat16).contiguous())]
        llm.forward_hidden_prefill_fp8_blockscaled(
            h, st['mcos'], st['msin'], 0, deepstack_by_layer=deep,
            run_lm_head=False)

    def _capture_prefill_graph(self, st, P, S, a, b):
        import torch

        llm = self.llm
        gs = llm._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                self._prefill_graph_body(st, P, S, a, b)
        gs.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gs), torch.no_grad():
            self._prefill_graph_body(st, P, S, a, b)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        return graph

    def prefill_graph(self):
        import torch

        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill_graph()')
        p = self._prompt
        key = p.get('pg_key')
        if key is None:
            return self.prefill()

        P, S, a, b = key
        st = self._pg_buffers.get(key)
        if st is None:
            self._prefill_graphs.pop(key, None)
            self._stage_prefill_inputs(P, S, (a, b))
            st = self._pg_buffers[key]
        graph = self._graph_cache_get(self._prefill_graphs, key)
        if graph is None:
            graph = self._capture_prefill_graph(st, P, S, a, b)
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
        logits = self.llm._write_logits_from_hidden(
            self.llm._last_hidden_buf[0, S - 1:S])
        torch.cuda.synchronize()
        return logits

    def decode_step(self, token_id, cache_pos: int):
        if self._prompt is None:
            rope_pos = cache_pos
        else:
            rope_pos = int(self._prompt['mrope_max']) + 1 + (
                cache_pos - int(self._prompt['S']))
        cos, sin = self.llm._rope_cos_sin(rope_pos)
        return self.llm.forward_own_decode_fp8(token_id, cos, sin, cache_pos)

    def _ensure_decode_graph(self, cache_pos: int, rope_pos: int):
        import torch

        key = (int(cache_pos), int(rope_pos))
        if not isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs = collections.OrderedDict(
                self._decode_graphs)
        graph = self._graph_cache_get(self._decode_graphs, key)
        if graph is not None:
            return graph
        llm = self.llm
        cos, sin = llm._rope_cos_sin(rope_pos)
        gs = llm._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.inference_mode():
            for _ in range(2):
                llm.forward_own_decode_fp8(
                    llm._static_token_id, cos, sin, cache_pos)
        gs.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gs), torch.inference_mode():
            llm.forward_own_decode_fp8(
                llm._static_token_id, cos, sin, cache_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        self._decode_graphs[key] = graph
        if isinstance(self._decode_graphs, collections.OrderedDict):
            self._decode_graphs.move_to_end(key)
        self._trim_lru_graph_cache(
            self._decode_graphs, self.max_decode_graphs)
        return graph

    def decode_step_with_graph(self, token_id, cache_pos: int):
        if self._prompt is None:
            rope_pos = cache_pos
        else:
            rope_pos = int(self._prompt['mrope_max']) + 1 + (
                cache_pos - int(self._prompt['S']))
        llm = self.llm
        if hasattr(token_id, 'ndim'):
            if token_id.ndim == 1:
                token_id = token_id.view(1, 1)
            llm._static_token_id.copy_(token_id)
        else:
            llm._static_token_id.fill_(int(token_id))
        self._ensure_decode_graph(cache_pos, rope_pos).replay()
        return llm._logits_buf

    def warmup_decode_graphs(self, n_tokens: int) -> None:
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before warmup')
        base_slot = int(self._prompt['S'])
        base_rope = int(self._prompt['mrope_max']) + 1
        for i in range(int(n_tokens)):
            self._ensure_decode_graph(base_slot + i, base_rope + i)

    def generate(self, messages: list, *, max_new_tokens: int = 256,
                 use_graph: bool = True) -> str:
        self.set_prompt(messages)
        logits = self.prefill_graph() if use_graph else self.prefill()
        p = self._prompt
        assert p is not None
        base_slot = int(p['S'])

        tok = int(logits[0].float().argmax())
        out_ids = [tok]
        for i in range(max_new_tokens - 1):
            if tok in self._eos_token_ids:
                break
            if use_graph:
                logits = self.decode_step_with_graph(tok, base_slot + i)
            else:
                logits = self.decode_step(tok, base_slot + i)
            tok = int(logits[0].float().argmax())
            out_ids.append(tok)

        eos = [t for t in out_ids if t in self._eos_token_ids]
        if eos:
            out_ids = out_ids[:out_ids.index(eos[0])]
        return self.processor.tokenizer.decode(
            out_ids, skip_special_tokens=True)
