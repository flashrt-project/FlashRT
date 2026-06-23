"""FlashRT — PyTorch frontend for Qwen3-VL (multimodal) on RTX SM120.

Class name ``Qwen3VlTorchFrontendRtx`` follows the
``docs/adding_new_model.md`` §0 naming rule, and the direct-instantiation
pattern of the text sibling ``qwen3_rtx.Qwen3TorchFrontendRtx`` (the
Qwen3 LLM paths are not registered in ``_PIPELINE_MAP``; a server or test
constructs the frontend directly).

The language stack is the existing dense-Qwen3 NVFP4 W4A4 path, reused
unchanged via ``Qwen3TorchFrontendRtx``; this frontend adds the ViT
tower (``_qwen3_vl_vision_rtx``), the multimodal image-feature scatter,
DeepStack injection into the first decoder layers, and interleaved-MRoPE.
The checkpoint is the self-contained directory produced by
``tools/quantize_qwen3_vl_nvfp4.py`` (NVFP4 language linears + BF16 vision
tower + combined config).

Per-prompt geometry (position ids, MRoPE / vision-RoPE tables, the
interpolated vision position embedding) is precomputed in ``set_prompt``
by ``_qwen3_vl_geometry``; the forward itself runs only kernels.
"""
from __future__ import annotations

import json
import os
from typing import Any

# Functions the Qwen3-VL ViT tower needs from the separate, opt-in
# ``flash_rt_qwen3_vl_kernels`` module (CMake ``-DFLASHRT_BUILD_QWEN3_VL=ON``).
_QWEN3_VL_KERNEL_FNS = (
    'rope_neox_qk_bf16',
    'layer_norm_to_fp8_block128_bf16',
    'gelu_tanh_to_fp8_block128_bf16',
    'gelu_tanh_bias_to_fp8_block128_bf16',
    'residual_add_bias_bf16',
    'qkv_split_bias_bf16',
)


def _check_qwen3_vl_kernels(module) -> None:
    """Raise a clear error if the Qwen3-VL kernel module is incomplete."""
    missing = [fn for fn in _QWEN3_VL_KERNEL_FNS if not hasattr(module, fn)]
    if missing:
        raise RuntimeError(
            'flash_rt_qwen3_vl_kernels is missing ' + ', '.join(missing)
            + '. Rebuild it with -DFLASHRT_BUILD_QWEN3_VL=ON (GPU_ARCH=120).')


def _require_qwen3_vl_kernels():
    """Import + validate the Qwen3-VL kernel module up front (fail-fast),
    before any weights are loaded. Returns the module."""
    try:
        from flash_rt import flash_rt_qwen3_vl_kernels as vlk
    except ImportError as e:
        raise RuntimeError(
            'flash_rt_qwen3_vl_kernels is not built. Configure with '
            '-DFLASHRT_BUILD_QWEN3_VL=ON (GPU_ARCH=120) and build the '
            'flash_rt_qwen3_vl_kernels target.') from e
    _check_qwen3_vl_kernels(vlk)
    return vlk


class Qwen3VlTorchFrontendRtx:
    """Qwen3-VL-8B-class multimodal inference frontend (RTX SM120, NVFP4).

    Public surface:
      __init__(checkpoint_path, *, device='cuda:0', max_seq=4096)
      set_prompt(messages)   -- preprocess image+text, build geometry
      prefill()              -- multimodal prefill, returns next-token logits
    """

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 max_seq: int = 4096, max_pixels: int | None = None) -> None:
        from transformers import AutoProcessor

        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
            Qwen3VlVisionRtx,
        )
        from flash_rt.frontends.torch.qwen3_rtx import Qwen3TorchFrontendRtx

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)

        # Fail fast if the opt-in kernel module is missing, before loading
        # the (large) language and vision weights.
        _require_qwen3_vl_kernels()

        cfg = json.load(
            open(os.path.join(checkpoint_path, 'config.json')))
        self._image_token_id = int(cfg['image_token_id'])
        self._video_token_id = int(cfg['video_token_id'])
        self._vision_start_token_id = int(cfg['vision_start_token_id'])
        vc = cfg['vision_config']
        self._merge = int(vc['spatial_merge_size'])
        self._vis_head_dim = vc['hidden_size'] // vc['num_heads']
        self._num_grid_per_side = int(vc['num_position_embeddings'] ** 0.5)
        self._deepstack_layers = len(vc['deepstack_visual_indexes'])
        self._rope_theta = float(cfg['rope_theta'])
        self._head_dim = int(cfg['head_dim'])
        self._mrope_section = tuple(cfg['rope_scaling']['mrope_section'])
        eos = cfg.get('eos_token_id')
        if eos is None:
            self._eos_token_ids: set = set()
        else:
            self._eos_token_ids = set(eos if isinstance(eos, list) else [eos])

        # Language core (reused unchanged) + ViT tower.
        self.llm = Qwen3TorchFrontendRtx(
            checkpoint_path, device=device, max_seq=max_seq,
            max_q_seq=max_seq)
        self.vision = Qwen3VlVisionRtx(checkpoint_path, device=device)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        # Optional resolution cap. The patch count (≈ pixels / patch_size^2)
        # sets both the ViT cost and the number of vision tokens the LLM
        # prefills, so it is the dominant TTFT knob; capping it triggers the
        # processor's smart_resize (rounds to the patch grid). None keeps the
        # checkpoint default (full resolution).
        self.max_pixels = max_pixels
        if max_pixels is not None:
            for proc in (getattr(self.processor, 'image_processor', None),
                         getattr(self.processor, 'video_processor', None)):
                size = getattr(proc, 'size', None)
                if isinstance(size, dict) and 'longest_edge' in size:
                    size['longest_edge'] = int(max_pixels)

        self._prompt: dict[str, Any] | None = None
        # cache-slot -> captured decode CUDA Graph (rope baked at the
        # MRoPE-continuation position, which differs from the slot).
        self._decode_graphs: dict[int, Any] = {}
        # Captured single-image prefill: (P,S,a,b) -> graph, plus the static
        # input buffers set_prompt stages into and the persistent output
        # buffers the replay + eager lm_head write.
        self._prefill_graphs: dict = {}
        self._pg_buffers: dict = {}
        import torch as _torch
        hidden = self.llm._cfg['hidden_size']
        vocab = self.llm._cfg['vocab_size']
        self._pg_last_hidden = _torch.empty(
            self.max_seq, hidden, dtype=_torch.bfloat16, device=device)
        self._pg_logits = _torch.empty(
            1, vocab, dtype=_torch.bfloat16, device=device)

    # ── Prompt ──

    def set_prompt(self, messages: list) -> None:
        """Preprocess a chat ``messages`` list (image + text) and build the
        per-prompt geometry consumed by ``prefill``."""
        import torch

        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        # device=self.device runs the fast image processor's resize /
        # normalize / patchify on the GPU (~10x over the CPU path), so the
        # whole preprocess is off the CPU.
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt',
            device=self.device).to(self.device)
        input_ids = inputs['input_ids'][0]
        S = int(input_ids.shape[0])
        if S > self.max_seq:
            raise ValueError(
                f'prompt length {S} exceeds max_seq {self.max_seq}')

        m = self._merge
        image_grid = inputs.get('image_grid_thw')
        video_grid = inputs.get('video_grid_thw')
        pix_img = inputs.get('pixel_values')
        pix_vid = inputs.get('pixel_values_videos')
        if pix_img is not None:
            pix_img = pix_img.to(torch.bfloat16)
        if pix_vid is not None:
            pix_vid = pix_vid.to(torch.bfloat16)
        # Videos are split into per-frame rows (t -> 1) so each frame is a
        # vision segment encoded like an image; the inter-frame timestamp
        # text tokens carry the temporal position (HF get_rope_index).
        # Walk the image/video token runs (sequence order; videos split per
        # frame). Raises ValueError for a text-only prompt or a grid mismatch.
        segs = geo.vision_segments(
            input_ids.cpu(), image_grid, video_grid,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id, spatial_merge_size=m)
        seg_pix: list = []
        seg_grids: list = []
        spans: list[tuple[int, int]] = []
        seg_patches: list[int] = []
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
        pixel_values = torch.cat(seg_pix, dim=0).contiguous()

        pos_ids = geo.mrope_position_ids(
            input_ids.cpu(),
            image_grid.cpu() if image_grid is not None else None,
            video_grid.cpu() if video_grid is not None else None,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            vision_start_token_id=self._vision_start_token_id,
            spatial_merge_size=m)
        mcos, msin = geo.mrope_cos_sin(
            pos_ids, head_dim=self._head_dim, rope_theta=self._rope_theta,
            mrope_section=self._mrope_section, device=self.device)
        vcos, vsin = geo.vision_rope_cos_sin(
            seg_grid, head_dim=self._vis_head_dim,
            spatial_merge_size=m, device=self.device)
        pos_embeds = geo.vision_pos_embeds(
            seg_grid, self.vision.pos_embed,
            num_grid_per_side=self._num_grid_per_side,
            spatial_merge_size=m, device=self.device)

        self._prompt = {
            'input_ids': input_ids, 'pixel_values': pixel_values,
            'spans': spans, 'seg_patches': seg_patches,
            'img_start': spans[0][0], 'img_end': spans[0][1],
            'mcos': mcos, 'msin': msin, 'vcos': vcos, 'vsin': vsin,
            'pos_embeds': pos_embeds, 'S': S,
            'mrope_max': int(pos_ids.max()),
        }
        # Single image: stage the captured-prefill static inputs here (this is
        # one-time set-up glue), so prefill_graph's hot path is just replay +
        # lm_head with no per-request torch copy.
        if len(spans) == 1:
            self._prompt['pg_key'] = self._stage_prefill_inputs(
                seg_patches[0], S, spans[0])

    _PG_KEYS = ('input_ids', 'pixel_values', 'pos_embeds',
                'vcos', 'vsin', 'mcos', 'msin')

    def _stage_prefill_inputs(self, P: int, S: int, span):
        """Copy the per-prompt tensors into persistent static buffers keyed by
        (P, S, a, b). The captured prefill graph reads only from these, so the
        replay path needs no torch copy. Returns the key."""
        p = self._prompt
        key = (P, S, span[0], span[1])
        bufs = self._pg_buffers.get(key)
        if bufs is None:
            bufs = {k: p[k].clone() for k in self._PG_KEYS}
            self._pg_buffers[key] = bufs
        else:
            for k in self._PG_KEYS:
                bufs[k].copy_(p[k])
        return key

    # ── Forward ──

    def prefill(self):
        """Multimodal prefill. Returns ``(1, vocab)`` bf16 next-token logits.

        Vision tower → scatter image features into the embedding stream →
        Qwen3 NVFP4 decoder layers (interleaved-MRoPE cos/sin) with
        DeepStack injection at the first layers → final norm + lm_head.
        """
        import torch

        from flash_rt import flash_rt_kernels as fvk

        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill()')
        p = self._prompt
        llm = self.llm
        llm.reset_state()
        stream = torch.cuda.current_stream().cuda_stream
        hidden = llm._cfg['hidden_size']
        vocab = llm._cfg['vocab_size']
        eps = float(llm._cfg['rms_norm_eps'])
        n_layers = llm._cfg['num_hidden_layers']
        S = p['S']
        spans = p['spans']

        # Vision tower, once per image (each image is an independent
        # attention window — running the tower per segment reproduces HF's
        # block-diagonal cu_seqlens attention without a varlen kernel).
        embed = llm._weights.anchors[0]
        h = embed[p['input_ids']].to(torch.bfloat16).view(1, S, hidden)
        h = h.contiguous()
        seg_deepstacks: list = []
        off = 0
        for (a, b), n_patch in zip(spans, p['seg_patches']):
            sl = slice(off, off + n_patch)
            off += n_patch
            emb, ds = self.vision.forward(
                p['pixel_values'][sl], p['pos_embeds'][sl],
                p['vcos'][sl], p['vsin'][sl])
            h[0, a:b] = emb.to(torch.bfloat16)
            seg_deepstacks.append(ds)

        # Decoder layers with MRoPE; DeepStack added at the first layers,
        # each image's features into its own span.
        cur = h
        for layer in range(n_layers):
            cur = llm._layer_forward_full_nvfp4_prefill(
                layer, cur, p['mcos'], p['msin'], 0, S)
            if layer < self._deepstack_layers:
                cur = cur.clone()
                for (a, b), ds in zip(spans, seg_deepstacks):
                    fvk.residual_add(
                        cur[0, a:b].data_ptr(),
                        ds[layer].to(torch.bfloat16).data_ptr(),
                        (b - a) * hidden, stream)
                cur = cur.contiguous()

        # Final RMSNorm + BF16 lm_head on the last row.
        x = cur.view(S, hidden)[-1:].contiguous()
        xn = torch.empty(1, hidden, dtype=torch.bfloat16, device=self.device)
        fvk.rms_norm(x.data_ptr(), int(llm._weights.ptrs['final_norm_w']),
                     xn.data_ptr(), 1, hidden, eps, stream)
        logits = torch.empty(
            1, vocab, dtype=torch.bfloat16, device=self.device)
        fvk.bf16_matmul_qwen36_bf16(
            xn.data_ptr(), int(llm._weights.ptrs['lm_head_w']),
            logits.data_ptr(), 1, vocab, hidden, stream)
        torch.cuda.synchronize()
        return logits

    # ── Prefill via a captured CUDA Graph (single image) ──

    def _prefill_graph_body(self, st, P, S, a, b):
        """Capture body: embed gather -> ViT tower -> scatter image features
        -> NVFP4 decoder layers with DeepStack inject -> final RMSNorm into
        ``self._pg_last_hidden[:S]``. No lm_head (run eager post-replay so one
        graph serves all real_S <= bucket). All torch glue (gather/scatter/
        clone/copy) traces into the graph, so replay is pure-kernel.

        ``st`` is the dict of static input buffers; the graph reads only from
        them. Mirrors the eager ``prefill`` single-image path exactly so
        captured and eager hidden state match bit-for-bit.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        llm = self.llm
        bf16 = torch.bfloat16
        stream = torch.cuda.current_stream().cuda_stream
        hidden = llm._cfg['hidden_size']
        eps = float(llm._cfg['rms_norm_eps'])
        n_layers = llm._cfg['num_hidden_layers']

        emb, ds = self.vision.forward(
            st['pixel_values'], st['pos_embeds'], st['vcos'], st['vsin'])
        h = llm._weights.anchors[0][st['input_ids']].to(bf16).view(
            1, S, hidden)
        h = h.contiguous()
        h[0, a:b] = emb.to(bf16)

        cur = h
        for layer in range(n_layers):
            cur = llm._layer_forward_full_nvfp4_prefill(
                layer, cur, st['mcos'], st['msin'], 0, S)
            if layer < self._deepstack_layers:
                cur = cur.clone()
                fvk.residual_add(
                    cur[0, a:b].data_ptr(),
                    ds[layer].to(bf16).data_ptr(), (b - a) * hidden, stream)
                cur = cur.contiguous()

        h2 = cur.view(S, hidden).contiguous()
        fvk.rms_norm(
            h2.data_ptr(), int(llm._weights.ptrs['final_norm_w']),
            self._pg_last_hidden[:S].data_ptr(), S, hidden, eps, stream)

    def _capture_prefill_graph(self, st, P, S, a, b):
        """2-iter warmup then capture the prefill body for one (P,S,a) bucket.
        Mirrors qwen3_rtx._ensure_prefill_graph (inference_mode + capture
        stream)."""
        import torch

        llm = self.llm
        gs = llm._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        # no_grad (not inference_mode) to match the ViT forward_graph and the
        # decode-graph capture this composes with; inference_mode flags the
        # ViT's per-call scratch as inference tensors and trips capture_begin.
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
        """Pure-kernel prefill via a captured CUDA Graph (single image).

        Falls back to the eager ``prefill`` for multi-image / video (variable
        segment structure is not graph-capturable). Returns ``(1, vocab)``
        bf16 next-token logits.
        """
        import torch
        from flash_rt import flash_rt_kernels as fvk

        if self._prompt is None:
            raise RuntimeError('call set_prompt() before prefill_graph()')
        p = self._prompt
        if p.get('pg_key') is None:           # multi-image / video: eager
            return self.prefill()

        llm = self.llm
        hidden = llm._cfg['hidden_size']
        vocab = llm._cfg['vocab_size']
        S = p['S']
        key = p['pg_key']
        P, _, a, b = key
        st = self._pg_buffers[key]            # already staged by set_prompt
        # No reset_state: the captured layers write K/V[0:S] (which is all the
        # causal prefill reads), and the subsequent decode overwrites
        # K/V[>=S] before reading it (same as qwen3_rtx.prefill_with_graph).

        graph = self._prefill_graphs.get(key)
        if graph is None:
            graph = self._capture_prefill_graph(st, P, S, a, b)
            self._prefill_graphs[key] = graph

        # Hot path: pure-kernel graph replay + eager lm_head (one buffer, no
        # per-request torch alloc/copy).
        graph.replay()
        stream = torch.cuda.current_stream().cuda_stream
        last = self._pg_last_hidden[S - 1:S].contiguous()
        fvk.bf16_matmul_qwen36_bf16(
            last.data_ptr(), int(llm._weights.ptrs['lm_head_w']),
            self._pg_logits.data_ptr(), 1, vocab, hidden, stream)
        torch.cuda.synchronize()
        return self._pg_logits

    def _ensure_decode_graph(self, cache_pos: int, rope_pos: int):
        """Lazy-capture a decode CUDA Graph for one cache slot.

        Mirrors ``qwen3_rtx._ensure_decode_graph`` but bakes the RoPE
        slice at ``rope_pos`` (the MRoPE continuation position) rather
        than the cache slot. Reuses the language core's static token
        buffer, capture stream and decode kernels.
        """
        import torch

        llm = self.llm
        g = self._decode_graphs.get(cache_pos)
        if g is not None:
            return g
        cos, sin = llm._rope_cos_sin(rope_pos)
        gs = llm._graph_stream
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs), torch.no_grad():
            for _ in range(2):
                llm.forward_own_decode_nvfp4(
                    llm._static_token_id, cos, sin, cache_pos)
        gs.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=gs), torch.no_grad():
            llm.forward_own_decode_nvfp4(
                llm._static_token_id, cos, sin, cache_pos)
        gs.synchronize()
        torch.cuda.current_stream().wait_stream(gs)
        self._decode_graphs[cache_pos] = g
        return g

    def warmup_decode_graphs(self, n_tokens: int) -> None:
        """Pre-capture decode graphs for the current prompt's next
        ``n_tokens`` steps so generation replays warm graphs."""
        if self._prompt is None:
            raise RuntimeError('call set_prompt() before warmup')
        p = self._prompt
        for i in range(n_tokens):
            self._ensure_decode_graph(p['S'] + i, p['mrope_max'] + 1 + i)

    def generate(self, messages: list, *, max_new_tokens: int = 256,
                 use_graph: bool = True) -> str:
        """Greedy multimodal generation. Returns the decoded text.

        Runs the multimodal prefill (which fills the KV cache), then the
        reused Qwen3 NVFP4 decode loop. After the image the MRoPE position
        is scalar, so the decode RoPE position continues from the prompt's
        max position while the KV-cache slot advances from the
        image-compressed prompt length. ``use_graph`` replays a per-slot
        captured CUDA Graph for each decode step.
        """
        self.set_prompt(messages)
        logits = self.prefill_graph()
        llm = self.llm
        p = self._prompt
        base_slot, base_rope = p['S'], p['mrope_max'] + 1

        tok = int(logits[0].float().argmax())
        out_ids = [tok]
        for i in range(max_new_tokens - 1):
            if tok in self._eos_token_ids:
                break
            if use_graph:
                logits = self._decode_step_graph(
                    tok, base_slot + i, base_rope + i)
            else:
                cos, sin = llm._rope_cos_sin(base_rope + i)
                logits = llm.forward_own_decode_nvfp4(
                    tok, cos, sin, base_slot + i)
            tok = int(logits[0].float().argmax())
            out_ids.append(tok)

        eos = [t for t in out_ids if t in self._eos_token_ids]
        if eos:
            out_ids = out_ids[:out_ids.index(eos[0])]
        return self.processor.tokenizer.decode(
            out_ids, skip_special_tokens=True)

    def _decode_step_graph(self, token: int, cache_pos: int, rope_pos: int):
        llm = self.llm
        llm._static_token_id.fill_(int(token))
        self._ensure_decode_graph(cache_pos, rope_pos).replay()
        return llm._logits_buf[:1]
