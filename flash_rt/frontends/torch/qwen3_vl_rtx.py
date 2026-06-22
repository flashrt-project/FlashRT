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


class Qwen3VlTorchFrontendRtx:
    """Qwen3-VL-8B-class multimodal inference frontend (RTX SM120, NVFP4).

    Public surface:
      __init__(checkpoint_path, *, device='cuda:0', max_seq=4096)
      set_prompt(messages)   -- preprocess image+text, build geometry
      prefill()              -- multimodal prefill, returns next-token logits
    """

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0',
                 max_seq: int = 4096) -> None:
        from transformers import AutoProcessor

        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
            Qwen3VlVisionRtx,
        )
        from flash_rt.frontends.torch.qwen3_rtx import Qwen3TorchFrontendRtx

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.max_seq = int(max_seq)

        cfg = json.load(
            open(os.path.join(checkpoint_path, 'config.json')))
        self._image_token_id = int(cfg['image_token_id'])
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

        self._prompt: dict[str, Any] | None = None
        # cache-slot -> captured decode CUDA Graph (rope baked at the
        # MRoPE-continuation position, which differs from the slot).
        self._decode_graphs: dict[int, Any] = {}

    # ── Prompt ──

    def set_prompt(self, messages: list) -> None:
        """Preprocess a chat ``messages`` list (image + text) and build the
        per-prompt geometry consumed by ``prefill``."""
        import torch

        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids'][0]
        grid = inputs['image_grid_thw']
        pixel_values = inputs['pixel_values'].to(torch.bfloat16)
        S = int(input_ids.shape[0])
        if S > self.max_seq:
            raise ValueError(
                f'prompt length {S} exceeds max_seq {self.max_seq}')

        # Image-token spans, one contiguous run per image (multi-image:
        # the runs appear in image order, matching the grid rows and the
        # concatenated pixel_values / vision tables).
        mask = (input_ids == self._image_token_id).cpu().tolist()
        spans: list[tuple[int, int]] = []
        i = 0
        while i < S:
            if mask[i]:
                j = i
                while j < S and mask[j]:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        # Patches fed to the ViT per image (t*h*w) and the merged-token span
        # length (t*(h/m)*(w/m)) it produces; they must line up with spans.
        m = self._merge
        seg_patches = [int(t * h * w) for t, h, w in grid.tolist()]
        for (a, b), (t, h, w) in zip(spans, grid.tolist()):
            assert b - a == int(t) * (int(h) // m) * (int(w) // m), (
                'image-token span does not match its grid')

        pos_ids = geo.mrope_position_ids(
            input_ids.cpu(), grid.cpu(),
            image_token_id=self._image_token_id,
            vision_start_token_id=self._vision_start_token_id,
            spatial_merge_size=self._merge)
        mcos, msin = geo.mrope_cos_sin(
            pos_ids, head_dim=self._head_dim, rope_theta=self._rope_theta,
            mrope_section=self._mrope_section, device=self.device)
        vcos, vsin = geo.vision_rope_cos_sin(
            grid.cpu(), head_dim=self._vis_head_dim,
            spatial_merge_size=self._merge, device=self.device)
        pos_embeds = geo.vision_pos_embeds(
            grid.cpu(), self.vision.pos_embed,
            num_grid_per_side=self._num_grid_per_side,
            spatial_merge_size=self._merge, device=self.device)

        self._prompt = {
            'input_ids': input_ids, 'pixel_values': pixel_values,
            'spans': spans, 'seg_patches': seg_patches,
            'img_start': spans[0][0], 'img_end': spans[0][1],
            'mcos': mcos, 'msin': msin, 'vcos': vcos, 'vsin': vsin,
            'pos_embeds': pos_embeds, 'S': S,
            'mrope_max': int(pos_ids.max()),
        }

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
        logits = self.prefill()
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
