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

        # Language core (reused unchanged) + ViT tower.
        self.llm = Qwen3TorchFrontendRtx(
            checkpoint_path, device=device, max_seq=max_seq,
            max_q_seq=max_seq)
        self.vision = Qwen3VlVisionRtx(checkpoint_path, device=device)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)

        self._prompt: dict[str, Any] | None = None

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

        # Contiguous image-token span (single image).
        mask = (input_ids == self._image_token_id)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        img_start = int(idx[0])
        img_end = int(idx[-1]) + 1

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
            'img_start': img_start, 'img_end': img_end,
            'mcos': mcos, 'msin': msin, 'vcos': vcos, 'vsin': vsin,
            'pos_embeds': pos_embeds, 'S': S,
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
        stream = torch.cuda.current_stream().cuda_stream
        hidden = llm._cfg['hidden_size']
        vocab = llm._cfg['vocab_size']
        eps = float(llm._cfg['rms_norm_eps'])
        n_layers = llm._cfg['num_hidden_layers']
        S = p['S']
        a, b = p['img_start'], p['img_end']

        # Vision tower → image features + DeepStack.
        image_embeds, deepstack = self.vision.forward(
            p['pixel_values'], p['pos_embeds'], p['vcos'], p['vsin'])

        # Embedding lookup + scatter the (contiguous) image span.
        embed = llm._weights.anchors[0]
        h = embed[p['input_ids']].to(torch.bfloat16).view(1, S, hidden)
        h = h.contiguous()
        h[0, a:b] = image_embeds.to(torch.bfloat16)

        # Decoder layers with MRoPE; DeepStack added at the first layers.
        cur = h
        for layer in range(n_layers):
            cur = llm._layer_forward_full_nvfp4_prefill(
                layer, cur, p['mcos'], p['msin'], 0, S)
            if layer < self._deepstack_layers:
                cur = cur.clone()
                fvk.residual_add(
                    cur[0, a:b].data_ptr(),
                    deepstack[layer].to(torch.bfloat16).data_ptr(),
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
