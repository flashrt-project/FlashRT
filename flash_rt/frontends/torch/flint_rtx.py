"""FlashRT -- Flint (RecursiveSI/flint) official NVFP4 GGUF, torch frontend for RTX (SM120).

Flint is a 49-layer MoE chat model (256 experts, top-6 sigmoid routing plus a
shared expert, sliding-window/global attention with sinks). This frontend
reads the official NVFP4 GGUF with FlashRT's own GGUF reader, repacks the
NVFP4 weights losslessly into the kernel layout, and runs the native W4A16
pipeline: chunked tensor-core prefill, then a decode step captured into one
CUDA graph and replayed per token. GGUF is only the weight container; no
llama.cpp code is involved.

    from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
    fe = FlintTorchFrontendRtx("/models/flint-nvfp4.gguf", max_seq=131072)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Constructed directly rather than through ``load_model`` (a text decoder has no
VLA ``predict`` surface). See ``docs/flint_usage.md``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from flash_rt.models.flint.config import FlintConfig
from flash_rt.models.flint.pipeline_rtx import FlintRuntime
from flash_rt.models.flint.tokenizer import FlintTokenizer

__all__ = ["FlintTorchFrontendRtx"]


class FlintTorchFrontendRtx:
    """Greedy generation over the native Flint runtime (batch 1)."""

    def __init__(self, gguf_path: str, max_seq: int = 32768, prefill_chunk: int = 8192,
                 attn_splits: int = 16, device: int = 0, check_block: int = 16):
        self.rt = FlintRuntime(gguf_path, max_seq=max_seq, prefill_chunk=prefill_chunk,
                               attn_splits=attn_splits, device=device)
        self.cfg: FlintConfig = self.rt.cfg
        self.tokenizer = FlintTokenizer(self.rt.gguf)
        self.eog = set(self.cfg.eog_token_ids)
        self.check_block = int(check_block)
        self.rt.capture()

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None) -> List[int]:
        """Greedy continuation of ``prompt_ids``; stops at an end-of-generation token (included)."""
        n = len(prompt_ids)
        if n + max_new_tokens + self.check_block > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens exceeds max_seq {self.rt.max_seq}")
        stop = self.eog if stop_ids is None else set(stop_ids)
        rt = self.rt
        rt.prefill(list(prompt_ids))           # writes tokens[n] (first generated token)
        out: List[int] = []
        produced = 1                            # tokens[n] exists after prefill
        with torch.cuda.stream(rt.stream):
            while True:
                hi = n + produced
                new = rt.tokens[n + len(out):hi].tolist()   # synchronizes the stream
                for t in new:
                    out.append(int(t))
                    if t in stop or len(out) >= max_new_tokens:
                        return out
                step = min(self.check_block, max_new_tokens - produced)
                rt.replay(step)
                produced += step

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens)
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)
