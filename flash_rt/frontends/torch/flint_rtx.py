"""FlashRT -- Flint (RecursiveSI/flint) official NVFP4 GGUF, torch frontend for RTX (SM120).

Flint is a 49-layer MoE chat model (256 experts, top-6 sigmoid routing plus a
shared expert, sliding-window/global attention with sinks). This frontend
reads the official NVFP4 GGUF with FlashRT's own GGUF reader, repacks the
NVFP4 weights losslessly into the kernel layout, and runs the native W4A16
pipeline: chunked tensor-core prefill, then a decode step captured into one
CUDA graph and replayed per token. GGUF is only the weight container; no
llama.cpp code is involved.

    from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
    fe = FlintTorchFrontendRtx("<path-to-flint-nvfp4.gguf>", max_seq=131072)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Constructed directly rather than through ``load_model`` (a text decoder has no
VLA ``predict`` surface). See ``docs/flint_usage.md``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import torch

from flash_rt.models.flint.config import FlintConfig
from flash_rt.frontends.torch._flint_rtx_runtime import FlintRuntime
from flash_rt.models.flint.tokenizer import FlintTokenizer

__all__ = ["FlintTorchFrontendRtx"]


class FlintTorchFrontendRtx:
    """Greedy generation over the native Flint runtime (batch 1)."""

    def __init__(self, gguf_path: str, max_seq: int = 32768, prefill_chunk: Optional[int] = None,
                 attn_splits: int = 16, device: int = 0, check_block: int = 16):
        if check_block < 1:
            raise ValueError("check_block must be positive")
        self.rt = FlintRuntime(gguf_path, max_seq=max_seq, prefill_chunk=prefill_chunk,
                               attn_splits=attn_splits, device=device)
        self.cfg: FlintConfig = self.rt.cfg
        self.tokenizer = FlintTokenizer(self.rt.gguf)
        self.eog = set(self.cfg.eog_token_ids)
        self.check_block = int(check_block)
        self._host_tokens = None
        self._kv_ids: List[int] = []     # logical prefix; runtime tracks which sliding KV is still resident
        self.min_reuse = 64             # shorter common prefixes are recomputed
        self.last_reused = 0
        self.rt.capture()

    def stream(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
               stop_ids: Optional[Sequence[int]] = None, reuse_prefix: bool = False) -> Iterator[List[int]]:
        """Greedy continuation of ``prompt_ids`` as it is produced: the first token right after prefill,
        then blocks of up to ``check_block`` tokens. Stops after an end-of-generation token (included);
        ``stop_ids=()`` disables stopping. The next decode block is queued before the current one is read
        back, so the GPU does not wait for the host; a stop discards at most one extra block.

        ``reuse_prefix``: keep the KV cache of the longest common prefix with the previous sequence (its
        prompt and generated tokens) and prefill only the rest. If the sliding-window history needed at
        the reuse position has been overwritten, recompute the full prompt."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(t < 0 or t >= self.cfg.vocab_size for t in prompt_ids):
            raise ValueError("prompt token id outside the checkpoint vocabulary")
        n = len(prompt_ids)
        if n + max_new_tokens + 2 * self.check_block > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens exceeds max_seq {self.rt.max_seq}")
        stop = self.eog if stop_ids is None else set(stop_ids)
        rt = self.rt
        if self._host_tokens is None or self._host_tokens.numel() < rt.max_seq:
            self._host_tokens = torch.empty(rt.max_seq, dtype=torch.int32, pin_memory=True)
        host = self._host_tokens
        ids = list(prompt_ids)
        start = 0
        if reuse_prefix and self._kv_ids:
            start = min(self.common_prefix(ids), n - 1)
            history_start = max(0, start - self.cfg.sliding_window + 1)
            if start < self.min_reuse or history_start < rt.swa_kv_start:
                start = 0
        self._kv_ids = []                        # invalid until this request finishes
        rt.prefill(ids, start=start)             # writes tokens[n] (first generated token)
        self.last_reused = start
        blocks = []                              # (lo, hi, event) queued read-backs, oldest first

        def enqueue_readback(lo, hi):
            host[lo:hi].copy_(rt.tokens[lo:hi], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(rt.stream)
            blocks.append((lo, hi, ev))

        generated: List[int] = []
        with torch.cuda.stream(rt.stream):
            enqueue_readback(n, n + 1)
            produced = 1
            done = 0
            while True:
                step = min(self.check_block, max_new_tokens - produced)
                if step > 0:
                    rt.replay(step)
                    enqueue_readback(n + produced, n + produced + step)
                    produced += step
                lo, hi, ev = blocks.pop(0)
                ev.synchronize()
                new = host[lo:hi].tolist()
                for i, t in enumerate(new):
                    if t in stop or done + i + 1 >= max_new_tokens:
                        rt.stream.synchronize()
                        # KV holds the prompt and every generated token except the last one
                        self._kv_ids = ids + generated + new[:i]
                        yield new[:i + 1]
                        return
                done += len(new)
                generated.extend(new)
                yield new

    def common_prefix(self, ids: Sequence[int]) -> int:
        """Length of the common prefix of ``ids`` and the sequence whose KV is cached."""
        kv = self._kv_ids
        n = min(len(kv), len(ids))
        i = 0
        while i < n and kv[i] == ids[i]:
            i += 1
        return i

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None) -> List[int]:
        """Greedy continuation of ``prompt_ids``; stops at an end-of-generation token (included)."""
        out: List[int] = []
        for block in self.stream(prompt_ids, max_new_tokens, stop_ids):
            out.extend(block)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens)
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)
