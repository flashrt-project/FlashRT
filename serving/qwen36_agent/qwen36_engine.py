"""Qwen3.6 frontend adapter for the agent-serving policy layer."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Sequence

from .engine import DecodeChunk


class Qwen36FrontendAgentEngine:
    """Adapter from ``Qwen36TorchFrontend*`` to ``AgentEngine``.

    The adapter is intentionally thin.  It owns tokenizer normalization and
    frontend method selection, while session policy remains in ``service.py``.
    The first wired backend supports the committed short-context split.  Long
    context and append-prefill are separate frontend gates because pretending
    to reuse cache by rebuilding would hide the latency issue this example is
    designed to solve.
    """

    def __init__(self, frontend: Any, *, model_name: str = "qwen36-27b"):
        self.fe = frontend
        self.model_name = model_name
        self.max_seq = int(getattr(frontend, "_user_max_seq", 0) or 0)
        self._last_prompt_tokens = 0
        self._last_prefill_ms = 0.0

    @classmethod
    def from_checkpoint(
            cls, checkpoint: str, *, device: str = "cuda",
            max_seq: int = 262208, model_name: str = "qwen36-27b"):
        """Load the hardware-matched Qwen3.6 frontend."""
        import torch

        cap = torch.cuda.get_device_capability()
        if cap == (11, 0):
            from flash_rt.frontends.torch.qwen36_thor import (
                Qwen36TorchFrontendThor as Frontend,
            )
        else:
            from flash_rt.frontends.torch.qwen36_rtx import (
                Qwen36TorchFrontendRtx as Frontend,
            )
        fe = Frontend(checkpoint, quant="nvfp4", device=device,
                      max_seq=max_seq)
        return cls(fe, model_name=model_name)

    def tokenize_chat(self, messages, tools=None, *,
                      enable_thinking: bool = False) -> List[int]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            normalized.append(msg)
        prompt = self.fe._tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return list(self.fe._tokenizer(
            prompt, add_special_tokens=False).input_ids)

    def prefill(self, token_ids: Sequence[int], *,
                cached_tokens: int = 0,
                max_tokens: int = 1,
                K: int = 6) -> None:
        import torch

        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        if self.max_seq and len(token_ids) + int(max_tokens) > self.max_seq:
            raise ValueError(
                f"prompt + max_tokens = {len(token_ids) + int(max_tokens)} "
                f"exceeds max_seq {self.max_seq}")
        if cached_tokens == len(token_ids):
            return

        input_ids = torch.tensor(
            [list(int(t) for t in token_ids)],
            device=getattr(self.fe, "device", "cuda"),
            dtype=torch.long,
        )
        prompt_len = int(input_ids.shape[1])
        use_long = (
            getattr(self.fe, "_long_ctx_mode", False)
            and hasattr(self.fe, "_should_use_long_ctx_route")
            and self.fe._should_use_long_ctx_route(prompt_len, int(max_tokens))
        )
        if use_long:
            raise NotImplementedError(
                "long-context committed split is not wired yet; this adapter "
                "currently exposes the short committed stream only")

        t0 = time.perf_counter()
        if cached_tokens:
            self.fe.append_own_speculative_nvfp4_agent(
                input_ids,
                start_pos=int(cached_tokens),
                max_new_tokens=int(max_tokens),
                K=int(K),
            )
        else:
            self.fe.prefill_own_speculative_nvfp4_agent(
                input_ids, max_new_tokens=int(max_tokens), K=int(K))
        self._last_prompt_tokens = prompt_len
        self._last_prefill_ms = (time.perf_counter() - t0) * 1000.0

    def generate_stream(self, *, max_tokens: int,
                        K: int) -> Iterable[DecodeChunk]:
        for token_chunk in self.fe.decode_own_speculative_nvfp4_committed_stream(
                max_new_tokens=int(max_tokens), K=int(K)):
            ids = tuple(int(t) for t in token_chunk)
            text = self.fe._tokenizer.decode(
                list(ids), skip_special_tokens=False)
            yield DecodeChunk(token_ids=ids, text=text, accepted=len(ids))
