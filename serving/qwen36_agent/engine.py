"""Engine protocol for the Qwen3.6 agent-serving host.

This module deliberately defines interfaces only.  The production backend will
wrap ``Qwen36TorchFrontendRtx`` and expose split prefill/decode operations while
the serving policy stays independent of Torch internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence


@dataclass(frozen=True)
class DecodeChunk:
    token_ids: tuple[int, ...]
    text: str
    accepted: int


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    cached_tokens: int
    new_prefill_tokens: int
    prefill_ms: float
    first_delta_ms: float
    decode_ms: float
    decode_tok_per_s: float
    graph_misses: int = 0


class AgentEngine(Protocol):
    """Minimal hot-path surface needed by the serving policy."""

    model_name: str
    max_seq: int

    def tokenize_chat(self, messages, tools=None, *,
                      enable_thinking: bool = False) -> list[int]:
        ...

    def prefill(self, token_ids: Sequence[int], *,
                cached_tokens: int = 0) -> None:
        ...

    def generate_stream(self, *, max_tokens: int, K: int) -> Iterable[DecodeChunk]:
        ...
