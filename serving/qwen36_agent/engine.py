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
    """A stream chunk of session-committed, client-visible tokens.

    ``token_ids`` are the visible tokens to commit to the session journal.

    Qwen3.6 speculative decode verifies a whole chunk at once. A committed
    frontend should roll KV/recurrent state back when a stop token lands
    mid-chunk, so ``state_lookahead`` is normally zero. If a backend cannot
    rollback and reports tokens after the visible stop boundary,
    ``state_lookahead`` counts them; a nonzero value means the GPU state leads
    the visible journal, so the session must be rebuilt rather than hot-appended.
    """

    token_ids: tuple[int, ...]
    text: str
    accepted: int
    stop: bool = False
    state_lookahead: int = 0


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
                cached_tokens: int = 0,
                max_tokens: int = 1,
                K: int = 6) -> None:
        """Bring the hot frontend state to ``token_ids``.

        ``cached_tokens`` is an exact prefix already resident in the hot
        contiguous session state.  Implementations must only prefill the suffix
        and must leave the state at the end of ``token_ids``.
        """
        ...

    def generate_stream(self, *, max_tokens: int, K: int) -> Iterable[DecodeChunk]:
        """Yield committed decode chunks.

        Chunks may contain more than one token because FlashRT flushes at
        speculative accept boundaries.  They must not include uncommitted
        lookahead tokens.
        """
        ...
