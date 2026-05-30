"""Qwen3.6 frontend adapter for the agent-serving policy layer."""

from __future__ import annotations

import re
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
        self._last_route = "unknown"
        self._last_enable_thinking = False

    @property
    def spec_enabled(self) -> bool:
        """True if the MTP head is loaded (speculative decode available). When
        False the committed-stream path has no draft chain and decode runs the
        slower non-spec fallback — set FLASHRT_QWEN36_MTP_CKPT_DIR to enable."""
        weights = getattr(self.fe, "_weights", None)
        ptrs = getattr(weights, "ptrs", None) if weights is not None else None
        return bool(ptrs and ptrs.get("mtp") is not None)

    @classmethod
    def from_checkpoint(
            cls, checkpoint: str, *, device: str = "cuda",
            max_seq: int = 262208, model_name: str = "qwen36-27b",
            route_min_seq: int | None = None,
            graph_cache_max: int | None = None):
        """Load the hardware-matched Qwen3.6 frontend."""
        import torch

        cap = torch.cuda.get_device_capability()
        if cap[0] >= 12:
            # The production Qwen3.6 server should use the same SM120 fast
            # decode/verify kernels as the documented benchmark path unless a
            # caller explicitly opts out before startup.
            import os
            os.environ.setdefault("FLASHRT_QWEN36_DECODE_FASTGEMM", "1")
            os.environ.setdefault("FLASHRT_QWEN36_VERIFY_WARPSPLIT", "1")
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
        if route_min_seq is not None and getattr(fe, "_long_ctx_mode", False):
            fe._long_ctx_route_min_seq = max(0, min(
                int(route_min_seq), int(getattr(fe, "_user_max_seq", max_seq))))
        if graph_cache_max is not None:
            fe.GRAPH_CACHE_MAX = int(graph_cache_max)
        return cls(fe, model_name=model_name)

    def render_chat(self, messages, tools=None, *,
                    add_generation_prompt: bool = True,
                    enable_thinking: bool = False) -> str:
        self._last_enable_thinking = bool(enable_thinking)
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            normalized.append(msg)
        return self.fe._tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def tokenize_text(self, text: str) -> List[int]:
        return list(self.fe._tokenizer(
            text, add_special_tokens=False).input_ids)

    def tokenize_chat(self, messages, tools=None, *,
                      enable_thinking: bool = False) -> List[int]:
        prompt = self.render_chat(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self.tokenize_text(prompt)

    def append_suffix_tokens_for_messages(
            self, previous_messages, incoming_messages, *,
            tools=None, enable_thinking: bool = False) -> List[int] | None:
        """Tokenize only the message suffix after a completed assistant turn."""
        if tools:
            return None
        if not previous_messages or len(incoming_messages) <= len(previous_messages):
            return None
        if incoming_messages[:len(previous_messages)] != previous_messages:
            return None
        last = previous_messages[-1]
        if last.get("role") != "assistant":
            return None
        content = last.get("content") or ""
        rendered_content = content.rstrip()
        rendered = self.render_chat(
            incoming_messages,
            tools=None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        idx = rendered.rfind(rendered_content)
        if idx < 0:
            return None
        suffix = rendered[idx + len(rendered_content):]
        if not suffix:
            return None
        return self.tokenize_text(suffix)

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
        t0 = time.perf_counter()
        if use_long:
            if cached_tokens:
                self.fe.append_long_ctx_nvfp4_agent(
                    input_ids,
                    start_pos=int(cached_tokens),
                    max_new_tokens=int(max_tokens),
                    K=int(K),
                )
            else:
                self.fe.prefill_long_ctx_nvfp4_agent(
                    input_ids, max_new_tokens=int(max_tokens), K=int(K))
            self._last_route = "long"
            self._last_prompt_tokens = prompt_len
            self._last_prefill_ms = (time.perf_counter() - t0) * 1000.0
            return

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
        self._last_route = "short"
        self._last_prompt_tokens = prompt_len
        self._last_prefill_ms = (time.perf_counter() - t0) * 1000.0

    def generate_stream(self, *, max_tokens: int,
                        K: int) -> Iterable[DecodeChunk]:
        if self._last_route == "long":
            chunks = self.fe.decode_long_ctx_nvfp4_committed_stream(
                max_new_tokens=int(max_tokens), K=int(K))
        else:
            chunks = self.fe.decode_own_speculative_nvfp4_committed_stream(
                max_new_tokens=int(max_tokens), K=int(K))
        stop_ids = self._visible_stop_token_ids()
        for token_chunk in chunks:
            ids = tuple(int(t) for t in token_chunk)
            stop_at = next(
                (i for i, tok in enumerate(ids) if tok in stop_ids), None)
            visible_ids = ids if stop_at is None else ids[:stop_at]
            text = self.fe._tokenizer.decode(
                list(visible_ids), skip_special_tokens=True)
            if not self._last_enable_thinking:
                text = re.sub(r"<think>.*?</think>\s*", "", text,
                              flags=re.DOTALL)
                text = text.replace("<think>", "").replace("</think>", "")
            if stop_at is None:
                yield DecodeChunk(
                    token_ids=ids, text=text, accepted=len(ids))
                continue
            # Stop token mid-chunk: the frontend committed `ids` to KV state, but
            # only `visible_ids` belong to the transcript. Commit the visible
            # tokens to the journal and report the post-stop tokens (the stop
            # token itself + any verified tail) as state lookahead so the session
            # is rebuilt rather than falsely hot-appended.
            yield DecodeChunk(
                token_ids=visible_ids, text=text, accepted=len(visible_ids),
                stop=True, state_lookahead=len(ids) - len(visible_ids))
            break

    def _visible_stop_token_ids(self) -> set[int]:
        tokenizer = self.fe._tokenizer
        out = set()
        for attr in ("eos_token_id", "pad_token_id"):
            tok = getattr(tokenizer, attr, None)
            if tok is not None:
                out.add(int(tok))
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            for token in ("<|im_end|>", "<|endoftext|>"):
                try:
                    tok = convert(token)
                except Exception:
                    tok = None
                if isinstance(tok, int) and tok >= 0:
                    out.add(tok)
        return out

    def dummy_token_ids(self, prompt_len: int):
        """Build exact-length token ids for startup graph warmup."""
        import torch

        prompt_len = int(prompt_len)
        if prompt_len <= 0:
            raise ValueError("prompt_len must be > 0")
        token_ids = self.fe._tokenizer(
            " warmup", add_special_tokens=False).input_ids
        token = int(token_ids[0] if token_ids else 1)
        return torch.full(
            (1, prompt_len),
            token,
            device=getattr(self.fe, "device", "cuda"),
            dtype=torch.long,
        )

    def warmup_committed_stream(
            self,
            shapes: Sequence[tuple[int, int]],
            *,
            K: int = 6,
            committed_max_prompt: int = 1024,
            long_decode_graphs: bool = True,
            long_prefill_graphs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Move committed-stream graph capture out of the first request.

        Small and medium shapes run the real agent split prefill+committed
        stream once. Large long-context shapes use the frontend's graph-only
        warmup hooks so startup does not spend minutes in synthetic prefill.
        """
        import torch

        out: List[Dict[str, Any]] = []
        for prompt_len, max_tokens in shapes:
            prompt_len = int(prompt_len)
            max_tokens = int(max_tokens)
            if prompt_len <= 0 or max_tokens <= 0:
                continue
            if self.max_seq and prompt_len + max_tokens > self.max_seq:
                out.append({
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "route": "skip",
                    "reason": "exceeds max_seq",
                })
                continue

            use_long = (
                getattr(self.fe, "_long_ctx_mode", False)
                and hasattr(self.fe, "_should_use_long_ctx_route")
                and self.fe._should_use_long_ctx_route(prompt_len, max_tokens)
            )
            t0 = time.perf_counter()
            if use_long and prompt_len > int(committed_max_prompt):
                prefill_warmed = []
                decode_warmed = []
                if (long_prefill_graphs
                        and hasattr(self.fe,
                                    "warmup_long_ctx_prefill_graphs")):
                    prefill_warmed = self.fe.warmup_long_ctx_prefill_graphs(
                        [(prompt_len, max_tokens)])
                if (long_decode_graphs
                        and hasattr(self.fe,
                                    "warmup_long_ctx_decode_graphs")):
                    decode_warmed = self.fe.warmup_long_ctx_decode_graphs(
                        [(prompt_len, max_tokens)], K=K)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                out.append({
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "route": "long_graphs",
                    "prefill_graphs": len(prefill_warmed),
                    "decode_graphs": len(decode_warmed),
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                })
                continue

            ids = self.dummy_token_ids(prompt_len)
            self.prefill(ids.view(-1).tolist(), max_tokens=max_tokens, K=K)
            _ = list(self.generate_stream(max_tokens=max_tokens, K=K))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out.append({
                "prompt_len": prompt_len,
                "max_tokens": max_tokens,
                "route": self._last_route,
                "prefill_ms": self._last_prefill_ms,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
            })
        return out
