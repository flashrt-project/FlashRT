"""Qwen3.6 frontend adapter for the agent-serving policy layer."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

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
        cls._set_agent_runtime_env_defaults(cap[0])
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

    @staticmethod
    def _set_agent_runtime_env_defaults(cap_major: int) -> None:
        """Set agent-serving runtime defaults before frontend construction.

        The fixed-shape benchmark path benefits from exact-position CUDA Graph
        replay. A long-lived agent session does not: every new generated
        position can be a never-seen graph key, so the first real coding-agent
        turn pays capture on the hot path and drops to cold-capture throughput.
        Keep the fast SM120 kernels on, but default the agent host to direct
        long-decode kernel launch for verify/MTP-chain unless the caller opts
        back into exact graph replay before startup.
        """
        import os

        if int(cap_major) >= 12:
            os.environ.setdefault("FLASHRT_QWEN36_DECODE_FASTGEMM", "1")
            os.environ.setdefault("FLASHRT_QWEN36_VERIFY_WARPSPLIT", "1")
            os.environ.setdefault("FLASHRT_QWEN36_TQ_VERIFY_GRAPH", "0")
            os.environ.setdefault("FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH", "0")

    def render_chat(self, messages, tools=None, *,
                    add_generation_prompt: bool = True,
                    enable_thinking: bool = False) -> str:
        self._last_enable_thinking = bool(enable_thinking)
        return self._render_chat(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def _render_chat(self, messages, tools=None, *,
                     add_generation_prompt: bool = True,
                     enable_thinking: bool = False) -> str:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            else:
                msg = dict(msg)
            if msg.get("tool_calls"):
                msg["tool_calls"] = self._normalize_tool_calls(
                    msg.get("tool_calls"))
            normalized.append(msg)
        return self.fe._tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> Any:
        """Qwen's chat template expects assistant tool-call arguments as a
        mapping so it can emit one <parameter=...> block per key. OpenAI wire
        format carries ``function.arguments`` as a JSON string; normalize it
        before handing history back to the tokenizer."""
        if not isinstance(tool_calls, list):
            return tool_calls
        out = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                out.append(tc)
                continue
            tc = dict(tc)
            fn = tc.get("function")
            if isinstance(fn, dict):
                fn = dict(fn)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                    except Exception:
                        parsed = {"arguments": args}
                    if isinstance(parsed, dict):
                        fn["arguments"] = parsed
                    else:
                        fn["arguments"] = {"arguments": parsed}
                tc["function"] = fn
            out.append(tc)
        return out

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

    def tokenize_chat_for_validation(self, messages, tools=None, *,
                                     enable_thinking: bool = False) -> List[int]:
        """Tokenize a request for HTTP pre-validation without mutating hot
        decode state. The normal ``tokenize_chat`` records ``enable_thinking``
        for committed-stream filtering; this method is safe to call before a
        StreamingResponse is returned while another stream owns the engine."""
        prompt = self._render_chat(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self.tokenize_text(prompt)

    def append_suffix_tokens_for_messages(
            self, previous_messages, incoming_messages, *,
            tools=None, enable_thinking: bool = False) -> List[int] | None:
        """Tokenize only the message suffix after a completed assistant turn.

        Agent clients replay tool-call history through their own OpenAI/SDK
        adapters.  The replayed full prompt may not be byte-identical to the
        exact tool-call text the model generated, but the continuation after
        the completed assistant message is still appendable from the hot GPU
        state.  Prefer a message-boundary suffix over full-prompt token prefix
        matching whenever the previous visible messages are a strict prefix of
        the incoming messages.
        """
        if not previous_messages or len(incoming_messages) <= len(previous_messages):
            return None
        if incoming_messages[:len(previous_messages)] != previous_messages:
            return None

        previous_rendered = self.render_chat(
            previous_messages,
            tools=tools or None,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        rendered = self.render_chat(
            incoming_messages,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not rendered.startswith(previous_rendered):
            # Qwen3.6's non-thinking generation prompt still injects an
            # internal ``<think>\n\n</think>\n\n`` control block before decode.
            # That block is part of the hot KV state but is intentionally not
            # replayed by OpenAI clients as visible assistant content. When the
            # visible message prefix is already known-equivalent, append only
            # the newly added messages after the committed stop-token boundary
            # instead of requiring the full visible render to be a byte prefix
            # of the internal prompt journal.
            suffix_messages = incoming_messages[len(previous_messages):]
            if not suffix_messages:
                return None
            suffix = self.fe._tokenizer.apply_chat_template(
                suffix_messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if suffix:
                suffix = "\n" + suffix
        else:
            suffix = rendered[len(previous_rendered):]
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

    def _to_input_ids(self, token_ids: Sequence[int]):
        import torch

        return torch.tensor(
            [[int(t) for t in token_ids]],
            device=getattr(self.fe, "device", "cuda"),
            dtype=torch.long,
        )

    def _uses_long_route(self, prompt_len: int, max_tokens: int) -> bool:
        return bool(
            getattr(self.fe, "_long_ctx_mode", False)
            and hasattr(self.fe, "_should_use_long_ctx_route")
            and self.fe._should_use_long_ctx_route(int(prompt_len), int(max_tokens))
        )

    def supports_capsule(self) -> bool:
        """True if the frontend exposes the capsule snapshot/restore API."""
        return (
            hasattr(self.fe, "snapshot_capsule")
            and hasattr(self.fe, "restore_capsule")
            and hasattr(self.fe, "capsule_aligned_len")
        )

    def capsule_aligned_len(self, prompt_len: int, max_tokens: int) -> int:
        """Chunk-aligned boundary to pin a shared prefix at for this prompt's
        route, or 0 if it cannot be pinned (short route, or shorter than one
        prefill chunk). Snapshotting at this boundary makes restore + append
        token-identical to a cold full prefill (see the frontend capsule API)."""
        if not self.supports_capsule():
            return 0
        if not self._uses_long_route(prompt_len, max_tokens):
            return 0
        return int(self.fe.capsule_aligned_len(int(prompt_len)))

    def prefill_and_pin(self, token_ids: Sequence[int], *, aligned_len: int,
                        max_tokens: int = 1, K: int = 6):
        """Cold-prefill the chunk-aligned head, snapshot it into a capsule, then
        append the remainder so the stream is ready to decode. Returns the opaque
        capsule. Long route only (``aligned_len`` > 0 comes from
        ``capsule_aligned_len``)."""
        if aligned_len <= 0:
            raise ValueError("aligned_len must be > 0 to pin a capsule")
        ids = self._to_input_ids(token_ids)
        prompt_len = int(ids.shape[1])
        if aligned_len > prompt_len:
            raise ValueError("aligned_len exceeds prompt length")
        self.fe.prefill_long_ctx_nvfp4_agent(
            ids[:, :aligned_len], max_new_tokens=int(max_tokens), K=int(K))
        cap = self.fe.snapshot_capsule()
        if prompt_len > aligned_len:
            self.fe.append_long_ctx_nvfp4_agent(
                ids, start_pos=int(aligned_len),
                max_new_tokens=int(max_tokens), K=int(K))
        self._last_route = "long"
        self._last_prompt_tokens = prompt_len
        return cap

    def prefill_from_capsule(self, capsule, token_ids: Sequence[int], *,
                             max_tokens: int = 1, K: int = 6) -> None:
        """Restore a pinned capsule and append the remainder of ``token_ids`` after
        the snapshot boundary, leaving the stream ready to decode. Token-identical
        to a cold full prefill of ``token_ids`` (capsule correctness contract)."""
        aligned = int(capsule["cur_pos"])
        ids = self._to_input_ids(token_ids)
        prompt_len = int(ids.shape[1])
        self.fe.restore_capsule(capsule)
        if prompt_len > aligned:
            self.fe.append_long_ctx_nvfp4_agent(
                ids, start_pos=aligned,
                max_new_tokens=int(max_tokens), K=int(K))
        self._last_route = "long"
        self._last_prompt_tokens = prompt_len

    def generate_stream(self, *, max_tokens: int,
                        K: int) -> Iterable[DecodeChunk]:
        stop_ids = self._visible_stop_token_ids()
        if self._last_route == "long":
            chunks = self._committed_stream(
                self.fe.decode_long_ctx_nvfp4_committed_stream,
                max_tokens=max_tokens, K=K, stop_ids=stop_ids)
        else:
            chunks = self._committed_stream(
                self.fe.decode_own_speculative_nvfp4_committed_stream,
                max_tokens=max_tokens, K=K, stop_ids=stop_ids)
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
            # Stop token mid-chunk: the token itself is the chat-template
            # boundary that a later full-history prompt will contain, so keep it
            # in the cache journal while hiding it from client-visible text. Any
            # tokens verified after the stop are not part of the transcript and
            # still force a rebuild.
            committed_ids = ids[:stop_at + 1]
            yield DecodeChunk(
                token_ids=committed_ids, text=text, accepted=len(visible_ids),
                stop=True, state_lookahead=len(ids) - len(committed_ids))
            break

    @staticmethod
    def _committed_stream(fn, *, max_tokens: int, K: int, stop_ids: set[int]):
        try:
            return fn(max_new_tokens=int(max_tokens), K=int(K),
                      stop_token_ids=tuple(int(t) for t in stop_ids))
        except TypeError as exc:
            if "stop_token_ids" not in str(exc):
                raise
            return fn(max_new_tokens=int(max_tokens), K=int(K))

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
            on_result: Callable[[int, int, Dict[str, Any]], None] | None = None,
    ) -> List[Dict[str, Any]]:
        """Move committed-stream graph capture out of the first request.

        Small and medium shapes run the real agent split prefill+committed
        stream once. Large long-context shapes use the frontend's graph-only
        warmup hooks so startup does not spend minutes in synthetic prefill.
        """
        import torch

        out: List[Dict[str, Any]] = []
        for prompt_len, max_tokens in shapes:
            index = len(out) + 1
            prompt_len = int(prompt_len)
            max_tokens = int(max_tokens)
            if prompt_len <= 0 or max_tokens <= 0:
                continue
            if self.max_seq and prompt_len + max_tokens > self.max_seq:
                item = {
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "route": "skip",
                    "reason": "exceeds max_seq",
                }
                out.append(item)
                if on_result is not None:
                    on_result(index, len(shapes), item)
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
                item = {
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "route": "long_graphs",
                    "prefill_graphs": len(prefill_warmed),
                    "decode_graphs": len(decode_warmed),
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                }
                out.append(item)
                if on_result is not None:
                    on_result(index, len(shapes), item)
                continue

            ids = self.dummy_token_ids(prompt_len)
            self.prefill(ids.view(-1).tolist(), max_tokens=max_tokens, K=K)
            _ = list(self.generate_stream(max_tokens=max_tokens, K=K))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            item = {
                "prompt_len": prompt_len,
                "max_tokens": max_tokens,
                "route": self._last_route,
                "prefill_ms": self._last_prefill_ms,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
            }
            out.append(item)
            if on_result is not None:
                on_result(index, len(shapes), item)
        return out
