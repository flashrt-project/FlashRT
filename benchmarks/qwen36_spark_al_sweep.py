"""Spark AL/K sweep for Qwen3.6 NVFP4 speculative decode.

The script keeps one frontend/model loaded and varies only runtime policy
knobs between generations. It is intended for Spark/SM121 acceptance-length
tuning before changing kernels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import torch


GRAPH_CACHE_ATTRS = (
    "_captured_graphs",
    "_captured_verify_graphs",
    "_captured_mtp_graphs",
    "_captured_chain_graphs",
    "_captured_graphs_tq",
    "_captured_verify_graphs_tq",
    "_captured_prefill_graphs_tq",
    "_captured_verify_graphs_fp8kv",
    "_captured_prefill_graphs_fp8kv",
    "_captured_verify_graphs_dflash",
    "_captured_drafter_graphs_dflash",
)


PROMPTS = {
    "repeat": (
        "Qwen Spark benchmark prompt. Keep the answer concise and factual. "
        "This text is repeated only to create a deterministic context."
    ),
    "explain": (
        "Explain CUDA graphs and speculative decoding to an engineer. "
        "Use a direct, technical style and continue the explanation "
        "without changing topics."
    ),
    "code": (
        "Write a clean, commented Python implementation of topological sort. "
        "Continue with edge cases, tests, and complexity analysis."
    ),
    "json": (
        "Extract the following deployment notes into compact JSON records "
        "with fields name, value, and reason. Keep the schema stable."
    ),
    "math": (
        "Solve the optimization problem step by step, show intermediate "
        "equations, and keep the derivation consistent."
    ),
}


@dataclass(frozen=True)
class Case:
    prompt_name: str
    ctx: int
    k_label: str
    tail: str
    tail_kv_only: str


def _parse_csv(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _prompt_text(name: str) -> str:
    if "=" in name:
        label, text = name.split("=", 1)
        if not label.strip() or not text.strip():
            raise argparse.ArgumentTypeError(
                "custom prompt format is label=text")
        return text.strip()
    try:
        return PROMPTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPTS))
        raise argparse.ArgumentTypeError(
            f"unknown prompt {name!r}; choose one of {choices}, "
            "or pass label=text") from exc


def _prompt_label(name: str) -> str:
    return name.split("=", 1)[0].strip() if "=" in name else name


def _build_prompt_ids(tokenizer, device: torch.device, ctx: int,
                      seed_text: str) -> torch.Tensor:
    text = seed_text
    ids = tokenizer(text, return_tensors="pt").input_ids
    while int(ids.shape[1]) < ctx:
        text = f"{text}\n{seed_text}"
        ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, :ctx].contiguous().to(device)


def _tokens(obj) -> list[int]:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().view(-1).tolist()
    if isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item.detach().cpu().view(-1).tolist()
    return list(obj)


def _clear_graph_caches(fe) -> int:
    if hasattr(fe, "clear_graphs"):
        try:
            fe.clear_graphs()
        except TypeError:
            pass
    cleared = 0
    for name in GRAPH_CACHE_ATTRS:
        cache = getattr(fe, name, None)
        if hasattr(cache, "clear"):
            cache.clear()
            cleared += 1
    return cleared


def _set_case_env(case: Case, *, direct: bool) -> int:
    old_k = os.environ.pop("FLASHRT_QWEN36_TQ_SPEC_K", None)
    if case.k_label != "auto":
        os.environ["FLASHRT_QWEN36_TQ_SPEC_K"] = case.k_label
    os.environ["FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL"] = case.tail
    os.environ["FLASHRT_QWEN36_LONG_MTP_TAIL_KV_ONLY"] = case.tail_kv_only
    if direct:
        os.environ["FLASHRT_QWEN36_TQ_VERIFY_GRAPH"] = "0"
        os.environ["FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH"] = "0"
    else:
        os.environ["FLASHRT_QWEN36_TQ_VERIFY_GRAPH"] = "1"
        os.environ["FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH"] = "1"
    return 0 if old_k is None else 1


def _effective_k(fe, ctx: int, k_label: str, caller_k: int,
                 max_new: int) -> int:
    if k_label != "auto":
        return int(k_label)
    if hasattr(fe, "_long_tq_effective_k"):
        return int(fe._long_tq_effective_k(ctx, caller_k, max_new))
    return int(caller_k)


def _caller_k_for_case(k_label: str, caller_k: int) -> int:
    if k_label != "auto":
        return int(k_label)
    return int(caller_k)


def _run_once(fe, ids: torch.Tensor, *, max_new: int,
              caller_k: int) -> tuple[list[int], dict]:
    torch.cuda.synchronize()
    out = fe.generate_own_speculative_KN_nvfp4(
        ids, max_new_tokens=max_new, K=caller_k)
    torch.cuda.synchronize()
    toks = _tokens(out)
    prefill_ms = float(getattr(fe, "_long_ctx_prefill_ms", 0.0) or 0.0)
    decode_ms = float(getattr(fe, "_long_ctx_decode_ms", 0.0) or 0.0)
    attempts = int(getattr(fe, "_spec_attempts", 0) or 0)
    accepts = int(getattr(fe, "_spec_accepts", 0) or 0)
    full = int(getattr(fe, "_spec_full", 0) or 0)
    route = str(getattr(fe, "_long_ctx_route", ""))
    generated = max(0, len(toks) - int(ids.shape[1]))
    return toks, {
        "generated": generated,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tok_s": (generated * 1000.0 / decode_ms) if decode_ms > 0 else 0.0,
        "spec_attempts": attempts,
        "spec_accepts": accepts,
        "accept_per_attempt": (accepts / attempts) if attempts > 0 else 0.0,
        "full_accept_rate": (full / attempts) if attempts > 0 else 0.0,
        "spec_full": full,
        "route": route,
    }


def _writer(path: str):
    fields = (
        "prompt",
        "ctx",
        "K",
        "effective_K",
        "tail",
        "tail_kv_only",
        "direct",
        "rep",
        "generated",
        "prefill_ms",
        "decode_ms",
        "tok_s",
        "spec_attempts",
        "spec_accepts",
        "accept_per_attempt",
        "full_accept_rate",
        "spec_full",
        "same_as_warmup",
        "route",
    )
    if path == "-":
        return sys.stdout, csv.DictWriter(sys.stdout, fieldnames=fields)
    fh = open(path, "w", newline="", encoding="utf-8")
    return fh, csv.DictWriter(fh, fieldnames=fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-dir", default="")
    parser.add_argument("--ctx", type=_parse_csv_ints, default=[128])
    parser.add_argument("--K", type=_parse_csv, default=["auto", "3", "5", "6"])
    parser.add_argument("--tails", type=_parse_csv, default=["auto", "0", "128"])
    parser.add_argument("--tail-kv-only", type=_parse_csv, default=["1"])
    parser.add_argument("--prompts", type=_parse_csv,
                        default=["repeat", "explain", "code", "json"])
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--caller-k", type=int, default=6)
    parser.add_argument("--long-route-min", type=int, default=None)
    parser.add_argument("--long-kv-cache", choices=("fp8", "tq"), default=None)
    parser.add_argument("--frontend", choices=("auto", "rtx", "spark"),
                        default="auto")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--out", default="-")
    parser.add_argument("--use-exec", action="store_true",
                        help="Leave FLASHRT_QWEN36_USE_EXEC enabled.")
    args = parser.parse_args()

    if args.mtp_dir:
        os.environ["FLASHRT_QWEN36_MTP_CKPT_DIR"] = args.mtp_dir
    if not args.use_exec:
        os.environ["FLASHRT_QWEN36_USE_EXEC"] = "0"
    if args.long_route_min is not None:
        os.environ["FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ"] = str(
            args.long_route_min)
    if args.long_kv_cache is not None:
        os.environ["FLASHRT_QWEN36_LONG_KV_CACHE"] = args.long_kv_cache

    frontend_name = args.frontend
    if frontend_name == "auto":
        cap = tuple(int(x) for x in torch.cuda.get_device_capability())
        frontend_name = "spark" if cap == (12, 1) else "rtx"
    if frontend_name == "spark":
        from flash_rt.frontends.torch.qwen36_spark import (
            Qwen36TorchFrontendSpark as Frontend,
        )
    else:
        from flash_rt.frontends.torch.qwen36_rtx import (
            Qwen36TorchFrontendRtx as Frontend,
        )

    fe = Frontend(args.model, quant="nvfp4", max_seq=args.max_seq)
    tokenizer = getattr(fe, "tokenizer", getattr(fe, "_tokenizer"))

    prompt_texts = {name: _prompt_text(name) for name in args.prompts}
    prompt_labels = {name: _prompt_label(name) for name in args.prompts}
    prompt_ids: dict[tuple[str, int], torch.Tensor] = {}
    for raw_name, seed in prompt_texts.items():
        for ctx in args.ctx:
            prompt_ids[(raw_name, ctx)] = _build_prompt_ids(
                tokenizer, fe.device, ctx, seed)

    fh, writer = _writer(args.out)
    try:
        writer.writeheader()
        if fh is not sys.stdout:
            fh.flush()
        for raw_prompt in args.prompts:
            for ctx in args.ctx:
                ids = prompt_ids[(raw_prompt, ctx)]
                for k_label in args.K:
                    if k_label != "auto":
                        int(k_label)
                    for tail in args.tails:
                        if tail != "auto":
                            int(tail)
                        for tail_kv_only in args.tail_kv_only:
                            if tail_kv_only not in ("0", "1"):
                                raise ValueError(
                                    "--tail-kv-only values must be 0 or 1")
                            case = Case(
                                prompt_labels[raw_prompt], ctx, k_label,
                                tail, tail_kv_only)
                            _set_case_env(case, direct=args.direct)
                            _clear_graph_caches(fe)
                            case_caller_k = _caller_k_for_case(
                                k_label, args.caller_k)
                            warm_toks: list[int] | None = None
                            for _ in range(args.warmup):
                                warm_toks, _ = _run_once(
                                    fe, ids, max_new=args.max_new,
                                    caller_k=case_caller_k)
                            effective_k = _effective_k(
                                fe, ctx, k_label, case_caller_k,
                                args.max_new)
                            for rep in range(args.reps):
                                toks, metrics = _run_once(
                                    fe, ids, max_new=args.max_new,
                                    caller_k=case_caller_k)
                                row = {
                                    "prompt": case.prompt_name,
                                    "ctx": case.ctx,
                                    "K": case.k_label,
                                    "effective_K": effective_k,
                                    "tail": case.tail,
                                    "tail_kv_only": case.tail_kv_only,
                                    "direct": int(args.direct),
                                    "rep": rep,
                                    "same_as_warmup": int(
                                        warm_toks == toks
                                        if warm_toks is not None else True),
                                    **metrics,
                                }
                                writer.writerow(row)
                                if fh is not sys.stdout:
                                    fh.flush()
                                else:
                                    sys.stdout.flush()
    finally:
        if fh is not sys.stdout:
            fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
