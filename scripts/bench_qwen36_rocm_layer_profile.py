from __future__ import annotations
import os

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_rt.frontends.torch.qwen36_rocm_owned import Qwen36OwnedDecodeRunner
from flash_rt.frontends.torch.qwen36_rocm_weights import extract_weights_qwen36_fp8_rocm
from transformers import AutoTokenizer


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def summarize(rows: list[dict]) -> None:
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row["ms"])
    total = sum(row["ms"] for row in rows)
    print("layer_total_ms", f"{total:.3f}")
    for typ, vals in sorted(by_type.items()):
        print(
            "layer_type",
            typ,
            "count",
            len(vals),
            "sum_ms",
            f"{sum(vals):.3f}",
            "mean_ms",
            f"{statistics.mean(vals):.3f}",
            "max_ms",
            f"{max(vals):.3f}",
        )
    print("slowest_layers")
    for row in sorted(rows, key=lambda x: x["ms"], reverse=True)[:10]:
        print(
            "  layer",
            row["idx"],
            row["type"],
            f"{row['ms']:.3f}ms",
        )


def profile_layers(runner: Qwen36OwnedDecodeRunner, h, *, pos_start: int, kv_seq: int):
    rows = []
    input_norm = None
    layers = runner.handles.ptrs["layers"]
    for idx, layer in enumerate(layers):
        typ = layer["type"]
        next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
        next_idx = idx + 1 if next_layer is not None else None
        if typ == "linear_attention":
            (h, input_norm), ms = timed(
                lambda layer=layer, idx=idx, h=h, input_norm=input_norm, next_layer=next_layer, next_idx=next_idx: runner.linear_layer(
                    h,
                    layer,
                    idx,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
            )
        elif int(h.shape[0]) == 1:
            (h, input_norm), ms = timed(
                lambda layer=layer, idx=idx, h=h, input_norm=input_norm, next_layer=next_layer, next_idx=next_idx: runner.full_layer_decode(
                    h,
                    layer,
                    idx,
                    pos_start=pos_start,
                    kv_seq=kv_seq,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
            )
        else:
            (h, input_norm), ms = timed(
                lambda layer=layer, idx=idx, h=h, input_norm=input_norm, next_layer=next_layer, next_idx=next_idx: runner.full_layer_prefill(
                    h,
                    layer,
                    idx,
                    pos_start=pos_start,
                    kv_seq=kv_seq,
                    input_norm=input_norm,
                    next_layer=next_layer,
                    next_idx=next_idx,
                )
            )
        rows.append({"idx": idx, "type": typ, "ms": ms})
    return h, rows


def profile_final_head(runner: Qwen36OwnedDecodeRunner, h):
    def run_norm():
        return runner.norm_quant(h, runner.top("final_norm_eff_w"), "profile_final_norm")

    (_, final_q, final_s), norm_ms = timed(run_norm)

    def run_head():
        out = runner.b(
            "profile_lm_head_fp8_logits",
            (int(final_q.shape[0]), int(runner.top("lm_head_fp8_w").shape[0])),
        )
        runner.aiter.gemm_a8w8_blockscale_ck(
            final_q,
            runner.top("lm_head_fp8_w"),
            final_s,
            runner.top("lm_head_fp8_s"),
            out,
        )
        return out

    logits, head_ms = timed(run_head)
    return logits, norm_ms, head_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="FlashRT ROCm prompt prefill")
    parser.add_argument("--model", default=os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B-FP8"))
    args = parser.parse_args()

    weights = extract_weights_qwen36_fp8_rocm(args.model, weight_mode="fp8_fnuz_cached")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].to("cuda")
    prompt_len = int(ids.numel())
    runner = Qwen36OwnedDecodeRunner(
        weights, persistent_linear_state=True, use_full_attn_kv=True
    )
    embed_w = runner.top("embed_w")

    # Warm all AITER/FA/custom kernels and allocate owned workspaces before
    # profiling. The first live call includes lazy kernel/import overhead and is
    # not useful for fusion decisions.
    runner.reset_linear_states()
    h_warm = embed_w.index_select(0, ids).contiguous()
    logits_warm = runner.full_logits_fp8(h_warm, pos_start=0, kv_seq=prompt_len)
    torch.cuda.synchronize()
    warm_next = int(logits_warm[-1].float().argmax().item())
    h_warm_decode = embed_w.index_select(
        0, torch.tensor([warm_next], device="cuda", dtype=torch.long)
    ).contiguous()
    runner.full_logits_fp8(h_warm_decode, pos_start=prompt_len, kv_seq=prompt_len + 1)
    torch.cuda.synchronize()

    print("prompt_len", prompt_len)
    print("mode", "prefill_seq")
    runner.reset_linear_states()
    h_prefill = embed_w.index_select(0, ids).contiguous()
    h_prefill, prefill_rows = profile_layers(
        runner, h_prefill, pos_start=0, kv_seq=prompt_len
    )
    logits_prefill, final_norm_ms, lm_head_ms = profile_final_head(runner, h_prefill)
    summarize(prefill_rows)
    print("final_norm_ms", f"{final_norm_ms:.3f}")
    print("lm_head_ms", f"{lm_head_ms:.3f}")
    print("prefill_total_with_head_ms", f"{sum(r['ms'] for r in prefill_rows) + final_norm_ms + lm_head_ms:.3f}")
    print("prefill_top_tokens", logits_prefill.float().argmax(dim=1).tolist())

    print("mode", "decode_seq1_after_prefill")
    runner.reset_linear_states()
    h_prefill = embed_w.index_select(0, ids).contiguous()
    logits_prompt = runner.full_logits_fp8(h_prefill, pos_start=0, kv_seq=prompt_len)
    next_token = int(logits_prompt[-1].float().argmax().item())
    h_decode = embed_w.index_select(
        0, torch.tensor([next_token], device="cuda", dtype=torch.long)
    ).contiguous()
    h_decode, decode_rows = profile_layers(
        runner, h_decode, pos_start=prompt_len, kv_seq=prompt_len + 1
    )
    logits_decode, final_norm_ms, lm_head_ms = profile_final_head(runner, h_decode)
    summarize(decode_rows)
    print("final_norm_ms", f"{final_norm_ms:.3f}")
    print("lm_head_ms", f"{lm_head_ms:.3f}")
    print("decode_total_with_head_ms", f"{sum(r['ms'] for r in decode_rows) + final_norm_ms + lm_head_ms:.3f}")
    print("decode_input_token", next_token)
    print("decode_top_token", int(logits_decode[0].float().argmax().item()))


if __name__ == "__main__":
    main()
