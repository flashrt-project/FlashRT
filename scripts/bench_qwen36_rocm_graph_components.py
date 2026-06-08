from __future__ import annotations
import os

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_rt.frontends.torch.qwen36_rocm_owned import Qwen36RocmOwnedFP8Frontend


def mean_ms(vals: list[float]) -> float:
    return statistics.mean(vals) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="FlashRT ROCm prompt prefill")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    frontend = Qwen36RocmOwnedFP8Frontend(
        os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B-FP8"),
        warmup_graph=False,
        persistent_linear_state=True,
        use_full_attn_kv=True,
    )
    ids = frontend.tokenizer(args.prompt, return_tensors="pt").input_ids[0].to("cuda")
    prompt_len = int(ids.numel())
    steps = int(args.max_new_tokens)
    frontend.capture_prefill_graph(prompt_len)
    frontend.capture_decode_graph_table(prompt_len, steps, token_id=0)

    # Warm a full generate once so all graphs and workspaces are live.
    frontend.generate_from_ids_graph(ids, max_new_tokens=steps, collect_logits=False)

    prefill_times = []
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = frontend.replay_prefill_graph(ids, sync=False)
        torch.cuda.synchronize()
        prefill_times.append(time.perf_counter() - t0)
    next_token = int(logits[-1].float().argmax().item())

    decode_times = []
    for _ in range(args.repeat):
        frontend.replay_prefill_graph(ids, sync=False)
        token = next_token
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(steps):
            pos = prompt_len + step
            frontend.input_ids_buf.fill_(token)
            frontend._decode_graphs[(pos, pos + 1)].replay()
            token = int(frontend.logits[0].float().argmax().item())
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t0)

    full_times = []
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = frontend.generate_from_ids_graph(
            ids, max_new_tokens=steps, collect_logits=False
        )
        torch.cuda.synchronize()
        full_times.append(time.perf_counter() - t0)

    print("prompt_len", prompt_len)
    print("max_new_tokens", steps)
    print("generated", result["generated_tokens"])
    print("prefill_graph_ms_mean", mean_ms(prefill_times))
    print("decode_graph_ms_mean", mean_ms(decode_times))
    print("decode_tok_s", steps / statistics.mean(decode_times))
    print("full_generate_ms_mean", mean_ms(full_times))
    print("full_generate_tok_s", steps / statistics.mean(full_times))
    print("OK bench_qwen36_rocm_graph_components")


if __name__ == "__main__":
    main()
