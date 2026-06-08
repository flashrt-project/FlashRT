from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync() -> None:
    torch.cuda.synchronize()


def bench(fn, *, warmup: int, repeat: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times)
    return {
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": times_sorted[len(times_sorted) // 2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    print("torch", torch.__version__, "hip", torch.version.hip)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval()
    sync()
    print("LOAD_S", time.perf_counter() - t0)

    base = "FlashRT ROCm Qwen3 baseline latency measurement. "
    text = (base * ((args.prompt_len // 8) + 16)).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.prompt_len].to("cuda")
    if ids.shape[1] < args.prompt_len:
        pad = ids[:, -1:].expand(1, args.prompt_len - ids.shape[1])
        ids = torch.cat([ids, pad], dim=1)
    attn = torch.ones_like(ids)
    print("PROMPT_TOKENS", int(ids.shape[1]))

    @torch.inference_mode()
    def prefill_only():
        return model(input_ids=ids, attention_mask=attn, use_cache=True)

    @torch.inference_mode()
    def ttft_generate():
        return model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )

    prefill_stats = bench(prefill_only, warmup=args.warmup, repeat=args.repeat)
    print("PREFILL_ONLY", prefill_stats)
    ttft_stats = bench(ttft_generate, warmup=args.warmup, repeat=args.repeat)
    print("TTFT_GENERATE_1", ttft_stats)

    with torch.inference_mode():
        out = prefill_only()
        past = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        position = ids.shape[1]
        decode_times = []
        for _ in range(args.decode_steps):
            pos = torch.tensor([[position]], device="cuda", dtype=torch.long)
            t0 = time.perf_counter()
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
                position_ids=pos,
            )
            sync()
            decode_times.append((time.perf_counter() - t0) * 1000.0)
            past = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            position += 1
    decode_sorted = sorted(decode_times)
    decode_stats = {
        "steps": len(decode_times),
        "times_ms": decode_times,
        "mean_ms": sum(decode_times) / len(decode_times),
        "min_ms": min(decode_times),
        "max_ms": max(decode_times),
        "p50_ms": decode_sorted[len(decode_sorted) // 2],
        "tok_per_s": 1000.0 / (sum(decode_times) / len(decode_times)),
    }
    print("DECODE_ONE_BY_ONE", decode_stats)
    print("MAX_MEM_GB", torch.cuda.max_memory_allocated() / 1024**3)


if __name__ == "__main__":
    main()
