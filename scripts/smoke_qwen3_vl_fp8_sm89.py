#!/usr/bin/env python3
"""End-to-end smoke/bench for Qwen3-VL official-FP8 SM89 frontends."""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch
from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _event_time_ms(fn, iters: int) -> list[float]:
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _summarize(times: list[float]) -> str:
    return (
        f"median={statistics.median(times):.3f} ms "
        f"mean={statistics.mean(times):.3f} ms min={min(times):.3f} ms")


def _bench_text(args) -> None:
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
        Qwen3VlFp8Sm89TextFrontend,
    )

    fe = Qwen3VlFp8Sm89TextFrontend(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        max_prefill_seq=args.max_prefill_seq,
        fuse_gate_up=args.fuse_gate_up,
        fuse_qk_postproc=not args.no_fuse_qk_postproc,
        use_fp8_lm_head=args.fp8_lm_head)
    hidden = int(fe._cfg["hidden_size"])
    embed = fe._weights.anchors[0]
    ids = torch.arange(args.text_token_base, args.text_token_base + args.text_s,
                       device=args.device, dtype=torch.long)
    h = embed[ids].to(torch.bfloat16).view(args.text_s, hidden).contiguous()
    cos = fe._rope_cos_table[:args.text_s]
    sin = fe._rope_sin_table[:args.text_s]

    fe.reset_state()
    fe.forward_hidden_prefill_fp8_blockscaled(h, cos, sin, 0)
    torch.cuda.synchronize()
    prefill_times = []
    logits = None
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fe.reset_state()
        logits = fe.forward_hidden_prefill_fp8_blockscaled(h, cos, sin, 0)
        torch.cuda.synchronize()
        prefill_times.append((time.perf_counter() - t0) * 1000.0)
    logits_f = logits.detach().float().clone()

    fe.reset_state()
    loop_logits = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for pos in range(args.text_s):
        loop_logits = fe.forward_hidden_decode_fp8(
            h[pos], cos[pos:pos + 1], sin[pos:pos + 1], pos)
    torch.cuda.synchronize()
    loop_ms = (time.perf_counter() - t0) * 1000.0
    loop_f = loop_logits.detach().float().clone()
    cosv = torch.nn.functional.cosine_similarity(logits_f, loop_f).item()

    fe.reset_state()
    fe.decode_step_with_graph(args.decode_token, args.decode_pos)
    torch.cuda.synchronize()
    decode_times = _event_time_ms(
        lambda: fe.decode_step_with_graph(args.decode_token, args.decode_pos),
        args.iters)

    print("--- text ---")
    print(f"S={args.text_s} prefill {_summarize(prefill_times)}")
    print(f"token_loop_ms={loop_ms:.3f} "
          f"prefill_speedup={loop_ms / statistics.median(prefill_times):.2f}x "
          f"logit_cos={cosv:.6f} top_prefill={int(logits_f.argmax())} "
          f"top_loop={int(loop_f.argmax())}")
    print(f"graph_decode_pos={args.decode_pos} {_summarize(decode_times)} "
          f"top={int(fe._logits_buf.argmax())} "
          f"finite={torch.isfinite(fe._logits_buf).all().item()}")


def _bench_multimodal(args) -> None:
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )

    fe = Qwen3VlFp8Sm89Frontend(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        max_prefill_seq=args.max_prefill_seq, max_pixels=args.max_pixels,
        fuse_gate_up=args.fuse_gate_up,
        fuse_qk_postproc=not args.no_fuse_qk_postproc,
        use_fp8_lm_head=args.fp8_lm_head,
        vision_bf16_first_blocks=args.vision_bf16_first_blocks,
        vision_bf16_block_linears=(
            None if not args.vision_block2_bf16_linears else {
                2: tuple(
                    name.strip()
                    for name in args.vision_block2_bf16_linears.split(",")
                    if name.strip())
            }))
    img = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.prompt},
        ],
    }]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fe.set_prompt(messages)
    torch.cuda.synchronize()
    set_prompt_ms = (time.perf_counter() - t0) * 1000.0
    set_prompt_warm_times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fe.set_prompt(messages)
        torch.cuda.synchronize()
        set_prompt_warm_times.append((time.perf_counter() - t0) * 1000.0)
    p = fe._prompt
    assert p is not None
    llm = fe.llm
    S = int(p["S"])
    hidden = int(llm._cfg["hidden_size"])
    embed = llm._weights.anchors[0]
    h = embed[p["input_ids"]].to(torch.bfloat16).view(S, hidden).contiguous()

    vision_times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        off = 0
        for n_patch in p["seg_patches"]:
            sl = slice(off, off + n_patch)
            off += n_patch
            fe.vision.forward(
                p["pixel_values"][sl], p["pos_embeds"][sl],
                p["vcos"][sl], p["vsin"][sl])
        torch.cuda.synchronize()
        vision_times.append((time.perf_counter() - t0) * 1000.0)

    llm.reset_state()
    llm.forward_hidden_prefill_fp8_blockscaled(
        h, p["mcos"], p["msin"], 0, run_lm_head=False)
    torch.cuda.synchronize()
    lang_times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.reset_state()
        llm.forward_hidden_prefill_fp8_blockscaled(
            h, p["mcos"], p["msin"], 0, run_lm_head=False)
        torch.cuda.synchronize()
        lang_times.append((time.perf_counter() - t0) * 1000.0)

    fe.prefill()
    torch.cuda.synchronize()
    prefill_times = []
    logits = None
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = fe.prefill()
        torch.cuda.synchronize()
        prefill_times.append((time.perf_counter() - t0) * 1000.0)

    prefill_logits_f = logits.detach().float().clone()
    graph_logits_f = prefill_logits_f
    graph_prefill_times = None
    if len(p["spans"]) == 1:
        fe.prefill_graph()
        torch.cuda.synchronize()
        graph_prefill_times = _event_time_ms(
            lambda: fe.prefill_graph(), args.iters)
        graph_logits_f = fe.llm._logits_buf.detach().float().clone()

    tok = int(graph_logits_f.argmax())
    cache_pos = int(p["S"])
    eager = fe.decode_step(tok, cache_pos)
    eager_f = eager.detach().float().clone()
    graph = fe.decode_step_with_graph(tok, cache_pos)
    graph_f = graph.detach().float().clone()
    torch.cuda.synchronize()
    decode_times = _event_time_ms(
        lambda: fe.decode_step_with_graph(tok, cache_pos), args.iters)
    cosv = torch.nn.functional.cosine_similarity(eager_f, graph_f).item()

    generated = ""
    if args.generate_tokens > 0:
        generated = fe.generate(
            messages, max_new_tokens=args.generate_tokens, use_graph=True)

    print("--- multimodal ---")
    print(f"image={args.image} S={p['S']} "
          f"pixel_shape={tuple(p['pixel_values'].shape)} "
          f"spans={p['spans']} set_prompt_ms={set_prompt_ms:.3f}")
    print(f"set_prompt_warm {_summarize(set_prompt_warm_times)}")
    print(f"vision_only {_summarize(vision_times)}")
    print(f"language_only_no_mm_scatter {_summarize(lang_times)}")
    print(f"prefill {_summarize(prefill_times)} "
          f"top={int(prefill_logits_f.argmax())} "
          f"finite={torch.isfinite(prefill_logits_f).all().item()}")
    if graph_prefill_times is not None:
        graph_cos = torch.nn.functional.cosine_similarity(
            prefill_logits_f, graph_logits_f).item()
        print(f"prefill_graph {_summarize(graph_prefill_times)} "
              f"cos_vs_eager={graph_cos:.6f} "
              f"top={int(graph_logits_f.argmax())} "
              f"finite={torch.isfinite(graph_logits_f).all().item()}")
    print(f"graph_decode_cache_pos={cache_pos} {_summarize(decode_times)} "
          f"cos_vs_eager={cosv:.6f} top_eager={int(eager_f.argmax())} "
          f"top_graph={int(graph_f.argmax())}")
    if args.generate_tokens > 0:
        print(f"generate_tokens={args.generate_tokens} text={generated!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--max-prefill-seq", type=int, default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--multimodal", action="store_true")
    p.add_argument("--text-s", type=int, default=79)
    p.add_argument("--text-token-base", type=int, default=100)
    p.add_argument("--decode-token", type=int, default=100)
    p.add_argument("--decode-pos", type=int, default=63)
    p.add_argument("--fuse-gate-up", action="store_true")
    p.add_argument("--no-fuse-qk-postproc", action="store_true",
                   help="use the older two-launch Q/K postprocess path")
    p.add_argument("--fp8-lm-head", action="store_true")
    p.add_argument("--vision-bf16-first-blocks", type=int, default=3)
    p.add_argument("--vision-block2-bf16-linears", default="",
                   help="experimental SM89-only override for vision block 2; "
                        "comma-separated subset of qkv,proj,fc1,fc2")
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--image", default=str(REPO_ROOT / "FlashRT.png"))
    p.add_argument("--prompt",
                   default="Describe this image in one sentence.")
    p.add_argument("--generate-tokens", type=int, default=2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(args.device))
    if args.warmup != 1:
        print("note: --warmup is accepted for CLI symmetry; "
              "frontends perform their own first-call warmup")
    run_text = args.text_only or not args.multimodal
    run_mm = args.multimodal or not args.text_only
    print(f"device={torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"checkpoint={args.checkpoint}")
    if run_text:
        _bench_text(args)
    if run_mm:
        _bench_multimodal(args)


if __name__ == "__main__":
    main()
