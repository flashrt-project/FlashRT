#!/usr/bin/env python3
"""Smoke: Qwen3-4B BF16 prefill on this device, compared with HuggingFace.

    python examples/qwen3_4b_bf16_smoke.py --checkpoint <Qwen3-4B dir> \\
        --prompt "The capital of France is"

Runs the device's FlashRT frontend and ``Qwen3ForCausalLM`` on the same
prompt, then prints the full-logits cosine and both next-token predictions.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from flash_rt.hardware import detect_arch, resolve_pipeline_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    frontend = resolve_pipeline_class("qwen3_4b", "torch", detect_arch())
    fe = frontend(args.checkpoint, max_seq=512)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].cuda()

    try:
        hf = Qwen3ForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16)
    except TypeError:   # transformers < 4.56
        hf = Qwen3ForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    hf = hf.cuda().eval()
    with torch.no_grad():
        ref = hf(input_ids=ids[None]).logits[0].float()
    out = fe.forward_prefill(ids).float()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"{frontend.__name__}: logits {tuple(out.shape)}, full-logits cosine vs HF {cos:.6f}")
    print(f"next token  FlashRT {tok.decode(out[-1].argmax())!r}  HF {tok.decode(ref[-1].argmax())!r}")


if __name__ == "__main__":
    main()
