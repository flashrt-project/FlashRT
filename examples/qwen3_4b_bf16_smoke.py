#!/usr/bin/env python3
"""Smoke: Qwen3-4B BF16 prefill with random weights (no Hub download)."""
from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from flash_rt.frontends.torch._qwen3_4b_rtx_bf16_weights import (
    qwen3_4b_config_dict,
)
from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx


def main() -> None:
    device = "cuda:0"
    cfg = Qwen3Config(**{
        k: v for k, v in qwen3_4b_config_dict().items()
        if k not in ("architectures",)
    })
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = {k: v.detach().contiguous() for k, v in hf.state_dict().items()}

    fe = Qwen3_4BTorchFrontendRtx(
        None, device=device, max_seq=64, max_q_seq=16,
        state_dict=sd, config=cfg.to_dict(),
    )
    ids = torch.randint(0, cfg.vocab_size, (8,), device=device)
    with torch.no_grad():
        logits_hf = hf(input_ids=ids.unsqueeze(0)).logits[0].float()
        logits_frt = fe.forward_prefill(ids).float()
    cos = torch.nn.functional.cosine_similarity(
        logits_frt.flatten(), logits_hf.flatten(), dim=0).item()
    print(f"shape={tuple(logits_frt.shape)} cosine={cos:.6f}")
    if cos < 0.999:
        raise SystemExit(f"cosine {cos:.6f} below 0.999")


if __name__ == "__main__":
    main()
