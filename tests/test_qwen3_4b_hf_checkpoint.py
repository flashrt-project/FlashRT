#!/usr/bin/env python3
"""Qwen3-4B BF16 prefill vs HuggingFace on a real checkpoint (full logits).

Runs when ``QWEN3_4B_CHECKPOINT`` names a Qwen/Qwen3-4B checkpoint directory:
HF ``from_pretrained(dtype=bf16)`` (fp32 RoPE ``inv_freq``) against the
frontend of the device's target loading the same directory, full-sequence
logits cosine >= 0.999 at short / sensitive / long prompt lengths. Kept apart
from the random-weight tests so the two model pairs are never resident
together.

Usage:
  QWEN3_4B_CHECKPOINT=<dir> PYTHONPATH=. pytest -q tests/test_qwen3_4b_hf_checkpoint.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from _helpers.qwen3_4b import cosine, frontend_cls  # noqa: E402

COSINE_FLOOR = 0.999
SEQS = [8, 256, 1024]


def test_logits_match_hf_checkpoint():
    ckpt = os.environ.get("QWEN3_4B_CHECKPOINT")
    if not ckpt:
        pytest.skip("set QWEN3_4B_CHECKPOINT to a Qwen/Qwen3-4B checkpoint directory")
    cls = frontend_cls()
    from transformers import Qwen3ForCausalLM

    try:
        hf = Qwen3ForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    except TypeError:   # transformers < 4.56
        hf = Qwen3ForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    hf = hf.to("cuda").eval()
    fe = cls(ckpt, device="cuda", max_seq=max(SEQS))
    # Random ids, not a repeated sentence: strongly repetitive prompts make
    # the BF16 reference itself fragile (see docs/qwen3_4b_bf16.md).
    g = torch.Generator(device="cuda").manual_seed(123)
    all_ids = torch.randint(0, hf.config.vocab_size, (max(SEQS),), device="cuda",
                            generator=g)
    for S in SEQS:
        ids = all_ids[:S]
        with torch.no_grad():
            ref = hf(input_ids=ids[None]).logits[0].float()
        cos = cosine(fe.forward_prefill(ids).float(), ref)
        assert cos >= COSINE_FLOOR, f"S={S}: full-logits cosine {cos:.6f}"
