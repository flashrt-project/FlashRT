"""Qwen3-4B FlashRT BF16 vs HuggingFace cosine similarity (random weights).

Uses ``Qwen3Config`` / ``Qwen3ForCausalLM.from_config`` — no pretrained
weight download. Requires CUDA + compiled ``flash_rt_kernels`` /
``flash_rt_fa2``.

Default frontend path keeps HF-parity only where leave-one-out showed a
material cosine drop: ``use_hf_qkn_rope`` + ``use_sep_qkv`` (plus bf16
``inv_freq`` RoPE tables). SDPA / HF-norm / sep-gateup default off.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _require_cuda_and_kernels():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as e:
        pytest.skip(f"flash_rt kernels not built: {e}")
    need = (
        "rms_norm",
        "residual_add",
        "residual_add_rms_norm",
        "silu_mul_qwen36_bf16",
        "qwen3_q_norm_rope_qstage_prefill_bf16",
        "qwen3_k_norm_rope_kvwrite_prefill_bf16",
    )
    missing = [n for n in need if not hasattr(fvk, n)]
    if missing:
        pytest.skip(f"missing kernels: {missing}")
    # FA2 is optional: Thor (sm_110) uses SDPA via frontend auto-default.


def _make_hf_qwen3_4b(device: str = "cuda:0"):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from flash_rt.frontends.torch._qwen3_4b_rtx_bf16_weights import (
        qwen3_4b_config_dict,
    )

    cfg = Qwen3Config(**{
        k: v for k, v in qwen3_4b_config_dict().items()
        if k not in ("architectures",)
    })
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, cfg


def _run_cosine(S: int, *, all_fvk: bool = False) -> tuple[float, float]:
    from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx

    device = "cuda:0"
    hf, cfg = _make_hf_qwen3_4b(device)
    sd = {k: v.detach().contiguous() for k, v in hf.state_dict().items()}

    kwargs = dict(
        checkpoint_path=None,
        device=device,
        max_seq=max(S, 64),
        max_q_seq=max(S, 32),
        state_dict=sd,
        config=cfg.to_dict(),
    )
    if all_fvk:
        kwargs.update(
            use_hf_norm=False,
            use_hf_qkn_rope=False,
            use_sep_gateup=False,
            use_sep_qkv=False,
            # keep use_sdpa=None so Thor auto-selects SDPA when FA2 missing
        )
    frt = Qwen3_4BTorchFrontendRtx(**kwargs)

    torch.manual_seed(123)
    input_ids = torch.randint(
        0, cfg.vocab_size, (1, S), device=device, dtype=torch.long)

    with torch.no_grad():
        hf_out = hf(input_ids=input_ids).logits[0].float()  # [S, V]
        frt_out = frt.forward_prefill(input_ids[0]).float()

    assert frt_out.shape == hf_out.shape, (frt_out.shape, hf_out.shape)
    cos = F.cosine_similarity(
        frt_out.flatten(), hf_out.flatten(), dim=0).item()
    cos_last = F.cosine_similarity(
        frt_out[-1], hf_out[-1], dim=0).item()
    return cos, cos_last


@pytest.mark.gpu
def test_qwen3_4b_hf_logits_cosine():
    """Random-weight full-seq logits: FlashRT (default) vs HF cosine >= 0.999."""
    _require_cuda_and_kernels()
    cos, cos_last = _run_cosine(8)
    print(f"Qwen3-4B default-path cosine S=8 full={cos:.6f} last={cos_last:.6f}")
    assert cos >= 0.999, f"cosine {cos:.6f} < 0.999 (last={cos_last:.6f})"


@pytest.mark.gpu
def test_qwen3_4b_hf_logits_cosine_seq1024():
    """Long prefill: default precision path must stay >= 0.999 at S=1024."""
    _require_cuda_and_kernels()
    cos, cos_last = _run_cosine(1024)
    print(f"Qwen3-4B default-path cosine S=1024 full={cos:.6f} last={cos_last:.6f}")
    assert cos >= 0.999, f"cosine {cos:.6f} < 0.999 (last={cos_last:.6f})"


@pytest.mark.gpu
def test_qwen3_4b_pipeline_map_resolves():
    """Resolver finds the BF16 frontend on rtx_sm89 / rtx_sm120 / thor."""
    from flash_rt.hardware import resolve_pipeline_class

    for arch in ("rtx_sm89", "rtx_sm120", "thor"):
        cls = resolve_pipeline_class("qwen3_4b", "torch", arch)
        assert cls.__name__ == "Qwen3_4BTorchFrontendRtx"
