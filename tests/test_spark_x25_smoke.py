#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spark-X2.5-4B wiring smoke test — no checkpoint, no model forward.

Covers the six things that break quietly when a text-LLM integration is
half-wired, in the order a reviewer checks them:

1. registry      — ``(spark_x25, torch, rtx_sm120)`` resolves to the frontend
2. dims          — the config defaults are the shipped checkpoint's geometry
3. validation    — a checkpoint that is not Spark-X2.5-4B is rejected up front
4. redirect      — ``load_model("spark_x25")`` explains the direct construction
5. gated kernels — the SM120-only module exposes its kernels, or is absent
6. FA2           — the window entry this model's prefill needs exists

The numerical kernel tests live in ``tests/test_spark_x25_kernels.py`` and need
a GPU. This file must pass on a machine with no accelerator, so every check that
needs a built extension is behind ``pytest.importorskip``.

Run:  python -m pytest tests/test_spark_x25_smoke.py -q
"""
from __future__ import annotations

import json

import pytest


# ── 1. registry ─────────────────────────────────────────────────────────────

def test_pipeline_map_entry_exists():
    from flash_rt.hardware import _PIPELINE_MAP

    entry = _PIPELINE_MAP.get(("spark_x25", "torch", "rtx_sm120"))
    assert entry == ("flash_rt.frontends.torch.spark_x25_rtx",
                     "SparkX25TorchFrontendRtx"), entry


def test_resolve_pipeline_class_imports_the_frontend():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("spark_x25", "torch", "rtx_sm120")
    assert cls.__name__ == "SparkX25TorchFrontendRtx"


def test_no_unrelated_arch_resolves():
    """SM120 only: the kernels build into a separate module gated on 120."""
    from flash_rt.hardware import _PIPELINE_MAP

    others = [k for k in _PIPELINE_MAP if k[0] == "spark_x25" and k[2] != "rtx_sm120"]
    assert others == [], f"unexpected spark_x25 registrations: {others}"


# ── 2. dims ─────────────────────────────────────────────────────────────────

def test_config_defaults_are_the_shipped_geometry():
    from flash_rt.models.spark_x25 import SparkX25Config

    c = SparkX25Config()
    assert (c.num_hidden_layers, c.hidden_size) == (36, 2560)
    assert (c.num_attention_heads, c.num_key_value_heads) == (16, 4)
    assert (c.head_dim, c.intermediate_size) == (256, 10240)
    assert (c.vocab_size, c.sliding_window) == (131072, 512)
    assert c.max_position_embeddings == 1048576
    assert c.tie_word_embeddings is True


def test_derived_dims_match_the_kernel_contract():
    from flash_rt.models.spark_x25 import SparkX25Config

    c = SparkX25Config()
    assert c.q_dim == 16 * 256
    assert c.kv_dim == 4 * 256
    assert c.qkv_dim == 16 * 256 + 2 * 4 * 256
    assert c.num_kv_groups == 4            # the kernels assume 4:1 GQA


def test_rope_geometry_is_per_layer_type():
    from flash_rt.models.spark_x25 import SparkX25Config

    c = SparkX25Config()
    assert c.rope_dim("full_attention") == int(256 * c.rope_partial_full) == 64
    assert c.rope_dim("sliding_attention") == 256
    assert c.rope_theta("full_attention") == 5_000_000.0
    assert c.rope_theta("sliding_attention") == 10_000.0


# ── 3. validation ───────────────────────────────────────────────────────────

def _write_ckpt(tmp_path, **overrides):
    """A config.json in the shipped checkpoint's schema, with overrides applied.

    Mirrors exactly the keys ``load_config`` reads (note ``rope_parameters`` is
    nested per layer type), so a bad override is rejected by the checkpoint
    validator rather than by a KeyError in the loader.
    """
    cfg = {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 10240,
        "vocab_size": 131072,
        "max_position_embeddings": 1048576,
        "sliding_window": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "layer_types": ["sliding_attention"] * 27 + ["full_attention"] * 9,
        "rope_parameters": {
            "full_attention": {"rope_theta": 5000000.0,
                               "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000.0,
                                  "partial_rotary_factor": 1.0},
        },
    }
    cfg.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def test_validate_accepts_the_shipped_geometry(tmp_path):
    from flash_rt.frontends.torch.spark_x25_rtx import validate_spark_x25_checkpoint

    cfg = validate_spark_x25_checkpoint(_write_ckpt(tmp_path))
    assert cfg.num_hidden_layers == 36
    assert len(cfg.full_layer_idx) == 9       # property, not a method
    assert len(cfg.sliding_layer_idx) == 27


@pytest.mark.parametrize("bad,field", [
    ({"num_hidden_layers": 32}, "layer count"),
    ({"head_dim": 128}, "head_dim"),
    ({"num_attention_heads": 12}, "GQA group"),
    ({"sliding_window": 1024}, "sliding window"),
])
def test_validate_rejects_a_foreign_checkpoint(tmp_path, bad, field):
    from flash_rt.frontends.torch.spark_x25_rtx import validate_spark_x25_checkpoint

    with pytest.raises(ValueError):
        validate_spark_x25_checkpoint(_write_ckpt(tmp_path, **bad))


# ── 4. redirect ─────────────────────────────────────────────────────────────

def test_load_model_redirects_to_direct_construction():
    """A text LLM is not served through load_model's VLA wrapper."""
    import flash_rt

    with pytest.raises(NotImplementedError) as ei:
        flash_rt.load_model(config="spark_x25", checkpoint="/nonexistent")
    msg = str(ei.value)
    assert "SparkX25TorchFrontendRtx" in msg
    assert "docs/spark_x25_usage.md" in msg


# ── 5. gated kernel symbols ─────────────────────────────────────────────────

# The decode path this model owns: attention over the E4M3 KV cache, the KV
# writer, the boundary quantiser, the gate, gproj and the device-side advance.
EXPECTED_KERNELS = (
    "attn_scores_bf16",
    "attn_softmax_bf16",
    "attn_pv_bf16",
    "attn_pv_combine_bf16",
    "attn_state_init_bf16",
    "kv_dequant_bf16",
    "qkv_post_rope_kvwrite_bf16",
    "residual_add_rms_norm_to_nvfp4_bf16",
    "attn_out_gate_to_nvfp4_bf16",
    "gelu_mul_to_nvfp4_swizzled_bf16",
    "gproj_bf16",
    "step_positions_bf16",
    "argmax_bf16",
)


def test_gated_kernel_module_exposes_its_symbols():
    """flash_rt_sparkx25 is built only for GPU_ARCH 120; skip elsewhere."""
    sk = pytest.importorskip(
        "flash_rt.flash_rt_sparkx25",
        reason="Spark-X2.5 kernels are SM120-only (GPU_ARCH=120)")

    missing = [n for n in EXPECTED_KERNELS if not hasattr(sk, n)]
    assert missing == [], f"flash_rt_sparkx25 is missing {missing}"


def test_no_unrelated_kernel_leaked_into_the_shared_module():
    """The model's kernels must not be reachable without its own module."""
    fvk = pytest.importorskip("flash_rt.flash_rt_kernels")
    leaked = [n for n in EXPECTED_KERNELS if hasattr(fvk, n)]
    assert leaked == [], f"spark_x25 kernels leaked into flash_rt_kernels: {leaked}"


# ── 6. FA2 window entry ─────────────────────────────────────────────────────

def test_fa2_window_entry_is_available():
    """Sliding-layer prefill needs the window-carrying FA2 entry."""
    fa2 = pytest.importorskip("flash_rt.flash_rt_fa2")
    assert hasattr(fa2, "fwd_bf16_window")
    assert hasattr(fa2, "fwd_bf16_causal")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
