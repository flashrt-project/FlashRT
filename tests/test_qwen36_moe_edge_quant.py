"""Structural tests for Qwen3.6-MoE edge checkpoint utilities."""

from __future__ import annotations

import torch

from qwen36_moe_edge.probe import _dequant_int4
from qwen36_moe_edge.route_trace import simulate_lru
from qwen36_moe_edge.quantize_experts import (
    HIDDEN,
    INTERMEDIATE,
    _hadamard16,
    _int4_weight,
    _int8_weight,
    _layout,
    _rht16,
    quantize_expert,
)


def test_int8_per_channel_round_trip_is_precise():
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(64, 256, generator=generator)

    quantized, scale = _int8_weight(weight)
    restored = quantized.float() * scale.float()[:, None]

    cosine = torch.nn.functional.cosine_similarity(
        weight.flatten(), restored.flatten(), dim=0)
    assert cosine > 0.9999


def test_int4_grouped_round_trip_matches_packed_layout():
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(64, 256, generator=generator)

    packed, scale = _int4_weight(weight, 32)
    restored = _dequant_int4(packed, scale, 256, 32)

    assert packed.shape == (64, 128)
    assert scale.shape == (64, 8)
    cosine = torch.nn.functional.cosine_similarity(
        weight.flatten(), restored.flatten(), dim=0)
    assert cosine > 0.99


def test_rht16_is_orthonormal_and_preserves_matmul():
    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(3, 32, generator=generator)
    weight = torch.randn(5, 32, generator=generator)
    transform = _hadamard16(weight.device)

    identity = transform @ transform.T
    rotated_activation = _rht16(activation)
    rotated_weight = _rht16(weight)

    assert torch.equal(identity, torch.eye(16))
    torch.testing.assert_close(
        rotated_activation @ rotated_weight.T,
        activation @ weight.T,
        rtol=1e-5,
        atol=1e-5,
    )


def test_expert_block_size_matches_manifest_layout():
    gate_up = torch.zeros(2 * INTERMEDIATE, HIDDEN)
    down = torch.zeros(HIDDEN, INTERMEDIATE)

    for quant_format, group_size in (
        ("int8", 32),
        ("int4", 32),
        ("int4-rht", 16),
    ):
        block = quantize_expert(
            gate_up,
            down,
            quant_format=quant_format,
            group_size=group_size,
            device="cpu",
        )
        assert len(block) == sum(
            _layout(quant_format, group_size).values())


def test_route_trace_lru_separates_prompt_and_decode():
    trace = [
        [[0, 1], [0, 2], [0, 1], [2, 3]],
        [[4, 5], [4, 6], [4, 5], [6, 7]],
    ]

    result = simulate_lru(trace, prompt_tokens=2, quota=2)

    assert result == {
        "prompt_hit_rate": 0.25,
        "decode_hit_rate": 0.25,
        "decode_misses_per_token": 3.0,
    }
