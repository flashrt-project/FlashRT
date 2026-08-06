"""Structural tests for Qwen3.6-MoE edge checkpoint utilities."""

from __future__ import annotations

import torch

from qwen36_moe_edge.kernel_parity import CASES, _binding_of
from qwen36_moe_edge.expert_quality import (
    SCHEMES,
    expert_forward,
    score_expert,
)
from qwen36_moe_edge.route_trace import (
    cold_prefill_blocks,
    global_frequency,
    read_volume,
    simulate_lru,
    simulate_two_tier,
    simulate_warm_lru,
)
from qwen36_moe_edge.quantize_experts import (
    BLOCK_ALIGNMENT,
    HIDDEN,
    INTERMEDIATE,
    _hadamard16,
    _int4_weight,
    _int8_weight,
    _layout,
    _rht16,
    dequantize_int4,
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

    packed, scale, global_scale = _int4_weight(weight, 32)
    restored = dequantize_int4(packed, scale, 256, 32, global_scale)

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
        block, alphas = quantize_expert(
            gate_up,
            down,
            quant_format=quant_format,
            group_size=group_size,
            device="cpu",
        )
        assert len(block) == sum(
            _layout(quant_format, group_size).values())
        assert len(alphas) == 2


def test_expert_blocks_are_aligned_for_direct_io():
    # An 8 GiB unified-memory device cannot stream the experts through the
    # page cache, so the reader needs O_DIRECT and every block offset and
    # length has to be aligned.
    for quant_format, group_size in (
        ("int8", 32),
        ("int4", 16),
        ("int4", 32),
        ("int4-rht", 16),
    ):
        layout = _layout(quant_format, group_size)
        block_bytes = sum(layout.values())
        assert block_bytes % BLOCK_ALIGNMENT == 0, (
            quant_format, group_size, block_bytes)
        assert layout["padding"] < BLOCK_ALIGNMENT
        assert list(layout)[-1] == "padding"

    # INT4 group-16 is aligned on its own; INT8 needs 2048 bytes of pad.
    assert _layout("int4-rht", 16)["padding"] == 0
    assert _layout("int8", 32)["padding"] == 2048
    assert sum(_layout("int8", 32).values()) == 3153920


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


def test_two_tier_warm_set_survives_prefill():
    # The prompt only ever selects expert 0, so it is the warm set. Decode
    # reuses it once and touches an unseen expert once.
    trace = [[[0], [0], [1], [0]]]

    result = simulate_two_tier(
        trace, prompt_tokens=2, pinned=1, stream=0)

    assert result == {
        "decode_hit_rate": 0.5,
        "warm_hit_rate": 0.5,
        "decode_misses_per_token": 0.5,
    }


def test_two_tier_stream_ring_serves_repeats():
    # No warm set at all: every decode hit has to come from the ring.
    trace = [[[0], [1], [1]]]

    result = simulate_two_tier(
        trace, prompt_tokens=1, pinned=0, stream=1)

    assert result["warm_hit_rate"] == 0.0
    assert result["decode_hit_rate"] == 0.5
    assert result["decode_misses_per_token"] == 0.5


def test_two_tier_oracle_warm_bounds_the_prompt_heuristic():
    # Decode routes somewhere the prompt never went, so a prompt-derived warm
    # set misses everything while a decode-derived one hits everything.
    trace = [[[0], [0], [0], [1], [1], [1]]]

    prompt_warm = simulate_two_tier(
        trace, prompt_tokens=3, pinned=1, stream=0)
    oracle_warm = simulate_two_tier(
        trace, prompt_tokens=3, pinned=1, stream=0, warm_from="decode")

    assert prompt_warm["decode_hit_rate"] == 0.0
    assert oracle_warm["decode_hit_rate"] == 1.0


def test_read_volume_converts_misses_to_bandwidth_limits():
    result = read_volume(
        2.0, block_bytes=1_000_000, bandwidths=(1.0, 2.0))

    assert result["mb_per_token"] == 2.0
    assert result["tok_s_at_1gbps"] == 500.0
    assert result["tok_s_at_2gbps"] == 1000.0


def test_warm_start_removes_the_first_touch_of_a_frequent_expert():
    # A trace whose decode phase only ever wants expert 5. Cold, the first
    # touch misses; warm-started from statistics that name expert 5, it does
    # not.
    trace = [[[5], [5], [5], [5]]]
    frequency = global_frequency([[[5], [5]]])

    cold = simulate_warm_lru(trace, prompt_tokens=2, quota=4)
    warm = simulate_warm_lru(
        trace, prompt_tokens=2, quota=4, preload=frequency)

    assert cold["decode_misses_per_token"] == 0.5
    assert warm["decode_misses_per_token"] == 0.0


def test_windowing_does_not_reduce_reads_an_lru_already_serves():
    # Verifying several tokens at once requests the union of their selections.
    # An LRU already holds a repeated expert, so grouping changes nothing.
    trace = [[[0], [0, 1], [1], [0, 1]]]

    single = simulate_warm_lru(trace, prompt_tokens=0, quota=8, window=1)
    grouped = simulate_warm_lru(trace, prompt_tokens=0, quota=8, window=2)

    assert single["decode_misses_per_token"] == grouped[
        "decode_misses_per_token"]


def test_requests_keep_the_router_order_so_eviction_matches_the_cache():
    # Within one request the insertion order decides which entry is oldest, so
    # it changes what the next eviction picks. The cache dedupes preserving the
    # router's order; a simulator that iterated a set instead would model a
    # different cache. Quota 2 with three distinct experts makes the choice
    # observable: after [7, 3] the oldest is 7, so requesting 9 must evict 7 and
    # leave 3 -- and the following request for 3 must then hit.
    trace = [[[7, 3], [9], [3]]]

    result = simulate_warm_lru(trace, prompt_tokens=0, quota=2)

    assert result["decode_misses_per_token"] == 1.0    # 7, 3, 9 miss; 3 hits
    assert result["distinct_hit_rate"] == 0.25


def test_cold_prefill_counts_the_union_prefill_touches():
    # Prefill routes each token independently, so a layer costs the union of
    # its tokens' selections, less whatever is already resident.
    trace = [[[0, 1], [1, 2], [9]], [[3], [4], [9]]]

    without = cold_prefill_blocks(
        trace, prompt_tokens=2, resident=[set(), set()])
    with_resident = cold_prefill_blocks(
        trace, prompt_tokens=2, resident=[{0, 1}, {3}])

    assert without == 3 + 2          # {0,1,2} and {3,4}
    assert with_resident == 1 + 1    # {2} and {4}


def test_parity_cases_name_bindings_that_exist_in_the_tiers():
    # The parity harness resolves a case name to a binding name. If a kernel is
    # renamed and this mapping is not updated, the case silently reports
    # "binding absent" and a real regression passes unnoticed.
    expected = {
        "bf16_matvec": "bf16_matvec_sm120_bf16",
        "moe_router_topk": "moe_router_topk_sm120_bf16",
        "silu_mul": "silu_mul_sm120_bf16",
        "sigmoid_mul": "sigmoid_mul_sm120_bf16",
        "moe_weighted_sum": "moe_weighted_sum_sm120_bf16",
        "w16a16_gemm": "w16a16_gemm_sm120_bf16",
        "lin_split_qkv": "qwen35moe_lin_split_qkv_broadcast_bf16",
        "split_q_gate": "qwen35moe_split_q_gate_bf16",
        "e0m3_dequant": "qwen35moe_e0m3_dequant_bf16",
    }

    assert set(CASES) == set(expected)
    for case, binding in expected.items():
        assert _binding_of(case) == binding


def test_expert_forward_is_a_swiglu_over_the_gate_up_split():
    generator = torch.Generator().manual_seed(13)
    activation = torch.randn(2, HIDDEN, generator=generator)
    gate_up = torch.randn(
        2 * INTERMEDIATE, HIDDEN, generator=generator) * 0.02
    down = torch.randn(HIDDEN, INTERMEDIATE, generator=generator) * 0.02

    projected = activation @ gate_up.T
    expected = (
        torch.nn.functional.silu(projected[:, :INTERMEDIATE])
        * projected[:, INTERMEDIATE:]
    ) @ down.T

    torch.testing.assert_close(
        expert_forward(activation, gate_up, down), expected)


def _outlier_weight(rows, columns, generator):
    """One large value in every group of 16 -- what the transform targets."""
    weight = torch.randn(rows, columns, generator=generator) * 0.02
    weight = weight.reshape(rows, columns // 16, 16)
    weight[:, :, 0] += torch.randn(
        rows, columns // 16, generator=generator).abs()
    return weight.reshape(rows, columns)


def test_rht16_reduces_int4_error_on_outlier_heavy_weights():
    generator = torch.Generator().manual_seed(11)
    activation = torch.randn(4, HIDDEN, generator=generator)
    gate_up = _outlier_weight(2 * INTERMEDIATE, HIDDEN, generator)
    down = _outlier_weight(HIDDEN, INTERMEDIATE, generator)

    plain = score_expert(
        activation, gate_up, down, scheme="w4a16", group_size=16)
    rotated = score_expert(
        activation, gate_up, down, scheme="w4a16_rht16", group_size=16)

    # The transform only pays off when groups have outliers, and once the
    # two-level scale is correct the win is modest -- about 9 % here. An
    # earlier single-level scale showed 38 %, but most of that was the
    # transform compensating for scale error rather than doing its own job.
    assert rotated["relative_l2"] < 0.97 * plain["relative_l2"]
    assert rotated["cosine"] > plain["cosine"]


def test_two_level_scale_keeps_group_scales_out_of_e4m3_subnormals():
    # Real expert weights have per-group amax around 0.02. With a single level
    # the scale is amax/7 ~ 0.003, below e4m3's smallest normal 2**-6, where
    # the format keeps about three bits; the scale error then swamps the 4-bit
    # value grid. Factoring out a global scale moves the stored bytes into
    # e4m3's normal range.
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(64, 512, generator=generator) * 0.02

    _, scale_bytes, global_scale = _int4_weight(weight, 16)
    stored = scale_bytes.view(torch.float8_e4m3fn).float()

    # Every stored byte is now in e4m3's normal range: that is the fix.
    assert (stored[stored > 0] >= 2.0 ** -6).all()
    assert global_scale > 0.0

    packed, scale_bytes, global_scale = _int4_weight(weight, 16)
    restored = dequantize_int4(packed, scale_bytes, 512, 16, global_scale)
    error = ((restored - weight).norm() / weight.norm()).item()

    # A signed 4-bit grid over a per-group amax has step amax/7, so uniform
    # quantization noise is step/sqrt(12). For Gaussian groups of 16, amax is
    # about 2 sigma, giving ~8.6 % -- and that is what this measures, meaning
    # the scale contributes nothing on top of the value grid.
    assert error < 0.10, error


def test_weight_only_schemes_order_by_bit_width():
    generator = torch.Generator().manual_seed(17)
    activation = torch.randn(4, HIDDEN, generator=generator)
    gate_up = _outlier_weight(2 * INTERMEDIATE, HIDDEN, generator)
    down = _outlier_weight(HIDDEN, INTERMEDIATE, generator)

    scores = {
        scheme: score_expert(
            activation, gate_up, down, scheme=scheme, group_size=16)
        for scheme in SCHEMES
    }

    assert scores["w8a16"]["relative_l2"] < scores["w4a16"]["relative_l2"]
    for values in scores.values():
        assert 0.0 < values["cosine"] <= 1.0
