"""LingBot-VLA-v2 RDNA routing, weights, HIP precision and model regressions.

Full-model tests use FLASH_RT_LINGBOT_VLA_V2_RDNA_CKPT,
FLASH_RT_LINGBOT_VLA_V2_PROCESSOR and
FLASH_RT_LINGBOT_VLA_V2_REFERENCE for independent official BF16 controls.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from flash_rt.amd.hardware.rdna.moe import RdnaMoe
from flash_rt.amd.hardware.rdna35.attention import RdnaAttention
from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend
from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
from flash_rt.amd.models.lingbot_vla_v2.pipeline_rdna import _TransformerOps, llm_rope
from flash_rt.executors.torch_weights import Cat, DictSource, MultiSafetensorsSource
from flash_rt.executors.weight_loader import LoaderContext, ModelWeightSpec, WeightLoader
from flash_rt.models.lingbot_vla_v2.weight_spec import build_spec

_VLM = "model.qwenvl_with_expert.qwenvl.model"
_EXPERT = "model.qwenvl_with_expert.qwen_expert.model"


@pytest.fixture(scope="module")
def device():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm and an RDNA GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1100", "gfx1151", "gfx1201"):
        pytest.skip("requires gfx1100, gfx1151 or gfx1201")
    return "cuda"


@pytest.fixture(scope="module")
def kernels(device):
    from flash_rt.amd import flash_rt_amd_kernels

    return flash_rt_amd_kernels


# Routing and observation inputs


@pytest.mark.parametrize("hardware", ["amd_rdna3", "amd_rdna35", "amd_rdna4"])
@pytest.mark.parametrize("precision", ["auto", "bf16"])
def test_load_model_forwards_lingbot_v2_setup(monkeypatch, hardware, precision):
    import flash_rt
    import flash_rt.hardware

    from flash_rt.amd.frontends.torch.lingbot_vla_v2_rdna import LingbotVlaV2TorchFrontendAmdRdna

    assert flash_rt.hardware.resolve_pipeline_class(
        "lingbot_vla_v2", "torch", hardware
    ) is LingbotVlaV2TorchFrontendAmdRdna
    calls = []

    class Frontend:
        def __init__(self, checkpoint, **kwargs):
            calls.append((checkpoint, kwargs))

    monkeypatch.setattr(flash_rt.hardware, "resolve_pipeline_class", lambda *args: Frontend)
    model = flash_rt.load_model(
        "/checkpoint", config="lingbot_vla_v2", hardware=hardware, precision=precision,
        num_views=3, num_steps=4, action_horizon=25, device="cuda:0",
    )
    assert isinstance(model.pipeline, Frontend)
    assert calls == [
        (
            "/checkpoint",
            dict(
                num_views=3, num_steps=4, action_horizon=25, hardware=hardware,
                use_fp8=False, device="cuda:0",
            ),
        )
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hardware": "thor"}, {"framework": "jax"}, {"precision": "fp8"},
        {"action_dim": 7},
    ],
)
def test_unsupported_routes_and_precision_fail_before_loading(monkeypatch, kwargs):
    import flash_rt
    import flash_rt.hardware

    def unexpected_load(*args):
        pytest.fail("unsupported request reached frontend loading")

    monkeypatch.setattr(flash_rt.hardware, "resolve_pipeline_class", unexpected_load)
    with pytest.raises(ValueError):
        flash_rt.load_model(
            "/unused", config="lingbot_vla_v2", **{"hardware": "amd_rdna35", **kwargs}
        )


def _policy():
    return {
        "align_params": {},
        "vlm_causal": True,
        "chunk_size": 50,
        "adanorm_time": True,
        "final_norm_adanorm": False,
        "token_top_k": 4,
        "routed_scaling_factor": 4.0,
    }


def test_official_yaml_and_basic_checkpoint_config(tmp_path):
    from flash_rt.amd.frontends.torch.lingbot_vla_v2_rdna import _load_policy_config
    import yaml

    (tmp_path / "config.json").write_text('{"vlm_family": "qwen3_vl"}')
    defaults = _load_policy_config(tmp_path, None, None)
    assert defaults["n_action_steps"] == 50
    assert defaults["num_steps"] == 10
    assert defaults["tokenizer_path"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert defaults["align_params"]["num_task_tokens"] == 8
    assert defaults["align_params"]["video"]["share_future_depth_query"]

    path = tmp_path / "lingbotvla_cli.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"adanorm_time": True, "tokenizer_path": "/local/processor"},
                "train": {**_policy(), "chunk_size": 25, "num_steps": 6},
            }
        )
    )
    config = _load_policy_config(tmp_path, None, 4)
    assert config["n_action_steps"] == 25
    assert config["num_steps"] == 4
    assert config["tokenizer_path"] == "/local/processor"
    assert config["align_params"] == {}
    assert _load_policy_config(tmp_path, None, None)["num_steps"] == 6
    path.unlink()
    (tmp_path / "config.json").write_text('{"tokenizer_path": "/offline/processor"}')
    config = _load_policy_config(tmp_path, None, None)
    assert config["tokenizer_path"] == "/offline/processor"
    assert config["align_params"] == defaults["align_params"]


def test_aux_layout_and_processor_injection():
    from flash_rt.amd.frontends.torch.lingbot_vla_v2_rdna import LingbotVlaV2TorchFrontendAmdRdna

    frontend = LingbotVlaV2TorchFrontendAmdRdna.__new__(LingbotVlaV2TorchFrontendAmdRdna)
    frontend.num_views = 3
    frontend.qwen_config = {
        "vision_config": {
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
        }
    }
    processor = object()
    frontend.set_hf_processor(processor)
    assert frontend._hf_processor() is processor
    aux = {
        "images": torch.zeros(1, 3, 4, 1536),
        "image_grid_thw": torch.tensor([[[1, 2, 2]] * 3]),
        "img_masks": torch.tensor([True, False, True]),
        "lang_tokens": torch.tensor([3, 4, 0]),
        "lang_masks": torch.tensor([True, True, False]),
    }
    pixels, metadata = frontend._observation_aux(aux)
    assert pixels.shape == (12, 1536)
    assert metadata["lang_tokens"].shape == (1, 3)
    assert metadata["img_masks"].tolist() == [[True, False, True]]
    with pytest.raises(ValueError, match="packed patches"):
        frontend._observation_aux({**aux, "images": torch.zeros(3, 3, 256, 256)})
    with pytest.raises(ValueError, match="batch size one"):
        frontend._observation_aux(
            {**aux, "lang_tokens": torch.zeros(2, 3), "lang_masks": torch.ones(2, 3)}
        )


def test_existing_model_routes_are_unchanged():
    from flash_rt.hardware import _PIPELINE_MAP

    for hardware in ("amd_rdna3", "amd_rdna35", "amd_rdna4"):
        assert _PIPELINE_MAP[("pi05", "torch", hardware)] == (
            "flash_rt.amd.frontends.torch.pi05_rdna", "Pi05TorchFrontendAmdRdna",
        )
        assert _PIPELINE_MAP[("groot_n17", "torch", hardware)] == (
            "flash_rt.amd.frontends.torch.groot_n17_rdna", "GrootN17TorchFrontendAmdRdna",
        )

# Checkpoint layouts


def _block_spec(name, item_names, num_layers=1):
    block = next(b for b in build_spec().blocks if b.name == name)
    return ModelWeightSpec(
        framework="torch",
        blocks=[
            replace(
                block, num_layers=num_layers,
                items=[it for it in block.items if it.name in item_names],
            )
        ],
    )


def _load(spec, tensors):
    target = SimpleNamespace()
    WeightLoader(source=DictSource(tensors), target=target, spec=spec).run()
    return target


def _matrix(rows, cols, offset=0, dtype=torch.bfloat16):
    return (torch.arange(rows * cols).reshape(rows, cols) + offset).to(dtype)


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_llm_qkv_and_gate_up_layout(source_dtype):
    # Different Q/K/V values and two layers catch component/layer reordering.
    tensors = {}
    for i in range(2):
        p = f"{_VLM}.language_model.layers.{i}"
        for proj, rows, offset in (("q", 4, 0), ("k", 2, 20), ("v", 2, 40)):
            tensors[f"{p}.self_attn.{proj}_proj.weight"] = _matrix(
                rows, 3, offset + i * 100, source_dtype
            )
        tensors[f"{p}.mlp.gate_proj.weight"] = _matrix(5, 3, i * 100, source_dtype)
        tensors[f"{p}.mlp.up_proj.weight"] = _matrix(5, 3, 50 + i * 100, source_dtype)
        # FP16 overflows the first value; FP16 -> BF16 double rounding changes
        # the second. Cat must cast directly to BF16.
        tensors[f"{p}.self_attn.q_proj.weight"][0, :2] = torch.tensor(
            [100000.0, 1.0041], dtype=source_dtype
        )

    target = _load(_block_spec("qwen3vl_llm", {"qkv_w", "gate_up_w"}, 2), tensors)
    for i in range(2):
        p = f"{_VLM}.language_model.layers.{i}"
        qkv = torch.cat(
            [tensors[f"{p}.self_attn.{proj}_proj.weight"].bfloat16() for proj in ("q", "k", "v")]
        )
        gate_up = torch.cat(
            [tensors[f"{p}.mlp.{proj}_proj.weight"].bfloat16() for proj in ("gate", "up")]
        )
        for actual, expected in (
            (target._llm_qkv_w[i], qkv.T), (target._llm_gate_up_w[i], gate_up.T),
        ):
            assert actual.dtype == torch.bfloat16
            assert actual.is_contiguous()
            assert torch.equal(actual, expected)


def test_action_qkv_bias_and_projection():
    p = f"{_EXPERT}.layers.0.self_attn"
    tensors = {}
    for proj, rows, offset in (("q", 4, 0), ("k", 2, 20), ("v", 2, 40)):
        tensors[f"{p}.{proj}_proj.weight"] = _matrix(rows, 3, offset)
        tensors[f"{p}.{proj}_proj.bias"] = torch.arange(rows).bfloat16() + offset
    target = _load(_block_spec("action_expert", {"qkv_w", "qkv_b"}), tensors)

    x = torch.tensor([[1.0, 2.0, -1.0]])
    expected = torch.cat(
        [
            F.linear(
                x, tensors[f"{p}.{proj}_proj.weight"].float(),
                tensors[f"{p}.{proj}_proj.bias"].float(),
            )
            for proj in ("q", "k", "v")
        ],
        dim=-1,
    )
    actual = x @ target._expert_qkv_w[0].float() + target._expert_qkv_b[0].float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert target._expert_qkv_w[0].dtype == target._expert_qkv_b[0].dtype == torch.bfloat16


def test_moe_keeps_expert_layout_and_fp32_router():
    p = f"{_EXPERT}.layers.0.mlp"
    tensors = {
        f"{p}.gate.weight": _matrix(4, 3, 1),
        f"{p}.e_score_correction_bias": torch.tensor([0.01, -0.02, 0.03, 0.04]).bfloat16(),
        f"{p}.experts.gate_proj": _matrix(4 * 2, 3, 10).reshape(4, 2, 3),
        f"{p}.experts.up_proj": _matrix(4 * 2, 3, 40).reshape(4, 2, 3),
        f"{p}.experts.down_proj": _matrix(4 * 3, 2, 70).reshape(4, 3, 2),
        f"{p}.shared_expert.gate_proj.weight": _matrix(5, 3, 100),
        f"{p}.shared_expert.up_proj.weight": _matrix(5, 3, 130),
        f"{p}.shared_expert.down_proj.weight": _matrix(3, 5, 160),
    }
    target = _load(
        _block_spec(
            "action_expert",
            {
                "router_w", "router_bias", "gate_w", "up_w", "down_w", "shared_gate_up_w",
                "shared_down_w",
            },
        ),
        tensors,
    )

    assert target._moe_router_w[0].dtype == target._moe_router_bias[0].dtype == torch.float32
    assert torch.equal(target._moe_router_w[0], tensors[f"{p}.gate.weight"].float())
    assert torch.equal(target._moe_router_bias[0], tensors[f"{p}.e_score_correction_bias"].float())
    for proj in ("gate", "up", "down"):
        actual = getattr(target, f"_moe_{proj}_w")[0]
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, tensors[f"{p}.experts.{proj}_proj"])
    shared = torch.cat(
        [tensors[f"{p}.shared_expert.{proj}_proj.weight"] for proj in ("gate", "up")]
    )
    assert torch.equal(target._moe_shared_gate_up_w[0], shared.T)
    assert torch.equal(
        target._moe_shared_down_w[0], tensors[f"{p}.shared_expert.down_proj.weight"].T
    )


def test_official_checkpoint_key_coverage():
    # Read shard headers without loading model tensors.
    from safetensors import safe_open

    checkpoint = Path(
        os.environ.get("FLASH_RT_LINGBOT_VLA_V2_RDNA_CKPT", "")
    )
    shards = sorted(checkpoint.glob("model*.safetensors"))
    if not shards:
        pytest.skip("LingBot-VLA-v2-6B checkpoint not available")
    source = MultiSafetensorsSource(shards, device="cpu")
    requested = []
    spec = build_spec()
    for block in spec.blocks:
        for i in range(block.num_layers):
            ctx = LoaderContext(source, None, prefix=block.prefix_fmt.format(i=i), layer_idx=i)
            for item in block.items:
                keys = item.key.keys if isinstance(item.key, Cat) else [item.key]
                requested.extend(ctx.subkey(k) for k in keys)
    requested.extend(it.key for it in spec.singletons)
    assert len(requested) == len(set(requested))
    assert all(source.has(key) for key in requested)
    keys = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    training_heads = (
        "model.depth_align_head.", "model.future_depth_align_head.",
        "model.current_video_align_head.", "model.future_video_align_head.",
    )
    expected = {key for key in keys if not key.startswith(training_heads)}
    assert set(requested) == expected

# BF16 schedule and rotary tables


@pytest.mark.parametrize("steps", [1, 3, 10, 20])
def test_bf16_euler_schedule(steps):
    from flash_rt.amd.models.lingbot_vla_v2.pipeline_rdna import bf16_time_schedule

    times, dt = bf16_time_schedule(steps)
    expected_time = torch.tensor(1.0, dtype=torch.bfloat16)
    expected_dt = torch.tensor(-1 / steps, dtype=torch.bfloat16)
    expected = []
    while expected_time >= -expected_dt / 2:
        expected.append(expected_time.clone())
        expected_time += expected_dt
    assert times.dtype == dt.dtype == torch.bfloat16
    assert torch.equal(times, torch.stack(expected))
    if steps == 10:
        assert times.tolist() == [
            1.0, 0.8984375, 0.796875, 0.6953125, 0.59375, 0.494140625, 0.39453125, 0.294921875,
            0.1953125, 0.09521484375,
        ]


def _text_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64, eos_token_id=1, rope_theta=5000000,
        rope_scaling={
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
    )
    config._attn_implementation = "eager"
    return config


def test_mrope_preserves_dtype_rounded_inverse_frequencies():
    device = "cpu"
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
    )

    config = _text_config()
    config.head_dim = 128
    config.rope_scaling["mrope_section"] = [24, 20, 20]
    rotary = Qwen3VLTextRotaryEmbedding(config).bfloat16().to(device)
    torch.manual_seed(151)
    positions = torch.randint(0, 1000, (3, 1, 13), device=device)
    q = torch.randn(1, 13, 4, 128, device=device)
    k = torch.randn(1, 13, 2, 128, device=device)
    cos, sin = llm_rope(positions, config.to_dict(), weight_dtype=torch.bfloat16)
    expected_cos, expected_sin = rotary(q, positions)
    assert cos.dtype == sin.dtype == torch.float32
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    assert torch.equal(_TransformerOps().fp32_rope(q, cos, sin), expected_q)
    assert torch.equal(_TransformerOps().fp32_rope(k, cos, sin), expected_k)

# Native HIP operations


@pytest.mark.parametrize("shape", [(2, 7, 32)])
def test_shared_gemm_provider_preserves_linear_layout(device, kernels, shape):
    ops = _TransformerOps(kernels, gemm=Rdna35GemmBackend(kernels, autotune=False, native_nt=True))
    torch.manual_seed(137)
    x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    original = torch.randn(48, 32, dtype=torch.bfloat16, device=device)
    weight = ops.prepare_weight(original.t().contiguous())
    bias = torch.randn(48, dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x, original, bias)
        actual = ops.linear(x, weight, bias)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("padding", [16])
@pytest.mark.parametrize("biased", [False, True])
def test_exact_gemm_calibration_preserves_samples_and_output_stride(
    device, kernels, padding, biased,
):
    torch.manual_seed(157)
    rows, columns, inner = 7, 48, 32
    baseline = Rdna35GemmBackend(kernels, autotune=False, native_nt=True)
    calibrated = Rdna35GemmBackend(kernels, autotune=False, native_nt=True)
    weight = torch.randn(columns, inner, dtype=torch.bfloat16, device=device)
    bias = torch.randn(columns, dtype=torch.bfloat16, device=device) if biased else None
    samples = [
        (torch.zeros(rows, inner, dtype=torch.bfloat16, device=device), weight, bias),
        (torch.randn(rows, inner, dtype=torch.bfloat16, device=device), weight, bias),
        (
            torch.randn(rows, inner, dtype=torch.bfloat16, device=device),
            torch.randn_like(weight), torch.randn_like(bias) if biased else None,
        ),
    ]
    storage = torch.full((rows, columns + padding), 12.5, dtype=weight.dtype, device=device)
    output = storage[:, :columns]
    expected = []
    with torch.inference_mode():
        for x, w, b in samples:
            baseline.linear(output, x, w, b, weight_transposed=True)
            expected.append(output.clone())
        calibrated.begin_exact_calibration(rows, columns, inner, biased=biased)
        for (x, w, b), reference in zip(samples, expected):
            calibrated.linear(output, x, w, b, weight_transposed=True)
            assert torch.equal(output, reference)
        assert calibrated.end_exact_calibration() >= 1
        for (x, w, b), reference in zip(samples, expected):
            calibrated.linear(output, x, w, b, weight_transposed=True)
            assert torch.equal(output, reference)
        assert torch.all(storage[:, columns:] == 12.5)
        input_buffer = samples[0][0].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            calibrated.linear(output, input_buffer, weight, bias, weight_transposed=True)
        for i in (1, 0):
            input_buffer.copy_(samples[i][0])
            graph.replay()
            assert torch.equal(output, expected[i])


@pytest.mark.parametrize(
    "shape",
    [(17, 80), (51, 768), (712, 2560), (22784, 128)],
)
def test_hip_affine_rms_and_fp32_ada_rounding(device, kernels, shape):
    torch.manual_seed(201)
    rows, width = shape
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    weight = torch.randn(width, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(1, width, device=device)
    beta = torch.randn_like(gamma)
    ops = Rdna35TensorOps(kernels)
    for eps in (1e-6, 1e-5):
        value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        expected = value.bfloat16() * weight
        assert torch.equal(ops.rms_norm_affine(x, weight, eps), expected)
        adaptive = ((1 + gamma) * (value * weight) + beta).bfloat16()
        assert torch.equal(ops.rms_norm_affine(x, weight, eps, modulation=(gamma, beta)), adaptive)


@pytest.mark.parametrize(
    "shape", [(17, 80), (51, 768)]
)
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_hip_residual_affine_norm_preserves_materialized_rounding(
    device, kernels, shape, routed, adaptive,
):
    torch.manual_seed(203)
    rows, width = shape
    ops = Rdna35TensorOps(kernels)
    weight = torch.randn(width, device=device, dtype=torch.bfloat16)
    modulation = tuple(torch.randn(1, width, device=device) for _ in range(2)) if adaptive else None
    for scale in (0.01, 1.0, 1000.0):
        residual = torch.randn(rows, width, device=device, dtype=torch.bfloat16) * scale
        update = (
            torch.randn(
                rows, width, device=device, dtype=torch.float32 if routed else torch.bfloat16
            )
            * scale
        )
        shared = torch.randn_like(residual) if routed else None
        merged = update.bfloat16()
        if routed:
            merged = merged + shared
        expected_sum = merged + residual
        for eps in (1e-6, 1e-5):
            expected = ops.rms_norm_affine(expected_sum, weight, eps, modulation=modulation)
            summed, actual = ops.residual_rms_affine(
                update, residual, weight, eps, modulation=modulation, shared=shared
            )
            assert torch.equal(summed, expected_sum)
            assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "dtype,fp32", [(torch.bfloat16, False), (torch.bfloat16, True), (torch.float32, True)]
)
def test_hip_rotate_half_strided_input_and_rounding(device, kernels, dtype, fp32):
    torch.manual_seed(205)
    packed = torch.randn(51, 3, 8, 128, dtype=dtype, device=device)
    x = packed[:, 1]
    tables = torch.randn(51, 2, 128, dtype=torch.float32 if fp32 else dtype, device=device)
    cos, sin = tables.unbind(1)
    first, second = x.float().chunk(2, -1)
    expected = (
        x.float() * cos[:, None].float() + torch.cat((-second, first), -1) * sin[:, None].float()
    )
    expected = expected.to(torch.float32 if fp32 else dtype)
    actual = Rdna35TensorOps(kernels).rotate_half_hip(x, cos, sin, fp32=fp32)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_hip_silu_preserves_bf16_activation_before_multiplication(device, kernels):
    # Exercise every finite BF16 input, including activation underflow.
    values = torch.arange(65536, device=device, dtype=torch.int32).short().view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    torch.manual_seed(207)
    up = torch.randn_like(values)
    ops = Rdna35TensorOps(kernels)
    expected = torch.nn.functional.silu(values)
    assert torch.equal(ops.silu_bf16(values), expected)
    assert torch.equal(ops.silu_bf16(values, up), expected * up)


@pytest.mark.parametrize(
    "rows,reference_rows,inner,columns",
    [
        (51, 72, 768, 704), (226, 286, 2560, 9728),
    ],
)
def test_hip_paired_projections_preserve_reference_and_graph_updates(
    device, kernels, rows, reference_rows, inner, columns
):
    ops = _TransformerOps(kernels, gemm=Rdna35GemmBackend(kernels, autotune=False, native_nt=True))
    torch.manual_seed(239)
    x = torch.randn(1, rows, inner, device=device, dtype=torch.bfloat16)
    weight = ops.prepare_weight(
        torch.randn(inner, 2 * columns, device=device, dtype=torch.bfloat16)
    )
    gate, up = weight.chunk(2, dim=1)
    expected = tuple(ops.linear(x, w, reference_rows=reference_rows) for w in (gate, up))
    actual = ops.linear_pair(x, weight, reference_rows=reference_rows)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ops.linear_pair(x, weight, reference_rows=reference_rows)
    x.mul_(0.5)
    weight.add_(0.125)
    graph.replay()
    expected = tuple(ops.linear(x, w, reference_rows=reference_rows) for w in (gate, up))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("rows,prefix", [(231, 286)])
def test_hip_prefix_norm_rope_layout_and_graph_updates(device, kernels, rows, prefix):
    torch.manual_seed(263)
    ops, provider = _TransformerOps(kernels), RdnaAttention(kernels)
    q = torch.randn(1, rows, 32, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, rows, 8, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qw, kw = (torch.randn(128, device=device, dtype=torch.bfloat16) for _ in range(2))
    angles = torch.randn(1, rows, 64, device=device).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    positions = torch.randperm(prefix, device=device)[:rows].sort().values
    row_map = torch.full((prefix,), -1, device=device, dtype=torch.long)
    row_map[positions] = torch.arange(rows, device=device)

    def reference():
        qr = ops.fp32_rope(ops.rms_norm(q, qw), cos, sin)
        kr = ops.fp32_rope(ops.rms_norm(k, kw), cos, sin)
        expected_q = qr.view(rows, 8, 4, 128).permute(1, 0, 2, 3).reshape(8, rows * 4, 128)
        expected_k = kr.new_zeros(1, prefix, 8, 128).index_copy_(1, positions, kr)
        expected_v = kr.new_zeros(1, prefix, 8, 128).index_copy_(1, positions, v.float())
        return expected_q, expected_k, expected_v

    expected = reference()
    cache = provider.prepare_prefix_cache(k, v, kw, cos, sin, row_map, 32, 51, 1e-6)
    packed_q = provider.prepare_prefix_query(q, qw, cos, sin, 4, 1e-6)
    assert all(
        torch.equal(a, b)
        for a, b in zip((packed_q, cache["key_states"], cache["value_states"]), expected)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache = provider.prepare_prefix_cache(k, v, kw, cos, sin, row_map, 32, 51, 1e-6)
        packed_q = provider.prepare_prefix_query(q, qw, cos, sin, 4, 1e-6)
    q.add_(0.125)
    k.mul_(0.5)
    v.add_(0.25)
    graph.replay()
    expected = reference()
    assert all(
        torch.equal(a, b)
        for a, b in zip((packed_q, cache["key_states"], cache["value_states"]), expected)
    )


def test_hip_router_input_preserves_bf16_store(device, kernels):
    torch.manual_seed(271)
    ops = Rdna35TensorOps(kernels)
    update, residual = (
        torch.randn(1, 51, 768, device=device, dtype=torch.bfloat16) for _ in range(2)
    )
    weight = torch.randn(768, device=device, dtype=torch.bfloat16)
    modulation = tuple(torch.randn(1, 1, 768, device=device) for _ in range(2))
    expected = ops.residual_rms_affine(update, residual, weight, modulation=modulation)
    summed, normalized, router = ops.residual_rms_router(update, residual, weight, modulation, 1e-6)
    assert torch.equal(summed, expected[0]) and torch.equal(normalized, expected[1])
    assert torch.equal(router, normalized.float())


@pytest.mark.parametrize(
    "queries,prefix,suffix,heads,kv_heads,dim",
    [
        (51, 286, 51, 32, 8, 128), (286, 286, 0, 32, 8, 128), (7, 12, 4, 4, 2, 8),
    ],
)
@pytest.mark.parametrize("strided_query", [False, True])
def test_native_attention_masking_and_split_cache(
    device, kernels, queries, prefix, suffix, heads, kv_heads, dim, strided_query
):
    torch.manual_seed(211)
    q = torch.randn(1, queries * (2 if strided_query else 1), heads, dim, device=device)
    if strided_query:
        q = q[:, ::2]
    k = torch.randn(1, prefix, kv_heads, dim, device=device)
    v = torch.randn_like(k)
    sk = torch.randn(1, suffix, kv_heads, dim, device=device)
    packed = torch.randn(1, suffix, 3, kv_heads, dim, device=device, dtype=torch.bfloat16)
    sv = packed[:, :, 2]
    mask = torch.rand(1, queries, prefix + suffix, device=device) > 0.2
    mask[:, 0] = False
    provider = RdnaAttention(kernels)
    original = RdnaAttention(kernels)
    original._grouped_gqa = False
    for changed in (False, True):
        if changed:
            q.mul_(0.5)
            k.add_(0.2)
            sv.mul_(0.75)
        expected = _TransformerOps().prefix_attention(
            q, torch.cat((k, sk), 1), torch.cat((v, sv.float()), 1), mask
        ).bfloat16()
        actual = provider(q, k, v, mask, suffix=(sk, sv))
        assert torch.equal(actual, original(q, k, v, mask, suffix=(sk, sv)))
        assert torch.isfinite(actual).all()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine >= 0.9999


@pytest.mark.parametrize("queries,keys", [(51, 337)])
def test_native_attention_calibrated_pv_graph_updates(device, kernels, queries, keys):
    if dict(kernels.build_info())["gpu_arch"] != "gfx1151":
        pytest.skip("grouped FP32 attention requires gfx1151")
    torch.manual_seed(239)
    q = torch.randn(1, queries, 32, 128, device=device)
    k = torch.randn(1, keys, 8, 128, device=device)
    v = torch.randn_like(k)
    mask = torch.rand(1, queries, keys, device=device) > 0.2
    mask[:, 0] = False
    original, provider = RdnaAttention(kernels), RdnaAttention(kernels)
    provider.begin_exact_calibration()
    for _ in range(3):
        q.mul_(0.75)
        v.add_(0.125)
        assert torch.equal(provider(q, k, v, mask), original(q, k, v, mask))
    provider.end_exact_calibration()
    assert torch.equal(provider(q, k, v, mask), original(q, k, v, mask))
    actual_pv = next(iter(provider._workspace.values()))[5]
    expected_pv = next(iter(original._workspace.values()))[5]
    assert torch.equal(actual_pv.view(torch.int32), expected_pv.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(q, k, v, mask)
    q.add_(0.25)
    k.mul_(0.5)
    v.neg_()
    mask[:, 1] = False
    graph.replay()
    assert torch.equal(actual, original(q, k, v, mask))
    assert torch.equal(actual_pv.view(torch.int32), expected_pv.view(torch.int32))


@pytest.mark.parametrize(
    "prefix,queries,heads,kv_heads",
    [
        (286, 51, 32, 8), (80, 17, 20, 4),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_hip_rotary_attention_preserves_cache_and_graph_updates(
    device, kernels, prefix, queries, heads, kv_heads, strided
):
    if dict(kernels.build_info())["gpu_arch"] != "gfx1151":
        pytest.skip("grouped rotary attention is enabled on gfx1151")
    torch.manual_seed(241)
    dim = 128
    q = torch.randn(
        1, queries, 2 if strided else 1, heads, dim, device=device, dtype=torch.bfloat16
    )[:, :, 0]
    kv = torch.randn(1, queries, 3, kv_heads, dim, device=device, dtype=torch.bfloat16)
    sk, sv = kv[:, :, 0], kv[:, :, 2]
    tables = torch.randn(1, queries, 3, dim, device=device)
    cos, sin = tables[:, :, 0], tables[:, :, 2]
    k = torch.randn(1, prefix, kv_heads, dim, device=device)
    v = torch.randn_like(k)
    mask = torch.rand(1, queries, prefix + queries, device=device) > 0.2
    mask[:, 0] = False
    ops, provider = _TransformerOps(kernels), RdnaAttention(kernels)
    cache = provider.prepare_cache(k, v, heads, queries)
    prefix_k = cache.packed_key[:prefix].clone()
    prefix_v = cache.packed_value[:prefix].clone()

    def reference():
        qf, kf = (ops.fp32_rope(x, cos, sin) for x in (q, sk))
        out = provider(qf, k, v, mask, suffix=(kf, sv), cache=cache).clone()
        packed_q = next(w[2] for key, w in provider._workspace.items() if key[0] == queries).clone()
        return out, packed_q, cache.packed_key[prefix:].clone(), cache.packed_value[prefix:].clone()

    expected, pq, pk, pv = reference()
    actual = provider(q, k, v, mask, suffix=(sk, sv), cache=cache, rotary=(cos, sin))
    assert torch.equal(actual, expected)
    assert torch.equal(
        next(w[2] for key, w in provider._workspace.items() if key[0] == queries).view(torch.int32),
        pq.view(torch.int32),
    )
    assert torch.equal(cache.packed_key[prefix:].view(torch.int32), pk.view(torch.int32))
    assert torch.equal(cache.packed_value[prefix:].view(torch.int32), pv.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = provider(q, k, v, mask, suffix=(sk, sv), cache=cache, rotary=(cos, sin))
    q.mul_(0.5)
    sk.add_(0.25)
    sv.mul_(0.75)
    cos.add_(0.125)
    sin.neg_()
    graph.replay()
    result = actual.clone()
    expected, _, _, _ = reference()
    assert torch.equal(result, expected)
    assert torch.equal(cache.packed_key[:prefix], prefix_k)
    assert torch.equal(cache.packed_value[:prefix], prefix_v)


@pytest.mark.parametrize("length", [256])
def test_native_vision_attention_bf16_scores_and_probabilities(device, kernels, length):
    torch.manual_seed(223)
    qkv = torch.randn(1, length, 3, 16, 64, device=device, dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)
    q, k = q.contiguous(), k.contiguous()
    expected = _TransformerOps().attention(q, k, v)
    actual = RdnaAttention(kernels).vision(q, k, v)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("ties", [False, True])
def test_native_router_topk_normalization_and_ties(device, kernels, top_k, ties):
    torch.manual_seed(227)
    logits = torch.randn(51, 32, device=device)
    bias = torch.randn(32, device=device) * 0.1
    if ties:
        logits.zero_()
        bias.zero_()
    scores = logits.sigmoid()
    expected_indices = torch.topk(scores + bias, top_k, -1).indices
    expected = scores.gather(1, expected_indices)
    expected = (expected / (expected.sum(-1, keepdim=True) + 1e-20) * 2.0).bfloat16()
    route = torch.empty_like(expected)
    selected = torch.empty(expected_indices.shape, dtype=torch.int32, device=device)
    kernels.moe_topk_rdna(
        logits.data_ptr(), bias.data_ptr(), route.data_ptr(), selected.data_ptr(), 51, 32, top_k,
        2.0, int(torch.cuda.current_stream().cuda_stream),
    )
    assert torch.equal(selected, expected_indices)
    assert torch.equal(route, expected)

# Routed MoE and lossless packing


def _assert_moe_reference(native, x, route, selected, gate, up, down):
    """Check route order and rounding against independent PyTorch projections."""
    inter = torch.empty_like(native.inter)
    routes_out = torch.empty_like(native.routes_out)
    for expert in range(gate.shape[0]):
        rows, slots = (selected == expert).nonzero(as_tuple=True)
        count = rows.numel()
        assert native.counts[expert] == count
        assert torch.equal(native.rows[expert, :count].long(), rows)
        assert torch.equal(native.slots[expert, :count].long(), slots)
        if count == 0:
            continue
        values = x[rows].float()
        projected_gate = values @ gate[expert].float().t()
        projected_up = values @ up[expert].float().t()
        activated = projected_gate * projected_gate.sigmoid()
        inter[rows, slots] = (
            activated * projected_up * route[rows, slots, None].float()
        ).bfloat16()
        # Check the down reduction using the actual BF16 rounding boundary.
        routes_out[rows, slots] = native.inter[rows, slots].float() @ down[expert].float().t()
    # FP32 GEMM reductions can differ before the BF16 intermediate store.
    torch.testing.assert_close(
        native.inter.float(), inter.float(), rtol=0.008,
        atol=float(inter.float().abs().max()) * 1e-6,
    )
    torch.testing.assert_close(
        native.routes_out, routes_out, rtol=2e-5, atol=float(routes_out.abs().max()) * 2e-6
    )
    output = torch.zeros_like(native.out)
    for slot in range(native.top_k):
        output += native.routes_out[:, slot]
    assert torch.equal(native.out, output)


@pytest.mark.parametrize(
    "shape", [(1, 64, 32, 4, 2), (17, 64, 32, 4, 4), (51, 80, 48, 4, 2), (51, 768, 512, 32, 4)]
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("weight_format", ["unpacked", "packed", "compact"])
def test_native_moe_matches_reference_and_reuses_workspace(device, shape, scale, weight_format):
    tokens, hidden, width, experts, top_k = shape
    torch.manual_seed(191)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) * scale
    gate = torch.randn(experts, width, hidden, device=device, dtype=torch.bfloat16) * scale
    up = torch.randn_like(gate) * scale
    down = torch.randn(experts, hidden, width, device=device, dtype=torch.bfloat16) * scale
    route = (torch.rand(tokens, top_k, device=device) * 2).bfloat16()
    concentrated = torch.arange(top_k, device=device)[None].expand(tokens, -1).contiguous()
    random = torch.randn(tokens, experts, device=device).topk(top_k).indices
    reference = RdnaMoe(tokens, hidden, experts, top_k, width, device=device)
    native = RdnaMoe(tokens, hidden, experts, top_k, width, device=device)
    if weight_format != "unpacked":
        native.prepare_weights(gate, up, down, compress=weight_format == "compact")
    output_ptr = native.out.data_ptr()
    first = None
    # Switch expert populations, then return to the first routing assignment.
    for selected in (concentrated, random, concentrated):
        reference(x, route, selected, gate, up, down)
        actual = native(x, route, selected, gate, up, down)
        _assert_moe_reference(native, x, route, selected, gate, up, down)
        assert actual.data_ptr() == output_ptr
        for name in ("counts", "inter", "routes_out", "out"):
            assert torch.equal(getattr(native, name), getattr(reference, name)), name
        for expert, count in enumerate(reference.counts.tolist()):
            assert torch.equal(native.rows[expert, :count], reference.rows[expert, :count])
            assert torch.equal(native.slots[expert, :count], reference.slots[expert, :count])
        if first is None:
            first = actual.clone()
        elif selected is concentrated:
            assert torch.equal(actual, first)


@pytest.mark.parametrize("weight_format", ["unpacked", "packed", "compact"])
def test_native_moe_graph_reads_changed_inputs_on_current_stream(device, weight_format):
    torch.manual_seed(217)
    tokens, hidden, width, experts, top_k = (
        (51, 80, 48, 4, 2) if weight_format == "unpacked" else (51, 64, 32, 4, 2)
    )
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    gate = torch.randn(experts, width, hidden, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    down = torch.randn(experts, hidden, width, device=device, dtype=torch.bfloat16)
    route = torch.rand(tokens, top_k, device=device).bfloat16()
    selected = torch.randn(tokens, experts, device=device).topk(top_k).indices.int()
    native = RdnaMoe(tokens, hidden, experts, top_k, width, device=device)
    if weight_format != "unpacked":
        native.prepare_weights(gate, up, down, compress=weight_format == "compact")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        baseline = native(x, route, selected, gate, up, down).clone()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = native(x, route, selected, gate, up, down)
    with torch.cuda.stream(stream):
        x.mul_(0.5)
        route.mul_(0.75)
        selected.copy_((selected + 1) % experts)
        expected = native(x, route, selected, gate, up, down).clone()
        graph.replay()
    stream.synchronize()
    assert torch.equal(captured, expected)
    _assert_moe_reference(native, x, route, selected, gate, up, down)
    assert not torch.equal(captured, baseline)


@pytest.mark.parametrize("hidden,width", [(32, 32), (96, 96)])
def test_compact_moe_mixes_encoded_and_original_tiles(device, kernels, hidden, width):
    if dict(kernels.build_info())["gpu_arch"] != "gfx1151":
        pytest.skip("compact weight dispatch is enabled on gfx1151")
    torch.manual_seed(219)
    tokens, experts, top_k = 17, 4, 2
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) * 0.01
    gate = torch.empty(experts, width, hidden, device=device, dtype=torch.bfloat16).uniform_(
        0.5, 0.75
    )
    up = torch.empty_like(gate).uniform_(-0.75, -0.5)
    down = torch.empty(experts, hidden, width, device=device, dtype=torch.bfloat16).uniform_(
        0.5, 0.75
    )
    # Mix raw tiles and distinct exponent groups in both projections.
    for k in range(0, hidden, 16):
        scale = (0, 1, 2**-24, 256)[(k // 16) % 4]
        gate[:, :, k : k + 16] *= scale
        up[:, :, k : k + 16] *= scale
    for k in range(0, width, 16):
        down[:, :, k : k + 16] *= (0, 1, 2**-24, 256)[(k // 16) % 4]
    route = torch.rand(tokens, top_k, device=device).bfloat16()
    selected = torch.randn(tokens, experts, device=device).topk(top_k).indices.int()
    native = RdnaMoe(tokens, hidden, experts, top_k, width, device=device)
    native.prepare_weights(gate, up, down, compress=True)
    gate_metadata, down_metadata = native._compressed_weights[gate.data_ptr()]
    bases = gate_metadata >> 24
    for metadata in (gate_metadata, down_metadata):
        assert ((metadata >> 24) == 0).any() and ((metadata >> 24) != 0).any()
    if hidden >= 64:
        assert bases.unique().numel() == 4
    native(x, route, selected, gate, up, down)
    expected = {name: getattr(native, name).clone() for name in ("inter", "routes_out", "out")}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native(x, route, selected, gate, up, down)
    x.mul_(0.5)
    selected.copy_((selected + 1) % experts)
    native(x, route, selected, gate, up, down)
    changed = captured.clone()
    graph.replay()
    assert torch.equal(captured, changed)
    assert not torch.equal(captured, expected["out"])
    expected = {name: getattr(native, name).clone() for name in ("inter", "routes_out", "out")}
    # Rebinding uncompressed weights also discards the previous encoding.
    native.prepare_weights(gate, up, down)
    assert gate.data_ptr() not in native._compressed_weights
    native(x, route, selected, gate, up, down)
    for name, value in expected.items():
        assert torch.equal(getattr(native, name), value), name


@pytest.mark.parametrize("gate", [False, True])
def test_compact_encoding_preserves_all_bf16_bit_patterns(device, gate):
    import numpy as np

    elements = 512 if gate else 256
    raw = (
        torch.arange(65536, dtype=torch.int32, device=device)
        .short()
        .view(torch.bfloat16)
        .view(-1, elements)
    )
    native = RdnaMoe(1, 32, 1, 1, 32, device=device)
    blob, metadata = native._pack_compact(raw, gate=gate)
    data = blob.cpu().numpy()
    descriptors = metadata.cpu().numpy().astype(np.uint32)
    assert (descriptors >> 24 == 0).any() and (descriptors >> 24 != 0).any()
    assert blob.numel() + metadata.numel() * 4 < raw.numel() * 2
    restored = np.empty((len(raw), elements), dtype=np.uint16)
    for tile, descriptor in enumerate(descriptors):
        offset, base = int(descriptor & 0xFFFFFF) * 128, int(descriptor >> 24)
        if base == 0:
            restored[tile] = data[offset : offset + elements * 2].view(np.uint16)
            continue
        words = data[offset : offset + elements * 3 // 2].view(np.uint32).reshape(-1, 6)
        values = np.empty((elements // 16, 16), dtype=np.uint16)
        for element in range(16):
            low = (words[:, element // 4] >> ((element % 4) * 8)) & 255
            nibble = (words[:, 4 + element // 8] >> ((element % 8) * 4)) & 15
            high = (nibble & 7) | ((nibble & 8) << 4) | base
            values[:, element] = low | (high << 8)
        restored[tile] = values.reshape(-1)
    assert np.array_equal(restored, raw.view(torch.int16).cpu().numpy().view(np.uint16))

# Full-model references and Graph reuse


def _assert_reference(actual, expected):
    actual, expected = actual.float().cpu().flatten(), expected.float().cpu().flatten()
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    assert cosine >= 0.999, f"action cosine {float(cosine):.8f} < 0.999"


def _required_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} for the LingBot-VLA-v2 model tests")
    path = Path(value)
    assert path.exists(), f"{name} does not exist: {value}"
    return str(path)


@pytest.fixture(scope="module")
def runtime(device, tmp_path_factory):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    hardware = {"gfx1100": "amd_rdna3", "gfx1151": "amd_rdna35", "gfx1201": "amd_rdna4"}
    checkpoint = _required_path("FLASH_RT_LINGBOT_VLA_V2_RDNA_CKPT")
    processor = _required_path("FLASH_RT_LINGBOT_VLA_V2_PROCESSOR")
    reference = _required_path("FLASH_RT_LINGBOT_VLA_V2_REFERENCE")
    import flash_rt
    from transformers.models.qwen2 import modeling_qwen2 as qwen2
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3

    before = (
        qwen2.Qwen2DecoderLayer, qwen3.Qwen3VLTextDecoderLayer,
        qwen3.Qwen3VLPreTrainedModel._init_weights,
    )
    # Bind deployment assets without copying the checkpoint tensors.
    deployment = tmp_path_factory.mktemp("lingbot_vla_v2")
    for shard in Path(checkpoint).glob("model*.safetensors"):
        (deployment / shard.name).symlink_to(shard.resolve())
    (deployment / "config.json").write_text(
        json.dumps({"tokenizer_path": str(Path(processor).resolve())})
    )
    model = flash_rt.load_model(
        deployment, config="lingbot_vla_v2", hardware=hardware[arch], num_views=3,
        precision="bf16",
    )
    assert before == (
        qwen2.Qwen2DecoderLayer, qwen3.Qwen3VLTextDecoderLayer,
        qwen3.Qwen3VLPreTrainedModel._init_weights,
    )
    entries = torch.load(reference, map_location="cpu", weights_only=True)
    assert len(entries) >= 3
    return model, entries


def test_qwen_processor_matches_official_packed_observation(runtime):
    model, entries = runtime
    entry = entries[0]
    aux = model.pipeline.prepare_observation(entry["raw_images"], entry["prompt"])
    assert torch.equal(aux["images"], entry["images"].reshape(-1, 1536))
    for key in ("image_grid_thw", "lang_tokens", "lang_masks", "img_masks"):
        assert torch.equal(aux[key].reshape(-1), entry[key].reshape(-1)), key
    assert model.pipeline._metadata is None


def test_public_eager_graph_and_fresh_observations(runtime):
    model, entries = runtime
    first, second, third = entries[:3]
    aux = {key: first[key].clone() for key in model.pipeline._PROMPT_KEYS + ("images",)}
    model.set_prompt(aux=aux, prompt=first["prompt"])
    aux["images"].add_(0.5)
    aux["lang_tokens"].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="lang_tokens"):
        model.infer(first["state_normalized"], aux=aux)
    state, noise = first["state_normalized"].clone(), first["initial_noise"].clone()
    baseline = None
    for entry in (first, second):
        eager = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"],
            use_graph=False,
        )
        graph = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"]
        )
        assert eager.dtype == graph.dtype == torch.float32
        assert graph.shape == (1, 50, 55)
        _assert_reference(eager, entry["action_pred"])
        assert torch.equal(graph, eager)
        if baseline is None:
            baseline = graph.clone()
            captured = model.pipeline._graph
            capture_stream = model.pipeline._capture_stream
            assert capture_stream != model.pipeline.pipeline.shared_stream
        else:
            assert model.pipeline._graph is captured
            assert not torch.equal(graph, baseline)
    with pytest.raises(ValueError, match="set_prompt"):
        model.infer(third["state_normalized"], aux=third, initial_noise=third["initial_noise"])
    assert model.pipeline._graph is captured
    model.set_prompt(aux=third)
    assert model.pipeline._graph is None
    assert model.pipeline._capture_stream is capture_stream
    changed = model.infer(
        third["state_normalized"], aux=third, initial_noise=third["initial_noise"]
    )
    _assert_reference(changed, third["action_pred"])
    assert not torch.equal(changed, baseline)
    assert torch.equal(first["state_normalized"], state)
    assert torch.equal(first["initial_noise"], noise)
    frontend = model.pipeline
    frontend.action_horizon = 25
    try:
        trimmed = frontend.predict(
            third["state_normalized"].squeeze(0), aux=third,
            initial_noise=third["initial_noise"],
        )
        assert torch.equal(trimmed, changed[:, :25])
    finally:
        frontend.action_horizon = 50


def test_cached_prefix_sampling_and_invalidation(runtime):
    model, entries = runtime
    frontend = model.pipeline
    first, second, third = entries[:3]
    model.set_prompt(aux=first)
    with pytest.raises(RuntimeError, match="prefill_observation"):
        frontend.infer_cached(first["state_normalized"], initial_noise=first["initial_noise"])
    frontend.prefill_observation(aux=first)
    eager = frontend.infer_cached(
        first["state_normalized"], initial_noise=first["initial_noise"], use_graph=False
    )
    graph = frontend.infer_cached(first["state_normalized"], initial_noise=first["initial_noise"])
    assert torch.equal(graph, eager)
    _assert_reference(graph, first["action_pred"])
    prefix_graph, decode_graph = frontend._prefix_graph, frontend._decode_graph
    frontend.prefill_observation(aux=second)
    changed = frontend.infer_cached(
        second["state_normalized"], initial_noise=second["initial_noise"]
    )
    _assert_reference(changed, second["action_pred"])
    assert frontend._prefix_graph is prefix_graph and frontend._decode_graph is decode_graph
    assert not torch.equal(changed, graph)
    noise = second["initial_noise"] * 0.5
    eager = frontend.infer_cached(second["state_normalized"], initial_noise=noise, use_graph=False)
    replay = frontend.infer_cached(second["state_normalized"], initial_noise=noise)
    assert torch.equal(eager, replay)
    frontend.prefill_observation(aux=third)
    assert frontend._prefix_graph is not prefix_graph and frontend._decode_graph is None
    changed = frontend.infer_cached(third["state_normalized"], initial_noise=third["initial_noise"])
    _assert_reference(changed, third["action_pred"])
    model.infer(third["state_normalized"], aux=third, initial_noise=third["initial_noise"])
    with pytest.raises(RuntimeError, match="prefill_observation"):
        frontend.infer_cached(third["state_normalized"], initial_noise=third["initial_noise"])
    frontend.prefill_observation(aux=third, use_graph=False)
    changed = frontend.infer_cached(third["state_normalized"], initial_noise=third["initial_noise"])
    _assert_reference(changed, third["action_pred"])


@pytest.mark.parametrize(
    "variant",
    ["masked_camera", "masked_cameras", "unpadded_prompt"],
)
def test_compact_prefill_matches_full_rows(runtime, variant):
    model, entries = runtime
    pipeline = model.pipeline.pipeline
    original = pipeline.compact_prefill
    if not original:
        pytest.skip("compact prefill is enabled on gfx1151")
    entry = dict(entries[0])
    if variant.startswith("masked_camera"):
        entry["img_masks"] = entry["img_masks"].clone()
        if variant == "masked_cameras":
            entry["img_masks"].zero_()
        else:
            entry["img_masks"].reshape(-1)[1] = False
    else:
        valid = entry["lang_masks"].reshape(-1).bool()
        tokens = entry["lang_tokens"].reshape(1, -1)[:, valid]
        entry["lang_tokens"] = tokens
        entry["lang_masks"] = torch.ones_like(tokens, dtype=torch.bool)
    try:
        pipeline.compact_prefill = False
        model.set_prompt(aux=entry)
        expected = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"],
            use_graph=False,
        ).clone()
        pipeline.compact_prefill = True
        model.set_prompt(aux=entry)
        eager = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"],
            use_graph=False,
        )
        graph = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"]
        )
        replay = model.infer(
            entry["state_normalized"], aux=entry, initial_noise=entry["initial_noise"]
        )
        assert torch.equal(eager, expected)
        assert torch.equal(graph, expected)
        assert torch.equal(replay, expected)
    finally:
        pipeline.compact_prefill = original
        model.set_prompt(aux=entries[0])
