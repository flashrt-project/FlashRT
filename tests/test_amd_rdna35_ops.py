"""Parity checks for the correctness-first RDNA 3.5 providers."""

from __future__ import annotations

import pytest


class _Rdna35TestKernels:
    """Test-only raw-pointer launch helpers for operator parity checks."""

    def __init__(self, module, torch):
        self.fvk = module
        self.torch = torch

    def _stream(self, tensor):
        return int(self.torch.cuda.current_stream(tensor.device).cuda_stream)

    def qkv_rope(self, q, k, v, qkv, cos, sin, position_start):
        self.fvk.qkv_rope_rdna(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), qkv.data_ptr(),
            cos.data_ptr(), sin.data_ptr(), qkv.shape[0], position_start,
            self._stream(qkv))

    def layer_norm(self, out, x, weight, bias, eps):
        self.fvk.layer_norm_rdna(
            out.data_ptr(), x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
            x.numel() // x.shape[-1], x.shape[-1], eps, self._stream(x))

    def rms_norm(self, out, x, eps):
        self.fvk.rms_norm_rdna(
            out.data_ptr(), x.data_ptr(), x.numel() // x.shape[-1],
            x.shape[-1], eps, self._stream(x))

    def adarms(self, out, x, modulation, eps):
        self.fvk.adarms_rdna(
            out.data_ptr(), x.data_ptr(), modulation.data_ptr(),
            x.numel() // x.shape[-1], x.shape[-1], eps, self._stream(x))

    def gelu_mul(self, out, gate, up):
        self.fvk.gelu_mul_rdna(
            out.data_ptr(), gate.data_ptr(), up.data_ptr(), out.numel(),
            self._stream(out))

    def gelu_mul_merged(self, out, gate_up):
        self.fvk.gelu_mul_merged_rdna(
            out.data_ptr(), gate_up.data_ptr(), out.shape[0], out.shape[1],
            self._stream(out))

    def residual(self, out, update, residual, gate=None):
        self.fvk.residual_rdna(
            out.data_ptr(), update.data_ptr(), residual.data_ptr(),
            0 if gate is None else gate.data_ptr(), out.numel(), out.shape[-1],
            self._stream(out))

    def residual_rms(self, out_sum, out_norm, update, residual, eps):
        self.fvk.residual_rms_rdna(
            out_sum.data_ptr(), out_norm.data_ptr(), update.data_ptr(),
            residual.data_ptr(), out_sum.numel() // out_sum.shape[-1],
            out_sum.shape[-1], eps, self._stream(out_sum))

    def residual_adarms(
        self, out_sum, out_norm, update, residual, gate, modulation, eps,
    ):
        self.fvk.residual_adarms_rdna(
            out_sum.data_ptr(), out_norm.data_ptr(), update.data_ptr(),
            residual.data_ptr(), gate.data_ptr(), modulation.data_ptr(),
            out_sum.numel() // out_sum.shape[-1], out_sum.shape[-1], eps,
            self._stream(out_sum))

    def decoder_gqa(
        self, out, query, key, value, valid_prefix, suffix_start, *,
        split_key=False, keys_per_iteration=4,
    ):
        arguments = (
            out.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
            query.shape[0], key.shape[0], query.shape[1], query.shape[2],
            valid_prefix, suffix_start, 0.0625,
        )
        if split_key:
            self.fvk.attention_decoder_gqa_splitkey_rdna(
                *arguments, keys_per_iteration, self._stream(query))
        else:
            self.fvk.attention_decoder_gqa_rdna(
                *arguments, self._stream(query))

    def encoder_gqa(self, out, query, key, value, valid_kv_rows):
        self.fvk.attention_encoder_gqa_rdna(
            out.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
            query.shape[0], valid_kv_rows, query.shape[1], query.shape[2],
            self._stream(query))

    def smallm_gemm(
        self, out, x, weight_nt, bias=None, *, accumulate=False,
    ):
        function = (
            self.fvk.smallm_wmma_bf16_residual_rdna
            if accumulate else self.fvk.smallm_wmma_bf16_rdna
        )
        function(
            out.data_ptr(), x.data_ptr(), weight_nt.data_ptr(),
            0 if bias is None else bias.data_ptr(), x.shape[0], out.shape[1],
            x.shape[1], self._stream(x))


def _require_gfx1151():
    torch = pytest.importorskip("torch")
    if getattr(torch.version, "hip", None) is None:
        pytest.skip("requires ROCm PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("requires a visible ROCm device")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":", 1)[0] != "gfx1151":
        pytest.skip("requires gfx1151")
    return torch


def _require_rdna35_hip():
    torch = _require_gfx1151()
    try:
        from flash_rt.amd import flash_rt_amd_kernels as fvk

        info = dict(fvk.build_info())
        if info.get("backend") != "rdna35" or info.get("wave_size") != 32:
            raise RuntimeError("extension is not an RDNA 3.5 build")
        kernels = _Rdna35TestKernels(fvk, torch)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"RDNA 3.5 HIP extension is not available: {exc}")
    return torch, kernels


def test_bf16_gemm_matches_torch_reference():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(17, 64, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    weight = torch.randn(64, 96, generator=generator, device="cuda",
                         dtype=torch.bfloat16)
    bias = torch.randn(96, generator=generator, device="cuda",
                       dtype=torch.bfloat16)
    out = torch.empty(17, 96, device="cuda", dtype=torch.bfloat16)
    Rdna35GemmBackend(kernels.fvk).linear(out, x, weight, bias)
    torch.testing.assert_close(out, torch.addmm(bias, x, weight),
                               atol=0, rtol=0)

    packed_weight = torch.randn(
        64, 192, generator=generator, device="cuda", dtype=torch.bfloat16)
    strided_weight = packed_weight[:, :96]
    assert not strided_weight.is_contiguous()
    Rdna35GemmBackend(kernels.fvk).linear(out, x, strided_weight, bias)
    torch.testing.assert_close(
        out, torch.addmm(bias, x, strided_weight), atol=7e-2, rtol=3e-2)


def test_bf16_gemm_supports_strided_output_rows():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    generator = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(
        17, 64, generator=generator, device="cuda", dtype=torch.bfloat16)
    left_weight = torch.randn(
        64, 96, generator=generator, device="cuda", dtype=torch.bfloat16)
    right_weight = torch.randn_like(left_weight)
    packed_output = torch.empty(
        17, 192, device="cuda", dtype=torch.bfloat16)
    left = packed_output[:, :96]
    right = packed_output[:, 96:]
    assert not left.is_contiguous()
    assert left.stride() == right.stride() == (192, 1)

    backend = Rdna35GemmBackend(kernels.fvk)
    backend.linear(left, x, left_weight)
    backend.linear(right, x, right_weight)

    torch.testing.assert_close(
        left, x @ left_weight, atol=7e-2, rtol=3e-2)
    torch.testing.assert_close(
        right, x @ right_weight, atol=7e-2, rtol=3e-2)


def test_split_ffn_views_use_safe_activation_fallback():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps

    ops = Rdna35TensorOps(kernels.fvk, torch.bfloat16)
    ops.fused_decoder_ops = True
    generator = torch.Generator(device="cuda").manual_seed(10)
    packed = torch.randn(
        15, 8192, generator=generator, device="cuda", dtype=torch.bfloat16)
    gate = packed[:, :4096]
    up = packed[:, 4096:]
    output = torch.empty_like(gate, memory_format=torch.contiguous_format)

    ops.gelu_mul(output, gate, up)
    reference = (
        F.gelu(gate.float(), approximate="tanh") * up.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


def test_gemm_backend_does_not_mutate_torch_tunable_state():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    enabled = torch.cuda.tunable.is_enabled()
    tuning = torch.cuda.tunable.tuning_is_enabled()
    filename = torch.cuda.tunable.get_filename()
    backend = Rdna35GemmBackend(kernels.fvk)
    assert backend._runner is not None
    assert torch.cuda.tunable.is_enabled() == enabled
    assert torch.cuda.tunable.tuning_is_enabled() == tuning
    assert torch.cuda.tunable.get_filename() == filename


def test_pipeline_fallback_matches_float32_reference():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps

    ops = Rdna35TensorOps(kernels.fvk, torch.bfloat16)
    ops.fused_large_ops = False

    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(8, 1024, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    weight = torch.randn(1024, generator=generator, device="cuda",
                         dtype=torch.bfloat16)
    bias = torch.randn(1024, generator=generator, device="cuda",
                       dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ops.layer_norm(out, x, weight, bias)
    ref = F.layer_norm(x.float(), (1024,), weight.float(), bias.float()).to(
        torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=8e-3, rtol=3e-2)


def test_sdpa_gqa_shape_and_finiteness():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    generator = torch.Generator(device="cuda").manual_seed(13)
    q = torch.randn(15, 8, 256, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn(527, 1, 256, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty(15, 2048, device="cuda", dtype=torch.bfloat16)
    result = Rdna35AttentionBackend(kernels.fvk).gqa(
        q, k, v, valid_prefix=512, prefix_capacity=512, out=out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.isfinite(out).all()


def test_compact_sdpa_gqa_matches_padded_layout():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    generator = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn(15, 8, 256, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    prefix = torch.randn(527, 1, 256, generator=generator, device="cuda",
                         dtype=torch.bfloat16)
    suffix = torch.randn(15, 1, 256, generator=generator, device="cuda",
                         dtype=torch.bfloat16)
    compact_k = torch.cat((prefix, suffix))
    compact_v = torch.randn_like(compact_k)
    padded_k = torch.zeros(727, 1, 256, device="cuda", dtype=torch.bfloat16)
    padded_v = torch.zeros_like(padded_k)
    padded_k[:527].copy_(compact_k[:527])
    padded_v[:527].copy_(compact_v[:527])
    padded_k[712:].copy_(compact_k[527:])
    padded_v[712:].copy_(compact_v[527:])

    backend = Rdna35AttentionBackend(kernels.fvk)
    compact = backend.gqa(
        q, compact_k, compact_v, valid_prefix=527, prefix_capacity=712)
    padded = backend.gqa(
        q, padded_k, padded_v, valid_prefix=527, prefix_capacity=712)
    torch.testing.assert_close(compact, padded, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("rows,position_start", [(15, 527), (559, 0)])
def test_hip_qkv_rope_matches_aten(rows, position_start):
    torch, kernels = _require_rdna35_hip()

    generator = torch.Generator(device="cuda").manual_seed(rows + 101)
    qkv = torch.randn(rows, 2560, generator=generator, device="cuda",
                      dtype=torch.bfloat16)
    phase = torch.randn(742, 128, generator=generator, device="cuda")
    rope_cos = phase.cos().to(torch.bfloat16)
    rope_sin = phase.sin().to(torch.bfloat16)
    q = torch.empty(rows, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(rows, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    kernels.qkv_rope(
        q, k, v, qkv, rope_cos, rope_sin, position_start)

    positions = torch.arange(
        position_start, position_start + rows, device="cuda")
    cos = rope_cos[positions].unsqueeze(1)
    sin = rope_sin[positions].unsqueeze(1)

    def reference(value):
        pair = value.view(rows, value.shape[1], 128, 2)
        real, imag = pair.unbind(dim=-1)
        return torch.stack(
            (real * cos - imag * sin, imag * cos + real * sin), dim=-1
        ).reshape_as(value)

    q_flat, k_flat, v_flat = torch.split(qkv, (2048, 256, 256), dim=-1)
    torch.testing.assert_close(
        q, reference(q_flat.view(rows, 8, 256)), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        k, reference(k_flat.view(rows, 1, 256)), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v, v_flat.view_as(v), atol=0, rtol=0)


def test_hip_decoder_fusions_match_aten():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(137)
    x = torch.randn(15, 1024, generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    modulation = torch.randn(
        1, 3072, generator=generator, device="cuda", dtype=torch.bfloat16)
    scale, shift, gate = modulation.chunk(3, dim=-1)

    out = torch.empty_like(x)
    kernels.adarms(out, x, modulation, 1e-6)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    reference = (
        normalized * (1.0 + scale.float()) + shift.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    update = torch.randn_like(x)
    out_sum = torch.empty_like(x)
    out_norm = torch.empty_like(x)
    kernels.residual_adarms(
        out_sum, out_norm, update, x, gate, modulation, 1e-6)
    reference_sum = (x.float() + update.float() * gate.float()).to(
        torch.bfloat16)
    normalized = reference_sum.float() * torch.rsqrt(
        reference_sum.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    reference_norm = (
        normalized * (1.0 + scale.float()) + shift.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(out_sum, reference_sum, atol=0, rtol=0)
    torch.testing.assert_close(
        out_norm, reference_norm, atol=2e-2, rtol=2e-2)

    gate_values = torch.randn(
        15, 4096, generator=generator, device="cuda", dtype=torch.bfloat16)
    up_values = torch.randn_like(gate_values)
    activated = torch.empty_like(gate_values)
    kernels.gelu_mul(activated, gate_values, up_values)
    reference_activation = (
        F.gelu(gate_values.float(), approximate="tanh") * up_values.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        activated, reference_activation, atol=2e-2, rtol=2e-2)

    merged = torch.cat((gate_values, up_values), dim=-1)
    merged_activated = torch.empty_like(gate_values)
    kernels.gelu_mul_merged(merged_activated, merged)
    torch.testing.assert_close(
        merged_activated, reference_activation, atol=2e-2, rtol=2e-2)


def test_hip_large_shape_primitives_match_aten():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(173)
    encoder = torch.randn(
        559, 2048, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    encoder_update = torch.randn_like(encoder)
    encoder_out = torch.empty_like(encoder)
    encoder_sum = torch.empty_like(encoder)

    kernels.rms_norm(encoder_out, encoder, 1e-6)
    reference_norm = encoder.float()
    reference_norm *= torch.rsqrt(
        reference_norm.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(
        encoder_out, reference_norm.to(torch.bfloat16),
        atol=2e-2, rtol=2e-2)

    kernels.residual_rms(
        encoder_sum, encoder_out, encoder_update, encoder, 1e-6)
    reference_sum = (encoder_update.float() + encoder.float()).to(
        torch.bfloat16)
    reference_norm = reference_sum.float()
    reference_norm *= torch.rsqrt(
        reference_norm.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(encoder_sum, reference_sum, atol=0, rtol=0)
    torch.testing.assert_close(
        encoder_out, reference_norm.to(torch.bfloat16),
        atol=2e-2, rtol=2e-2)

    vision = torch.randn(
        512, 1152, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    vision_update = torch.randn_like(vision)
    weight = torch.randn(
        1152, generator=generator, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    vision_out = torch.empty_like(vision)

    kernels.layer_norm(vision_out, vision, weight, bias, 1e-6)
    reference = F.layer_norm(
        vision.float(), (1152,), weight.float(), bias.float(), 1e-6)
    torch.testing.assert_close(
        vision_out, reference.to(torch.bfloat16), atol=2e-2, rtol=2e-2)

    kernels.residual(vision_out, vision_update, vision)
    reference = (vision_update.float() + vision.float()).to(torch.bfloat16)
    torch.testing.assert_close(vision_out, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "query_rows,query_heads,kv_rows,valid_prefix,suffix_start,split_key",
    [
        (15, 8, 574, 559, 559, False),
        (15, 8, 574, 559, 559, True),
        (15, 8, 727, 527, 712, False),
        (15, 8, 727, 527, 712, True),
        (1, 1, 1025, 1025, 1025, True),
        (1, 16, 2048, 2048, 2048, True),
    ],
)
def test_hip_decoder_gqa_matches_sdpa(
    query_rows, query_heads, kv_rows, valid_prefix, suffix_start, split_key,
):
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(
        query_rows + query_heads + kv_rows + valid_prefix)
    query = torch.randn(
        query_rows, query_heads, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        kv_rows, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty(
        query_rows, query_heads * 256, device="cuda", dtype=torch.bfloat16)

    kernels.decoder_gqa(
        output, query, key, value, valid_prefix, suffix_start,
        split_key=split_key)

    mask = None
    if suffix_start != valid_prefix:
        mask = torch.zeros(
            query_rows, kv_rows, dtype=torch.bool, device="cuda")
        mask[:, :valid_prefix] = True
        mask[:, suffix_start:] = True
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(output)

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "rows,valid_kv_rows,query_heads",
    [
        (559, 559, 8),
        (712, 559, 8),
        (1025, 1, 8),
        (4096, 1, 8),
        (4096, 4096, 1),
        (1, 1, 1),
        (17, 17, 1),
        (17, 17, 16),
    ],
)
def test_hip_encoder_gqa_matches_sdpa(rows, valid_kv_rows, query_heads):
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(
        rows + valid_kv_rows + query_heads)
    query = torch.randn(
        rows, query_heads, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        rows, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty(
        rows, query_heads * 256, device="cuda", dtype=torch.bfloat16)

    kernels.encoder_gqa(
        output, query, key, value, valid_kv_rows)
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key[:valid_kv_rows].transpose(0, 1).unsqueeze(0),
        value[:valid_kv_rows].transpose(0, 1).unsqueeze(0),
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(output)

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


def test_encoder_backend_routes_native_compact_and_padded(monkeypatch):
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1")
    generator = torch.Generator(device="cuda").manual_seed(1319)
    backend = Rdna35AttentionBackend(kernels.fvk)

    for rows, valid_kv_rows, prefix_capacity in (
        (559, 559, 559),
        (712, 559, 712),
    ):
        query = torch.randn(
            rows, 8, 256, generator=generator, device="cuda",
            dtype=torch.bfloat16)
        key = torch.randn(
            rows, 1, 256, generator=generator, device="cuda",
            dtype=torch.bfloat16)
        value = torch.randn_like(key)
        output = backend.gqa(
            query,
            key,
            value,
            valid_prefix=valid_kv_rows,
            prefix_capacity=prefix_capacity,
        )
        reference = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key[:valid_kv_rows].transpose(0, 1).unsqueeze(0),
            value[:valid_kv_rows].transpose(0, 1).unsqueeze(0),
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1).reshape_as(output)
        torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


def test_encoder_backend_routes_native_at_extended_row_limit(monkeypatch):
    torch, _kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    class RecordNativeLaunch:
        called = False

        @staticmethod
        def attention_encoder_gqa_rdna(*_args):
            RecordNativeLaunch.called = True

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1")
    generator = torch.Generator(device="cuda").manual_seed(1321)
    rows = 4096
    query = torch.randn(
        rows, 8, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        rows, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)

    output = Rdna35AttentionBackend(RecordNativeLaunch()).gqa(
        query, key, value, valid_prefix=1, prefix_capacity=rows)
    assert RecordNativeLaunch.called
    assert output.shape == (rows, 2048)


def test_encoder_backend_keeps_dense_extended_shape_on_sdpa(monkeypatch):
    torch, _kernels = _require_rdna35_hip()
    import torch.nn.functional as F
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    class RejectNativeLaunch:
        @staticmethod
        def attention_encoder_gqa_rdna(*_args):
            raise AssertionError("dense extended shape reached native GQA")

    called = False

    def fake_sdpa(query, _key, _value, **_kwargs):
        nonlocal called
        called = True
        return torch.zeros_like(query)

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1")
    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    rows = 4096
    query = torch.zeros(
        rows, 8, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(
        rows, 1, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)

    output = Rdna35AttentionBackend(RejectNativeLaunch()).gqa(
        query, key, value, valid_prefix=rows, prefix_capacity=rows)
    assert called
    assert output.shape == (rows, 2048)


@pytest.mark.parametrize("keys_per_iteration", [1, 2, 4, 8])
def test_hip_decoder_gqa_splitkey_variants(keys_per_iteration):
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(
        251 + keys_per_iteration)
    query = torch.randn(
        15, 8, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        574, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty(
        15, 2048, device="cuda", dtype=torch.bfloat16)

    kernels.decoder_gqa(
        output, query, key, value, 559, 559, split_key=True,
        keys_per_iteration=keys_per_iteration)
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(output)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "rows,inner,columns,with_bias",
    [
        (15, 1024, 2560, False),
        (15, 1024, 4096, True),
        (15, 4096, 1024, True),
        (15, 16384, 2048, False),
        (17, 1024, 32, True),
        (32, 1024, 32, True),
        (48, 1024, 32, True),
    ],
)
def test_hip_smallm_wmma_matches_torch(rows, inner, columns, with_bias):
    torch, kernels = _require_rdna35_hip()

    generator = torch.Generator(device="cuda").manual_seed(
        rows + inner + columns)
    x = torch.randn(
        rows, inner, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    weight = torch.randn(
        inner, columns, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    weight_nt = weight.t().contiguous()
    bias = None
    if with_bias:
        bias = torch.randn(
            columns, generator=generator, device="cuda",
            dtype=torch.bfloat16)
    output = torch.empty(
        rows, columns, device="cuda", dtype=torch.bfloat16)

    kernels.smallm_gemm(output, x, weight_nt, bias)
    reference = x @ weight
    if bias is not None:
        reference = reference + bias
    torch.testing.assert_close(output, reference, atol=7e-2, rtol=3e-2)


def test_hip_attention_and_smallm_support_graph_replay():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(211)
    query = torch.randn(
        15, 8, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        574, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_output = torch.empty(
        15, 2048, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(
        15, 1024, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    weight_nt = torch.randn(
        32, 1024, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    bias = torch.randn(
        32, generator=generator, device="cuda", dtype=torch.bfloat16)
    gemm_output = torch.empty(
        15, 32, device="cuda", dtype=torch.bfloat16)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        kernels.decoder_gqa(
            attention_output, query, key, value, 559, 559, split_key=True)
        kernels.smallm_gemm(gemm_output, x, weight_nt, bias)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.decoder_gqa(
            attention_output, query, key, value, 559, 559, split_key=True)
        kernels.smallm_gemm(gemm_output, x, weight_nt, bias)

    query.copy_(torch.randn_like(query))
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()

    attention_reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(attention_output)
    gemm_reference = x @ weight_nt.t() + bias
    torch.testing.assert_close(
        attention_output, attention_reference, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        gemm_output, gemm_reference, atol=7e-2, rtol=3e-2)


def test_hip_encoder_attention_supports_graph_replay():
    torch, kernels = _require_rdna35_hip()
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda").manual_seed(217)
    query = torch.randn(
        559, 8, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    key = torch.randn(
        559, 1, 256, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty(
        559, 2048, device="cuda", dtype=torch.bfloat16)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        kernels.encoder_gqa(output, query, key, value, 559)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.encoder_gqa(output, query, key, value, 559)

    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1).reshape_as(output)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


def test_hipblaslt_runner_supports_graph_replay():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    generator = torch.Generator(device="cuda").manual_seed(219)
    x = torch.randn(
        15, 1024, generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        1024, 2560, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    output = torch.empty(
        15, 2560, device="cuda", dtype=torch.bfloat16)
    backend = Rdna35GemmBackend(kernels.fvk)

    # The first eager call selects and caches an algorithm. Graph capture must
    # then contain only the selected hipBLASLt launch, never the timing loop.
    backend.linear(output, x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.linear(output, x, weight)

    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, x @ weight, atol=7e-2, rtol=3e-2)


@pytest.mark.parametrize("rows", [15, 48])
def test_gemm_backend_routes_prepared_action_projection(monkeypatch, rows):
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_SMALLM", "1")
    generator = torch.Generator(device="cuda").manual_seed(223)
    x = torch.randn(
        rows, 1024, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    weight = torch.randn(
        1024, 32, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    bias = torch.randn(
        32, generator=generator, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(
        rows, 32, device="cuda", dtype=torch.bfloat16)

    backend = Rdna35GemmBackend(kernels.fvk)
    backend.prepare_smallm_weight(weight)
    backend.linear(output, x, weight, bias)

    torch.testing.assert_close(
        output, torch.addmm(bias, x, weight), atol=7e-2, rtol=3e-2)


def test_prepared_smallm_rejects_invalid_bias_before_raw_launch(monkeypatch):
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_SMALLM", "1")
    x = torch.zeros(15, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(1024, 32, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(15, 32, device="cuda", dtype=torch.bfloat16)
    invalid_bias = torch.zeros(31, device="cuda", dtype=torch.bfloat16)

    backend = Rdna35GemmBackend(kernels.fvk)
    backend.prepare_smallm_weight(weight)
    with pytest.raises(ValueError, match="GEMM bias"):
        backend.linear(output, x, weight, invalid_bias)


def test_gemm_backend_rejects_invalid_rank_before_stride_access():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    backend = Rdna35GemmBackend(kernels.fvk)
    x = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(1024, 32, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(32, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="linear rank"):
        backend.linear(output, x, weight)
    with pytest.raises(ValueError, match="linear-residual rank"):
        backend.linear_residual(output, x, weight)


def test_native_attention_rejects_invalid_output_shape():
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35AttentionBackend

    query = torch.zeros(15, 8, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(574, 1, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    invalid_output = torch.empty(
        15, 1024, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="GQA output shape"):
        Rdna35AttentionBackend(kernels.fvk).gqa(
            query, key, value, valid_prefix=559, prefix_capacity=559,
            out=invalid_output)


@pytest.mark.parametrize("rows", [15, 48])
def test_gemm_backend_fuses_prepared_action_residual(monkeypatch, rows):
    torch, kernels = _require_rdna35_hip()
    from flash_rt.amd.hardware.rdna35 import Rdna35GemmBackend

    monkeypatch.setenv("FLASHRT_RDNA35_HIP_SMALLM", "1")
    generator = torch.Generator(device="cuda").manual_seed(227)
    x = torch.randn(
        rows, 1024, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    weight = torch.randn(
        1024, 32, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    bias = torch.randn(
        32, generator=generator, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        rows, 32, generator=generator, device="cuda",
        dtype=torch.bfloat16)
    output = residual.clone()

    backend = Rdna35GemmBackend(kernels.fvk)
    backend.prepare_smallm_weight(weight)
    assert backend.linear_residual(output, x, weight, bias)

    projected = torch.addmm(bias, x, weight).to(torch.bfloat16)
    reference = (residual.float() + projected.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, atol=7e-2, rtol=3e-2)
