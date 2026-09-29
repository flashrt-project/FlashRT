"""Host-side contract validation of the HyVLA fused-kernel pybind APIs.

Validation throws before any CUDA work, so these tests pass dummy pointers
and assert the Python-visible ValueError. Requires the built module (CUDA
host toolchain), but not a running device for the negative cases.
"""

import pytest

pytest.importorskip("torch")

try:
    import flash_rt.flash_rt_kernels as fvk
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)

if not hasattr(fvk, "hyvla_rope_qknorm_kvwrite_bf16"):
    pytest.skip("HyVLA kernels require FLASHRT_ENABLE_HYVLA", allow_module_level=True)


def test_rope_qknorm_rejects_wrong_head_dim():
    with pytest.raises(ValueError, match="hd==128"):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=8, nq=4, nkv=1, hd=64, S_tot=8, off=0)


def test_rope_qknorm_rejects_bad_offset_window():
    with pytest.raises(ValueError, match="invalid offset"):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=8, nq=4, nkv=1, hd=128, S_tot=8, off=4)


def test_rope_qknorm_rejects_nonpositive_shapes():
    with pytest.raises(ValueError):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=0, nq=4, nkv=1, hd=128, S_tot=8, off=0)
    with pytest.raises(ValueError, match="eps"):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=8, nq=4, nkv=1, hd=128, S_tot=8, off=0,
            eps=0.0)
    with pytest.raises(ValueError, match="kv_rep"):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=8, nq=4, nkv=1, hd=128, S_tot=8, off=0,
            kv_rep=0)
    with pytest.raises(ValueError, match="nq == nkv \\* kv_rep"):
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, S=8, nq=4, nkv=2, hd=128,
            S_tot=8, off=0, kv_rep=1)


def test_vit_add_layer_norm_rejects_odd_dim():
    with pytest.raises(ValueError, match="even"):
        fvk.hyvla_vit_add_layer_norm_bf16(0, 0, 0, 0, 0, rows=4, dim=127)


def test_vit_add_layer_norm_rejects_nonpositive_rows():
    with pytest.raises(ValueError):
        fvk.hyvla_vit_add_layer_norm_bf16(0, 0, 0, 0, 0, rows=0, dim=128)


@pytest.mark.skipif(not hasattr(fvk, "hyvla_quant_fp8_dyn_bf16"),
                    reason="Thor-only kernel")
def test_quant_fp8_dyn_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n>0"):
        fvk.hyvla_quant_fp8_dyn_bf16(0, 0, 0, 0)


@pytest.mark.skipif(
    not hasattr(fvk, "hyvla_quant_fp8_dyn_bf16"),
    reason="Thor-only kernel",
)
def test_quant_fp8_dyn_handles_odd_tail():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("requires a Thor SM110 device")
    for n in (1, 3, 513):
        x = torch.linspace(-2.0, 3.0, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(n, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, device="cuda", dtype=torch.float32)
        fvk.hyvla_quant_fp8_dyn_bf16(
            x.data_ptr(), out.data_ptr(), scale.data_ptr(), n,
            torch.cuda.current_stream().cuda_stream)
        expected_scale = torch.clamp(x.float().abs().amax() / 448.0, min=1e-12)
        expected = torch.clamp(
            x.float() / expected_scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        torch.cuda.synchronize()
        assert torch.equal(out.float(), expected.float())
        assert torch.equal(scale, expected_scale.reshape(1))


@pytest.mark.skipif(not hasattr(fvk, "hyvla_ffn_gu_silu_bf16"),
                    reason="Thor-only kernel")
def test_ffn_gu_silu_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="Nout%32"):
        fvk.hyvla_ffn_gu_silu_bf16(0, 0, 0, M=1, K=1024, Nout=1000, sx=0, sgu=1.0)
    with pytest.raises(ValueError, match="K%16"):
        fvk.hyvla_ffn_gu_silu_bf16(0, 0, 0, M=1, K=1000, Nout=1024, sx=0, sgu=1.0)


@pytest.mark.skipif(not hasattr(fvk, "hyvla_ffn_dn_res_bf16"),
                    reason="Thor-only kernel")
def test_ffn_dn_res_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="N%32"):
        fvk.hyvla_ffn_dn_res_bf16(0, 0, 0, 0, M=1, K=1024, N=1000, sa=0, sdn=1.0)


# ── RTX SM120 kernels ────────────────────────────────────────────────
# Required-symbol contract for the RTX (SM120/SM121) plan. A missing symbol
# must surface here rather than as a deep AttributeError during capture.
_RTX_REQUIRED_SYMBOLS = (
    "hyvla_prefill_attn_bf16",
    "hyvla_euler_update_bf16_fp32",
    "hyvla_vit_pos_add_bf16",
    "hyvla_merger_pool_bf16",
    "hyvla_fa2_denoise_gather_o_fp8_block128_bf16",
)


@pytest.mark.skipif(not hasattr(fvk, "hyvla_prefill_attn_bf16"),
                    reason="RTX-only HyVLA kernels (FLASHRT_ENABLE_HYVLA)")
def test_rtx_required_symbols_are_exported():
    missing = [name for name in _RTX_REQUIRED_SYMBOLS
               if not hasattr(fvk, name)]
    assert not missing, f"missing required RTX HyVLA kernels: {missing}"


@pytest.mark.skipif(not hasattr(fvk, "hyvla_prefill_attn_bf16"),
                    reason="RTX-only kernel")
def test_prefill_attn_rejects_nonpositive_shapes():
    for s, h in ((0, 4), (8, 0)):
        with pytest.raises(ValueError, match="S>0, H>0"):
            fvk.hyvla_prefill_attn_bf16(
                0, 0, 0, 0, 0, 0, S=s, H=h,
                q_stride_h=1, k_stride_h=1, v_stride_h=1)


@pytest.mark.skipif(
    not hasattr(fvk, "hyvla_fa2_denoise_gather_o_fp8_block128_bf16"),
    reason="RTX-only kernel")
def test_gather_o_fp8_rejects_misaligned_shape():
    with pytest.raises(ValueError, match=r"\(H\*D\)%128"):
        fvk.hyvla_fa2_denoise_gather_o_fp8_block128_bf16(
            0, 0, 0, S=41, H=3, D=8)


@pytest.mark.skipif(
    not hasattr(fvk, "hyvla_fa2_denoise_gather_o_nvfp4_bf16"),
    reason="RTX-only kernel")
def test_gather_o_nvfp4_rejects_misaligned_shape():
    with pytest.raises(ValueError, match=r"\(H\*D\)%16"):
        fvk.hyvla_fa2_denoise_gather_o_nvfp4_bf16(
            0, 0, 0, S=41, H=3, D=5)


def test_fa2_smallq_tile_binding_present_or_fallback():
    # The RTX plan prefers the short-query tile but documents a seqused
    # fallback; either symbol must exist so the frontend's getattr default is
    # reachable rather than silently missing.
    fa2 = pytest.importorskip("flash_rt.flash_rt_fa2")
    assert hasattr(fa2, "fwd_bf16_tile") or hasattr(fa2, "fwd_bf16_seqused")


# ── shared bf16 activations (csrc/kernels/act_bf16.cu) ───────────────
@pytest.mark.skipif(not hasattr(fvk, "silu_bf16"),
                    reason="shared act_bf16 kernels not built")
def test_act_bf16_matches_torch_silu_gelu():
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.current_stream().cuda_stream
    # n not a multiple of 8 exercises the 8-wide vector body + scalar tail.
    n = 8 * 256 * 4 + 5
    x = torch.linspace(-6.0, 6.0, n, device="cuda", dtype=torch.bfloat16)
    s = x.clone()
    fvk.silu_bf16(s.data_ptr(), n, stream)
    g = x.clone()
    fvk.gelu_erf_bf16(g.data_ptr(), n, stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(s, F.silu(x), atol=1e-2, rtol=0)
    torch.testing.assert_close(
        g, F.gelu(x, approximate="none"), atol=1e-2, rtol=0)
