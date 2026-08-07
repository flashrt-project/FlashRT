"""Host-side contract validation of the HyVLA fused-kernel pybind APIs.

Validation throws before any CUDA work, so these tests pass dummy pointers
and assert the Python-visible ValueError. Requires the built module (CUDA
host toolchain), but not a running device for the negative cases.
"""

import pytest

pytest.importorskip("torch")

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"flash_rt_kernels is not built: {exc}", allow_module_level=True)

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
