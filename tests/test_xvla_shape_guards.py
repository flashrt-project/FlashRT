"""Fail-fast guards for the X-VLA frontend and small-M GEMM bindings."""

import pytest
import torch

pytest.importorskip("flash_rt")
try:
    from flash_rt import flash_rt_fa2, flash_rt_kernels  # noqa: F401
except ImportError:
    pytest.skip("FlashRT kernel extensions are not built", allow_module_level=True)

from flash_rt.frontends.torch.xvla_rtx import XVLATorchFrontendRtx


def test_invalid_precision_is_rejected_before_checkpoint_load():
    with pytest.raises(ValueError, match="precision"):
        XVLATorchFrontendRtx("/nonexistent-checkpoint", precision="int8")


def test_invalid_num_views_is_rejected_before_checkpoint_load():
    with pytest.raises(ValueError, match="num_views"):
        XVLATorchFrontendRtx("/nonexistent-checkpoint", num_views=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_smallm_gemm_rejects_unsupported_n_before_touching_pointers():
    from flash_rt import flash_rt_kernels as fvk

    if not hasattr(fvk, "xvla_fp4_gemm_bias_bf16out"):
        pytest.skip("small-M X-VLA GEMM is not built for this arch")
    # N % 128 != 0 must be rejected by the entry's shape guard. Null pointers
    # are safe here because the guard returns before the TMA maps are built.
    rc = fvk.xvla_fp4_gemm_bias_bf16out(0, 0, 0, 0, 32, 127, 256, 0, 0, 1.0, 0)
    assert rc == 1
    rc = fvk.xvla_fp4_gemm_bias_gelu_fp4out(
        0, 0, 0, 0, 0, 0, 0, 32, 127, 256, 1.0, 0
    )
    assert rc == 1
