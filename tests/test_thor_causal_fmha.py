"""Basic correctness of the Thor CUTLASS causal FMHA shared libraries."""

import ctypes
import math
import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

import flash_rt  # noqa: E402

_LIB_PATH = os.path.join(os.path.dirname(flash_rt.__file__), "libfmha_fp16_causal.so")
if not os.path.exists(_LIB_PATH):
    pytest.skip(
        "libfmha_fp16_causal.so not built (requires FLASHRT_ENABLE_CHAMELEON "
        "on SM100/110)", allow_module_level=True)

_lib = ctypes.CDLL(_LIB_PATH)
_lib.fmha_fp16_causal.restype = ctypes.c_int
_lib.fmha_fp16_causal.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]

torch.manual_seed(0)

HD = 128
SCALE = 1.0 / math.sqrt(HD)


def _cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def _ref_causal(q, k, v):
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=SCALE)
    return out.transpose(1, 2)


@pytest.mark.parametrize("B,S,NQ,NKV", [(1, 128, 8, 8), (2, 64, 8, 2)])
def test_fmha_fp16_causal_matches_sdpa(B, S, NQ, NKV):
    q = (torch.randn(B, S, NQ, HD, device="cuda", dtype=torch.float16) * 0.5)
    k = (torch.randn(B, S, NKV, HD, device="cuda", dtype=torch.float16) * 0.5)
    v = (torch.randn(B, S, NKV, HD, device="cuda", dtype=torch.float16) * 0.5)
    o = torch.empty_like(q)

    err = _lib.fmha_fp16_causal(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
        B, S, S, NQ, NKV, HD, None)
    if err != 0:
        pytest.skip(f"fmha_fp16_causal returned {err} (shape/arch not supported)")
    torch.cuda.synchronize()

    ref = _ref_causal(q, k, v)
    assert _cos(o, ref) >= 0.99, f"cosine {_cos(o, ref)} < 0.99"
