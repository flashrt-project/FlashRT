#!/usr/bin/env python3
"""AOT-export the vendored FA4 forward for the ggml adapter's vision path.

Compiles the FA4 SM100-compatible forward (vendored under
csrc/attention/flash_attn_4_src) at the padded SigLIP shape (head_dim 80,
f16, no mask) and writes fa4_siglip_fwd.h / fa4_siglip_fwd.o into this
directory. Run on the target device with the thor-fa4 deps installed:

    CUTE_DSL_ARCH=sm_110a python export_fa4_siglip.py
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from flash_rt.hardware.thor import fa4_backend  # noqa: E402

assert fa4_backend.is_available(), fa4_backend.status()
fwd = fa4_backend.fa4_fwd()

import flashrt_fa4.cute.interface_fwd_sm100 as ifw  # noqa: E402

# Strip --enable-tvm-ffi so the cache holds a plain JitCompiledFunction:
# only that variant carries the classic C-header exporter (embedded cubin
# plus a plain-C host launch entry). The compiled object is exported
# without ever being executed (the tvm-ffi call convention differs).
_holder = []
_orig_compile = ifw.cute.compile


class _NoCall:
    def __init__(self, inner):
        self._inner = inner

    def __call__(self, *args, **kwargs):
        return None


def _compile_no_ffi(*args, **kwargs):
    kwargs.pop("options", None)
    obj = _orig_compile(*args, **kwargs)
    _holder.append(obj)
    return _NoCall(obj)


ifw.cute.compile = _compile_no_ffi
ifw._flash_attn_fwd.compile_cache.clear()

# vision attention: padded head_dim 80, MHA
NV, SQ, NH, HD = 2, 256, 16, 80
q = torch.zeros(NV, SQ, NH, HD, dtype=torch.float16, device="cuda")
k = torch.zeros_like(q)
v = torch.zeros_like(q)
out = torch.empty_like(q)
fwd(q, k, v, causal=False, num_splits=1, pack_gqa=False, out=out)
torch.cuda.synchronize()

assert _holder, "FA4 compile did not run"
_holder[0].export_to_c(str(_HERE), "fa4_siglip_fwd")

# prefill self-attention: head_dim 256, GQA with one KV head
_holder.clear()
ifw._flash_attn_fwd.compile_cache.clear()
B, SQ2, HQ, HK, HD2 = 1, 559, 8, 1, 256
q2 = torch.zeros(B, SQ2, HQ, HD2, dtype=torch.float16, device="cuda")
k2 = torch.zeros(B, SQ2, HK, HD2, dtype=torch.float16, device="cuda")
v2 = torch.zeros_like(k2)
out2 = torch.empty_like(q2)
fwd(q2, k2, v2, softmax_scale=HD2 ** -0.5, causal=False,
    num_splits=1, pack_gqa=True, out=out2)
torch.cuda.synchronize()

assert _holder, "FA4 prefill compile did not run"
_holder[0].export_to_c(str(_HERE), "fa4_prefill_fwd")
print("exported:", sorted(p.name for p in _HERE.glob("fa4_*_fwd.*")))
