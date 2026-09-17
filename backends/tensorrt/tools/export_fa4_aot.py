#!/usr/bin/env python3
"""AOT-export FlashRT's FlashAttention-4 (CuTe DSL) forward modules.

The kernels come from FlashRT's vendored FA4 tree
(csrc/attention/flash_attn_4_src) through the same loader and entry point the
FlashRT torch pipeline uses (flash_rt.hardware.thor.fa4_backend). Each module
is compiled with the arguments the production call sites pass, then written as
a C header plus object file:

  fa4_hd256_fwd     head_dim 256, 8 query heads / 1 KV head, non-causal,
                    seq_q * 8 > 128 (pi0.5 encoder prefill)
  fa4_hd256_q1_fwd  the same with seq_q * 8 <= 128 (pi0.5 decoder, 10 queries)
  fa4_hd72_fwd      head_dim 72, 16 heads MHA, non-causal, seq_q > 128
                    (pi0.5 SigLIP)

Within one module the compile key does not depend on batch or sequence
length; the only length-dependent key field is the query stage count
(q_stage = 2 when seq_q * q_heads_per_kv_head > 128, else 1). The script calls
each entry at several shapes and fails if one module needs a second kernel.

usage: python backends/tensorrt/tools/export_fa4_aot.py [out_dir]
"""
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "csrc/attention/fa4_aot").resolve()
out_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.environ.get("FLASHRT_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))))

import torch  # noqa: E402

from flash_rt.hardware.thor import fa4_backend  # noqa: E402

assert fa4_backend.is_available(), fa4_backend.status()
fwd = fa4_backend.fa4_fwd()

import flashrt_fa4.cute.interface_fwd_sm100 as ifw  # noqa: E402

# The exporter needs the plain JitCompiledFunction (classic C header with an
# embedded cubin and a plain-C launch entry), so compile without tvm-ffi and
# never execute the result.
compiled = []
_compile = ifw.cute.compile


def _compile_for_export(*args, **kwargs):
    kwargs.pop("options", None)
    compiled.append(_compile(*args, **kwargs))
    return lambda *a, **k: None


ifw.cute.compile = _compile_for_export

CONFIGS = {
    # name: (num_q_heads, num_kv_heads, head_dim, pack_gqa, [(batch, seq_q, seq_k), ...])
    "fa4_hd256_fwd": (8, 1, 256, True, [(1, 976, 976), (1, 559, 559), (1, 17, 1024)]),
    "fa4_hd256_q1_fwd": (8, 1, 256, True, [(1, 10, 986), (1, 10, 10), (1, 16, 1024)]),
    "fa4_hd72_fwd": (16, 16, 72, False, [(3, 256, 256), (2, 256, 256), (1, 129, 129)]),
}

for name, (hq, hk, hd, pack_gqa, shapes) in CONFIGS.items():
    compiled.clear()
    ifw._flash_attn_fwd.compile_cache.clear()
    for b, sq, sk in shapes:
        q = torch.zeros(b, sq, hq, hd, dtype=torch.float16, device="cuda")
        k = torch.zeros(b, sk, hk, hd, dtype=torch.float16, device="cuda")
        v = torch.zeros_like(k)
        o = torch.empty_like(q)
        fwd(q, k, v, causal=False, num_splits=1, pack_gqa=pack_gqa, out=o)
    torch.cuda.synchronize()
    assert len(compiled) == 1, f"{name}: {len(compiled)} kernels compiled across {shapes}"
    compiled[0].export_to_c(str(out_dir), name)
    print(f"{name}: one kernel for shapes {shapes}")

print("CUTE_DSL_ARCH", os.environ.get("CUTE_DSL_ARCH"), "FLASH_ATTENTION_ARCH",
      os.environ.get("FLASH_ATTENTION_ARCH"))
print("exported:", sorted(p.name for p in out_dir.glob("fa4_*")))
