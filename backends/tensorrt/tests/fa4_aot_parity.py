"""FA4 AOT modules (fa4_aot/) vs the JIT FA4 the FlashRT torch pipeline runs:
bitwise output equality at the pi0.5 shapes, contiguous and strided inputs.

usage: python backends/tensorrt/tests/fa4_aot_parity.py <libfa4_aot_runner.so>
"""
import ctypes
import math
import os
import sys

sys.path.insert(0, os.environ.get("FLASHRT_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))))

import torch  # noqa: E402

from flash_rt.hardware.thor import fa4_backend  # noqa: E402

lib = ctypes.CDLL(sys.argv[1])
fwd = fa4_backend.fa4_fwd()
assert fwd is not None, fa4_backend.status()
dev = torch.device("cuda")


def desc(t):
    shape = (ctypes.c_int * 4)(*t.shape)
    stride = (ctypes.c_longlong * 3)(*t.stride()[:3])
    assert t.stride(3) == 1
    return [ctypes.c_void_p(t.data_ptr()), shape, stride]


def run_aot(module, q, k, v, o):
    stream = torch.cuda.current_stream()
    rc = getattr(lib, module + "_run")(*desc(q), *desc(k), *desc(v), *desc(o),
                                        ctypes.c_float(1.0 / math.sqrt(q.shape[3])),
                                        ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    assert rc == 0, f"{module} returned {rc}"


def case(module, b, sq, sk, hq, hk, hd, packed=False, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    if packed:  # SigLIP production layout: Q/K/V interleaved per row in one QKV buffer
        qkv = torch.randn(b, sq, 3 * hq * hd, generator=g, device=dev, dtype=torch.float16)
        q = qkv[..., : hq * hd].unflatten(-1, (hq, hd))
        k = qkv[..., hq * hd: 2 * hq * hd].unflatten(-1, (hq, hd))
        v = qkv[..., 2 * hq * hd:].unflatten(-1, (hq, hd))
    else:
        q = torch.randn(b, sq, hq, hd, generator=g, device=dev, dtype=torch.float16)
        k = torch.randn(b, sk, hk, hd, generator=g, device=dev, dtype=torch.float16)
        v = torch.randn(b, sk, hk, hd, generator=g, device=dev, dtype=torch.float16)
    ref = torch.empty(b, sq, hq, hd, device=dev, dtype=torch.float16)
    got = torch.full_like(ref, 7.0)
    fwd(q, k, v, causal=False, num_splits=1, pack_gqa=hk != hq, out=ref)
    torch.cuda.synchronize()
    run_aot(module, q, k, v, got)
    same = torch.equal(got, ref)
    d = (got.float() - ref.float()).abs().max().item()
    print(f"{module:17s} B={b} Sq={sq:4d} Sk={sk:4d} H={hq}/{hk} HD={hd} "
          f"{'packed' if packed else 'contig'}: bitwise={same} max|d|={d:.3g}")
    return same


ok = True
for s in range(2):
    ok &= case("fa4_hd256_fwd", 1, 976, 976, 8, 1, 256, seed=s)
    ok &= case("fa4_hd256_fwd", 1, 559, 559, 8, 1, 256, seed=s)
    ok &= case("fa4_hd256_q1_fwd", 1, 10, 986, 8, 1, 256, seed=s)
    ok &= case("fa4_hd256_q1_fwd", 1, 10, 976, 8, 1, 256, seed=s)
    ok &= case("fa4_hd72_fwd", 3, 256, 256, 16, 16, 72, seed=s)
    ok &= case("fa4_hd72_fwd", 3, 256, 256, 16, 16, 72, packed=True, seed=s)
    ok &= case("fa4_hd72_fwd", 1, 256, 256, 16, 16, 72, packed=True, seed=s)
print("FA4_AOT_PARITY_" + ("PASS" if ok else "FAIL"))
