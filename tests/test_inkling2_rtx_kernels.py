"""Kernel checks of ``flash_rt_inkling2`` on random inputs (an SM120 GPU and the module built with
FLASHRT_ENABLE_INKLING2; no checkpoint)."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("needs an SM120 GPU", allow_module_level=True)
_built = list((Path(__file__).resolve().parents[1] / "flash_rt").glob("flash_rt_inkling2*.so"))
if _built:   # a built module that does not import is a failure, not a skip
    from flash_rt import flash_rt_inkling2 as ik
else:
    ik = pytest.importorskip("flash_rt.flash_rt_inkling2")

D = 2304


def P(t):
    return t.data_ptr()


@pytest.mark.parametrize("name,n,k,xf,yf,pro", [
    ("plain", 64, 4608, 0, 0, "GEMV_PLAIN"), ("norm f32 in", 3072, 2304, 1, 0, "GEMV_NORM"),
    ("norm bf16 in", 384, 2304, 0, 0, "GEMV_NORM"), ("gate", 2304, 3072, 0, 0, "GEMV_GATE"),
    ("norm f32 out", 128, 2304, 1, 1, "GEMV_NORM")])
def test_gemv_rows_equal_one_row_launches(name, n, k, xf, yf, pro):
    """Multi-row GEMV (verification rows): every row bit-equal to a one-row launch on that row."""
    torch.manual_seed(0)
    R, dev, s = ik.MAX_ROWS, "cuda", torch.cuda.current_stream().cuda_stream
    pro = getattr(ik, pro)
    W = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(R, k, device=dev) * 3
    x = x if xf else x.to(torch.bfloat16)
    gate = pro == ik.GEMV_GATE
    aux = (torch.randn(R, k, device=dev) if gate else torch.rand(k, device=dev) + 0.5).to(torch.bfloat16)
    yd = torch.float32 if yf else torch.bfloat16
    rows = torch.empty(R, n, device=dev, dtype=yd)
    ik.gemv_multi([(P(W), P(x), P(rows), n, k, xf, yf, pro, P(aux), R)], [], s)
    for r in range(R):
        one = torch.empty(n, device=dev, dtype=yd)
        ik.gemv_multi([(P(W), P(x[r]), P(one), n, k, xf, yf, pro, P(aux[r] if gate else aux))], [], s)
        torch.cuda.synchronize()
        assert torch.equal(one, rows[r]), f"{name}: row {r}"


def test_gate_mix_tiles_track_the_row_kernel():
    """Message gates of a prompt chunk on the tiled MMA kernel against the per-row kernel (FP32 sums in another
    order)."""
    torch.manual_seed(0)
    R, dev, s = 512, "cuda", torch.cuda.current_stream().cuda_stream
    F = [torch.randn(D, 64, device=dev).mul(0.05).bfloat16() for _ in range(3)]
    G = [torch.randn(D, 64, device=dev).mul(0.05).bfloat16() for _ in range(3)]
    S = [torch.randn(R, 64, device=dev).bfloat16() for _ in range(3)]
    h, m = torch.randn(R, 3, D, device=dev), torch.randn(R, 4, D, device=dev)

    def run(rows, out):
        for r0 in range(0, R, rows):
            ik.gate_mix(dict(rows=rows, h_stride=3 * D, m_stride=4 * D, out_stride=D, forget=[P(f) for f in F],
                             gain=[P(g) for g in G], states=[P(x) + r0 * 64 * 2 for x in S],
                             h=[P(h) + (r0 * 3 * D + g * D) * 4 for g in range(3)],
                             m=[P(m) + (r0 * 4 * D + g * D) * 4 for g in range(3)],
                             out=[P(out[g]) + r0 * D * 4 for g in range(3)]), s)
        torch.cuda.synchronize()

    a = [torch.zeros(R, D, device=dev) for _ in range(3)]
    b = [torch.zeros(R, D, device=dev) for _ in range(3)]
    run(ik.MAX_ROWS, a)
    run(R, b)
    for x, y in zip(a, b):
        assert ((x - y).norm() / x.norm()).item() < 1e-5


def test_row_limits_are_enforced():
    s = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="kMaxChunk rows"):
        # every other input present (non-null), so only the row count can be refused
        ik.attn_flash(dict(rows=ik.MAX_CHUNK + 1, sliding=1, q_pre=16, rel_t=16, stage_kv=16), s)
    with pytest.raises(ValueError, match="at most 8 rows"):
        ik.lm_head(dict(rows=ik.MAX_ROWS + 1, blocks=1), s)
