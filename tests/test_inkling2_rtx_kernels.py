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


def _gemm(a, b, c, ws, s):
    ik.gemm(dict(m=a.shape[0], n=b.shape[0], k=a.shape[1], a=P(a), lda=a.stride(0), b=P(b), ldb=b.stride(0), c=P(c),
                 ldc=c.stride(0), out_f32=int(c.dtype == torch.float32), workspace=P(ws), workspace_bytes=ws.numel()),
            s)


def _torch_matmul_flags():
    mm = torch.backends.cuda.matmul
    return mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction


@pytest.mark.parametrize("m,n,k,out", [(512, 3072, 2304, torch.float32), (512, 1024, 2304, torch.bfloat16),
                                       (17, 128, 2304, torch.float32), (40, 64, 4608, torch.bfloat16)])
def test_gemm_against_fp64(m, n, k, out):
    """C = A B^T (BF16 inputs, FP32 sums), A as a row slice of a wider tensor (lda > k)."""
    torch.manual_seed(0)
    wide = torch.randn(m, k + 64, device="cuda").bfloat16()
    a, b = wide[:, :k], (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
    c = torch.empty(m, n, device="cuda", dtype=out)
    ws = torch.empty(ik.GEMM_WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
    _gemm(a, b, c, ws, torch.cuda.current_stream().cuda_stream)
    ref = a.double() @ b.double().t()
    scale = a.double().abs() @ b.double().abs().t()          # the FP32 sums' error bound scales with sum |a b|
    bound = 1e-5 * scale + (2.0 ** -8 * ref.abs() if out == torch.bfloat16 else 0.0)
    assert ((c.double() - ref).abs() <= bound).all()


def test_gemm_does_not_read_or_change_torch_matmul_settings():
    torch.manual_seed(0)
    a, b = torch.randn(512, 2304, device="cuda").bfloat16(), torch.randn(3072, 2304, device="cuda").bfloat16()
    ws = torch.empty(ik.GEMM_WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
    s, mm = torch.cuda.current_stream().cuda_stream, torch.backends.cuda.matmul
    saved, out = _torch_matmul_flags(), []
    try:
        for fp32, reduced in (("ieee", False), ("tf32", True)):
            mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction = fp32, reduced
            for dt in (torch.float32, torch.bfloat16):
                c = torch.empty(512, 3072, device="cuda", dtype=dt)
                _gemm(a, b, c, ws, s)
                out.append(c)
            assert _torch_matmul_flags() == (fp32, reduced)
    finally:
        mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction = saved
    assert torch.equal(out[0], out[2]) and torch.equal(out[1], out[3])


def test_gemm_from_concurrent_threads():
    """Several threads on their own streams and workspaces: every result equals the sequential one bit for bit."""
    import threading
    torch.manual_seed(0)
    a, b = torch.randn(256, 2304, device="cuda").bfloat16(), torch.randn(1024, 2304, device="cuda").bfloat16()
    ws0 = torch.empty(ik.GEMM_WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
    ref = torch.empty(256, 1024, device="cuda")
    _gemm(a, b, ref, ws0, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    flags, errors = _torch_matmul_flags(), []

    def work():
        try:
            st = torch.cuda.Stream()
            ws = torch.empty(ik.GEMM_WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
            outs = [torch.empty(256, 1024, device="cuda") for _ in range(20)]
            for c in outs:
                _gemm(a, b, c, ws, st.cuda_stream)
            st.synchronize()
            if not all(torch.equal(c, ref) for c in outs):
                errors.append("a result differs from the sequential one")
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert _torch_matmul_flags() == flags


def test_gemm_call_keeps_its_operands_alive():
    """A recorded product holds raw pointers: the call keeps its tensors referenced, so an output passed as a
    temporary is not freed (and its memory not reused by the next allocation) before the product runs."""
    import gc
    from flash_rt.frontends.torch._inkling2_rtx_runtime import gemm_call
    torch.manual_seed(0)
    a, b = torch.randn(64, 2304, device="cuda").bfloat16(), torch.randn(128, 2304, device="cuda").bfloat16()
    ws = torch.empty(ik.GEMM_WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
    call = gemm_call(a, b, torch.empty(64, 128, device="cuda"), ws)
    gc.collect()
    fillers = [torch.full((64, 128), 7.0, device="cuda") for _ in range(4)]   # would reuse a freed block
    call[0](call[1], torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert all(bool((f == 7.0).all()) for f in fillers)
    out = call[1]["_keep"][2]
    assert out.data_ptr() == call[1]["c"]
    assert torch.allclose(out, (a.float() @ b.float().t()), rtol=1e-3, atol=1e-2)


# ---------------------------------------------------------------------------------- tensor-core GEMVs
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _mma_weights(kind, n, k):
    """(w, aux, aux2) of a tensor-core GEMV and the weights it represents (FP64)."""
    from flash_rt.frontends.torch._inkling2_rtx_runtime import e4m3_rows, nvfp4_rows
    W = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    if kind == "gemv_mma":
        return (W, 0, 0), W.double()
    if kind == "gemv_fp8":
        codes, scale = e4m3_rows(W)
        return (codes, scale, 0), codes.view(torch.float8_e4m3fn).double() * scale.double()[:, None]
    codes, gs, rs = nvfp4_rows(W)
    nib = torch.stack([codes & 15, codes >> 4], -1).view(n, k).long()
    vals = torch.tensor(_E2M1, device="cuda", dtype=torch.float64)[nib]
    g = gs.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1)
    return (codes, rs, gs), vals * g * rs.double()[:, None]


def _mma_task(w, x, y, n, k, rows, yf=0):
    w, aux, aux2 = w
    t = (P(w), P(x), P(y), n, k, 0, yf, ik.GEMV_PLAIN, P(aux) if torch.is_tensor(aux) else 0, rows, 0)
    return t + ((P(aux2),) if torch.is_tensor(aux2) else ())


@pytest.mark.parametrize("kind", ["gemv_mma", "gemv_fp8", "gemv_fp4w"])
@pytest.mark.parametrize("yf", [0, 1])
def test_tensor_core_gemv_rows_are_independent(kind, yf):
    """Every row of a multi-row launch bit-equal to a one-row launch on that row (verification rows equal decode
    steps), whatever the row count; FP32 sums close to the represented weights' product."""
    torch.manual_seed(0)
    n, k, s = 2304, 4608, torch.cuda.current_stream().cuda_stream
    w, ref_w = _mma_weights(kind, n, k)
    fn = getattr(ik, kind)
    x = torch.randn(ik.MAX_MMA_ROWS, k, device="cuda").bfloat16()
    yd = torch.float32 if yf else torch.bfloat16
    outs = {}
    for R in (ik.MAX_MMA_ROWS, 7, 1):
        y = torch.empty(R, n, device="cuda", dtype=yd)
        fn([_mma_task(w, x, y, n, k, R, yf)], [], s)
        outs[R] = y
    torch.cuda.synchronize()
    assert torch.equal(outs[7], outs[ik.MAX_MMA_ROWS][:7]) and torch.equal(outs[1], outs[7][:1])
    ref = x.double() @ ref_w.t()
    err = (outs[ik.MAX_MMA_ROWS].double() - ref).norm() / ref.norm()
    assert err < (3e-3 if yf == 0 else 1e-5)          # BF16 outputs: their rounding; FP32: the sums' order


def test_fp8_gemv_norm_prologue_equals_norm_rows():
    """One row with the norm in the prologue (decode) equals norm_rows then the plain product (verification rows),
    FP32 input rows at a stride."""
    from flash_rt.frontends.torch._inkling2_rtx_runtime import e4m3_rows
    torch.manual_seed(0)
    n, s = 3072, torch.cuda.current_stream().cuda_stream
    codes, scale = e4m3_rows((torch.randn(n, D, device="cuda") * 0.02).bfloat16())
    nw = (torch.rand(D, device="cuda") + 0.5).bfloat16()
    x = torch.randn(1, 3, D, device="cuda") * 3
    for g in range(3):
        xg = P(x) + g * D * 4
        a = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        ik.gemv_fp8([(P(codes), xg, P(a), n, D, 1, 0, ik.GEMV_NORM, P(scale), 1, 3 * D, P(nw))], [], s)
        xn = torch.empty(1, D, device="cuda", dtype=torch.bfloat16)
        ik.norm_rows([(xg, 1, 3 * D, P(nw), P(xn), 1, D)], s)
        b = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        ik.gemv_fp8([(P(codes), P(xn), P(b), n, D, 0, 0, ik.GEMV_PLAIN, P(scale), 1, 0)], [], s)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"query {g}"


# ---------------------------------------------------------------------------------- message gates
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("gates", [3, 1])
def test_msg_gate_equals_the_three_kernels(rows, gates):
    """The cluster kernel (controllers' down projections, gate_mix, input norms) bit-equal to gemv_multi, gate_mix
    and norm_rows: the attention side (three gates, FP32 inputs at a row stride of 3 D, normalized rows) and the
    MLP side (one gate, no norm)."""
    torch.manual_seed(0)
    R, s, G = rows, torch.cuda.current_stream().cuda_stream, gates
    hs = 3 * D if G == 3 else D
    down = [(torch.randn(64, 2 * D, device="cuda") * 0.02).bfloat16() for _ in range(G)]
    gin = [torch.randn(R, 2 * D, device="cuda").bfloat16() for _ in range(G)]
    F = [(torch.randn(D, 64, device="cuda") * 0.05).bfloat16() for _ in range(G)]
    Gn = [(torch.randn(D, 64, device="cuda") * 0.05).bfloat16() for _ in range(G)]
    nw = [(torch.rand(D, device="cuda") + 0.5).bfloat16() for _ in range(G)]
    h, m = torch.randn(R, hs, device="cuda"), torch.randn(R, 4, D, device="cuda")
    hp = [P(h) + g * D * 4 for g in range(G)]
    mp = [P(m) + g * D * 4 for g in range(G)]
    common = dict(rows=R, h_stride=hs, m_stride=4 * D, out_stride=D, forget=[P(f) for f in F],
                  gain=[P(g) for g in Gn], h=hp, m=mp)

    def bufs():
        return ([torch.zeros(R, 64, device="cuda", dtype=torch.bfloat16) for _ in range(G)],
                [torch.zeros(R, D, device="cuda") for _ in range(G)],
                [torch.zeros(R, D, device="cuda", dtype=torch.bfloat16) for _ in range(G)])

    st_a, mx_a, nr_a = bufs()
    ik.gemv_multi([(P(down[g]), P(gin[g]), P(st_a[g]), 64, 2 * D, 0, 0, ik.GEMV_PLAIN, 0, R) for g in range(G)],
                  [], s)
    ik.gate_mix(dict(common, states=[P(t) for t in st_a], out=[P(t) for t in mx_a]), s)
    if G == 3:
        ik.norm_rows([(P(mx_a[g]), 1, 0, P(nw[g]), P(nr_a[g]), R, D) for g in range(G)], s)
    st_b, mx_b, nr_b = bufs()
    norm = dict(norm_w=[P(t) for t in nw], norm_out=[P(t) for t in nr_b]) if G == 3 else {}
    ik.msg_gate(dict(common, gin_stride=2 * D, down=[P(t) for t in down], gin=[P(t) for t in gin],
                     states=[P(t) for t in st_b], out=[P(t) for t in mx_b], **norm), s)
    torch.cuda.synchronize()
    for g in range(G):
        assert torch.equal(st_a[g], st_b[g]) and torch.equal(mx_a[g], mx_b[g]), f"gate {g}"
        if G == 3:
            assert torch.equal(nr_a[g], nr_b[g]), f"norm {g}"

