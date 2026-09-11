"""Equivalence tests for the WY front-matter kernels.

These replaced a chain of tensor ops -- two l2 normalisations, a per-chunk gate
cumulative sum, a GQA broadcast and the chunk-major packings -- so the
reference here is that chain, written out again rather than imported, because
the point is to check the kernel against what it replaced and not against
itself.

The l2 normalisation is compared by relative error, not exactly: the kernel
reduces 128 elements in butterfly order and the tensor path reduces them in
torch's order. The packing and the broadcast are pure data movement and are
compared exactly -- a permutation that is nearly right is wrong.
"""

import pytest
import torch
import torch.nn.functional as F

NK, NV, HD, CH = 16, 32, 128, 64
QKG = NV // NK


def _load_fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the WY prefill kernel tests")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    for name in ("gdn_wy_norm_pack_q_cumsum_edge_bf16",
                 "gdn_wy_pack_v_edge_bf16"):
        if not hasattr(fvk, name):
            pytest.skip(f"{name} not in this build")
    return fvk


def _ptr(x):
    return x.data_ptr()


def _ref_l2(x):
    """The l2 normalisation the sequential scan and the tensor path both use."""
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16)


def _ref_pack(x, ch=CH):
    """(S, H, D) -> (chunks, H, ch, D), zero-padded past S."""
    s, hh, d = x.shape
    pad = (-s) % ch
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
    return x.reshape(-1, ch, hh, d).permute(0, 2, 1, 3).contiguous()


def _ref_gcumsum(g, ch=CH):
    s = g.shape[0]
    pad = (-s) % ch
    gp = F.pad(g, (0, 0, 0, pad)) if pad else g
    return torch.cumsum(gp.float().reshape(-1, ch, g.shape[1]), 1).reshape(
        -1, g.shape[1])[:s].to(torch.bfloat16)


@pytest.mark.parametrize("S", [1, 64, 100, 512, 2048, 4097])
def test_norm_pack_cumsum_matches_the_chain_it_replaced(S):
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(3 + S)

    # q and k arrive GQA-broadcast across the v-head slots, which is the form
    # the conv split writes and the form the kernel expects.
    q16 = torch.randn(S, NK, HD, generator=g, device=dev, dtype=torch.bfloat16)
    k16 = torch.randn(S, NK, HD, generator=g, device=dev, dtype=torch.bfloat16)
    qb = q16.repeat_interleave(QKG, 1).contiguous()
    kb = k16.repeat_interleave(QKG, 1).contiguous()
    gate = torch.randn(S, NV, generator=g, device=dev,
                       dtype=torch.bfloat16) * 0.1

    chunks = (S + CH - 1) // CH
    k_l2 = torch.empty(S, NK, HD, dtype=torch.bfloat16, device=dev)
    q_pack = torch.empty(chunks, NV, CH, HD, dtype=torch.bfloat16, device=dev)
    gc = torch.empty(S, NV, dtype=torch.bfloat16, device=dev)
    fvk.gdn_wy_norm_pack_q_cumsum_edge_bf16(
        _ptr(qb), _ptr(kb), _ptr(gate), _ptr(k_l2), _ptr(q_pack), _ptr(gc),
        S, NK, NV, HD, QKG, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(dev)

    k_ref = _ref_l2(k16)
    q_ref_pack = _ref_pack(_ref_l2(q16).repeat_interleave(QKG, 1))
    gc_ref = _ref_gcumsum(gate)

    def rel(a, b):
        return ((a.float() - b.float()).norm()
                / b.float().norm().clamp_min(1e-9)).item()

    assert rel(k_l2, k_ref) < 5e-3, "k l2 normalisation drifted"
    assert rel(q_pack, q_ref_pack) < 5e-3, "packed q drifted"
    assert rel(gc, gc_ref) < 5e-3, "gate cumulative sum drifted"

    # The tail of the last chunk must be zero, not whatever was in the buffer:
    # output_o reads the whole chunk.
    if S % CH:
        assert torch.equal(q_pack[-1, :, S % CH:, :],
                           torch.zeros_like(q_pack[-1, :, S % CH:, :])), \
            "packed q tail is not zeroed"

    # Every v-head slot of a GQA group must carry the same vector -- the
    # broadcast is the part that never materialises, so it is the part most
    # able to go wrong silently.
    for r in range(1, QKG):
        assert torch.equal(q_pack[:, 0::QKG], q_pack[:, r::QKG]), \
            f"GQA group member {r} differs from its leader"


@pytest.mark.parametrize("S", [1, 64, 100, 512, 2048, 4097])
def test_pack_v_matches_the_reference_permutation(S):
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(11 + S)
    v = torch.randn(S, NV, HD, generator=g, device=dev, dtype=torch.bfloat16)

    chunks = (S + CH - 1) // CH
    v_pack = torch.empty(chunks, NV, CH, HD, dtype=torch.bfloat16, device=dev)
    fvk.gdn_wy_pack_v_edge_bf16(
        _ptr(v), _ptr(v_pack), S, NV, HD,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(dev)

    # Pure data movement: exact, including the zero tail.
    assert torch.equal(v_pack, _ref_pack(v))


@pytest.mark.parametrize("S", [64, 2048])
def test_reproducible(S):
    """Prefill seeds a decode that has to reproduce."""
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(5)
    qb = torch.randn(S, NV, HD, generator=g, device=dev, dtype=torch.bfloat16)
    kb = torch.randn(S, NV, HD, generator=g, device=dev, dtype=torch.bfloat16)
    gate = torch.randn(S, NV, generator=g, device=dev, dtype=torch.bfloat16)
    chunks = (S + CH - 1) // CH
    st = torch.cuda.current_stream().cuda_stream

    def run():
        k_l2 = torch.empty(S, NK, HD, dtype=torch.bfloat16, device=dev)
        q_pack = torch.empty(chunks, NV, CH, HD, dtype=torch.bfloat16,
                             device=dev)
        gc = torch.empty(S, NV, dtype=torch.bfloat16, device=dev)
        fvk.gdn_wy_norm_pack_q_cumsum_edge_bf16(
            _ptr(qb), _ptr(kb), _ptr(gate), _ptr(k_l2), _ptr(q_pack),
            _ptr(gc), S, NK, NV, HD, QKG, st)
        torch.cuda.synchronize(dev)
        return k_l2, q_pack, gc

    a = run()
    for _ in range(3):
        for x, y in zip(a, run()):
            assert torch.equal(x, y), "WY front matter is not reproducible"
