"""Does the vendored FA2 compute this model's attention, and on both windows?

Two windows matter and they are not the same test. A square block is what a
single-pass prefill asks for and torch already had a fused backend for it, so
the bar there is "no worse". A non-square block -- Sq queries against Sk
accumulated keys -- is what a chunked prefill asks for, torch had no fused
backend for it, and FA2's causal is bottom-right aligned, which is precisely
what that window means. Getting the alignment wrong is silent: it truncates
history and still returns plausible numbers.

Skipped where FA2 is not built, since that is a target property rather than a
failure.
"""

import pytest
import torch
import torch.nn.functional as F

NQ, NKV, HD = 16, 2, 256


def _fwd():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FA2 shape test")
    try:
        import flash_rt.frontends.torch._nexn2_rtx_forward as fwd
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"frontend not importable: {exc}")
    if fwd._get_fa2() is None:
        pytest.skip("FA2 is not built for this target")
    return fwd


def _reference(q, k, v, dev):
    """Bottom-right causal, fp32, scores materialised. Slow and unambiguous."""
    Sq, Sk = q.shape[1], k.shape[1]
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float().repeat_interleave(NQ // NKV, 1)
    vt = v.transpose(1, 2).float().repeat_interleave(NQ // NKV, 1)
    s = (qt @ kt.transpose(-1, -2)) * (HD ** -0.5)
    qi = torch.arange(Sk - Sq, Sk, device=dev).unsqueeze(1)
    mask = torch.arange(Sk, device=dev).unsqueeze(0) <= qi
    s = s.masked_fill(~mask, float("-inf"))
    return (F.softmax(s, -1) @ vt).transpose(1, 2)


def test_probe_accepts_this_target():
    fwd = _fwd()
    assert fwd._fa2_usable("cuda:0"), \
        "FA2 is built but its own probe rejects it here"


@pytest.mark.parametrize("Sq,Sk", [(64, 64), (512, 512), (1024, 1024)])
def test_square_window(Sq, Sk):
    fwd = _fwd()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(Sq)
    q = torch.randn(1, Sq, NQ, HD, generator=g, device=dev,
                    dtype=torch.bfloat16)
    k = torch.randn(1, Sk, NKV, HD, generator=g, device=dev,
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    o = fwd._fa2_causal_attn(q, k, v, dev, _probe=True)
    torch.cuda.synchronize(dev)
    ref = _reference(q, k, v, dev)
    rel = ((o.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3, f"square window off by {rel:.3e}"


@pytest.mark.parametrize("Sq,Sk", [(64, 256), (512, 2048), (256, 4096)])
def test_non_square_window_is_bottom_right(Sq, Sk):
    """The one that used to have no fused backend.

    Also checks the alignment explicitly: a top-left reading of the same
    request drops the history, and the two only coincide when Sq == Sk, so a
    kernel that quietly did the wrong one would pass every square case above.
    """
    fwd = _fwd()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(Sq * 31 + Sk)
    q = torch.randn(1, Sq, NQ, HD, generator=g, device=dev,
                    dtype=torch.bfloat16)
    k = torch.randn(1, Sk, NKV, HD, generator=g, device=dev,
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    o = fwd._fa2_causal_attn(q, k, v, dev, _probe=True)
    torch.cuda.synchronize(dev)

    ref = _reference(q, k, v, dev)
    rel = ((o.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3, f"non-square window off by {rel:.3e}"

    # Top-left would attend query i to keys [0, i] instead of [0, Sk-Sq+i].
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float().repeat_interleave(NQ // NKV, 1)
    vt = v.transpose(1, 2).float().repeat_interleave(NQ // NKV, 1)
    s = (qt @ kt.transpose(-1, -2)) * (HD ** -0.5)
    tl = torch.arange(Sk, device=dev).unsqueeze(0) <= torch.arange(
        Sq, device=dev).unsqueeze(1)
    topleft = (F.softmax(s.masked_fill(~tl, float("-inf")), -1)
               @ vt).transpose(1, 2)
    rel_tl = ((o.float() - topleft).norm() / topleft.norm()).item()
    assert rel_tl > 0.05, \
        "output matches the top-left window; the alignment is wrong"
