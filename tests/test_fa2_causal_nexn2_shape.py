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

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

NQ, NKV, HD = 16, 2, 256


def test_causal_wrapper_respects_slim_hdim_matrix():
    """Python and native adapters must use the same macro-gated dispatch.

    Thor builds only hdim=256. An older Python-only branch referenced the
    hdim=128 templates unconditionally, so the module linked but failed to
    import with an undefined symbol.
    """
    source = (
        Path(__file__).parents[1]
        / "csrc/attention/fa2_wrapper_causal.cu"
    ).read_text()
    bf16_body = source.split(
        'extern "C" void fvk_attention_fa2_fwd_bf16_causal(', 1
    )[1].split("#else  // !FA2_HAS_BF16", 1)[0]

    assert "FLASHRT_FA2_NATIVE_BUILD" not in bf16_body
    for head_dim in (128, 256):
        guard = (
            "#if defined(FA2_HAS_BF16) && "
            f"defined(FA2_HAS_HDIM_{head_dim})"
        )
        assert bf16_body.count(guard) == 2


def _fwd():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FA2 shape test")
    import flash_rt.frontends.torch._nexn2_rtx_forward as fwd
    if fwd._get_fa2() is None:
        pytest.skip("FA2 is not built for this target")
    return fwd


def test_optional_fa2_import_only_swallows_its_own_absence(monkeypatch):
    import importlib
    import flash_rt.frontends.torch._nexn2_rtx_forward as fwd

    fwd._FA2_MOD = None

    def absent(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", absent)
    assert fwd._get_fa2() is None

    fwd._FA2_MOD = None

    def broken(_name):
        raise ImportError("undefined symbol: run_mha_fwd")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ImportError, match="undefined symbol"):
        fwd._get_fa2()


def test_backend_optional_import_propagates_link_errors(monkeypatch):
    import importlib
    from flash_rt.hardware.rtx.attn_backend_nexn2 import _load_optional_fa2

    def broken(_name):
        raise ImportError("undefined symbol: run_mha_fwd")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ImportError, match="undefined symbol"):
        _load_optional_fa2()


def _backend_with_failed_probe():
    from flash_rt.hardware.rtx.attn_backend_nexn2 import (
        RtxFlashAttnBackendNexn2,
    )

    backend = RtxFlashAttnBackendNexn2.__new__(RtxFlashAttnBackendNexn2)
    backend.Q_buf = torch.empty(1, 1, 16, 256, dtype=torch.bfloat16)
    backend.K_cache = torch.empty(1, 8, 2, 256, dtype=torch.bfloat16)
    backend.V_cache = torch.empty_like(backend.K_cache)
    backend.O_buf = torch.empty_like(backend.Q_buf)

    def fail(*_args, **_kwargs):
        raise RuntimeError("probe launch failed")

    backend._launch_fa2 = fail
    return backend


def test_probe_failure_warns_before_automatic_fallback():
    backend = _backend_with_failed_probe()
    backend._require_fa2 = False
    with pytest.warns(RuntimeWarning, match="SDPA attention fallback"):
        assert backend._probe_fa2() is False


def test_probe_failure_is_fatal_when_fa2_is_explicit():
    backend = _backend_with_failed_probe()
    backend._require_fa2 = True
    with pytest.raises(RuntimeError, match="explicitly requested"):
        backend._probe_fa2()


def test_sdpa_fallback_uses_bottom_right_causal_window():
    from flash_rt.hardware.rtx.attn_backend_nexn2 import (
        RtxFlashAttnBackendNexn2,
    )

    q_seq, kv_seq = 3, 7
    generator = torch.Generator().manual_seed(19)
    backend = RtxFlashAttnBackendNexn2.__new__(RtxFlashAttnBackendNexn2)
    backend.Q_buf = torch.randn(
        1, q_seq, 16, 256, generator=generator, dtype=torch.bfloat16)
    backend.K_cache = torch.randn(
        1, kv_seq, 2, 256, generator=generator, dtype=torch.bfloat16)
    backend.V_cache = torch.randn_like(backend.K_cache)
    backend.O_buf = torch.empty_like(backend.Q_buf)

    scale = 256 ** -0.5
    backend._sdpa(0, q_seq, kv_seq, scale)
    produced = backend.O_buf.float()

    q = backend.Q_buf.transpose(1, 2).float()
    k = backend.K_cache.repeat_interleave(8, dim=2).transpose(1, 2).float()
    v = backend.V_cache.repeat_interleave(8, dim=2).transpose(1, 2).float()
    scores = (q @ k.transpose(-1, -2)) * scale
    q_positions = torch.arange(kv_seq - q_seq, kv_seq).unsqueeze(1)
    mask = torch.arange(kv_seq).unsqueeze(0) <= q_positions
    expected = (scores.masked_fill(~mask, float("-inf")).softmax(-1)
                @ v).transpose(1, 2)
    relative = ((produced - expected).norm()
                / expected.norm().clamp_min(1e-6)).item()
    assert relative < 5e-3


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
