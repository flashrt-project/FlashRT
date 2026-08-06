#!/usr/bin/env python3
"""Unit tests for ThorFlashAttnBackend construction + get_slot_ptrs.

These tests use fake (non-zero) integer pointers — no GPU, no fvk call.
Validation goals:
  * Spec shape checks fire correctly (wrong sites, wrong D).
  * Required slot keys enforced, null-pointer slots rejected.
  * SigLIP pointer arithmetic: Q/K/V offsets are (0, 2*D, 4*D) bytes.
  * Encoder/decoder per-layer K/V pointers advance by layer_stride.
  * Encoder/decoder O aliases Q.
  * layer_idx range checked.

Run: python3 tests/test_thor_attn_backend.py
"""

from __future__ import annotations

import sys
import traceback


def _fake_slots():
    """Build a self-consistent set of slot dicts with dummy pointers.

    Pointers are chosen as obviously-distinct values so mis-wiring would
    surface as a printable number in asserts.
    """
    D_sig = 16 * 72  # siglip num_q_heads * head_dim
    enc_stride = 1000 * 256 * 2  # some total_keys * HD * 2 (bytes)

    siglip = {"qkv": 0x1000, "O": 0x2000, "D": D_sig}
    encoder = {
        "Q_O": 0x3000,
        "Kc":  0x40000,
        "Vc":  0x50000,
        "logits": 0x6000,
        "layer_stride": enc_stride,
        "scale": 1.0 / (256 ** 0.5),
    }
    decoder = {
        "Q_O": 0x7000,
        "Kc":  0x40000,  # same base as encoder (shared cache)
        "Vc":  0x50000,
        "logits": 0x8000,
        "layer_stride": enc_stride,
        "scale": 1.0 / (256 ** 0.5),
    }
    return siglip, encoder, decoder, enc_stride, D_sig


def test_construct_ok():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600, chunk_size=10)

    class _DummyCtx:
        cpp = 0xdead

    b = ThorFlashAttnBackend(spec, _DummyCtx(),
                              siglip_slots=siglip,
                              encoder_slots=encoder,
                              decoder_slots=decoder)
    assert set(b.sites()) == {"siglip", "encoder", "decoder"}, b.sites()
    assert b.head_dim("encoder") == 256
    assert b.num_q_heads("encoder") == 8
    assert b.num_kv_heads("encoder") == 1
    print("  PASS  test_construct_ok")


def test_siglip_slot_ptrs():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, _, D_sig = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)

    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(),
                              siglip_slots=siglip,
                              encoder_slots=encoder,
                              decoder_slots=decoder)

    # layer_idx is ignored for siglip (all layers share one scratch)
    for li in (0, 5, 26):
        p = b.get_slot_ptrs("siglip", li)
        assert p["Q"] == 0x1000, p
        assert p["K"] == 0x1000 + D_sig * 2, p
        assert p["V"] == 0x1000 + 2 * D_sig * 2, p
        assert p["O"] == 0x2000, p
    print("  PASS  test_siglip_slot_ptrs")


def test_encoder_slot_ptrs():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, stride, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)

    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(),
                              siglip_slots=siglip,
                              encoder_slots=encoder,
                              decoder_slots=decoder)

    # layer 0 → base; layer 17 → base + 17*stride; O aliases Q.
    for li, expected_k in [(0, 0x40000), (1, 0x40000 + stride),
                           (17, 0x40000 + 17 * stride)]:
        p = b.get_slot_ptrs("encoder", li)
        assert p["Q"] == 0x3000 == p["O"], p
        assert p["K"] == expected_k, (li, hex(p["K"]), hex(expected_k))
        assert p["V"] == expected_k + (0x50000 - 0x40000), p
    print("  PASS  test_encoder_slot_ptrs")


def test_decoder_slot_ptrs():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, stride, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)

    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(),
                              siglip_slots=siglip,
                              encoder_slots=encoder,
                              decoder_slots=decoder)

    # Decoder Q_O is distinct from encoder Q_O; KV base shared.
    p0 = b.get_slot_ptrs("decoder", 0)
    p17 = b.get_slot_ptrs("decoder", 17)
    assert p0["Q"] == 0x7000 == p0["O"], p0
    assert p17["K"] == 0x40000 + 17 * stride, p17
    assert p0["K"] == b.get_slot_ptrs("encoder", 0)["K"], "shared cache"
    print("  PASS  test_decoder_slot_ptrs")


def test_layer_idx_oob():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    for bad in (-1, 18, 999):
        try:
            b.get_slot_ptrs("encoder", bad)
        except IndexError:
            pass
        else:
            raise AssertionError(f"expected IndexError for layer_idx={bad}")
    print("  PASS  test_layer_idx_oob")


def test_reject_null_ptr():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, _, _ = _fake_slots()
    siglip["O"] = 0
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    class _C: cpp = 0
    try:
        ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    except ValueError as e:
        assert "null" in str(e).lower(), str(e)
        print("  PASS  test_reject_null_ptr")
        return
    raise AssertionError("null pointer slot should have been rejected")


def test_reject_bad_D():
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )
    siglip, encoder, decoder, _, _ = _fake_slots()
    siglip["D"] = 42  # wrong — spec expects 16*72=1152
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    class _C: cpp = 0
    try:
        ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    except ValueError as e:
        assert "num_q_heads*head_dim" in str(e), str(e)
        print("  PASS  test_reject_bad_D")
        return
    raise AssertionError("bad D should have been rejected")


def test_reject_extra_site():
    from flash_rt.hardware.thor.attn_backend import ThorFlashAttnBackend
    from flash_rt.hardware.backend import AttentionSpec
    # Build a spec with an unexpected site.
    spec = AttentionSpec()
    spec.add_site("siglip",  num_layers=27, num_q_heads=16, num_kv_heads=16,
                             head_dim=72, max_q_seq=256)
    spec.add_site("encoder", num_layers=18, num_q_heads=8, num_kv_heads=1,
                             head_dim=256, max_q_seq=600)
    spec.add_site("decoder", num_layers=18, num_q_heads=8, num_kv_heads=1,
                             head_dim=256, max_q_seq=10, max_kv_seq=610)
    spec.add_site("rogue",   num_layers=1, num_q_heads=1, num_kv_heads=1,
                             head_dim=1, max_q_seq=1)

    siglip, encoder, decoder, _, _ = _fake_slots()
    class _C: cpp = 0
    try:
        ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    except ValueError as e:
        assert "sites" in str(e), str(e)
        print("  PASS  test_reject_extra_site")
        return
    raise AssertionError("extra site should have been rejected")


def test_required_fa4_unavailable_raises():
    from flash_rt.hardware.thor import fa4_backend
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )

    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    original_fwd = fa4_backend.fa4_fwd
    original_status = fa4_backend.status
    fa4_backend.fa4_fwd = lambda: None
    fa4_backend.status = lambda: "missing-test-runtime"
    try:
        class _C: cpp = 0
        try:
            ThorFlashAttnBackend(
                spec, _C(), siglip_slots=siglip, encoder_slots=encoder,
                decoder_slots=decoder, use_fa4=True)
        except RuntimeError as exc:
            assert "missing-test-runtime" in str(exc), str(exc)
        else:
            raise AssertionError("required FA4 must not select another kernel")
    finally:
        fa4_backend.fa4_fwd = original_fwd
        fa4_backend.status = original_status


def test_required_fa4_rejects_fixed_shape():
    from flash_rt.hardware.thor import fa4_backend
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )

    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    original_fwd = fa4_backend.fa4_fwd
    fa4_backend.fa4_fwd = lambda: object()
    try:
        class _C: cpp = 0
        backend = ThorFlashAttnBackend(
            spec, _C(), siglip_slots=siglip, encoder_slots=encoder,
            decoder_slots=decoder, use_fa4=True)
        try:
            backend.set_fixed_shape(True)
        except RuntimeError as exc:
            assert "does not support fixed-shape" in str(exc), str(exc)
        else:
            raise AssertionError("FA4 fixed-shape must fail explicitly")
    finally:
        fa4_backend.fa4_fwd = original_fwd


def test_fa4_writes_pipeline_owned_outputs():
    import flash_rt.hardware.thor.attn_backend as attn_backend
    from flash_rt.hardware.thor import fa4_backend
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi05_attention_spec,
    )

    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    original_fwd = fa4_backend.fa4_fwd
    original_view = attn_backend._fp16_tensor_from_ptr
    calls = []
    views = []

    class _TensorView:
        def __init__(self, ptr, shape, strides, offset):
            self.ptr = ptr
            self.shape = shape
            self.strides = strides
            self.offset = offset
            self.copied_from = None

        def copy_(self, source):
            self.copied_from = source

    def fake_view(ptr, shape, strides=None, offset=0):
        view = _TensorView(ptr, shape, strides, offset)
        views.append(view)
        return view

    def fake_fwd(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return kwargs["out"], None

    fa4_backend.fa4_fwd = lambda: fake_fwd
    attn_backend._fp16_tensor_from_ptr = fake_view
    try:
        class _C: cpp = 0
        backend = ThorFlashAttnBackend(
            spec, _C(), siglip_slots=siglip, encoder_slots=encoder,
            decoder_slots=decoder, use_fa4=True)
        backend._fvk = object()

        assert backend.run("siglip", 0, q_seq=256) == siglip["O"]
        assert calls[0][3]["out"].ptr == siglip["O"]
        assert calls[0][3]["out"].shape == (2, 256, 16, 72)
        assert calls[0][3]["pack_gqa"] is False

        assert backend.run("encoder", 0, q_seq=600) == encoder["Q_O"]
        assert calls[1][3]["out"].ptr == encoder["logits"]
        assert calls[1][3]["out"].shape == (1, 600, 8, 256)
        assert calls[1][3]["pack_gqa"] is True
        assert calls[1][0].copied_from is calls[1][3]["out"]
        assert not hasattr(backend, "_fa4_outputs")
    finally:
        fa4_backend.fa4_fwd = original_fwd
        attn_backend._fp16_tensor_from_ptr = original_view


def _make_pi0_backend():
    """Helper: build a Pi0-spec'd ThorFlashAttnBackend with fake ptrs."""
    from flash_rt.hardware.thor.attn_backend import (
        ThorFlashAttnBackend, make_pi0_attention_spec,
    )
    siglip, encoder, decoder, _, _ = _fake_slots()
    spec = make_pi0_attention_spec(num_views=2, enc_seq_max=600, S_dec=11)
    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    return b, spec


def test_pi0_spec_decoder_extra():
    """Pi0 decoder site carries extra={'kernel': 'state_masked'}."""
    from flash_rt.hardware.thor.attn_backend import make_pi0_attention_spec
    spec = make_pi0_attention_spec(num_views=2, enc_seq_max=600, S_dec=11)
    dec = spec.site("decoder")
    assert dec.extra.get("kernel") == "state_masked", dec.extra
    assert dec.max_q_seq == 11, dec.max_q_seq
    assert dec.max_kv_seq == 611, dec.max_kv_seq
    # Pi0.5 spec: default extra (no kernel key).
    from flash_rt.hardware.thor.attn_backend import make_pi05_attention_spec
    p05 = make_pi05_attention_spec(num_views=2, enc_seq_max=600)
    assert "kernel" not in p05.site("decoder").extra, p05.site("decoder").extra
    print("  PASS  test_pi0_spec_decoder_extra")


def test_state_masked_requires_state_nk():
    """run() on state_masked site without state_nk raises ValueError."""
    b, _ = _make_pi0_backend()
    try:
        b.run("decoder", 0, q_seq=11, kv_seq=611, stream=0)  # no state_nk
    except ValueError as e:
        assert "state_nk" in str(e), str(e)
        print("  PASS  test_state_masked_requires_state_nk")
        return
    raise AssertionError("state_masked without state_nk should have raised")


def test_state_masked_state_nk_range():
    """state_nk out of [1, kv_seq] range raises."""
    b, _ = _make_pi0_backend()
    for bad in (0, 612, -1):
        try:
            b.run("decoder", 0, q_seq=11, kv_seq=611, stream=0, state_nk=bad)
        except ValueError as e:
            assert "state_nk" in str(e), str(e)
            continue
        raise AssertionError(f"state_nk={bad} should have raised")
    print("  PASS  test_state_masked_state_nk_range")


def test_unknown_kernel_rejected():
    """A site with extra['kernel'] set to an unknown value raises at run()."""
    from flash_rt.hardware.thor.attn_backend import ThorFlashAttnBackend
    from flash_rt.hardware.backend import AttentionSpec
    spec = AttentionSpec()
    spec.add_site("siglip",  num_layers=27, num_q_heads=16, num_kv_heads=16,
                             head_dim=72, max_q_seq=256)
    spec.add_site("encoder", num_layers=18, num_q_heads=8, num_kv_heads=1,
                             head_dim=256, max_q_seq=600)
    spec.add_site("decoder", num_layers=18, num_q_heads=8, num_kv_heads=1,
                             head_dim=256, max_q_seq=10, max_kv_seq=610,
                             extra={"kernel": "bogus_kernel"})
    siglip, encoder, decoder, _, _ = _fake_slots()
    class _C: cpp = 0
    b = ThorFlashAttnBackend(spec, _C(), siglip_slots=siglip,
                              encoder_slots=encoder, decoder_slots=decoder)
    try:
        b.run("decoder", 0, q_seq=10, kv_seq=610, stream=0)
    except ValueError as e:
        assert "bogus_kernel" in str(e) or "unknown kernel" in str(e), str(e)
        print("  PASS  test_unknown_kernel_rejected")
        return
    raise AssertionError("unknown kernel should have raised")


def main() -> int:
    tests = [
        test_construct_ok,
        test_siglip_slot_ptrs,
        test_encoder_slot_ptrs,
        test_decoder_slot_ptrs,
        test_layer_idx_oob,
        test_reject_null_ptr,
        test_reject_bad_D,
        test_reject_extra_site,
        test_required_fa4_unavailable_raises,
        test_required_fa4_rejects_fixed_shape,
        test_fa4_writes_pipeline_owned_outputs,
        test_pi0_spec_decoder_extra,
        test_state_masked_requires_state_nk,
        test_state_masked_state_nk_range,
        test_unknown_kernel_rejected,
    ]
    print("=== ThorFlashAttnBackend unit tests ===")
    failures = 0
    for t in tests:
        try:
            t()
        except Exception:
            failures += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"--- {len(tests) - failures}/{len(tests)} passed ---")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
