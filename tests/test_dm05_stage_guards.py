#!/usr/bin/env python3
"""The DM05 stages reject a checkpoint they cannot compute.

Each stage's constants describe one specific checkpoint, and every one of them is
load-bearing: nothing downstream re-checks that the tower it was handed has 34
layers, or that its k projection is four heads wide. A mismatch would either
raise somewhere far from the cause or, for the sequence length, quietly produce a
different number of image tokens than the prefix expects.

These tests only build the stages, so they use CPU tensors of the right rank and
shape and never touch a device -- the constructors read shapes, not values. The
GemmRunner and the two extensions are stubbed for the same reason.

What is NOT here: any check of the arithmetic. That needs the checkpoint and the
reference implementation, and lives outside the repository as a bit-parity
comparison against the reference's own forward.

Usage:
  PYTHONPATH=. pytest -q tests/test_dm05_stage_guards.py
"""
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

TEXT_HIDDEN = 2560   # the projector's output width; not a stage constant


def _stage_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt.models.pi05 import pipeline_rtx_dm05 as m
    except ImportError as exc:
        pytest.skip(f"flash_rt_dm05 not built: {exc}")
    return m


def _t(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


class _Stub:
    """Stands in for the GemmRunner and the extension: the constructors under
    test never call into either, they only read shapes."""

    def __getattr__(self, name):
        return lambda *a, **k: None


def _vision_weights(m, layers=None):
    n = m.VIS_L if layers is None else layers
    return {
        "w_patch": _t(m.VIS_D, m.PATCH_FEAT), "b_patch": _t(m.VIS_D),
        "pos": _t(2 * m.VIS_SEQ_PER_VIEW, m.VIS_D),
        "post_w": _t(m.VIS_D), "post_b": _t(m.VIS_D),
        "mm_norm_w": _t(m.VIS_D), "mm_proj_w": _t(TEXT_HIDDEN, m.VIS_D),
        "layers": [{"qkv_w": _t(3 * m.VIS_D, m.VIS_D)} for _ in range(n)],
    }


class TestVisionStage:
    def _build(self, m, w, **kw):
        return m.Dm05VisionStage(w, {}, fvk=_Stub(), dm05=_Stub(), gemm=_Stub(), **kw)

    def test_accepts_the_shape_it_describes(self):
        m = _stage_module()
        self._build(m, _vision_weights(m), num_views=2)

    @pytest.mark.parametrize("key", ["w_patch", "mm_proj_w"])
    def test_rejects_a_wrong_tower_weight(self, key):
        m = _stage_module()
        w = _vision_weights(m)
        w[key] = _t(7, 7)
        with pytest.raises(ValueError, match="Dm05VisionStage"):
            self._build(m, w, num_views=2)

    def test_rejects_the_wrong_layer_count(self):
        """27 layers is this tower; a checkpoint with 26 would still build."""
        m = _stage_module()
        with pytest.raises(ValueError, match="vision layers"):
            self._build(m, _vision_weights(m, layers=m.VIS_L - 1), num_views=2)

    def test_rejects_the_wrong_head_layout(self):
        m = _stage_module()
        w = _vision_weights(m)
        w["layers"][0]["qkv_w"] = _t(2 * m.VIS_D, m.VIS_D)
        with pytest.raises(ValueError, match="qkv weight"):
            self._build(m, w, num_views=2)

    def test_rejects_the_wrong_view_count(self):
        """The prefix this feeds is built for two views."""
        m = _stage_module()
        with pytest.raises(ValueError, match="views"):
            self._build(m, _vision_weights(m), num_views=1)


def _prefix_weights(m, layers=None, max_seq=640):
    n = m.ENC_L if layers is None else layers
    ly = {"q": _t(m.ENC_NH * m.ENC_HD, m.ENC_D), "k": _t(m.ENC_NKV * m.ENC_HD, m.ENC_D),
          "q_norm": _t(m.ENC_HD), "k_norm": _t(m.ENC_HD)}
    return {
        "embed_w": _t(1000, m.ENC_D), "embed_scale": 50.0, "final_norm": _t(m.ENC_D),
        "layer_types": ["sliding_attention"] * n, "scale": 0.0625, "nsms": 1,
        "rope": {"sliding_attention": (_t(max_seq, m.ENC_HD), _t(max_seq, m.ENC_HD)),
                 "full_attention": (_t(max_seq, m.ENC_HD), _t(max_seq, m.ENC_HD))},
        "layers": [dict(ly) for _ in range(n)],
    }


class TestPrefixStage:
    def _build(self, m, w, max_seq=640):
        return m.Dm05PrefixStage(w, {}, max_seq=max_seq, fvk=_Stub(), gemm=_Stub(),
                                 fa2=_Stub())

    def test_accepts_the_shape_it_describes(self):
        m = _stage_module()
        self._build(m, _prefix_weights(m))

    def test_rejects_a_wrong_embedding_width(self):
        m = _stage_module()
        w = _prefix_weights(m)
        w["embed_w"] = _t(1000, 7)
        with pytest.raises(ValueError, match="embedding width"):
            self._build(m, w)

    def test_rejects_the_wrong_layer_count(self):
        m = _stage_module()
        with pytest.raises(ValueError, match="prefix layers"):
            self._build(m, _prefix_weights(m, layers=m.ENC_L - 1))

    def test_rejects_the_wrong_kv_head_count(self):
        """One KV head is the pi05 encoder's layout, not this one's."""
        m = _stage_module()
        w = _prefix_weights(m)
        w["layers"][0]["k"] = _t(m.ENC_HD, m.ENC_D)
        with pytest.raises(ValueError, match="k weight"):
            self._build(m, w)

    def test_rejects_a_missing_rope_table(self):
        """The two layer types use different bases; one table cannot serve both."""
        m = _stage_module()
        w = _prefix_weights(m)
        del w["rope"]["full_attention"]
        with pytest.raises(ValueError, match="rope table"):
            self._build(m, w)

    def test_rejects_a_max_seq_the_norms_cannot_address(self):
        m = _stage_module()
        with pytest.raises(ValueError, match="multiple of 8"):
            self._build(m, _prefix_weights(m, max_seq=637), max_seq=637)


class TestPrefixStagePromptBinding:
    """set_prompt is where a caller's run table meets the preallocated buffers."""

    def _stage(self, m, max_seq=640):
        return m.Dm05PrefixStage(_prefix_weights(m, max_seq=max_seq), {}, max_seq=max_seq,
                                 fvk=_Stub(), gemm=_Stub(), fa2=_Stub())

    def test_a_prompt_longer_than_max_seq_is_rejected(self):
        m = _stage_module()
        s = self._stage(m)
        n = 641
        with pytest.raises(ValueError, match="exceeds"):
            s.set_prompt(np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int32),
                         np.zeros(n, dtype=np.int32), [(0, n, n, True)])

    def test_a_segment_reading_past_the_prompt_is_rejected(self):
        """A run whose key limit exceeds the prompt would attend stale cache rows."""
        m = _stage_module()
        s = self._stage(m)
        with pytest.raises(ValueError, match="reads keys past"):
            s.set_prompt(np.zeros(8, dtype=np.int64), np.full(8, -1, dtype=np.int32),
                         np.zeros(8, dtype=np.int32), [(0, 8, 9, True)])

    def test_a_segment_outside_the_prompt_is_rejected(self):
        m = _stage_module()
        s = self._stage(m)
        with pytest.raises(ValueError, match="outside"):
            s.set_prompt(np.zeros(8, dtype=np.int64), np.full(8, -1, dtype=np.int32),
                         np.zeros(8, dtype=np.int32), [(4, 4, 4, True)])


def _action_weights(m, layers=None, steps=10):
    n = m.DEC_L if layers is None else layers
    return {
        "horizon": 10, "action_dim": m.ACTION_DIM, "scale": 0.0625, "nsms": 1,
        "fa2_splits": 16, "layer_types": ["sliding_attention"] * n,
        "in_mod": _t(steps, n, 3 * m.DEC_D), "mlp_mod": _t(steps, n, 3 * m.DEC_D),
        "fin_mod": _t(steps, 3 * m.DEC_D),
        "act_mask": torch.ones(m.ACTION_DIM, dtype=torch.float32),
        "rope": {"sliding_attention": (_t(650, m.DEC_HD), _t(650, m.DEC_HD)),
                 "full_attention": (_t(650, m.DEC_HD), _t(650, m.DEC_HD))},
        "layers": [{"q": _t(m.DEC_NH * m.DEC_HD, m.DEC_D)} for _ in range(n)],
    }


class _PrefixStub:
    def __init__(self, L, NKV, HD):
        self.L, self.NKV, self.HD, self.seq = L, NKV, HD, 0


class TestActionStage:
    def _prefix(self, m, L=None):
        return _PrefixStub(m.DEC_L if L is None else L, m.DEC_NKV, m.DEC_HD)

    def _build(self, m, w, p):
        return m.Dm05ActionStage(w, {}, p, fvk=_Stub(), gemm=_Stub(), fa2=_Stub())

    def test_accepts_the_shape_it_describes(self):
        m = _stage_module()
        p = self._prefix(m)
        self._build(m, _action_weights(m), p)

    def test_rejects_a_prefix_cache_of_the_wrong_geometry(self):
        """The expert writes its chunk's K/V into the prefix's cache, so the two
        have to agree on the layer count and the KV width or every write lands in
        the wrong place."""
        m = _stage_module()
        p = self._prefix(m, L=m.DEC_L - 1)
        with pytest.raises(ValueError, match="prefix cache"):
            self._build(m, _action_weights(m), p)

    def test_rejects_the_wrong_layer_count(self):
        m = _stage_module()
        p = self._prefix(m)
        with pytest.raises(ValueError, match="action layers"):
            self._build(m, _action_weights(m, layers=m.DEC_L - 1), p)

    @pytest.mark.parametrize("key", ["in_mod", "mlp_mod", "fin_mod"])
    def test_rejects_a_wrong_modulation(self, key):
        """The modulation shape encodes the step count and the layer count."""
        m = _stage_module()
        p = self._prefix(m)
        w = _action_weights(m)
        w[key] = _t(3, 3, 3)
        with pytest.raises(ValueError, match=key):
            self._build(m, w, p)

    def test_rejects_a_wrong_action_mask(self):
        m = _stage_module()
        p = self._prefix(m)
        w = _action_weights(m)
        w["act_mask"] = torch.ones(m.ACTION_DIM + 1, dtype=torch.float32)
        with pytest.raises(ValueError, match="action mask"):
            self._build(m, w, p)

    def test_rejects_a_missing_rope_table(self):
        m = _stage_module()
        p = self._prefix(m)
        w = _action_weights(m)
        del w["rope"]["full_attention"]
        with pytest.raises(ValueError, match="rope table"):
            self._build(m, w, p)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
