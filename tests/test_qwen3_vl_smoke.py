"""Smoke tests for the Qwen3-VL multimodal frontend.

CI-friendly: no checkpoint, no GPU. Covers the seams a reviewer needs to
trust the model is wired in -- frontend import, the fail-fast kernel-module
check, the pure geometry builders on synthetic image/multi-image/video
sequences, and the input-validation boundaries (text-only and grid
mismatch).

The full cos-vs-HF / argmax E2E tests require the checkpoint and the
SM120 kernels; see docs/qwen3_vl_nvfp4.md (Correctness section).

Run:
    PYTHONPATH=. python -m pytest tests/test_qwen3_vl_smoke.py -v
"""
from __future__ import annotations

import importlib

import pytest
import torch

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

IMG, VID, VSTART = 100, 101, 102
MERGE = 2


# ── wiring / fail-fast ──

def test_frontend_imports():
    """The frontend module + class import without a GPU or checkpoint."""
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx')
    assert hasattr(m, 'Qwen3VlTorchFrontendRtx')
    assert hasattr(m, '_require_qwen3_vl_kernels')


def test_require_kernels_fails_fast_on_missing_symbols():
    """The kernel-module check raises a clear, actionable RuntimeError (with
    the CMake flag) instead of crashing mid-forward after the weights load."""
    from flash_rt.frontends.torch.qwen3_vl_rtx import _check_qwen3_vl_kernels

    class _Empty:                      # stand-in for an incomplete module
        pass

    with pytest.raises(RuntimeError) as ei:
        _check_qwen3_vl_kernels(_Empty())
    assert 'FLASHRT_BUILD_QWEN3_VL' in str(ei.value)


def test_fail_fast_on_each_missing_symbol():
    """Dropping ANY single required symbol (e.g. a stale .so without the
    fused-bias kernels) must raise at the check, naming the missing one —
    not crash mid-ViT with an AttributeError."""
    from flash_rt.frontends.torch.qwen3_vl_rtx import (
        _QWEN3_VL_KERNEL_FNS,
        _check_qwen3_vl_kernels,
    )

    for missing in _QWEN3_VL_KERNEL_FNS:
        stub = type('Stub', (), {fn: (lambda *a, **k: None)
                                 for fn in _QWEN3_VL_KERNEL_FNS
                                 if fn != missing})()
        with pytest.raises(RuntimeError) as ei:
            _check_qwen3_vl_kernels(stub)
        assert missing in str(ei.value)


def test_kernel_module_complete_when_built():
    """If the gated module is built, it must expose every symbol the tower
    calls (otherwise skip -- this build did not enable it)."""
    try:
        from flash_rt import flash_rt_qwen3_vl_kernels as vlk
    except ImportError:
        pytest.skip('flash_rt_qwen3_vl_kernels not built')
    from flash_rt.frontends.torch.qwen3_vl_rtx import _check_qwen3_vl_kernels
    _check_qwen3_vl_kernels(vlk)        # must not raise


# ── vision_segments: walk + validation ──

def _image_seq(grids):
    """Build a synthetic id sequence with one image run per grid row."""
    ids = [1, 1]
    for t, h, w in grids:
        ids.append(VSTART)
        ids += [IMG] * (t * (h // MERGE) * (w // MERGE))
    ids.append(1)
    return torch.tensor(ids, dtype=torch.long)


def test_vision_segments_single_image():
    grids = [(1, 4, 6)]
    segs = geo.vision_segments(
        _image_seq(grids), torch.tensor(grids), None,
        image_token_id=IMG, video_token_id=VID, spatial_merge_size=MERGE)
    assert len(segs) == 1
    s = segs[0]
    assert s['kind'] == 'image' and s['grid'] == (1, 4, 6)
    assert s['patches'] == 1 * 4 * 6
    a, b = s['span']
    assert b - a == (4 // 2) * (6 // 2)


def test_vision_segments_multi_image_order():
    grids = [(1, 4, 6), (1, 2, 2)]
    segs = geo.vision_segments(
        _image_seq(grids), torch.tensor(grids), None,
        image_token_id=IMG, video_token_id=VID, spatial_merge_size=MERGE)
    assert [s['grid'] for s in segs] == grids
    assert [s['kind_index'] for s in segs] == [0, 1]


def test_vision_segments_video_splits_per_frame():
    """A (t,h,w) video grid is split into t per-frame segments (t->1)."""
    t, h, w = 3, 4, 6
    per_frame = (h // MERGE) * (w // MERGE)
    ids = [1]
    for _ in range(t):                 # one vision run per frame
        ids.append(VSTART)
        ids += [VID] * per_frame
    ids.append(1)
    segs = geo.vision_segments(
        torch.tensor(ids), None, torch.tensor([(t, h, w)]),
        image_token_id=IMG, video_token_id=VID, spatial_merge_size=MERGE)
    assert len(segs) == t
    assert all(s['kind'] == 'video' and s['grid'] == (1, h, w) for s in segs)
    assert [s['kind_index'] for s in segs] == list(range(t))


def test_vision_segments_text_only_raises():
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    with pytest.raises(ValueError) as ei:
        geo.vision_segments(
            ids, None, None, image_token_id=IMG, video_token_id=VID,
            spatial_merge_size=MERGE)
    assert 'at least one image or video' in str(ei.value)


def test_vision_segments_grid_mismatch_raises():
    ids = torch.tensor([VSTART, IMG, IMG, IMG, 1], dtype=torch.long)  # 3 tok
    with pytest.raises(ValueError) as ei:
        geo.vision_segments(
            ids, torch.tensor([(1, 4, 6)]), None,  # grid wants 6 tokens
            image_token_id=IMG, video_token_id=VID, spatial_merge_size=MERGE)
    assert 'does not match its grid' in str(ei.value)


# ── geometry builders: shapes on synthetic inputs ──

def test_mrope_cos_sin_shape():
    grids = [(1, 4, 6)]
    ids = _image_seq(grids)
    pos = geo.mrope_position_ids(
        ids, torch.tensor(grids), None, image_token_id=IMG,
        video_token_id=VID, vision_start_token_id=VSTART,
        spatial_merge_size=MERGE)
    assert pos.shape == (3, ids.numel())
    cos, sin = geo.mrope_cos_sin(
        pos, head_dim=128, rope_theta=5e6, mrope_section=(24, 20, 20),
        device='cpu')
    assert cos.shape == (ids.numel(), 64) == sin.shape


def test_vision_rope_and_pos_embeds_shapes():
    grids = torch.tensor([(1, 4, 6)])
    n_patch = 1 * 4 * 6
    cos, sin = geo.vision_rope_cos_sin(
        grids, head_dim=72, spatial_merge_size=MERGE, device='cpu')
    assert cos.shape == (n_patch, 36) == sin.shape
    table = torch.randn(48 * 48, 8)    # num_position_embeddings 2304 = 48^2
    pe = geo.vision_pos_embeds(
        grids, table, num_grid_per_side=48, spatial_merge_size=MERGE,
        device='cpu')
    assert pe.shape == (n_patch, 8)
