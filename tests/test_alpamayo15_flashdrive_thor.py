"""Alpamayo 1.5 FlashDrive backend on Thor: contracts shared with FlashDrive's own rollout."""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import (  # noqa: E402
    FlashDriveThor,
    contiguous_runs,
    euler_cache_schedule,
    rotate_keys_fp32,
)

BF16 = torch.bfloat16


def _fd():
    return pytest.importorskip("flashdrive")


@pytest.mark.parametrize("steps,cache", [(8, [3, 4, 5, 6]), (10, []), (6, [2, 4])])
def test_cache_schedule_matches_euler_with_cache(steps, cache):
    _fd()
    from flashdrive.diffusion import euler_with_cache

    calls = []

    def step_fn(x, t):
        calls.append(round(float(t.flatten()[0]) * steps))
        return torch.ones_like(x)

    diffusion = SimpleNamespace(x_dims=(4, 2))
    euler_with_cache(diffusion, noise=torch.zeros(1, 4, 2), batch_size=1, step_fn=step_fn,
                     cache_steps=cache, device="cpu", inference_step=steps)
    fresh = [i for i, cached in enumerate(euler_cache_schedule(steps, cache)) if not cached]
    assert calls == fresh


def test_cache_schedule_rejects_invalid_steps():
    with pytest.raises(ValueError, match="step 0"):
        euler_cache_schedule(8, [0, 3])
    with pytest.raises(ValueError, match="outside"):
        euler_cache_schedule(8, [8])


def test_expert_key_rotation_matches_flashdrive_formula():
    torch.manual_seed(0)
    S, H, D = 37, 3, 128
    k = torch.randn(S, H, D).to(BF16)
    ang = torch.randn(S, D // 2)
    emb = torch.cat((ang, ang), dim=-1)
    cos, sin = emb.cos(), emb.sin()                                       # FP32 tables, as the expert's rotary
    got = rotate_keys_fp32(k, cos[:, None, :], sin[:, None, :])
    _fd()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    # FlashDrive's own function on its [B, H, S, D] layout, (B, S, D) tables
    ref = apply_rotary_pos_emb_single(k.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)
    assert torch.equal(got.to(BF16), ref.to(BF16))


def _alpamayo15_layout(n_sys=40, n_views=4, n_frames=4, frame_len=6, gap=3, tail=9):
    """Token layout of an Alpamayo 1.5 prompt: system text, per view camera/frame labels between
    [VS..VE] frame blocks, then trajectory/text. Returns (frame_ranges, traj_text_range, P)."""
    pos, frame_ranges = n_sys, []
    for _ in range(n_views):
        view = []
        for _ in range(n_frames):
            pos += gap
            view.append((pos, pos + frame_len))
            pos += frame_len
        frame_ranges.append(view)
    return frame_ranges, (pos, pos + tail), pos + tail


def test_streaming_mask_is_causal_over_cache_positions():
    _fd()
    from flashdrive.streaming import create_streaming_attention_mask_sdpa

    frame_ranges, traj, P = _alpamayo15_layout()
    kv_len = P + 25
    mask = create_streaming_attention_mask_sdpa(frame_ranges=frame_ranges, traj_text_range=traj, kv_length=kv_len,
                                                valid_length=P, device="cpu", dtype=torch.float32,
                                                extended_vision_ranges=True)[0, 0]
    qpos = torch.cat([torch.arange(*v[-1]) for v in frame_ranges] + [torch.arange(*traj)])
    causal = torch.arange(kv_len)[None, :] <= qpos[:, None]
    assert torch.equal(mask == 0, causal)


def test_contiguous_runs_of_streaming_rows():
    assert contiguous_runs([3, 4, 5, 9, 10, 20]) == [(3, 6), (9, 11), (20, 21)]
    with pytest.raises(ValueError, match="increasing"):
        contiguous_runs([3, 5, 4])


def test_backend_requires_flashdrive_model():
    _fd()
    with pytest.raises(TypeError, match="FlashDrive model"):
        FlashDriveThor(torch.nn.Linear(2, 2), expert="flashdrive")
    with pytest.raises(ValueError, match="unknown expert backend"):
        FlashDriveThor(torch.nn.Linear(2, 2), expert="sdpa")


def test_action_rejects_unsupported_requests():
    eng = object.__new__(FlashDriveThor)
    eng.m = SimpleNamespace(diffusion=SimpleNamespace(num_inference_steps=10))
    cp = torch.arange(5, 9)
    with pytest.raises(ValueError, match="euler_with_cache"):
        eng._action(total_samples=1, position_ids=None, attention_mask=None, cache_position=cp,
                    diffusion_kwargs={"int_method": "euler"})
    with pytest.raises(ValueError, match="one trajectory sample"):
        eng._action(total_samples=6, position_ids=None, attention_mask=None, cache_position=cp,
                    diffusion_kwargs={"int_method": "euler_with_cache", "cache_steps": [3]})


def test_resolver_entry_and_load_model_redirect():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("alpamayo15_flashdrive", "torch", "thor")] == (
        "flash_rt.frontends.torch.alpamayo15_flashdrive_thor", "Alpamayo15FlashDriveTorchFrontendThor")
    import flash_rt
    with pytest.raises(NotImplementedError, match="Alpamayo15FlashDriveTorchFrontendThor"):
        flash_rt.load_model("unused", config="alpamayo15_flashdrive")


def test_rollout_defaults_match_flashdrive():
    _fd()
    import inspect

    from flashdrive._base import FlashDriveBaseMixin
    from flash_rt.frontends.torch.alpamayo15_flashdrive_thor import FlashDriveCheckpoints, FlashDriveRollout

    r = FlashDriveRollout()
    sig = inspect.signature(FlashDriveBaseMixin.sample_trajectories_streaming).parameters
    for name in ("max_new_tokens", "temperature", "top_p"):          # FlashDrive's own defaults
        assert getattr(r, name) == sig[name].default, name
    # scripts/infer.py: DIFFUSION_STEPS = 8, CACHE_STEPS = [3, 4, 5, 6], one trajectory sample per window
    assert (r.inference_step, list(r.cache_steps), r.num_traj_samples) == (8, [3, 4, 5, 6], 1)
    assert r.diffusion_kwargs() == {"inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"}
    c = FlashDriveCheckpoints()
    assert (c.paro, c.dflash) == (c.base + "-PARO", c.base + "-DFlash")


def test_reset_clears_flashdrive_stream_state():
    from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import reset_flashdrive_stream

    calls = []
    model = SimpleNamespace(
        _past_key_values=object(), _cached_streaming_attention_mask=torch.zeros(1),
        _compiled_step_registry={"encode": object(), "dflash_traj_forward": object()},
        vlm=SimpleNamespace(model=SimpleNamespace(
            visual=SimpleNamespace(reset_shape_caches=lambda: calls.append("visual")),
            language_model=SimpleNamespace(reset_shape_caches=lambda: calls.append("lm")))))
    reset_flashdrive_stream(model)
    assert model._past_key_values is None and model._cached_streaming_attention_mask is None
    assert model._compiled_step_registry == {} and sorted(calls) == ["lm", "visual"]


# ---------------------------------------------------------------------------
# kernels (GPU)
# ---------------------------------------------------------------------------
from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import FLASHDRIVE_KERNEL_SYMBOLS  # noqa: E402


def _kernels():
    if not torch.cuda.is_available():
        pytest.skip("FlashDrive backend kernels require CUDA")
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels is unavailable: {exc}")
    if not all(hasattr(fvk, n) for n in FLASHDRIVE_KERNEL_SYMBOLS):
        pytest.skip("built without -DFLASHRT_ENABLE_ALPAMAYO15=ON")
    return fvk


def test_flashdrive_symbol_inventory_is_atomic():
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    found = {n for n in dir(fvk) if n.startswith("flashdrive_")}
    if hasattr(fvk, "alpamayo15_rms_bf16"):       # same build gate as the Alpamayo 1.5 kernels
        assert found == set(FLASHDRIVE_KERNEL_SYMBOLS)
    else:
        assert found == set()


def _rot(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _head_norm(x, w, eps):  # Qwen3VLTextRMSNorm on BF16 activations
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)


def _bf16_close(a, e, max_frac=0.01):
    """Equal up to one BF16 ulp on a small fraction of elements (FP32 reduction order of the norm)."""
    a, e = a.float(), e.float()
    ulp = torch.where(e == 0, torch.full_like(e, 2.0 ** -133), e.abs() * 2.0 ** -7)
    d = (a - e).abs()
    assert bool((d <= ulp * 1.01).all()), f"max diff {d.max().item()}"
    assert float((d > 0).float().mean()) <= max_frac


def _fd_cache(L_S, Hk, D):  # FlashDrive StaticLayer layout [1, Hk, S, D]
    return torch.zeros(1, Hk, L_S, D, device="cuda", dtype=BF16)


@pytest.mark.parametrize("mode", ["row_map", "slot"])
def test_qk_prep_writes_unrotated_cache_and_rotated_copy(mode):
    fvk = _kernels()
    torch.manual_seed(11)
    rows, Hq, Hk, D, S, eps = 13, 32, 8, 128, 64, 1e-6
    q = (torch.randn(rows, Hq * D, device="cuda") * 2).to(BF16)
    kv = (torch.randn(rows, 2 * Hk * D + 64, device="cuda") * 2).to(BF16)   # k | v | pad: strided leading dim
    k, v = kv[:, :Hk * D], kv[:, Hk * D:2 * Hk * D]
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    if mode == "row_map":
        rmap = torch.tensor([2, 5, 6, 7, 20, 21, 22, 23, 24, 40, 41, 50, 63], device="cuda", dtype=torch.long)
        slot, cache_rows = None, rmap
    else:
        rmap, slot = None, torch.tensor([31], device="cuda", dtype=torch.long)
        cache_rows = torch.arange(31, 31 + rows, device="cuda")
    k_raw, v_c = _fd_cache(S, Hk, D), _fd_cache(S, Hk, D)
    k_rope = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)                  # attention copy [S, Hk, D]
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    fvk.flashdrive_qk_prep_bf16(q.data_ptr(), k.data_ptr(), v.data_ptr(), Hq * D, kv.shape[1], kv.shape[1],
                                qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 0, q_out.data_ptr(),
                                k_raw.data_ptr(), D, S * D, k_rope.data_ptr(), Hk * D, D, v_c.data_ptr(), D, S * D,
                                rmap.data_ptr() if rmap is not None else 0, slot.data_ptr() if slot is not None else 0,
                                rows, Hq, Hk, eps, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    from flashdrive._backbone import apply_rotary_pos_emb_single  # FlashDrive's own rotation (BF16 tables)
    n_q = _head_norm(q.view(rows, Hq, D), qw, eps)
    n_k = _head_norm(k.view(rows, Hk, D), kw, eps)
    rope = lambda n: apply_rotary_pos_emb_single(n.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1)
    _bf16_close(q_out, rope(n_q))
    got_raw = k_raw[0].transpose(0, 1)[cache_rows]                              # [rows, Hk, D]
    _bf16_close(got_raw, n_k)
    assert torch.equal(k_rope[cache_rows], rope(got_raw))                       # rotation of what was cached
    assert torch.equal(v_c[0].transpose(0, 1)[cache_rows], v.view(rows, Hk, D))
    untouched = torch.ones(S, dtype=torch.bool, device="cuda"); untouched[cache_rows] = False
    assert not k_raw[0].transpose(0, 1)[untouched].any() and not k_rope[untouched].any()


@pytest.mark.parametrize("fp32_rule", [0, 1])
def test_rope_rows_matches_flashdrive_rotation(fp32_rule):
    fvk = _kernels()
    torch.manual_seed(12)
    S, Hk, D = 96, 8, 128
    k_raw = (torch.randn(1, Hk, S, D, device="cuda") * 2).to(BF16)
    ang = torch.randn(S, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    tab = torch.float32 if fp32_rule else BF16
    cos, sin = emb.cos().to(tab).contiguous(), emb.sin().to(tab).contiguous()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    ref = apply_rotary_pos_emb_single(k_raw, cos[None], sin[None])[0].transpose(0, 1).to(BF16)   # [S, Hk, D]
    out = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)
    rows = torch.tensor([0, 7, 8, 33, 95], device="cuda", dtype=torch.long)
    fvk.flashdrive_rope_rows(k_raw.data_ptr(), D, S * D, out.data_ptr(), Hk * D, D, cos.data_ptr(), sin.data_ptr(),
                             fp32_rule, rows.data_ptr(), 0, 0, 0, rows.numel(), Hk, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out[rows], ref[rows])
    mask = torch.ones(S, dtype=torch.bool, device="cuda"); mask[rows] = False
    assert not out[mask].any()
    # contiguous range from a device slot, bounded by a device row count
    out.zero_()
    slot = torch.tensor([10], device="cuda", dtype=torch.long)
    n = torch.tensor([20], device="cuda", dtype=torch.int32)
    fvk.flashdrive_rope_rows(k_raw.data_ptr(), D, S * D, out.data_ptr(), Hk * D, D, cos.data_ptr(), sin.data_ptr(),
                             fp32_rule, 0, 5, slot.data_ptr(), n.data_ptr(), 64, Hk, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out[15:35], ref[15:35])
    assert not out[:15].any() and not out[35:].any()


def test_qk_prep_fp32_rule_matches_expert_autocast_rotation():
    fvk = _kernels()
    torch.manual_seed(13)
    rows, Hq, Hk, D, S, eps = 64, 16, 8, 128, 96, 1e-6
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)   # FlashDrive's fused expert projection
    ld = qkv.shape[1]
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()                   # FP32 expert tables
    slot = torch.tensor([17], device="cuda", dtype=torch.long)
    k_raw, v_c = _fd_cache(S, Hk, D), _fd_cache(S, Hk, D)
    k_rope = torch.zeros(S, Hk, D, device="cuda", dtype=BF16)
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    fvk.flashdrive_qk_prep_bf16(qkv.data_ptr(), qkv[:, Hq * D:].data_ptr(), qkv[:, (Hq + Hk) * D:].data_ptr(), ld, ld, ld,
                                qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 1, q_out.data_ptr(),
                                k_raw.data_ptr(), D, S * D, k_rope.data_ptr(), Hk * D, D, v_c.data_ptr(), D, S * D,
                                0, slot.data_ptr(), rows, Hq, Hk, eps, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    from flashdrive._backbone import apply_rotary_pos_emb_single
    rope = lambda n: apply_rotary_pos_emb_single(n.transpose(0, 1)[None], cos[None], sin[None])[0].transpose(0, 1).to(BF16)
    n_q = _head_norm(qkv[:, :Hq * D].view(rows, Hq, D), qw, eps)
    _bf16_close(q_out, rope(n_q))
    got_raw = k_raw[0].transpose(0, 1)[17:17 + rows]
    _bf16_close(got_raw, _head_norm(qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D), kw, eps))
    assert torch.equal(k_rope[17:17 + rows], rope(got_raw))
    assert torch.equal(v_c[0].transpose(0, 1)[17:17 + rows], qkv[:, (Hq + Hk) * D:].view(rows, Hk, D))


@pytest.mark.parametrize("rows", [1, 8, 802])
def test_fp16_input_consumers_equal_cast_then_bf16_kernels(rows):
    """The fused W4A8 path feeds Marlin's FP16 outputs straight into qk prep, the residual RMSNorm and SiLU-mul;
    each must equal FlashDrive's .to(torch.bfloat16) cast followed by the BF16 kernel, bit for bit."""
    fvk = _kernels()
    torch.manual_seed(14)
    Hq, Hk, D, S, H, F_, eps = 32, 8, 128, 1024, 4096, 12288, 1e-6
    st = torch.cuda.current_stream().cuda_stream
    bits = lambda t: t.view(torch.int16)
    f16 = lambda *shape: (torch.randn(*shape, device="cuda") * 40).to(torch.float16)  # finer than BF16 precision
    q, k, v = f16(rows, Hq * D), f16(rows, Hk * D), f16(rows, Hk * D)
    qw, kw = ((torch.rand(D, device="cuda") + 0.5).to(BF16) for _ in range(2))
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    slot = torch.tensor([3], device="cuda", dtype=torch.long)
    outs = []
    for f16_in, (qi, ki, vi) in ((1, (q, k, v)), (0, (q.to(BF16), k.to(BF16), v.to(BF16)))):
        o = {"q": torch.empty(rows, Hq, D, device="cuda", dtype=BF16), "kr": _fd_cache(S, Hk, D),
             "k": torch.zeros(S, Hk, D, device="cuda", dtype=BF16), "v": _fd_cache(S, Hk, D)}
        fvk.flashdrive_qk_prep_bf16(qi.data_ptr(), ki.data_ptr(), vi.data_ptr(), Hq * D, Hk * D, Hk * D,
                                    qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(), 0, o["q"].data_ptr(),
                                    o["kr"].data_ptr(), D, S * D, o["k"].data_ptr(), Hk * D, D, o["v"].data_ptr(), D,
                                    S * D, 0, slot.data_ptr(), rows, Hq, Hk, eps, st, f16_in=f16_in)
        outs.append(o)
    for name in outs[0]:
        assert torch.equal(bits(outs[0][name]), bits(outs[1][name])), name

    x0 = (torch.randn(rows, H, device="cuda") * 3).to(BF16)
    d16 = f16(rows, H)
    w = (torch.rand(H, device="cuda") + 0.5).to(BF16)
    xa, xb = x0.clone(), x0.clone()
    ha, hb = torch.empty_like(x0), torch.empty_like(x0)
    fvk.flashdrive_rms_f16delta_bf16(xa.data_ptr(), d16.data_ptr(), w.data_ptr(), ha.data_ptr(), rows, H, eps, st)
    d_bf = d16.to(BF16)
    fvk.alpamayo15_rms_bf16(xb.data_ptr(), d_bf.data_ptr(), w.data_ptr(), hb.data_ptr(), rows, H, eps, st)
    torch.cuda.synchronize()
    assert torch.equal(bits(xa), bits(xb)) and torch.equal(bits(ha), bits(hb))

    g, u = f16(rows, F_) / 8, f16(rows, F_) / 8
    g[0, :4] = torch.tensor([0.0, -0.0, 65504.0, -65504.0], dtype=torch.float16)
    aa, ab = (torch.empty(rows, F_, device="cuda", dtype=BF16) for _ in range(2))
    fvk.flashdrive_silu_mul_f16in_bf16(g.data_ptr(), u.data_ptr(), aa.data_ptr(), rows * F_, st)
    gb, ub = g.to(BF16), u.to(BF16)
    fvk.silu_mul_qwen36_bf16(gb.data_ptr(), ub.data_ptr(), ab.data_ptr(), rows * F_, st)
    torch.cuda.synchronize()
    assert torch.equal(bits(aa), bits(ab))
    ref = (torch.nn.functional.silu(gb) * ub)                                  # FlashDrive's act_fn(gate) * up
    assert torch.equal(bits(aa[torch.isfinite(ref)]), bits(ref[torch.isfinite(ref)]))


@pytest.mark.parametrize("rows,K", [(1, 4096), (5, 4096), (802, 4096), (8, 12288)])
def test_w4a8_act_prep_bit_exact_with_paroquant_and_vllm(rows, K):
    fvk = _kernels()
    pytest.importorskip("paroquant")
    pytest.importorskip("vllm")
    import paroquant.kernels.cuda  # noqa: F401  (registers torch.ops.rotation.rotate)
    from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

    torch.manual_seed(14)
    x = (torch.randn(rows, K, device="cuda") * 3).to(BF16)
    x[0, :7] = torch.tensor([0.0, 1e-6, -2e-5, 65000.0, -3.0e4, 1e-8, 2.5], device="cuda").to(BF16)   # edge values
    sets = []
    for _ in range(3):
        theta = (torch.randn(8, K // 2, device="cuda") * 2).to(torch.float16)
        perm = torch.stack([torch.cat([torch.randperm(128) for _ in range(K // 128)]) for _ in range(8)])
        pairs = perm.to(torch.int16).cuda()                    # 64 disjoint in-group pairs per layer and group
        scales = (torch.rand(1, K, device="cuda") + 0.5).to(torch.float16)
        igs = torch.tensor(6.5e-6, device="cuda", dtype=torch.float32)
        sets.append((theta, pairs, scales, igs))
    xq = [torch.empty(rows, K, device="cuda", dtype=torch.int8) for _ in sets]
    a_s = [torch.empty(rows, device="cuda", dtype=torch.float32) for _ in sets]
    xr = [torch.empty(rows, K, device="cuda", dtype=torch.float16) for _ in sets]
    part = [torch.empty(rows, K // 128, device="cuda", dtype=torch.float32) for _ in sets]
    p = lambda ts: [t.data_ptr() for t in ts]
    fvk.flashdrive_w4a8_act_prep(x.data_ptr(), K, rows, K, p(s[0] for s in sets), p(s[1] for s in sets),
                                 p(s[2] for s in sets), p(s[3] for s in sets), p(xr), p(part), p(xq), p(a_s),
                                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    for (theta, pairs, scales, igs), q, a, r in zip(sets, xq, a_s, xr):
        x_rot = torch.ops.rotation.rotate(x.to(torch.float16), pairs, theta, scales, 128)
        q_ref, s_ref = per_token_quant_int8(x_rot)
        assert torch.equal(q, q_ref)
        assert torch.equal(a, (s_ref * igs).view(-1))
        assert torch.equal(r.view(torch.int16), x_rot.view(torch.int16))   # FP16 workspace, bitwise (inf/NaN rows)


def test_fa4_call_patterns_match_flashdrive_masks():
    """The backend's three FA4 calls against SDPA with FlashDrive's masks: streaming rows causal by cache position
    (shared-prefix varlen), verification rows causal over [0, cur + 8), the unmasked trajectory-start row. Each
    error must be far below the change an off-by-one key length produces."""
    _kernels()
    from flash_rt.hardware.thor import fa4_backend
    fa4 = fa4_backend.fa4_fwd()
    if fa4 is None:
        pytest.skip(f"FA4 unavailable: {fa4_backend.status()}")
    torch.manual_seed(17)
    S, Hq, Hk, D = 96, 32, 8, 128
    eng = SimpleNamespace(_fa4=fa4, n_heads=Hq, head_dim=D, scale=D ** -0.5,
                          _v_view=lambda st, li: st["v"][li].permute(0, 2, 1, 3))
    kvlm = torch.randn(1, S, Hk, D, device="cuda").to(BF16)                      # rotated keys [S, Hk, D]
    v = torch.randn(1, Hk, S, D, device="cuda").to(BF16)                         # FlashDrive's value cache
    st = {"kvlm": kvlm, "v": [v], "S": S, "keep": {}}

    def sdpa(q, pos, shift=0):          # q [n, Hq, D]; row i attends keys [0, pos[i] + shift]
        k_ = kvlm[0].float().repeat_interleave(Hq // Hk, 1).transpose(0, 1)     # [Hq, S, D]
        v_ = v[0].float().repeat_interleave(Hq // Hk, 0)                       # [Hq, S, D]
        sc = torch.einsum("nhd,hsd->hns", q.float(), k_) * D ** -0.5
        keep = torch.arange(S, device="cuda")[None, :] <= (torch.tensor(pos, device="cuda") + shift)[:, None]
        sc = sc.masked_fill(~keep[None], float("-inf"))
        return torch.einsum("hns,hsd->nhd", sc.softmax(-1), v_).reshape(len(pos), Hq * D)

    def check(out, q, pos):
        ref = sdpa(q, pos)
        err = (out.float() - ref).abs().mean()
        off = min((sdpa(q, pos, -1) - ref).abs().mean(), (sdpa(q, pos, 1) - ref).abs().mean())
        assert err < 0.1 * off, (float(err), float(off))

    runs = [(10, 20), (40, 52), (70, 86)]                                        # a streaming window's rows
    pos = [r for a, b in runs for r in range(a, b)]
    lens = [b - a for a, b in runs]
    st["keep"]["prefill_meta"] = {
        "cu_q": torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device="cuda"),
        "cu_k": torch.zeros(len(runs) + 1, dtype=torch.int32, device="cuda"),
        "used": torch.tensor([b for _, b in runs], dtype=torch.int32, device="cuda"), "max_q": max(lens)}
    st["kvlm"] = kvlm[0][None]                                                   # [layers, S, Hk, D]
    q = torch.randn(len(pos), Hq, D, device="cuda").to(BF16)
    out = torch.empty(len(pos), Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_prefill(eng, st, 0, q.view(len(pos), -1), out)
    check(out, q, pos)

    cur = 50                                                                     # verification block
    st["keep"]["dec"] = {"used": torch.tensor([cur + 8], dtype=torch.int32, device="cuda")}
    q = torch.randn(8, Hq, D, device="cuda").to(BF16)
    out = torch.empty(8, Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_verify(eng, st, 0, q.view(8, -1), out)
    check(out, q, list(range(cur, cur + 8)))

    q = torch.randn(1, Hq, D, device="cuda").to(BF16)                           # trajectory-start row: no mask
    out = torch.empty(1, Hq * D, device="cuda", dtype=BF16)
    FlashDriveThor._attend_all_rows(eng, st, 0, q.view(1, -1), out)
    ref = sdpa(q, [S - 1])
    assert float((out.float() - ref).abs().mean()) < 0.1 * float((sdpa(q, [S - 1], -1) - ref).abs().mean())


def _marlin_w4a8_buffers(K, N, seed):
    """Random ParoQuant-style Marlin W4A8 buffers packed by vLLM's own layout functions."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_zero_points
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    g = torch.Generator().manual_seed(seed)
    G = K // 128
    w = torch.randint(0, 16, (K, N), generator=g)
    z = torch.randint(0, 16, (G, N), generator=g)
    s = torch.randint(1, 4097, (G, N), generator=g).to(torch.int16)
    s[0, 0] = 4096
    qweight = marlin_weights(w, K, N, 4, get_weight_perm(4, is_a_8bit=True), is_a_8bit=True).cuda()
    scales = marlin_permute_scales(s.view(torch.float16), K, N, 128, is_a_8bit=True).cuda()
    qzeros = marlin_zero_points(z, G, N, 4, is_a_8bit=True).cuda()
    return w, z, s, qweight, scales, qzeros


@pytest.mark.parametrize("M,N,K", [(300, 384, 512), (802, 1024, 1024)])
def test_w4a8_int8_gemm_is_exact_and_matches_marlin(M, N, K):
    fvk = _kernels()
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx, marlin_make_workspace_new)
    from vllm.scalar_type import scalar_types
    from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import marlin_w4a8_to_int8
    w, z, s, qweight, scales, qzeros = _marlin_w4a8_buffers(K, N, seed=15)
    w8, sg = marlin_w4a8_to_int8(qweight, scales, qzeros, K, N)
    wd = w - z.repeat_interleave(128, 0)                                       # [K, N]
    assert torch.equal(w8.cpu(), wd.t().to(torch.int8)) and torch.equal(sg.cpu(), s.to(torch.int32))

    torch.manual_seed(16)
    xq = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    # exact integer total (float64 holds it exactly); per-token scales put each row's outputs up to ~1e4 (FP16 range)
    total = xq.double() @ (wd.double() * s.double().repeat_interleave(128, 0)).cuda()
    a_s = (1e4 * (torch.rand(M, 1, device="cuda", dtype=torch.float64) * 0.9 + 0.1)
           / total.abs().amax(dim=1, keepdim=True)).float().contiguous()
    nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, K)
    ws = torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device="cuda")
    ones = torch.ones(int(n_ones), dtype=torch.int32, device="cuda")
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ones.data_ptr(), a_s.data_ptr(),
                                   out.data_ptr(), M, N, K, ws.data_ptr(), ws.numel(),
                                   torch.cuda.current_stream().cuda_stream)
    ref = (total.float() * a_s).half()          # one conversion of the total to float, the per-token scale, FP16
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))

    dev = torch.device("cuda")
    marlin = ops.marlin_gemm(xq, None, qweight, None, scales, a_s, None, qzeros, marlin_make_empty_g_idx(dev),
                             marlin_make_empty_g_idx(dev), marlin_make_workspace_new(dev), scalar_types.uint4,
                             size_m=M, size_n=N, size_k=K, use_atomic_add=False, use_fp32_reduce=True,
                             is_zp_float=False)
    # Marlin sums FP32 partials of K slices: one FP16 ulp plus that sum's rounding (cancellation) apart at most
    r = marlin.float()
    tol = (r.abs() * 2.0 ** -10).clamp_min(2.0 ** -24) + r.abs().amax(dim=1, keepdim=True) * 2.0 ** -20
    assert bool(((out.float() - r).abs() <= tol).all())
    assert float((out == marlin).float().mean()) > 0.999


# ---------------------------------------------------------------------------
# backend vs FlashDrive on a short synthetic stream (opt-in: real checkpoints)
# ---------------------------------------------------------------------------
_CKPT = os.environ.get("FLASHRT_FLASHDRIVE_CKPT")   # dir holding Alpamayo-1.5-10B{,-PARO,-DFlash}


@pytest.fixture
def _release_model():
    """Each case loads the full checkpoint; the backend and the model reference each other, so free them before
    the next case loads another copy."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not _CKPT, reason="set FLASHRT_FLASHDRIVE_CKPT to the z-lab checkpoint directory")
@pytest.mark.usefixtures("_release_model")
@pytest.mark.parametrize("backends", [
    dict(expert="flashrt", vision="flashrt", prefill="flashrt", decode="flashrt", w4a8="fused"),
    dict(expert="flashrt", vision="flashrt", prefill="flashrt", decode="flashrt", w4a8="fused", w4a8_gemm="marlin"),
    dict(expert="flashdrive", vision="flashdrive", prefill="flashrt", decode="flashrt", w4a8="flashdrive"),
], ids=["all_flashrt", "all_flashrt_marlin", "mixed"])
def test_backend_primitives_match_flashdrive_on_identical_state(backends):
    _kernels()
    from alpamayo1_5 import helper
    from flashdrive.streaming import convert_to_streaming_window
    from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import load_flashdrive

    torch.set_float32_matmul_precision("high")
    root = _CKPT
    model = load_flashdrive(f"{root}/Alpamayo-1.5-10B", f"{root}/Alpamayo-1.5-10B-PARO", f"{root}/Alpamayo-1.5-10B-DFlash")
    names = ["_action", "_encode", "_dflash_prefill", "_shift_streaming_kv_cache", "_dflash_block_step",
             "_dflash_traj_forward"]
    ref = {n: getattr(model, n) for n in names}
    eng = FlashDriveThor(model, **backends)
    frt = {n: getattr(model, n) for n in names if getattr(model, n) is not ref[n]}   # primitives the backend replaced
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    seen = {n: [] for n in frt}

    def snap():
        pkv = model._past_key_values
        return [(l.keys.clone(), l.values.clone()) for l in pkv.layers] if pkv is not None and pkv.layers[0].is_initialized else None

    def restore(s_):
        for l, (k, v) in zip(model._past_key_values.layers, s_):
            l.keys.copy_(k); l.values.copy_(v)

    def kvlm_is_rotated_cache():
        st = eng._stream()
        fresh = torch.empty_like(st["kvlm"][0])
        for li in range(eng.n_layers):
            eng._rope_rows(st, li, fresh, st["cos"], st["sin"], 0, st["S"])
            if not torch.equal(fresh, st["kvlm"][li]):
                return False
        return True

    def tensors(o):
        if torch.is_tensor(o):
            return [o.detach().clone()]
        return [t for x in (o or []) for t in tensors(x)] if isinstance(o, (list, tuple)) else []

    def wrap(n):
        def f(*a, **k):
            pre, rng = snap(), gen.get_state()
            out = ref[n](*a, **k)                       # FlashDrive (kvlm is not touched: still the rotation of pre)
            r_out, post, rng_ref = tensors(out), snap(), gen.get_state()
            if pre is not None:
                restore(pre)
            gen.set_state(rng)
            f_out = tensors(frt[n](*a, **k))
            invariant = kvlm_is_rotated_cache()          # every FlashRT write keeps the rotated copy exact
            seen[n].append((r_out, f_out, pre, post, snap(), torch.equal(gen.get_state(), rng_ref), invariant))
            restore(post); gen.set_state(rng_ref)
            eng._sync_kvlm_all(eng._stream())            # continue from FlashDrive's state
            return out
        return f
    for n in frt:
        setattr(model, n, wrap(n))
    proc = helper.get_processor(model.tokenizer)
    vs, ve = (proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>"))
    g = torch.Generator().manual_seed(0)
    hist = (torch.zeros(1, 1, 16, 3), torch.eye(3).expand(1, 1, 16, 3, 3).contiguous())

    def stream(n_windows):
        frames = torch.randint(0, 256, (4, n_windows + 3, 3, 1080, 1920), dtype=torch.uint8, generator=g)
        for k in range(n_windows):
            msgs = helper.create_message(frames[:, k:k + 4].flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]))
            t = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                         return_dict=True, return_tensors="pt")
            w = {"tokenized_data": t, "ego_history_xyz": hist[0], "ego_history_rot": hist[1]}
            if k:
                w = convert_to_streaming_window(w, vs, ve)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.sample_trajectories_streaming(helper.to_device(w, "cuda"), num_traj_samples=1, diffusion_kwargs={
                    "inference_step": 8, "cache_steps": [3, 4, 5, 6], "int_method": "euler_with_cache"})
    torch.manual_seed(0)
    stream(4)
    eng.reset()                          # an unrelated second stream: FlashDrive's state and the stream graphs rebuilt
    stream(2)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
    for n in frt:
        assert seen[n], f"{n} was not exercised"
        for r_out, f_out, pre, post_r, post_f, rng_eq, invariant in seen[n]:
            assert rng_eq, f"{n}: random-number consumption differs from FlashDrive"
            assert invariant, f"{n}: rotated cache copy is stale"
            for a, b in zip(r_out, f_out):
                if a.dtype in (torch.long, torch.int32):
                    continue
                assert cos(a, b) > 0.995, n
            for li, ((kr, vr), (kf, vf)) in enumerate(zip(post_r, post_f)):
                if n == "_shift_streaming_kv_cache":
                    assert torch.equal(kr, kf) and torch.equal(vr, vf)
                    continue
                if pre is None:
                    continue
                rows = ((kr != pre[li][0]) | (kf != pre[li][0])).any(dim=(0, 1, 3))    # rows either side wrote
                if rows.any():
                    assert cos(kr[:, :, rows], kf[:, :, rows]) > 0.99 and cos(vr[:, :, rows], vf[:, :, rows]) > 0.99, n
    if "_dflash_block_step" in seen:
        proposals = [torch.equal(r[0], f[0]) for r, f, *_ in seen["_dflash_block_step"]]
        assert sum(proposals) >= 0.8 * len(proposals)      # greedy draft proposals from the same context
