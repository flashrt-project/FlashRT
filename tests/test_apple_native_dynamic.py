"""Mutable uniforms, padded attention and RoPE across bounded replay lengths."""

import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.plan import Attention, Plan, UIntParameter, Weight

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or not torch.backends.mps.is_available()
    or not (Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib").is_file(),
    reason="Built native plan library and Apple host GPU required",
)


@pytest.mark.parametrize("rows,split,suffix", [(400, False, False), (10, False, True), (10, True, True), (1, True, True)])
def test_dynamic_gqa_masks_stale_padding(rows, split, suffix, monkeypatch):
    torch.manual_seed(17)
    capacity, heads = 400, 2
    keys = capacity + (rows if suffix else 0)
    p = Plan()
    try:
        valid = UIntParameter(p, capacity, minimum=1, maximum=capacity)
        q = torch.randn(rows, heads, 256).bfloat16()
        k = torch.randn(keys, 1, 256).bfloat16()
        v = torch.randn_like(k)
        a, b, c, out = (p.allocate(shape) for shape in (q.shape, k.shape, v.shape, q.shape))
        p.write(a, q)
        Attention(p, split_decode=split).gqa(
            a, b, c, valid_prefix=valid, prefix_capacity=capacity, out=out,
        )
        p.finish()
        p.set_resident(True)
        bindings = tuple(valid._bindings)
        counts = p.dispatch_count, len(p.uniforms), p.copy_count
        def forbidden(*args, **kwargs):
            raise AssertionError("Replay attempted setup")
        monkeypatch.setattr(p, "allocate", forbidden)
        monkeypatch.setattr(p, "dispatch", forbidden)
        monkeypatch.setattr(p, "finish", forbidden)
        for length in (400, 1, 127, 128, 129, 257, 399, 1, 400):
            # Deliberately stale, large padding must never influence attention.
            actual_k, actual_v = k.clone(), v.clone()
            actual_k[length:capacity] = 64
            actual_v[length:capacity] = -64
            p.write(b, actual_k)
            p.write(c, actual_v)
            valid.set(length)
            selected_k = torch.cat((k[:length], k[capacity:])).float()
            selected_v = torch.cat((v[:length], v[capacity:])).float()
            scores = torch.einsum("rhd,khd->hrk", q.float(), selected_k.expand(-1, heads, -1)) / 16
            ref = torch.einsum("hrk,khd->rhd", scores.softmax(-1), selected_v.expand(-1, heads, -1)).numpy()
            results = []
            for execution in ("direct", "icb"):
                p.run(execution)
                actual = p.read(out)
                assert np.isfinite(actual).all()
                assert np.linalg.norm(actual - ref) / np.linalg.norm(ref) < .004
                results.append(actual)
            np.testing.assert_array_equal(*results)
        assert tuple(valid._bindings) == bindings
        assert (p.dispatch_count, len(p.uniforms), p.copy_count) == counts
        for invalid in (0, capacity + 1, -1, 1.5):
            with pytest.raises(ValueError, match="bounds"):
                valid.set(invalid)
    finally:
        p.close()
    with pytest.raises(RuntimeError, match="closed"):
        valid.set(1)


def test_dynamic_rope_does_not_alias_constant_uniforms():
    p = Plan()
    try:
        torch.manual_seed(4)
        rows, heads, dim = 3, 2, 256
        x = torch.randn(rows, (heads + 2) * dim).bfloat16()
        cos, sin = torch.randn(16, dim // 2).bfloat16(), torch.randn(16, dim // 2).bfloat16()
        a = p.allocate(x.shape)
        p.write(a, x)
        position = UIntParameter(p, 0, minimum=0, maximum=13)
        independent = UIntParameter(p, 0, minimum=0, maximum=13)
        outputs = []
        for offset in (0, position, position, independent):
            q = p.allocate((rows, heads, dim))
            k = p.allocate((rows, 1, dim))
            v = p.allocate(k.shape)
            p.ops.qkv_rope(q, k, v, a, Weight(cos, "cos"), Weight(sin, "sin"), offset)
            outputs.append((q, k, v))
        assert len(position._bindings) == len(independent._bindings) == 1
        with pytest.raises(ValueError, match="table capacity"):
            p.ops.qkv_rope(q, k, v, a, Weight(cos, "cos"), Weight(sin, "sin"),
                          UIntParameter(p, 0, minimum=0, maximum=14))
        p.finish()
        for offset in (13, 1, 0, 7):
            position.set(offset)
            p.run()
            for i, expected_offset in enumerate((0, offset, offset, 0)):
                q, k, v = [p.read(t) for t in outputs[i]]
                inp = x.reshape(rows, heads + 2, dim)[:, :heads + 1].reshape(rows, heads + 1, dim // 2, 2)
                c, s = cos[expected_offset:expected_offset + rows, None], sin[expected_offset:expected_offset + rows, None]
                real = inp[..., 0] * c - inp[..., 1] * s
                imag = inp[..., 1] * c + inp[..., 0] * s
                expected = torch.stack((real, imag), -1).reshape(rows, heads + 1, dim).float().numpy()
                np.testing.assert_array_equal(q, expected[:, :heads])
                np.testing.assert_array_equal(k, expected[:, heads:])
                np.testing.assert_array_equal(v, x.reshape(rows, heads + 2, dim)[:, -1:].float().numpy())
    finally:
        p.close()


def test_native_dynamic_upload_restores_padding_and_rejects_overflow():
    import threading

    from flash_rt.models.pi05.native_plan_mps import NativePi05Plan

    native = NativePi05Plan.__new__(NativePi05Plan)
    native.plan = p = Plan()
    native._lock = threading.Lock()
    native.probes = {}
    native.prompt_len = native.prompt_capacity = 4
    native.pad_prompt = True
    native._valid_prefix = UIntParameter(p, 6, minimum=3, maximum=6)
    try:
        native.input_images = p.allocate((4, 32))
        native.input_prompt = p.allocate((4, 32))
        native.input_noise = native.output = p.allocate((4, 32))
        # Destructive replay exposes stale padded uploads on the next call.
        p.ops.residual(native.input_prompt, native.input_images, native.input_prompt)
        p.ops.residual(native.output, native.input_prompt, native.input_noise)
        p.finish()
        images = np.ones((4, 32), np.float32)
        noise = np.full_like(images, 2)
        for length in (4, 1, 2, 4, 1):
            prompt = np.full((length, 32), length + 1, np.float32)
            output, _ = native.forward(images, prompt, noise)
            expected = images + noise
            expected[:length] += prompt
            np.testing.assert_array_equal(output, expected)
            assert native.prompt_len == length
            assert native._valid_prefix.value == 2 + length
        for shape in ((0, 32), (5, 32), (2, 31), (32,)):
            with pytest.raises(ValueError, match="Native prompt"):
                native.forward(images, np.zeros(shape, np.float32), noise)
    finally:
        native.close()


@pytest.mark.parametrize("int8", [False, True])
def test_compact_dynamic_projection_never_computes_padding(int8):
    import os

    from flash_rt.hardware.apple_native.int8 import INT8Plan
    from flash_rt.hardware.apple_native.plan import Tensor

    library = os.environ.get("FLASHRT_APPLE_INT8_LIBRARY")
    if int8 and not library:
        pytest.skip("INT8 library required")
    p = INT8Plan(library, stages=("prefix",)) if int8 else Plan()
    try:
        torch.manual_seed(31)
        capacity, width = 129, 256
        length = UIntParameter(p, capacity, minimum=1, maximum=capacity)
        rows = length.dimension()
        a, out, norm = (p.allocate((capacity, width)) for _ in range(3))
        x = torch.randn(capacity, width).bfloat16()
        w = (torch.randn(width, width) / width**.5).bfloat16()
        p.write(a, x)
        p.stage = "prefix"
        p.linear(out[:rows], a[:rows], Weight(w, "weight"))
        p.ops.rms_norm(norm[:rows], out[:rows])
        snapshot = p.allocate((rows, width))
        snapshot.copy_(out[:rows])
        p.finish()
        for n in (129, 1, 32, 33, 127, 128, 1, 129):
            p.write(out, np.full(out.shape, 31, np.float32))
            p.write(norm, np.full(norm.shape, 31, np.float32))
            length.set(n)
            p.run()
            actual = p.read(out)
            np.testing.assert_array_equal(actual[n:], 31)
            np.testing.assert_array_equal(p.read(norm)[n:], 31)
            np.testing.assert_array_equal(p.read(snapshot), actual[:n])
            ref = (x[:n].float() @ w.float()).numpy()
            assert np.linalg.norm(actual[:n] - ref) / np.linalg.norm(ref) < (.02 if int8 else .004)
            for meta in p.resolved_dispatches():
                assert meta["params"][0] == n
            view = Tensor(p, (rows, width), out.buffer)
            assert p.read(view).shape == (n, width)
        p.profile()
        np.testing.assert_array_equal(p.read(snapshot), actual)
    finally:
        p.close()


def test_compact_dynamic_kv_suffix_and_attention():
    p = Plan()
    try:
        torch.manual_seed(32)
        capacity, rows, heads, dim = 270, 10, 2, 256
        length = UIntParameter(p, capacity, minimum=1, maximum=capacity)
        prefix = length.dimension()
        qkv = torch.randn(rows, (heads + 2) * dim).bfloat16()
        cs = torch.randn(capacity + rows, dim // 2).bfloat16()
        sn = torch.randn_like(cs)
        a = p.allocate(qkv.shape)
        p.write(a, qkv)
        q = p.allocate((rows, heads, dim))
        k, v = (p.allocate((capacity + rows, 1, dim)) for _ in range(2))
        out = p.allocate(q.shape)
        p.ops.qkv_rope(q, k[prefix:prefix + rows], v[prefix:prefix + rows], a,
                      Weight(cs, "cos"), Weight(sn, "sin"), prefix)
        Attention(p).gqa(q, k[:prefix + rows], v[:prefix + rows], valid_prefix=prefix,
                         prefix_capacity=capacity, out=out)
        p.finish()
        for n in (270, 1, 118, 119, 128, 129, 270):
            p.write(k, np.full(k.shape, 31, np.float32))
            p.write(v, np.full(v.shape, 31, np.float32))
            length.set(n)
            p.run()
            keys, values, queries = p.read(k), p.read(v), p.read(q)
            np.testing.assert_array_equal(keys[:n], 31)
            np.testing.assert_array_equal(keys[n + rows:], 31)
            np.testing.assert_array_equal(values[:n], 31)
            np.testing.assert_array_equal(values[n + rows:], 31)
            np.testing.assert_array_equal(values[n:n + rows], qkv.reshape(rows, heads + 2, dim)[:, -1:].float().numpy())
            score = torch.einsum("rhd,khd->hrk", torch.from_numpy(queries),
                                 torch.from_numpy(keys[:n + rows]).expand(-1, heads, -1)) / 16
            ref = torch.einsum("hrk,khd->rhd", score.softmax(-1),
                               torch.from_numpy(values[:n + rows]).expand(-1, heads, -1)).numpy()
            actual = p.read(out)
            assert np.isfinite(actual).all()
            assert np.linalg.norm(actual - ref) / np.linalg.norm(ref) < .004
    finally:
        p.close()
