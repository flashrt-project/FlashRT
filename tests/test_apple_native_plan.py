"""Native plan kernels: independent FP32 oracle and direct/ICB dependency checks."""

import numpy as np
from pathlib import Path
import platform
import pytest
import torch
from flash_rt.hardware.apple_native.plan import Plan, Weight, Attention

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or not torch.backends.mps.is_available()
    or not (
        Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib"
    ).is_file(),
    reason="Built native plan library and Apple host GPU required",
)


def compare(plan, out, expected, tolerance=0.01):
    plan.finish()
    plan.set_resident(True)
    plan.run("direct")
    direct = plan.read(out)
    plan.run("icb")
    actual = plan.read(out)
    np.testing.assert_array_equal(actual, direct)
    profile = plan.profile()
    assert len(profile) == plan.dispatch_count + plan.copy_count
    assert all(row["gpu_us"] >= 0 for row in profile)
    np.testing.assert_array_equal(plan.read(out), actual)
    ref = expected.float().numpy()
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - ref) / max(np.linalg.norm(ref), 1e-12) < tolerance


def test_native_norm_and_modulation():
    p = Plan()
    try:
        torch.manual_seed(66)
        x = torch.randn(10, 1024).bfloat16()
        mod = torch.randn(1, 3072).bfloat16() * 0.1
        a = p.allocate(x.shape)
        y = p.allocate(x.shape)
        p.write(a, x)
        p.ops.adarms(y, a, None, None, None, modulation=Weight(mod, "mod"))
        f = x.float()
        scale, shift, _ = mod.float().chunk(3, -1)
        ref = (
            f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * (1 + scale)
            + shift
        ).bfloat16()
        compare(p, y, ref, 0.003)
    finally:
        p.close()


def test_native_rope_rounding_and_kv_suffix():
    p = Plan()
    try:
        torch.manual_seed(9)
        r, h, d, pos = 10, 8, 256, 522
        x = torch.randn(r, (h + 2) * d).bfloat16()
        phase = torch.randn(pos + r, d // 2)
        c = phase.cos().bfloat16()
        s = phase.sin().bfloat16()
        a = p.allocate(x.shape)
        q = p.allocate((r, h, d))
        kv = p.allocate((2, pos + r, 1, d))
        p.write(a, x)
        p.write(kv, torch.zeros(kv.shape, dtype=torch.bfloat16))
        k = kv[0, pos:]
        v = kv[1, pos:]
        p.ops.qkv_rope(q, k, v, a, Weight(c, "cos"), Weight(s, "sin"), pos)
        qr = x[:, : h * d].reshape(r, h, d // 2, 2)
        real, imag = qr.unbind(-1)
        ref = torch.stack(
            (
                real * c[pos:, None] - imag * s[pos:, None],
                imag * c[pos:, None] + real * s[pos:, None],
            ),
            -1,
        ).reshape(r, h, d)
        compare(p, q, ref, 0.000001)
        np.testing.assert_array_equal(
            p.read(v), x[:, -d:].reshape(r, 1, d).float().numpy()
        )
        assert not p.read(kv[0, :pos]).any()
    finally:
        p.close()


@pytest.mark.parametrize(
    "rows,keys,scale",
    [(10, 532, 1), (10, 671, 8), (1, 129, 1), (16, 257, 0), (17, 522, 1)],
)
@pytest.mark.parametrize("output_rank", [2, 3])
def test_native_gqa_online_softmax(rows, keys, scale, output_rank):
    p = Plan()
    try:
        torch.manual_seed(7)
        q = (torch.randn(rows, 8, 256) * scale).bfloat16()
        k = (torch.randn(keys, 1, 256) * scale).bfloat16()
        v = torch.randn_like(k)
        a = p.allocate(q.shape)
        b = p.allocate(k.shape)
        c = p.allocate(v.shape)
        o = p.allocate((rows, 2048) if output_rank == 2 else q.shape)
        for target, value in ((a, q), (b, k), (c, v)):
            p.write(target, value)
        Attention(p).gqa(
            a,
            b,
            c,
            valid_prefix=keys - rows if rows <= 16 else keys,
            prefix_capacity=keys,
            out=o,
        )
        score = (
            torch.einsum("rhd,khd->hrk", q.float(), k.float().expand(-1, 8, -1)) / 16
        )
        ref = (
            torch.einsum("hrk,khd->rhd", score.softmax(-1), v.float().expand(-1, 8, -1))
            .reshape(o.shape)
            .bfloat16()
        )
        compare(p, o, ref, 0.004)
    finally:
        p.close()


def test_foreign_tensor_cannot_alias_a_local_buffer_id():
    owner, other = Plan(), Plan()
    try:
        local = owner.allocate((4,))
        foreign = other.allocate((4,))
        out = owner.allocate((4,))
        assert local.buffer == foreign.buffer
        owner.write(local, torch.ones(4, dtype=torch.bfloat16))
        other.write(foreign, torch.full((4,), 7, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="another plan"):
            out.copy_(foreign)
        assert owner.dispatch_count == 0
        out.copy_(local)
        compare(owner, out, torch.ones(4), 0.000001)
    finally:
        owner.close()
        other.close()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rows,inner,columns", [(10, 1024, 2560), (522, 128, 256)])
def test_native_gemm_tail(rows, inner, columns, bias):
    p = Plan()
    try:
        torch.manual_seed(5)
        x = torch.randn(rows, inner).bfloat16()
        w = torch.randn(inner, columns).bfloat16() / inner**0.5
        a = p.allocate(x.shape)
        o = p.allocate((rows, columns))
        p.write(a, x)
        b = torch.randn(columns).bfloat16() * 0.03 if bias else None
        p.linear(o, a, Weight(w, "w"), Weight(b, "bias") if bias else None)
        ref = (x.float() @ w.float()).bfloat16()
        if bias:
            ref = ref + b
        compare(p, o, ref, 0.004)
    finally:
        p.close()


def test_native_patch_conv_and_position():
    """Compare native image packing/conv/bias/position to an independent conv."""
    p = Plan()
    try:
        torch.manual_seed(28)
        x = torch.randn(2, 224, 224, 3).bfloat16()
        w = (torch.randn(1152, 3, 14, 14) / 588**0.5).bfloat16()
        b = torch.randn(1152).bfloat16() * 0.1
        pos = torch.randn(256, 1152).bfloat16() * 0.1
        a = p.allocate(x.shape)
        out = p.allocate((2, 256, 1152))
        p.write(a, x)
        patch = p.patch(out, a, w, b, pos)
        ref = torch.nn.functional.conv2d(
            x.permute(0, 3, 1, 2).float(), w.float(), b.float(), stride=14
        ).bfloat16()
        expected = ref.flatten(2).transpose(1, 2) + pos
        compare(p, out, expected, 0.003)
        np.testing.assert_allclose(
            p.read(patch),
            ref.flatten(2).transpose(1, 2).float().numpy(),
            atol=0.03125,
            rtol=0.01,
        )
        # New input contents must be consumed on replay, without a plan rebuild.
        p.write(a, torch.zeros_like(x))
        p.run("icb")
        np.testing.assert_array_equal(
            p.read(out), (b[None, None] + pos[None]).expand(2, -1, -1).float().numpy()
        )
        timing = p.run()
        p.set_resident(False)
        p.set_resident(True)
        p.run("icb")
        np.testing.assert_array_equal(
            p.read(out), (b[None, None] + pos[None]).expand(2, -1, -1).float().numpy()
        )
        assert timing["gpu_start_s"] > timing["submit_host_s"]
        assert timing["return_host_s"] >= timing["gpu_end_s"]
        assert (
            abs(
                timing["wall_us"]
                - sum(
                    timing[k]
                    for k in (
                        "encode_us",
                        "commit_to_gpu_us",
                        "gpu_us",
                        "gpu_to_return_us",
                    )
                )
            )
            < 10
        )
    finally:
        p.close()


def test_plan_mapping_rejects_closed_owner():
    p = Plan()
    a = p.allocate((10,))
    p.write(a, np.ones(10, np.float32))
    actual = p.read(a)
    p.close()
    np.testing.assert_array_equal(actual, np.ones(10, np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        p.read(a)


@pytest.mark.parametrize("same_buffer", [False, True])
def test_native_blit_handles_bf16_offsets_and_byte_counts(same_buffer):
    p = Plan()
    try:
        source = p.allocate((8,))
        target = source if same_buffer else p.allocate((8,))
        values = torch.arange(8, dtype=torch.bfloat16)
        p.write(source, values)
        if not same_buffer:
            p.write(target, torch.full((8,), -9, dtype=torch.bfloat16))
        # A two-byte source offset and six-byte payload require no copy shader.
        out = target[4:7]
        out.copy_(source[1:4])
        assert p.dispatch_count == 0 and p.copy_count == 1
        compare(p, out, values[1:4], 0.000001)
        actual = p.read(target)
        np.testing.assert_array_equal(actual[:4], values[:4].float().numpy() if same_buffer else -9)
        assert actual[7] == (7 if same_buffer else -9)
    finally:
        p.close()


@pytest.mark.parametrize("capture_probes", [False, True])
def test_native_replay_and_profiler_restore_mutated_inputs(capture_probes):
    import threading
    from flash_rt.models.pi05.native_plan_mps import NativePi05Plan

    # Small real Metal graph with the model's destructive prompt/noise inputs.
    native = NativePi05Plan.__new__(NativePi05Plan)
    native.plan = p = Plan()
    native._lock = threading.Lock()
    native.probes = {}
    native.last_timing = None
    try:
        native.input_images = p.allocate((2, 32))
        prompt_storage = p.allocate((4, 32))
        p.write(prompt_storage, torch.full((4, 32), 31, dtype=torch.bfloat16))
        native.input_prompt = prompt_storage[1:3]
        native.input_noise = native.output = p.allocate((2, 32))
        p.ops.residual(native.input_prompt, native.input_images, native.input_prompt)
        p.ops.residual(native.output, native.input_prompt, native.input_noise)
        if capture_probes:
            snapshot = p.allocate((2, 32))
            snapshot.copy_(native.output)
            native.probes["snapshot"] = snapshot
        p.finish()
        p.set_resident(True)
        assert p.copy_count == int(capture_probes)
        assert p.dispatch_count == 2
        saved = None
        for seed in (0, 1, 0):
            images = torch.full((2, 32), seed + 1, dtype=torch.bfloat16)
            prompt = torch.full((2, 32), seed + 2, dtype=torch.bfloat16)
            noise = torch.full((2, 32), seed + 3, dtype=torch.bfloat16)
            expected = (images + prompt + noise).float().numpy()
            for execution in ("direct", "icb"):
                actual, probes = native.forward(images, prompt, noise,
                    execution=execution, capture_probes=capture_probes)
                np.testing.assert_array_equal(actual, expected)
                if capture_probes:
                    np.testing.assert_array_equal(probes["snapshot"], expected)
                    if saved is None:
                        saved = probes["snapshot"]
            for _ in range(2):
                actual, timings = native.profile(images, prompt, noise)
                np.testing.assert_array_equal(actual, expected)
                assert len(timings) == p.dispatch_count + p.copy_count
            np.testing.assert_array_equal(p.read(prompt_storage)[[0, 3]], 31)
        if saved is not None:
            np.testing.assert_array_equal(saved, 6)
    finally:
        native.close()
    with pytest.raises(RuntimeError, match="closed"):
        native.profile(images, prompt, noise)


@pytest.mark.parametrize("rows", [10, 67])
def test_native_split_gate_up_strided_output(rows):
    p = Plan()
    try:
        torch.manual_seed(43)
        inner, hidden = 39, 70
        # Binary fractions make the independent FP32 GEMM oracle exact.
        x = (torch.randint(-3, 4, (rows, inner)).float() / 8).bfloat16()
        w = (torch.randint(-3, 4, (inner, 2 * hidden)).float() / 8).bfloat16()
        a = p.allocate(x.shape)
        storage = p.allocate((rows + 2, 2 * hidden))
        out = p.allocate((rows, hidden))
        p.write(a, x)
        p.write(storage, torch.full(storage.shape, 31, dtype=torch.bfloat16))
        both = storage[1:rows + 1]
        gate, up = both[:, :hidden], both[:, hidden:]
        weight = Weight(w, "gate_up")
        p.linear(gate, a, weight[:, :hidden])
        p.linear(up, a, weight[:, hidden:])
        p.ops.gelu_mul(out, gate, up)
        product = (x.float() @ w.float()).bfloat16()
        g, u = product.float().chunk(2, -1)
        expected = (torch.nn.functional.gelu(g, approximate="tanh") * u).bfloat16()
        compare(p, out, expected, 0.0001)
        np.testing.assert_array_equal(p.read(both), product.float().numpy())
        np.testing.assert_array_equal(p.read(storage)[[0, -1]], 31)
        assert p.dispatch_count == 3
    finally:
        p.close()
