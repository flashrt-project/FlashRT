"""The shared qwen3_5_moe path must not change what it does on import.

Nex-N2 and Qwen3.6 share ``_nexn2_rtx_forward``. Two ways that sharing can go
wrong are covered here: importing the module changing process-global state that
another model then inherits, and a kernel appearing in a build silently
changing what an already-validated path calls. Both were real -- the first was
an ``os.environ.setdefault`` at module scope, the second a bare ``getattr``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _policy_module():
    return pytest.importorskip(
        "flash_rt.frontends.torch._nexn2_rtx_forward")


def test_importing_the_forward_does_not_touch_the_environment():
    """Run in a subprocess: this process may already have imported it.

    The variable below is read by a kernel shared with every other frontend,
    so setting it here would decide the autotune behaviour for a model loaded
    later in the same process that never asked for it.
    """
    script = (
        # The package root is imported first and its environment taken as the
        # baseline: this test is about what importing *this module* adds.
        "import os, sys, flash_rt;"
        "before = dict(os.environ);"
        "import flash_rt.frontends.torch._nexn2_rtx_forward;"
        "after = dict(os.environ);"
        "added = {k: after[k] for k in after if k not in before};"
        "changed = {k: (before[k], after[k]) for k in before"
        " if after.get(k) != before[k]};"
        "sys.exit(0 if not added and not changed"
        " else 'import touched ' + repr(added) + repr(changed))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + result.stderr)


def test_policy_defaults_are_the_shipped_ones():
    """Nex-N2 was validated with these; a new field must not move one."""
    policy = _policy_module().kernel_policy()
    assert policy.dense_cublaslt is True
    assert policy.cublaslt_max_algos == 1
    assert policy.wy_gdn is True
    assert policy.edge_w4a16 is True
    assert policy.route_kernel is True
    assert policy.fused_shared_combine is True
    assert policy.warp_router_topk is True
    assert policy.gdn_recurrent_edge is True
    assert policy.verify_k_rows is True


@pytest.mark.parametrize("env,field", [
    ("NEXN2_DENSE_CUBLASLT", "dense_cublaslt"),
    ("NEXN2_WY_GDN", "wy_gdn"),
    ("NEXN2_ROUTE_KERNEL", "route_kernel"),
    ("FLASHRT_QWEN35MOE_W4A16_EDGE", "edge_w4a16"),
    ("FLASHRT_QWEN35MOE_VERIFY_K_ROWS", "verify_k_rows"),
])
def test_the_existing_environment_variables_are_the_defaults(
        monkeypatch, env, field):
    """They predate the policy object, so they must still work unchanged."""
    module = _policy_module()
    monkeypatch.setenv(env, "0")
    assert getattr(module.KernelPolicy(), field) is False
    monkeypatch.setenv(env, "1")
    assert getattr(module.KernelPolicy(), field) is True


def test_an_argument_beats_the_environment(monkeypatch):
    module = _policy_module()
    monkeypatch.setenv("NEXN2_WY_GDN", "0")
    assert module.KernelPolicy(wy_gdn=True).wy_gdn is True


def test_installing_a_policy_returns_the_previous_one():
    module = _policy_module()
    original = module.kernel_policy()
    replacement = module.KernelPolicy(edge_w4a16=False)
    try:
        assert module.set_kernel_policy(replacement) is original
        assert module.kernel_policy() is replacement
        assert module.kernel_policy().edge_w4a16 is False
    finally:
        module.set_kernel_policy(original)
    assert module.kernel_policy() is original


def test_installing_something_that_is_not_a_policy_is_refused():
    module = _policy_module()
    with pytest.raises(TypeError):
        module.set_kernel_policy({"edge_w4a16": False})


def test_the_policy_decides_the_dispatch_not_the_symbol_table():
    """A kernel being present in the build is not a reason to call it.

    Both entries below fall back to the kernel they replace, so turning the
    field off must select the fallback even though the symbol is right there.
    """
    module = _policy_module()

    class FakeModule:
        w4a16_matvec_sm120_bf16 = "original"
        w4a16_matvec_edge_sm120_bf16 = "edge"
        moe_grouped_w4a16_sm120_bf16 = "original"
        moe_grouped_w4a16_edge_sm120_bf16 = "edge"

    fvk = FakeModule()
    original = module.kernel_policy()
    try:
        module.set_kernel_policy(module.KernelPolicy(edge_w4a16=True))
        assert module.w4a16_matvec(fvk) == "edge"
        assert module.moe_grouped_w4a16(fvk) == "edge"
        module.set_kernel_policy(module.KernelPolicy(edge_w4a16=False))
        assert module.w4a16_matvec(fvk) == "original"
        assert module.moe_grouped_w4a16(fvk) == "original"
    finally:
        module.set_kernel_policy(original)


def test_the_decode_dispatch_helpers_follow_the_policy_too():
    decode = pytest.importorskip(
        "flash_rt.frontends.torch._nexn2_rtx_decode")
    module = _policy_module()

    class FakeModule:
        moe_router_topk_sm120_bf16 = "block"
        moe_router_topk_warp_sm120_bf16 = "warp"
        gated_deltanet_recurrent_qwen36_bf16 = "shipped"
        gated_deltanet_recurrent_edge_qwen36_bf16 = "edge"

    fvk = FakeModule()
    original = module.kernel_policy()
    try:
        module.set_kernel_policy(module.KernelPolicy())
        assert decode.router_topk(fvk) == "warp"
        assert decode.gdn_recurrent(fvk) == "edge"
        module.set_kernel_policy(module.KernelPolicy(
            warp_router_topk=False, gdn_recurrent_edge=False))
        assert decode.router_topk(fvk) == "block"
        assert decode.gdn_recurrent(fvk) == "shipped"
    finally:
        module.set_kernel_policy(original)


def test_a_build_without_the_optional_kernels_still_dispatches():
    """The fallback is what makes these kernels optional rather than required."""
    decode = pytest.importorskip(
        "flash_rt.frontends.torch._nexn2_rtx_decode")
    module = _policy_module()

    class FakeModule:
        moe_router_topk_sm120_bf16 = "block"
        gated_deltanet_recurrent_qwen36_bf16 = "shipped"
        w4a16_matvec_sm120_bf16 = "original"
        moe_grouped_w4a16_sm120_bf16 = "original"

    fvk = FakeModule()
    assert decode.router_topk(fvk) == "block"
    assert decode.gdn_recurrent(fvk) == "shipped"
    assert module.w4a16_matvec(fvk) == "original"
    assert module.moe_grouped_w4a16(fvk) == "original"
