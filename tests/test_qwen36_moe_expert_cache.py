"""Sizing and admission tests for the streaming expert cache.

These exercise the parts that need no device and no bundle: the footprint
arithmetic, the budget refusal, and the quota rule that makes one token's
experts unable to evict each other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen36_moe_edge.expert_cache import (
    CacheBudgetError,
    CacheConfig,
    ExpertCache,
)


GIB = 2 ** 30
# The shipped INT4 group-16 bundle.
MANIFEST = {
    "block_bytes": 1769472,
    "block_alignment": 4096,
    "num_layers": 40,
    "num_experts": 256,
    "block_sizes": {},
}


def _config(**overrides) -> CacheConfig:
    values = {
        "bundle": Path("/nonexistent"),
        "slots_per_layer": 57,
        "staging_buffers": 4,
    }
    values.update(overrides)
    return CacheConfig(**values)


def test_plan_accounts_for_slots_staging_resident_and_reserve():
    config = _config(
        slots_per_layer=57,
        staging_buffers=4,
        resident_bytes=int(1.696 * GIB),
        reserve_bytes=int(1.5 * GIB),
    )

    plan = ExpertCache.plan(config, MANIFEST)

    assert plan["slot_bytes"] == 57 * 40 * 1769472
    assert plan["staging_bytes"] == 4 * 1769472
    assert plan["projected_bytes"] == (
        plan["slot_bytes"] + plan["staging_bytes"]
        + plan["resident_bytes"] + plan["reserve_bytes"])
    # 57 slots is what quantizing the GDN weights to INT8 pays for, and it has
    # to fit the 7 GiB target.
    assert plan["projected_bytes"] < 7.0 * GIB


def test_max_slots_shrinks_when_resident_weights_grow():
    # Leaving the GDN weights at BF16 costs 0.94 GiB of resident, which is
    # what the extra slots were bought with.
    lean = _config(
        budget_bytes=int(7.0 * GIB), reserve_bytes=int(1.5 * GIB),
        resident_bytes=int(1.696 * GIB))
    heavy = _config(
        budget_bytes=int(7.0 * GIB), reserve_bytes=int(1.5 * GIB),
        resident_bytes=int(2.637 * GIB))

    generous = ExpertCache.max_slots_per_layer(lean, MANIFEST)
    tight = ExpertCache.max_slots_per_layer(heavy, MANIFEST)

    assert generous >= 57
    assert tight < generous
    assert generous - tight >= 12


def test_max_slots_is_zero_when_the_budget_is_already_spent():
    config = _config(
        budget_bytes=int(2.0 * GIB), reserve_bytes=int(1.5 * GIB),
        resident_bytes=int(1.0 * GIB))

    assert ExpertCache.max_slots_per_layer(config, MANIFEST) == 0


def test_quota_below_experts_per_token_is_rejected(tmp_path):
    # Below the top-k a single token's experts would evict one another, so a
    # caller could not hold their pointers at the same time.
    (tmp_path / "manifest.json").write_text(_manifest_json())
    config = _config(
        bundle=tmp_path, slots_per_layer=4, experts_per_token=8)

    with pytest.raises(ValueError, match="experts_per_token"):
        ExpertCache(config)


def test_construction_refuses_a_cache_that_exceeds_the_budget(tmp_path):
    (tmp_path / "manifest.json").write_text(_manifest_json())
    config = _config(
        bundle=tmp_path,
        slots_per_layer=200,
        budget_bytes=int(7.0 * GIB),
        reserve_bytes=int(1.5 * GIB),
        resident_bytes=int(1.696 * GIB),
    )

    with pytest.raises(CacheBudgetError) as error:
        ExpertCache(config)
    # The message has to say what to do about it.
    assert "Reduce slots_per_layer" in str(error.value)


def test_a_bundle_whose_blocks_are_unaligned_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(
        _manifest_json(block_bytes=3151872))     # the unpadded INT8 payload

    with pytest.raises(ValueError, match="not a multiple"):
        ExpertCache(_config(bundle=tmp_path))


def _manifest_json(**overrides) -> str:
    import json

    manifest = dict(MANIFEST)
    manifest.update(overrides)
    return json.dumps(manifest)


def test_close_is_documented_to_release_the_slots():
    # The slot array is the largest allocation the runtime makes. A close that
    # only dropped descriptors would leak the whole cache on reconfiguration,
    # which on a budgeted device is a functional defect rather than untidiness.
    import inspect

    source = inspect.getsource(ExpertCache.close)

    assert "self.slots = None" in source
    assert "self._staging = []" in source


def test_components_follows_the_manifest_and_omits_the_pad():
    # A consumer must not reproduce the offset arithmetic; it reads the order
    # from the manifest, so it cannot drift from whatever wrote the bundle.
    import inspect

    source = inspect.getsource(ExpertCache.components)

    assert 'self.manifest["block_layout"]' in source
    assert 'name != "padding"' in source


def test_global_scales_validates_the_sidecar_size():
    # The scales live beside the blocks so a block stays exactly block_bytes
    # and aligned. A truncated sidecar has to be caught, not silently reshaped.
    import inspect

    source = inspect.getsource(ExpertCache.global_scales)

    assert "num_experts * 2 * 4" in source
    assert "raise ValueError" in source


def test_streaming_frontend_actually_asks_the_loader_to_skip():
    # A silently-unapplied edit left the argument off this call once, and the
    # run that followed reported a resident footprint identical to the ordinary
    # frontend's -- plausible enough to miss without comparing against the
    # documented baseline. Pin the wiring so it cannot regress quietly.
    import inspect

    from flash_rt.frontends.torch import nexn2_rtx

    source = inspect.getsource(nexn2_rtx.Nexn2TorchFrontendRtx)

    assert "stream_experts=self._stream_experts" in source


def test_lm_head_has_a_path_without_the_sm120_only_kernel():
    # fp4_w4a4_mma_sm120_full_n_bf16out is built only for GPU_ARCH 120/121, so
    # before this branch the lm_head decode had no implementation on any other
    # target, the Orin one included.
    import inspect

    from flash_rt.frontends.torch import _nexn2_rtx_decode

    # The projection moved into its own function when the draft head, which
    # ends the same way, needed it too.
    source = inspect.getsource(_nexn2_rtx_decode._lm_head)

    assert "hasattr(fvk, 'fp4_w4a4_mma_sm120_full_n_bf16out')" in source
    # The fallback goes through the resolver, which returns whichever
    # weight-only GEMV this build has; both members of that pair are in the
    # W4A16 tier, so either satisfies what this test is about.
    assert "w4a16_matvec(fvk)" in source


def test_staging_buffers_are_owned_for_the_duration_of_a_read():
    # Indexing a buffer by task number is unsafe: the pool bounds how many
    # tasks run at once, not the order they finish in, so task N and task
    # N + len(staging) can hold the same buffer simultaneously and read into
    # each other's memory. A task must acquire one and give it back.
    import inspect

    source = inspect.getsource(ExpertCache._fetch)

    assert "self._available.get()" in source
    assert "self._available.put(buffer)" in source
    assert "finally:" in source
    # And the submission must not hand a buffer index in at all.
    submit = inspect.getsource(ExpertCache.get_many)
    assert "len(self._staging)" not in submit


def test_fetch_rejects_an_out_of_range_expert():
    # A negative or oversized index turns into an invalid file offset, which a
    # direct read reports as a bare EINVAL that names nothing.
    import inspect

    source = inspect.getsource(ExpertCache._fetch)

    assert "0 <= expert < self.num_experts" in source
    assert "invalid file offset" in source


def test_streaming_replaces_only_the_routed_expert_gemvs():
    # An earlier version returned from the middle of the MoE layer, which
    # dropped the shared expert and its gate from every layer -- and returned
    # the wrong shape and dtype while doing it. The branch must set d_dn and
    # fall through to the shared tail.
    import inspect

    from flash_rt.frontends.torch import _nexn2_rtx_decode

    source = inspect.getsource(_nexn2_rtx_decode._moe_layer_decode)

    assert "d_dn = _moe_experts_streamed(" in source
    assert "return _moe_experts_streamed(" not in source
    # The shared expert and its gate are still reached on both paths.
    assert "shared_down_proj" in source
    assert "shared_gate_w_t" in source


def test_streamed_experts_rotate_the_activation_when_the_bundle_is_rotated():
    # The bundle stores H*W, so an unrotated activation gives a wrong product
    # that is still finite and plausible -- the failure mode that hides.
    import inspect

    from flash_rt.frontends.torch import _nexn2_rtx_decode

    source = inspect.getsource(_nexn2_rtx_decode._moe_experts_streamed)

    assert "cache.manifest.get('rht')" in source
    # Both GEMMs: the hidden state entering gate_up, and the gated result
    # entering down.
    assert source.count("_rotate16(") == 2


def test_attention_backend_probes_the_vendored_kernel():
    # Importing the module and finding its symbols proves neither that the
    # kernel runs nor that it computes. On an SM110 part it imported, exposed
    # every symbol, and then refused at run time while writing nothing --
    # which reads downstream as plausible-but-wrong attention, not a failure.
    import inspect

    from flash_rt.hardware.rtx import attn_backend_nexn2 as backend

    cls = backend.RtxFlashAttnBackendNexn2
    probe = inspect.getsource(cls._probe_fa2)
    run = inspect.getsource(cls.run)

    # The probe compares against a reference rather than checking a return code.
    assert "scaled_dot_product_attention" in probe
    assert "isfinite" in probe
    # And it goes through the same launch the hot path uses, so it cannot pass
    # while the real call fails.
    assert "_launch_fa2" in probe
    assert "_launch_fa2" in run
    assert "self._fa2_usable" in run
