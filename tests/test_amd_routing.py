"""AMD (ROCm/CDNA4) hardware routing — dispatch-table and load_model gates.

What this file protects:

  * the ``("pi05", "torch", "amd_cdna4")`` registration in
    ``flash_rt.hardware._PIPELINE_MAP`` (a silent unregistration would make
    every AMD load fall through to the "no pipeline" error);
  * the ``detect_arch()`` ROCm branch: gfx950 (any ``gfx950:...`` suffix
    variant) maps to ``"amd_cdna4"``, and every other ROCm arch REFUSES
    instead of silently borrowing an NVIDIA backend (detect_arch is
    documented "deliberately strict");
  * the ``load_model`` front-door failure modes: an unbuilt AMD extension
    fails with build instructions (not a bare ModuleNotFoundError), an
    unknown hardware string never resolves, and the Thor-only FP4 knobs
    are refused (or safely defused) on amd_cdna4.

Environment matrix:
  - No GPU / NVIDIA CI: everything here runs except the two tests that
    require the built AMD extension (they skip with a reason).
  - MI350X with the extension built: the "extension missing" test skips,
    everything else runs.
No checkpoints are needed anywhere in this file — every load_model call
that would reach checkpoint loading is aimed at an error that fires first,
or is asserted to fail on the (deliberately) nonexistent checkpoint path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


def _amd_ext_importable() -> bool:
    """True iff the compiled AMD module is importable in this env."""
    try:
        import flash_rt.amd.flash_rt_amd_kernels  # noqa: F401
        return True
    except ImportError:
        return False


def _amd_pipeline_importable() -> bool:
    """True iff the AMD pi05 frontend module imports in this env.

    Resolution of ("pi05","torch","amd_cdna4") imports the frontend, which
    pulls in the pipeline module and its third-party numeric helpers (the
    same ones the RTX pipeline imports, declared under an optional extra
    rather than the base install). Tests that must run *past* resolution
    gate on this so a lean environment skips instead of erroring.
    """
    try:
        import flash_rt.amd.frontends.torch.pi05  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# _PIPELINE_MAP registration
# ---------------------------------------------------------------------------


def test_pipeline_map_has_amd_pi05_entry():
    """The AMD Pi0.5 dispatch entry must stay registered verbatim.

    load_model resolves lazily through this dict; if the tuple drifts
    (module rename, class rename) resolution breaks only at runtime on an
    AMD box, so the exact strings are pinned here.
    """
    from flash_rt.hardware import _PIPELINE_MAP

    key = ("pi05", "torch", "amd_cdna4")
    assert key in _PIPELINE_MAP, "AMD Pi0.5 dispatch entry disappeared"
    assert _PIPELINE_MAP[key] == (
        "flash_rt.amd.frontends.torch.pi05", "Pi05TorchFrontendAmd")


def test_amd_cdna4_is_a_documented_arch_string():
    """detect_arch's docstring is the user-facing support matrix; the AMD
    entry must be documented there, not just implemented."""
    from flash_rt.hardware import detect_arch

    assert "amd_cdna4" in (detect_arch.__doc__ or "")


# ---------------------------------------------------------------------------
# detect_arch() — ROCm branch (monkeypatched, no GPU needed)
# ---------------------------------------------------------------------------


def _patch_rocm(monkeypatch, gcn_arch_name: str):
    """Fake a ROCm torch build reporting the given gcnArchName."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # torch.version.hip is None on CUDA builds; set it truthy.
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda idx: SimpleNamespace(gcnArchName=gcn_arch_name))


@pytest.mark.parametrize("gcn", [
    # MI350X reports the arch with feature suffixes; the router must split
    # them off. Bare "gfx950" must also work (containers/tools vary).
    "gfx950:sramecc+:xnack-",
    "gfx950",
])
def test_detect_arch_maps_gfx950_to_amd_cdna4(monkeypatch, gcn):
    from flash_rt.hardware import detect_arch

    _patch_rocm(monkeypatch, gcn)
    assert detect_arch() == "amd_cdna4"


@pytest.mark.parametrize("gcn", [
    "gfx942:sramecc+:xnack-",   # MI300X — not supported, must refuse
    "gfx90a",                   # MI200
    "gfx1100",                  # RDNA3
    "gfx9500",                  # prefix trap: startswith() would wrongly pass
])
def test_detect_arch_refuses_non_gfx950_rocm(monkeypatch, gcn):
    """Verified behavior: a non-gfx950 ROCm GPU raises RuntimeError (it does
    NOT map to amd_cdna4 and does NOT fall through to the CUDA SM table —
    flash_rt/hardware/__init__.py raises inside the hip branch). Silently
    routing MI300/RDNA through the CDNA4 backend would fail much later at
    the first MFMA kernel, so strictness here is the contract."""
    from flash_rt.hardware import detect_arch

    _patch_rocm(monkeypatch, gcn)
    with pytest.raises(RuntimeError, match="unsupported ROCm GPU arch"):
        detect_arch()


def test_detect_arch_rocm_error_names_the_supported_arch(monkeypatch):
    """The refusal must tell the operator what IS supported."""
    from flash_rt.hardware import detect_arch

    _patch_rocm(monkeypatch, "gfx942:sramecc+")
    with pytest.raises(RuntimeError, match="gfx950"):
        detect_arch()


# ---------------------------------------------------------------------------
# resolve_pipeline_class — unsupported combos never resolve
# ---------------------------------------------------------------------------


def test_resolver_rejects_unknown_arch_string():
    from flash_rt.hardware import resolve_pipeline_class

    with pytest.raises(RuntimeError, match="no pipeline"):
        resolve_pipeline_class("pi05", "torch", "no_such_arch")


def test_resolver_rejects_unported_amd_configs():
    """Only pi05/torch is registered for amd_cdna4 today. A GROOT or JAX
    request must fail at resolution (with the built-for hint), not import
    half a backend."""
    from flash_rt.hardware import resolve_pipeline_class

    for config, framework in [("groot", "torch"), ("pi05", "jax"),
                              ("pi0", "torch")]:
        with pytest.raises(RuntimeError, match="no pipeline"):
            resolve_pipeline_class(config, framework, "amd_cdna4")


# ---------------------------------------------------------------------------
# load_model failure modes (no GPU: hardware= is passed explicitly, and
# every asserted error fires before any device or checkpoint access)
# ---------------------------------------------------------------------------

_BOGUS_CKPT = "/nonexistent/flashrt-amd-test-ckpt"


@pytest.mark.skipif(
    _amd_ext_importable(),
    reason="flash_rt_amd_kernels is built here; the missing-extension gate "
           "cannot fire (covered by the ext-present tests below)")
def test_load_model_amd_without_extension_names_the_build():
    """Verified behavior (flash_rt/api.py, arch == "amd_cdna4" gate): the
    refusal is an ImportError raised BEFORE the frontend import, and its
    message must carry the module name and the build command — a bare
    ModuleNotFoundError reads as a broken install rather than a build the
    user still has to run."""
    import flash_rt

    with pytest.raises(ImportError) as caught:
        flash_rt.load_model(_BOGUS_CKPT, framework="torch", config="pi05",
                            hardware="amd_cdna4")
    msg = str(caught.value)
    assert "flash_rt_amd_kernels" in msg
    assert "not built" in msg
    # The operator must be handed the next step, not just the diagnosis.
    assert "build_amd.sh" in msg or "cmake" in msg


def test_load_model_unknown_hardware_string_errors():
    """A typo'd hardware= must never produce a model. Depending on which
    extensions this environment has built, the error is either the CUDA
    extension gate (ImportError, fires first in flash_rt.api) or the
    resolver's "no pipeline" RuntimeError — both are acceptable refusals;
    success is the only failure mode."""
    import flash_rt

    with pytest.raises((ImportError, RuntimeError)):
        flash_rt.load_model(_BOGUS_CKPT, framework="torch", config="pi05",
                            hardware="not_a_real_arch")


@pytest.mark.skipif(
    not _amd_ext_importable(),
    reason="flash_rt_amd_kernels not built (the amd_cdna4 extension gate "
           "fires before the FP4 checks, masking them)")
def test_load_model_use_fp4_decoder_rejected_on_amd():
    """Verified behavior: the Thor NVFP4 tier's explicit sub-flags
    (use_fp4_decoder etc.) raise ValueError on any non-Thor arch — the
    check in flash_rt/api.py fires before resolve_pipeline_class, so no
    checkpoint is touched. NOTE this check sits AFTER the AMD extension
    import gate, hence the skip above on boxes without the .so."""
    import flash_rt

    with pytest.raises(ValueError, match="hardware='thor'"):
        flash_rt.load_model(_BOGUS_CKPT, framework="torch", config="pi05",
                            hardware="amd_cdna4",
                            use_fp4=True, use_fp4_decoder=True)


@pytest.mark.skipif(
    not _amd_ext_importable(),
    reason="flash_rt_amd_kernels not built (the amd_cdna4 extension gate "
           "fires before the FA4 check, masking it)")
def test_load_model_use_fa4_rejected_on_amd():
    """FA4 is a Thor-only attention backend; verified: ValueError before
    resolution (flash_rt/api.py gates on config/framework/arch)."""
    import flash_rt

    with pytest.raises(ValueError, match="use_fa4"):
        flash_rt.load_model(_BOGUS_CKPT, framework="torch", config="pi05",
                            hardware="amd_cdna4", use_fa4=True)


@pytest.mark.skipif(
    not _amd_ext_importable(),
    reason="flash_rt_amd_kernels not built (the extension gate raises "
           "ImportError before the FP4 routing block is reached)")
@pytest.mark.skipif(
    not _amd_pipeline_importable(),
    reason="AMD pi05 frontend not importable here (optional numeric deps "
           "missing); this test asserts behaviour past class resolution")
def test_load_model_bare_use_fp4_falls_back_to_fp8_on_amd(caplog):
    """Verified behavior — this was checked in the source rather than
    assumed: bare ``use_fp4=True`` (no sub-flags) does NOT raise on
    amd_cdna4. flash_rt/api.py's FP4 routing block logs
    "use_fp4=True is only supported ..." and defuses the flag
    (``use_fp4 = False`` → FP8 path). We pin that contract: the load must
    proceed past the FP4 block (proven by it then failing on the
    deliberately nonexistent checkpoint with FileNotFoundError, not a
    ValueError/ImportError from the FP4 machinery) and the fallback must
    be announced in the log."""
    import flash_rt

    with caplog.at_level(logging.WARNING, logger="flash_rt.api"):
        with pytest.raises(FileNotFoundError):
            flash_rt.load_model(_BOGUS_CKPT, framework="torch",
                                config="pi05", hardware="amd_cdna4",
                                use_fp4=True)
    assert any("Falling back to FP8" in rec.getMessage()
               for rec in caplog.records), (
        "expected the documented use_fp4 → FP8 fallback warning")
