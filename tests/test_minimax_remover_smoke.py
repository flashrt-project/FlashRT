"""Smoke tests for MiniMax-Remover FlashRT integration.

These tests run in **any** build configuration:
  - default build (SM120 NVFP4 kernels absent): import succeeds,
    ``_load_kernels`` raises ``RuntimeError``, pipeline construction
    fails fast.
  - gated build (SM120 NVFP4 kernels present): the required NVFP4
    symbols are present and callable.

No GPU, no model checkpoint, no MiniMax-Remover source tree is required.
"""
import pytest


# ── 1. Package import always succeeds ──

def test_package_import():
    """Importing the model package must not require flash_rt_kernels."""
    from flash_rt.models.minimax_remover import MiniMaxRemoverPipeline
    assert MiniMaxRemoverPipeline is not None


def test_pipeline_module_import():
    """The pipeline module imports cleanly without flash_rt_kernels."""
    from flash_rt.models.minimax_remover import pipeline
    assert hasattr(pipeline, "MiniMaxRemoverPipeline")
    assert hasattr(pipeline, "_load_kernels")
    assert hasattr(pipeline, "_REQUIRED_NVFP4_SYMBOLS")
    assert "nvfp4_sf_swizzled_bytes" in pipeline._REQUIRED_NVFP4_SYMBOLS


# ── 2. _load_kernels validates the NVFP4 surface ──

def test_load_kernels_raises_when_symbols_absent():
    """Without the NVFP4 kernels, _load_kernels raises a clear RuntimeError."""
    from flash_rt.models.minimax_remover import pipeline

    class _NoNvfp4:
        # Module stub exposing none of the required symbols.
        pass

    import sys
    import types

    fake_mod = types.ModuleType("flash_rt.flash_rt_kernels")
    sys.modules["flash_rt.flash_rt_kernels"] = fake_mod
    try:
        with pytest.raises(RuntimeError) as excinfo:
            pipeline._load_kernels()
        msg = str(excinfo.value)
        assert "NVFP4" in msg
        assert "Missing symbols" in msg
        assert "nvfp4_sf_swizzled_bytes" in msg
    finally:
        del sys.modules["flash_rt.flash_rt_kernels"]


def test_load_kernels_succeeds_when_symbols_present():
    """With all required symbols, _load_kernels returns the kernels module."""
    from flash_rt.models.minimax_remover import pipeline

    import sys
    import types

    fake_mod = types.ModuleType("flash_rt.flash_rt_kernels")
    for s in pipeline._REQUIRED_NVFP4_SYMBOLS:
        setattr(fake_mod, s, lambda *a, **k: None)
    sys.modules["flash_rt.flash_rt_kernels"] = fake_mod
    try:
        fvk = pipeline._load_kernels()
        assert fvk is fake_mod
    finally:
        del sys.modules["flash_rt.flash_rt_kernels"]


# ── 3. Pipeline construction validates kernel availability ──

class _FakePipe:
    """Minimal stub matching the diffusers pipeline contract.

    Construction must fail at _load_kernels before any pipe attribute is
    touched, so the stub is never actually read.
    """


def test_pipeline_constructor_validates_kernels(monkeypatch):
    """Pipeline construction must fail before touching model internals."""
    from flash_rt.models.minimax_remover import pipeline

    def _raise_missing():
        raise RuntimeError(
            "MiniMax-Remover requires the SM120 NVFP4 kernels which are not "
            "compiled into flash_rt_kernels. Rebuild with the Blackwell NVFP4 "
            "build option enabled.")

    monkeypatch.setattr(pipeline, "_load_kernels", _raise_missing)
    with pytest.raises(RuntimeError, match="NVFP4"):
        pipeline.MiniMaxRemoverPipeline(_FakePipe())


def test_pipeline_constructor_calls_load_kernels(monkeypatch):
    """_load_kernels is invoked exactly once during construction."""
    from flash_rt.models.minimax_remover import pipeline

    calls = []

    def _fake_load():
        calls.append(1)
        raise RuntimeError("stop construction here")

    monkeypatch.setattr(pipeline, "_load_kernels", _fake_load)
    with pytest.raises(RuntimeError, match="stop construction"):
        pipeline.MiniMaxRemoverPipeline(_FakePipe())
    assert len(calls) == 1


# ── 4. Gated build: required NVFP4 symbols present and callable ──

def test_nvfp4_symbols_present_when_gated():
    """In a gated build, every required NVFP4 symbol is present & callable."""
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        try:
            import flash_rt_kernels as fvk  # type: ignore
        except ImportError:
            pytest.skip("flash_rt_kernels not built")

    from flash_rt.models.minimax_remover.pipeline import _REQUIRED_NVFP4_SYMBOLS
    missing = [s for s in _REQUIRED_NVFP4_SYMBOLS if not hasattr(fvk, s)]
    if missing:
        pytest.skip(f"SM120 NVFP4 kernels not compiled (missing: {', '.join(missing)})")

    for sym in _REQUIRED_NVFP4_SYMBOLS:
        assert callable(getattr(fvk, sym)), f"{sym} is not callable"


def test_nvfp4_symbols_absent_in_default_build():
    """In a default (non-NVFP4) build, _load_kernels documents the gap.

    This documents the 'compile option OFF' case end-to-end: if any
    required symbol is missing the pipeline refuses to construct.
    """
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        pytest.skip("flash_rt_kernels not built (treated as missing)")
    from flash_rt.models.minimax_remover.pipeline import _REQUIRED_NVFP4_SYMBOLS

    missing = [s for s in _REQUIRED_NVFP4_SYMBOLS if not hasattr(fvk, s)]
    if not missing:
        pytest.skip("this build has the NVFP4 kernels (gated build) — covered elsewhere")

    # Verify _load_kernels raises and names the missing symbols.
    import sys
    import types

    fake_mod = types.ModuleType("flash_rt.flash_rt_kernels")
    for s in _REQUIRED_NVFP4_SYMBOLS:
        if hasattr(fvk, s):
            setattr(fake_mod, s, getattr(fvk, s))
    sys.modules["flash_rt.flash_rt_kernels"] = fake_mod
    try:
        from flash_rt.models.minimax_remover import pipeline
        with pytest.raises(RuntimeError) as excinfo:
            pipeline._load_kernels()
        for s in missing:
            assert s in str(excinfo.value)
    finally:
        del sys.modules["flash_rt.flash_rt_kernels"]
