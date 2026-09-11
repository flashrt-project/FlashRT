"""LTX-2.5 runtime contracts: import, fallback, alignment, and residency.

These cover the parts that are easy to get wrong without a checkpoint on
disk: that the model package imports with none of its optional pieces
present, that a broken extension is not mistaken for an absent one, that
the FFN swap declines the shapes CUTLASS declines, and that the residency
lease can be ended twice and taken again.

The heavy paths (a real pipeline, real kernels, a device) are not
reachable here and are covered by the model's own benchmark runs.
"""

import importlib
import sys
import types

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402


# --------------------------------------------------------------------
# import smoke: the package must import with no LTX install, no kernels
# --------------------------------------------------------------------

def test_model_package_imports_without_optional_dependencies():
    """Neither the upstream LTX packages nor a built extension is required.

    Unconditional on purpose. A host with no extension must reach this
    import (the swaps fall back), and a host whose extension is broken must
    fail it -- so there is no environment in which skipping is the honest
    answer, and skipping is how the previous version of this test let an
    import contract regress unnoticed.
    """
    from flash_rt.models import ltx25
    from flash_rt.models.ltx25 import _attn_swap, _nvfp4_ffn_swap

    assert ltx25 is not None
    assert isinstance(_attn_swap.fvk_sage2_available(), bool)
    assert isinstance(_nvfp4_ffn_swap.fvk_ffn_available(), bool)


def test_frontend_module_imports_and_declares_its_surface():
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    for name in ("set_prompt", "infer", "release_resident", "close",
                 "get_latency_stats"):
        assert callable(getattr(Ltx25TorchFrontendRtx, name)), name


# --------------------------------------------------------------------
# the public API: what the documented entry point actually gives back
# --------------------------------------------------------------------

def _load_ltx(monkeypatch, **kwargs):
    """Run ``load_model(config="ltx25")`` against a recording frontend.

    The construction path is the thing under test -- which arguments reach
    the frontend, and what the caller is handed back -- so everything up to
    the frontend class is the real code and only the frontend itself is
    replaced. A checkpoint is never opened.
    """
    from flash_rt import api

    class _RecordingFrontend:
        instances = []

        def __init__(self, checkpoint, attention=None, fuse=True,
                     compile_mode=None, device=None, **rest):
            self.checkpoint = checkpoint
            self.attention = attention
            self.fuse = fuse
            self.compile_mode = compile_mode
            self.released = 0
            self.closed = 0
            type(self).instances.append(self)

        def release_resident(self):
            self.released += 1
            return 7

        def close(self):
            self.closed += 1
            return 11

    import flash_rt.hardware as hardware

    _RecordingFrontend.instances = []
    # load_model resolves the frontend class through this one call, which
    # is where the real code is interposed: everything before it (config
    # validation, arch detection, the argument forwarding under test) runs
    # unchanged.
    monkeypatch.setattr(hardware, "resolve_pipeline_class",
                        lambda *a, **k: _RecordingFrontend)
    monkeypatch.setattr(hardware, "detect_arch", lambda *a, **k: "rtx_sm120")
    model = api.load_model(checkpoint="unused", config="ltx25", **kwargs)
    return model, _RecordingFrontend.instances[-1]


def test_public_api_accepts_the_ltx_config():
    """``load_model`` must know the config name it documents."""
    from flash_rt import api

    with pytest.raises(ValueError, match="Unknown config"):
        api.load_model(checkpoint="unused", config="no_such_model")


def test_public_api_forwards_the_execution_assembly(monkeypatch):
    """attention/fuse/compile_mode must reach the frontend from load_model.

    Capture mode is the point of this runtime, and a caller who cannot ask
    for it through the documented entry point does not have it.
    """
    model, frontend = _load_ltx(
        monkeypatch, attention="sage2-fvk", fuse=True,
        compile_mode="capture")
    assert frontend.attention == "sage2-fvk"
    assert frontend.fuse is True
    assert frontend.compile_mode == "capture"
    assert model is not None


def test_public_api_leaves_frontend_defaults_alone(monkeypatch):
    """Unset arguments must not be forwarded as None over the defaults."""
    _, frontend = _load_ltx(monkeypatch)
    assert frontend.attention is None
    assert frontend.fuse is True, "the frontend's own default must survive"
    assert frontend.compile_mode is None


def test_public_model_exposes_the_release_surface():
    """The lifecycle calls the docs show have to exist on what is returned.

    ``load_model`` hands back a wrapper, not the frontend, so a method the
    documentation tells a caller to use is only real if the wrapper
    delegates it.
    """
    from flash_rt.api import VLAModel

    for name in ("release_resident", "close"):
        assert callable(getattr(VLAModel, name, None)), name


def test_public_model_delegates_release_and_close():
    from flash_rt.api import VLAModel

    class _Frontend:
        def __init__(self):
            self.released = self.closed = 0

        def release_resident(self):
            self.released += 1
            return 7

        def close(self):
            self.closed += 1
            return 11

    frontend = _Frontend()
    model = VLAModel(frontend, "torch")
    assert model.release_resident() == 7
    assert model.close() == 11
    assert (frontend.released, frontend.closed) == (1, 1)


def test_public_model_release_is_harmless_on_other_frontends():
    """A frontend that holds nothing answers 0 instead of refusing."""
    from flash_rt.api import VLAModel

    model = VLAModel(object(), "torch")
    assert model.release_resident() == 0
    assert model.close() == 0


# --------------------------------------------------------------------
# optional import: absent is a fallback, broken is a bug
# --------------------------------------------------------------------

class _RaisingFinder:
    """Meta-path finder that makes one module name fail on import.

    The swaps import through ``importlib``, so a fake ``__import__`` does
    not stand in the way: the simulation has to happen in the import
    machinery itself or it tests nothing. ``exc`` is raised when the name is
    looked up, which is what an absent (or broken) module does.
    """

    def __init__(self, name, exc):
        self.name = name
        self.exc = exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise self.exc
        return None


def _reimport(module_name, finder, monkeypatch):
    """Import ``module_name`` afresh with ``finder`` in the way.

    The simulated module has to leave ``sys.modules`` as well as the module
    under test: an already-imported extension is returned straight from the
    cache and no finder is ever consulted, which would quietly turn this
    into a test of nothing on any host where the extension is built.
    """
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)
    for name in [module_name, "flash_rt.models.ltx25", finder.name]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", [
    "flash_rt.models.ltx25._attn_swap",
    "flash_rt.models.ltx25._nvfp4_ffn_swap",
])
def test_absent_extension_is_a_fallback(module_name, monkeypatch):
    """No extension: the module imports and the swap knows it has none."""
    finder = _RaisingFinder(
        "flash_rt.flash_rt_kernels",
        ModuleNotFoundError("No module named 'flash_rt.flash_rt_kernels'",
                            name="flash_rt.flash_rt_kernels"))
    module = _reimport(module_name, finder, monkeypatch)
    assert module.fvk is None


@pytest.mark.parametrize("module_name", [
    "flash_rt.models.ltx25._attn_swap",
    "flash_rt.models.ltx25._nvfp4_ffn_swap",
])
def test_broken_extension_is_not_swallowed_as_absent(module_name, monkeypatch):
    """An extension that fails to load must not read as 'not built'.

    The swaps treat the extension's own absence as a fallback. Every other
    load failure -- an undefined symbol, an ABI mismatch, a transitive
    import error -- has to propagate, or a broken build silently runs the
    slow path and reports nothing.
    """
    finder = _RaisingFinder("flash_rt.flash_rt_kernels",
                            ImportError("undefined symbol: _Z9brokenABIv"))
    with pytest.raises(ImportError, match="undefined symbol"):
        _reimport(module_name, finder, monkeypatch)


@pytest.mark.parametrize("module_name", [
    "flash_rt.models.ltx25._attn_swap",
    "flash_rt.models.ltx25._nvfp4_ffn_swap",
])
def test_absent_extension_is_distinguished_from_a_missing_dependency(
        module_name, monkeypatch):
    """A *different* module going missing is not this extension's absence.

    The name check is what separates them: without it, any transitive
    ModuleNotFoundError raised while loading the extension would be read as
    "the extension is not built" and quietly fall back.
    """
    finder = _RaisingFinder(
        "flash_rt.flash_rt_kernels",
        ModuleNotFoundError("No module named 'some_transitive_dep'",
                            name="some_transitive_dep"))
    with pytest.raises(ModuleNotFoundError, match="some_transitive_dep"):
        _reimport(module_name, finder, monkeypatch)


def test_sage_package_probe_falls_back_only_on_absence(monkeypatch):
    """``auto`` may fall back to SDPA when sageattention is not installed.

    It may not do so when the package is installed and broken: that is an
    environment fault, and answering it with a quiet slowdown hides it.
    """
    from flash_rt.models.ltx25 import _attn_swap

    monkeypatch.setattr(_attn_swap, "fvk_sage2_available", lambda: False)

    absent = _RaisingFinder(
        "sageattention",
        ModuleNotFoundError("No module named 'sageattention'",
                            name="sageattention"))
    monkeypatch.setattr(sys, "meta_path", [absent] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "sageattention", raising=False)
    attn = _attn_swap.make_ltx25_attention("auto")
    assert attn is None or getattr(attn, "label", "") == "sdpa"

    broken = _RaisingFinder("sageattention",
                            ImportError("undefined symbol: _Z6brokenv"))
    monkeypatch.setattr(sys, "meta_path", [broken] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "sageattention", raising=False)
    with pytest.raises(ImportError, match="undefined symbol"):
        _attn_swap.make_ltx25_attention("auto")


# --------------------------------------------------------------------
# attention selection
# --------------------------------------------------------------------

def test_attention_selection_refuses_unknown_kinds():
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention

    with pytest.raises(ValueError):
        make_ltx25_attention("no_such_backend")


def test_sdpa_selection_never_depends_on_the_extension():
    """The baseline backend must be reachable on a host with no kernels."""
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention

    attn = make_ltx25_attention("sdpa")
    assert attn is None or getattr(attn, "label", "") == "sdpa"


def test_explicit_backend_fails_fast_when_unavailable(monkeypatch):
    """``sage2`` asked for by name must not silently become SDPA."""
    from flash_rt.models.ltx25 import _attn_swap

    monkeypatch.setattr(_attn_swap, "fvk_sage2_available", lambda: False)
    with pytest.raises(RuntimeError, match="sage2"):
        _attn_swap.make_ltx25_attention("sage2-fvk")


# --------------------------------------------------------------------
# FFN alignment: the swap declines exactly what the kernel declines
# --------------------------------------------------------------------

@pytest.mark.parametrize("rows,swapped", [
    (128, True), (256, True), (2688, True), (24576, True),
    (1, False), (126, False), (127, False), (129, False),
])
def test_ffn_swap_routes_by_row_alignment(rows, swapped):
    """M % 128 decides the arm, because that is what can_implement decides.

    The CUTLASS chain reports a validation failure for unaligned M without
    writing an output, so an unaligned call that reached it would produce
    silent garbage. The predicate is checked directly here: it holds with
    or without a device.
    """
    from flash_rt.models.ltx25._nvfp4_ffn_swap import rows_are_swappable

    assert rows_are_swappable(rows) is swapped


# --------------------------------------------------------------------
# residency lease
# --------------------------------------------------------------------

class _FakeModel(torch.nn.Module):
    """Stands in for the built transformer: a module with a dispose."""

    def __init__(self):
        super().__init__()
        self.disposed = 0

    def dispose(self):
        self.disposed += 1


class _FakeBuilder:
    """Inner builder standing in for the upstream stage builder."""

    def __init__(self):
        self.builds = 0

    def build(self, **kwargs):
        self.builds += 1
        return _FakeModel()


def _resident_builder(monkeypatch):
    from flash_rt.models.ltx25 import _resident_graph

    # the X0Model patch reaches into the upstream package; the lease
    # semantics under test do not need it
    monkeypatch.setattr(_resident_graph, "_patch_x0_dispose", lambda: None)
    return _resident_graph.ResidentSwapBuilder(_FakeBuilder(), [])


def test_residency_is_taken_once_and_reused(monkeypatch):
    builder = _resident_builder(monkeypatch)
    first = builder.build()
    assert builder.build() is first
    assert builder._inner.builds == 1
    assert builder.is_resident
    assert builder.keeps_gpu_resident_weights


def test_release_is_idempotent_and_disposes_once(monkeypatch):
    builder = _resident_builder(monkeypatch)
    model = builder.build()
    builder.release()
    assert not builder.is_resident
    assert model.disposed == 1
    builder.release()
    builder.release()
    assert model.disposed == 1, "a second release must not re-dispose"


def test_a_released_lease_can_be_taken_again(monkeypatch):
    builder = _resident_builder(monkeypatch)
    first = builder.build()
    builder.release()
    second = builder.build()
    assert second is not first
    assert builder.is_resident
    assert builder._inner.builds == 2


def test_rewrapped_builders_share_one_lease(monkeypatch):
    """The stage rewraps its builder per call; the lease must not fork."""
    builder = _resident_builder(monkeypatch)
    model = builder.build()
    rewrapped = builder._rewrap(builder._inner)
    assert rewrapped.build() is model
    rewrapped.release()
    assert not builder.is_resident


# --------------------------------------------------------------------
# prompt cache: a hit keeps the lease, a miss ends it
# --------------------------------------------------------------------

def test_cached_prompt_does_not_disturb_residency():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls, misses = [], []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: calls.append(prompts) or "embeds",
        on_miss=lambda: misses.append(1))

    assert encoder(["a fisherman"]) == "embeds"
    assert encoder(["a fisherman"]) == "embeds"
    assert len(calls) == 1, "a repeat prompt must not re-run the encoder"
    assert len(misses) == 1, "only the first encode ends the lease"


def test_new_prompt_ends_the_lease_before_encoding():
    """The release must happen *before* the encoder loads, not after.

    Ordering is the whole point: the encoder's weights and the resident
    transformer do not fit together, so a release that ran afterwards would
    free memory the encode had already failed to get.
    """
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    order = []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: order.append("encode") or "embeds",
        on_miss=lambda: order.append("release"))

    encoder(["first"])
    encoder(["second"])
    assert order == ["release", "encode", "release", "encode"]


def test_prompt_cache_keys_on_encoder_arguments():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls = []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: calls.append(kw) or "embeds")

    encoder(["p"], enhance=False)
    encoder(["p"], enhance=True)
    assert len(calls) == 2, "different encoder arguments are different work"


def test_prompt_cache_clear_is_idempotent():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls = []
    encoder = CachingPromptEncoder(lambda prompts, **kw: calls.append(1))
    encoder(["p"])
    encoder.clear()
    encoder.clear()
    encoder(["p"])
    assert len(calls) == 2


def test_release_entries_are_safe_before_a_pipeline_exists():
    """``release_resident``/``close`` must not require a loaded pipeline."""
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    frontend = Ltx25TorchFrontendRtx.__new__(Ltx25TorchFrontendRtx)
    frontend._pipe = None
    assert frontend.release_resident() == 0
    assert frontend.close() == 0
    assert frontend.close() == 0


def test_release_resident_is_a_no_op_outside_capture_mode():
    """Non-capture modes hold no lease, so there is nothing to release."""
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    frontend = Ltx25TorchFrontendRtx.__new__(Ltx25TorchFrontendRtx)
    stage = types.SimpleNamespace(_transformer_builder=object())
    frontend._pipe = types.SimpleNamespace(stage=stage)
    assert frontend.release_resident() == 0


def test_release_detaches_the_swapped_forwards():
    """Release must take back what the swap attached to the shell.

    The upstream builder caches model shells by structure and reuses them
    across builds, so a released model is not a collected one: the repacked
    FP4 weights the swap installed stay reachable through the shell unless
    the instance-level forward is removed by name.
    """
    from flash_rt.models.ltx25._nvfp4_ffn_swap import uninstall_nvfp4_ffn

    class _Shell(torch.nn.Module):
        def forward(self, x):
            return x

    shell = _Shell()
    packed = torch.zeros(8)
    swapped = lambda x: x                       # noqa: E731 - stands in
    swapped._flash_rt_keep = (packed,)
    shell.forward = swapped

    assert uninstall_nvfp4_ffn(shell) == 1
    assert "forward" not in shell.__dict__, "the class forward must be back"
    assert uninstall_nvfp4_ffn(shell) == 0, "uninstall is idempotent"


def test_release_drops_the_captured_block_loop(monkeypatch):
    """The capture runner holds a private graph pool; release must drop it.

    Freeing the loaded weights does not: upstream's dispose leaves the shell
    intact by design, and the runner hangs off the shell as an instance
    attribute over the class's own method.
    """
    from flash_rt.models.ltx25 import _resident_graph

    monkeypatch.setattr(_resident_graph, "_patch_x0_dispose", lambda: None)

    class _Runner:
        """Stands in for the capture runner and its graphs."""

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.disposed = 0

        def _process_transformer_blocks(self, *a):
            return a

        def dispose(self):
            self.disposed += 1

    model = _Model()
    runner = _Runner()
    model._process_transformer_blocks = runner

    class _Builder:
        def build(self, **kwargs):
            return model

    builder = _resident_graph.ResidentSwapBuilder(_Builder(), [])
    builder.build()
    builder.release()

    assert "_process_transformer_blocks" not in model.__dict__
    assert callable(model._process_transformer_blocks), (
        "the class's own block loop must be reachable again")
    assert model.disposed == 1


def test_release_evicts_the_mutated_shell(monkeypatch):
    """A shell whose parameters were freed must not be handed to a rebuild.

    Upstream caches model shells by structure and reassigns weights onto
    them, which is only sound while the parameters keep their shapes. The
    resident FFN swap frees the upstream projections it repacked, so the
    shell it leaves behind cannot be loaded into -- release evicts it.
    """
    from flash_rt.models.ltx25 import _resident_graph

    monkeypatch.setattr(_resident_graph, "_patch_x0_dispose", lambda: None)

    class _Registry:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    class _Inner:
        def __init__(self):
            self.registry = _Registry()

        def build(self, **kwargs):
            return _FakeModel()

    inner = _Inner()
    builder = _resident_graph.ResidentSwapBuilder(inner, [])
    builder.build()
    builder.release()
    assert inner.registry.cleared == 1
    builder.release()
    assert inner.registry.cleared == 1, "nothing to evict without a lease"


def test_shell_eviction_walks_wrapped_builders(monkeypatch):
    """The stage wraps builders; the registry may be several layers in."""
    from flash_rt.models.ltx25._resident_graph import _evict_cached_shell

    class _Registry:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    class _Base:
        registry = None

    base = _Base()
    base.registry = _Registry()
    wrapper = types.SimpleNamespace(_inner=types.SimpleNamespace(_inner=base))
    assert _evict_cached_shell(wrapper) is True
    assert base.registry.cleared == 1
    assert _evict_cached_shell(types.SimpleNamespace(_inner=None)) is False
