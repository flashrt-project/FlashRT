"""The AMD frontends must not mutate third-party module state.

An earlier revision worked around a transformers offline-mode bug by
replacing ``huggingface_hub.model_info`` — first permanently at import,
then temporarily around one call. Both are wrong for a library: the
substitution is visible to every other thread in the process, and a
save/restore pair is not safe under concurrency (two interleaved scopes
can leave a wrapper installed for good).

These gates pin the current contract: importing or constructing the AMD
frontends leaves third-party callables untouched, and the offline case
is handled by reporting it with an injection point instead.
"""
import importlib

import pytest

# Third-party callables an earlier revision replaced, plus the ones most
# likely to be reached for by a future workaround.
_WATCHED = [
    ("huggingface_hub", "model_info"),
    ("huggingface_hub", "snapshot_download"),
]

_AMD_FRONTENDS = [
    "flash_rt.amd.frontends.torch.pi05",
    "flash_rt.amd.frontends.torch.groot_n17",
]


def _snapshot():
    """Identity of each watched callable, skipping absent modules."""
    seen = {}
    for mod_name, attr in _WATCHED:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None:
            seen[(mod_name, attr)] = fn
    return seen


@pytest.mark.parametrize("frontend", _AMD_FRONTENDS)
def test_importing_a_frontend_leaves_third_party_callables_alone(frontend):
    """Importing a frontend must not swap out third-party functions."""
    before = _snapshot()
    if not before:
        pytest.skip("huggingface_hub is not installed here")
    try:
        importlib.import_module(frontend)
    except ImportError as exc:
        pytest.skip(f"{frontend} not importable here: {exc}")
    after = _snapshot()
    for key, original in before.items():
        assert after.get(key) is original, (
            f"importing {frontend} replaced {key[0]}.{key[1]}")


@pytest.mark.parametrize("frontend", _AMD_FRONTENDS)
def test_frontend_source_does_not_assign_to_third_party_globals(frontend):
    """Guard the pattern itself, not just its effect at import time.

    A workaround installed inside a function would pass the import test
    above while still mutating shared state when it runs, so the source
    must not assign to the watched attributes at all.
    """
    try:
        module = importlib.import_module(frontend)
    except ImportError as exc:
        pytest.skip(f"{frontend} not importable here: {exc}")
    source = open(module.__file__, encoding="utf-8").read()
    for mod_name, attr in _WATCHED:
        for alias in (mod_name, mod_name.split(".")[-1], "hh"):
            assert f"{alias}.{attr} =" not in source, (
                f"{frontend} assigns to {alias}.{attr}; a library must not "
                f"replace third-party callables, even temporarily")


def test_groot_frontend_exposes_a_processor_injection_point():
    """The supported alternative to patching must stay available.

    Offline callers build the processor themselves and inject it, which
    is why the frontend does not need to touch huggingface_hub.
    """
    try:
        mod = importlib.import_module(
            "flash_rt.amd.frontends.torch.groot_n17")
    except ImportError as exc:
        pytest.skip(f"GROOT AMD frontend not importable here: {exc}")
    cls = mod.GrootN17TorchFrontendAmd
    assert hasattr(cls, "set_hf_processor")

    class _Stub:
        pass

    stub = _Stub()
    # Bypass __init__ (it needs a device and a checkpoint): the setter and
    # the cache lookup are what this gate covers.
    obj = cls.__new__(cls)
    cls.set_hf_processor(obj, stub)
    assert cls._hf_processor(obj) is stub, (
        "an injected processor must be used without attempting a load")
