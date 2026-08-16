"""What a gate verdict records: the shape it holds for, and its memory.

Both are recording, not deciding. ``min_speedup`` is untouched, no new
refusal path exists, and every verdict this produces is the verdict the
gate produced before — with two facts attached that were previously lost:
the shape the measurement was taken at, and what the unit does to resident
memory. A refusal that reads as free may be the most expensive line in a
run, and until it says so that cannot even be discussed.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from flash_rt.structures import frontdoor  # noqa: E402


# --------------------------------------------------------------------
# the per-shape rule the specs have always declared
# --------------------------------------------------------------------

def test_specs_declaring_per_shape_are_read():
    """Thirteen specs declare it; the code has never looked at one."""
    from flash_rt.structures.registry import list_structures, load

    declared = [n for n in list_structures()
                if ((load(n).gates or {}).get("latency") or {}).get("per_shape")]
    assert declared, "no spec declares latency.per_shape"
    for name in declared:
        assert frontdoor._per_shape_rule(name) is True, name


def test_a_routed_unit_resolves_to_its_structure():
    """The gate's unit name is not always the structure's name."""
    assert frontdoor._per_shape_rule("attention_core_routed") is True


def test_an_unknown_unit_claims_nothing():
    assert frontdoor._per_shape_rule("not_a_structure") is False


def test_refusal_reasons_carry_the_shape_and_the_scope():
    """A verdict measured at one shape must not read as universal."""
    import inspect

    source = inspect.getsource(frontdoor.attach)
    assert "per_shape_rule" in source
    assert "holds for that shape and no other" in source


# --------------------------------------------------------------------
# the shape a verdict was measured at
# --------------------------------------------------------------------

class _Guarded(torch.nn.Module):
    def __init__(self, rows=None, capacity=None, bytes_=0):
        super().__init__()
        self._frt_guard = type("G", (), {"rows": rows,
                                         "row_capacity": capacity})()
        if bytes_:
            self.register_buffer("w", torch.empty(bytes_, dtype=torch.uint8))


def test_shape_comes_from_the_guards_when_the_binding_is_silent():
    """``m_profile`` is empty for hosts whose binding does not declare it,
    and the refusal then read "rows unrecorded" — a shape-scoped verdict
    with no shape on it. The guards always knew."""
    from flash_rt.structures.autobuild import AutoPlan

    paths = {"a": _Guarded(rows=4032), "b": _Guarded(rows=786432)}
    note = frontdoor._measured_shape(paths, AutoPlan())
    assert "4032" in note and "786432" in note


def test_a_capacity_armed_seam_says_so():
    from flash_rt.structures.autobuild import AutoPlan

    note = frontdoor._measured_shape({"a": _Guarded(capacity=2048)},
                                     AutoPlan())
    assert note == "rows<=2048"


def test_shape_falls_back_to_the_binding_profile():
    from flash_rt.structures.autobuild import AutoPlan
    from flash_rt.structures.discover import Seam

    plan = AutoPlan()
    plan.seams.append(Seam(structure="vision_ffn", path="p", parent_path="",
                           norm_attr=None, dims={}, variant={},
                           m_profile={"decode": {}}))
    assert "decode" in frontdoor._measured_shape({}, plan)


# --------------------------------------------------------------------
# what a unit does to resident memory
# --------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device")
def test_a_replacement_reports_what_it_saves():
    """A quantized form replacing host weights must read as a saving.

    The retained host lives inside its replacement for fallback, so a walk
    that counts it on both sides reports a form that halves the weights as
    costing nothing.
    """
    host = _Guarded(bytes_=8 << 20).cuda()
    bound = _Guarded(rows=8, bytes_=2 << 20).cuda()
    bound.host = host                       # retained for fallback
    bound._frt_host = lambda: host

    memory = frontdoor._memory_delta({"p": bound})
    assert memory["host_bytes"] == 8 << 20
    assert memory["bound_bytes"] == 2 << 20
    assert memory["delta_bytes"] == -(6 << 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device")
def test_a_form_that_brings_a_working_set_reports_it():
    """Nothing replaced, so everything it holds is new resident memory."""
    core = _Guarded(rows=8, bytes_=4 << 20).cuda()
    memory = frontdoor._memory_delta({"p": core})
    assert memory["delta_bytes"] == 4 << 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device")
def test_a_pooled_working_set_is_counted_once():
    """Sites sharing one staging set hold it once, not once each."""
    shared = torch.empty(4 << 20, dtype=torch.uint8, device="cuda")

    cores = {}
    for name in ("a", "b", "c"):
        core = _Guarded(rows=8).cuda()
        core.staging = shared               # a plain attribute, as pooled
        cores[name] = core
    memory = frontdoor._memory_delta(cores)
    assert memory["bound_bytes"] == 4 << 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device")
def test_tensors_held_inside_a_workspace_object_are_found():
    """A form's largest holding is often an artifact's workspace object,
    which is neither a parameter nor a buffer."""
    class _Workspace:
        def __init__(self):
            self.packed = torch.empty(3 << 20, dtype=torch.uint8,
                                      device="cuda")

    core = _Guarded(rows=8).cuda()
    core.workspace = _Workspace()
    assert frontdoor._memory_delta({"p": core})["bound_bytes"] == 3 << 20


def test_small_memory_moves_stay_out_of_the_reason():
    """The note exists to make a large consequence visible, not to add
    noise to every refusal."""
    assert frontdoor._memory_note({"delta_bytes": 1 << 20}) == ""
    assert "GiB" in frontdoor._memory_note({"delta_bytes": 3 << 30})
    assert frontdoor._memory_note({}) == ""


def test_recording_does_not_change_any_threshold():
    """The decision is the same scalar comparison it always was."""
    import inspect

    source = inspect.getsource(frontdoor.attach)
    assert 'timing["speedup"] < min_speedup' in source
    assert "delta_bytes" not in source.split("# ---- per unit")[1].split(
        "continue")[0].replace("_memory_delta(paths)", "")
