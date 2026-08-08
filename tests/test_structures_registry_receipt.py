"""The two seams that let structures land in parallel and be accepted.

The binder registry routes a structure's seams to a binder registered
from the structure's own module, so adding a structure does not edit the
central routing function. The receipt's environment fingerprint makes
two measurements comparable — a figure without the versions it was
measured under cannot be reviewed.
"""

import json

import pytest

from flash_rt.structures import autobuild
from flash_rt.structures.frontdoor import _environment


class _Seam:
    structure = "test_structure"
    path = "layers.0.thing"
    variant = {}


def test_registered_binder_is_consulted_first():
    calls = {}

    def binder(model, seam, cap, *, points, fmt, fmt_params):
        calls["seam"] = seam.path
        calls["fmt"] = fmt
        calls["fmt_params"] = fmt_params
        return "bound-sentinel"

    autobuild.register_structure_binder("test_structure", binder)
    try:
        out = autobuild._bind_auto(None, _Seam(), {}, None, {}, False,
                                   points=None, fmt="some_format",
                                   fmt_params={"alpha": 0.5})
        assert out == "bound-sentinel"
        assert calls == {"seam": "layers.0.thing", "fmt": "some_format",
                         "fmt_params": {"alpha": 0.5}}
    finally:
        autobuild._STRUCTURE_BINDERS.pop("test_structure", None)


def test_unregistered_structure_still_walls_on_unknown_format():
    with pytest.raises(ValueError, match="no impl variant"):
        autobuild._bind_auto(None, _Seam(), {}, None, {}, False,
                             points=None, fmt="some_format")


def test_environment_fingerprint_is_json_and_versioned():
    env = _environment()
    assert "python" in env and "torch" in env
    json.dumps(env)                     # must serialise as-is
    # transformers is optional, but when present it must be a version
    if "transformers" in env:
        assert env["transformers"][0].isdigit()
