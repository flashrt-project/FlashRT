"""Region resolution is a receipt consumer, never an experimenter.

The stability contract under test: the automatic tier obeys author
pin > decision cache > seated, and any receipt naming a form this box
cannot qualify — unknown, or with missing prerequisites — falls
through to the seated floor with the reason on the trail. A wrong
receipt may cost speed; it must never cost correctness or crash a
bind. The writer side is strict instead: a measurement run cannot
record a winner the family never declared.
"""

from __future__ import annotations

import pytest

from flash_rt.structures import decisions, regions


FAMILY = "dit_block"


def _install(monkeypatch, tmp_path, *candidates):
    monkeypatch.setenv("FRT_DECISION_CACHE",
                       str(tmp_path / "decisions.json"))
    monkeypatch.delenv("FRT_REGION_DIT_BLOCK", raising=False)
    fam = regions.RegionFamily(
        FAMILY, identify=lambda model: ("blocks.0",),
        candidates=list(candidates))
    regions.register_region_family(fam)
    return fam


def test_cold_box_runs_the_seated_floor(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    notes = {}
    winner, source = regions.resolve(FAMILY, notes=notes)
    assert (winner, source) == (regions.SEATED, "default")
    trail = notes["regions"][0]
    assert trail["winner"] == regions.SEATED
    assert trail["fell_through"] == []


def test_recorded_winner_seats_from_the_cache(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    regions.record(FAMILY, "fp4_chain", {"fp4_chain": 34.17,
                                         "seated": 36.68})
    winner, source = regions.resolve(FAMILY)
    assert (winner, source) == ("fp4_chain", "cache")


def test_author_pin_outranks_the_cache(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"),
             regions.RegionCandidate("other"))
    regions.record(FAMILY, "fp4_chain", {})
    monkeypatch.setenv("FRT_REGION_DIT_BLOCK", "other")
    winner, source = regions.resolve(FAMILY)
    assert (winner, source) == ("other", "pin")


def test_seated_pin_is_a_valid_author_choice(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    regions.record(FAMILY, "fp4_chain", {})
    monkeypatch.setenv("FRT_REGION_DIT_BLOCK", regions.SEATED)
    winner, source = regions.resolve(FAMILY)
    assert (winner, source) == (regions.SEATED, "pin")


def test_unqualified_receipt_falls_to_seated_with_reason(
        monkeypatch, tmp_path):
    """A receipt measured on a box whose hub package has the symbols
    must not crash a box whose package does not."""
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate(
                 "fp4_chain",
                 missing=lambda: ["nvfp4_gemm_bias_residual"]))
    regions.record(FAMILY, "fp4_chain", {})
    notes = {}
    winner, source = regions.resolve(FAMILY, notes=notes)
    assert (winner, source) == (regions.SEATED, "default")
    fell = notes["regions"][0]["fell_through"]
    assert fell[0]["source"] == "cache"
    assert "nvfp4_gemm_bias_residual" in fell[0]["reason"]


def test_unknown_cached_name_falls_to_seated(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    decisions.record(f"region:{FAMILY}", "renamed_away", {})
    notes = {}
    winner, source = regions.resolve(FAMILY, notes=notes)
    assert (winner, source) == (regions.SEATED, "default")
    assert notes["regions"][0]["fell_through"][0]["reason"] == (
        "unknown_candidate")


def test_bad_pin_falls_through_to_good_cache(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    regions.record(FAMILY, "fp4_chain", {})
    monkeypatch.setenv("FRT_REGION_DIT_BLOCK", "typo")
    notes = {}
    winner, source = regions.resolve(FAMILY, notes=notes)
    assert (winner, source) == ("fp4_chain", "cache")
    assert notes["regions"][0]["fell_through"][0]["source"] == "pin"


def test_the_writer_refuses_undeclared_winners(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    with pytest.raises(ValueError, match="not a candidate"):
        regions.record(FAMILY, "typo", {})
    regions.record(FAMILY, regions.SEATED, {})   # the floor is legal
    assert regions.resolve(FAMILY) == (regions.SEATED, "cache")


def test_region_keys_share_the_band_cache_file(monkeypatch, tmp_path):
    """One file, one transport channel: import_decisions and the
    band entries must coexist with region entries untouched."""
    _install(monkeypatch, tmp_path,
             regions.RegionCandidate("fp4_chain"))
    decisions.record("dit", "fp8", {"fp8": 14.66})
    regions.record(FAMILY, "fp4_chain", {"fp4_chain": 34.17})
    assert decisions.lookup("dit") == "fp8"
    assert decisions.lookup(f"region:{FAMILY}") == "fp4_chain"
