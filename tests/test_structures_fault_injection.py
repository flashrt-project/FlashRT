"""Fault injection — the negative controls as a public test family.

Every case plants a specific fault and asserts the system names it
instead of computing through it. These are the release's RQ3 exhibits
promoted from scattered probes into contracts.
"""

from __future__ import annotations

import json

from flash_rt.structures.gates import check_env, env_lock, verify_record


def _probe_style_record():
    rec = {"gate": "t", "metrics": {"speedup": 1.5, "tf": 1.0},
           "verdict": "PASS"}
    import hashlib

    rec["plan_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


def test_tampered_metric_fails_verification():
    rec = _probe_style_record()
    assert verify_record(rec)
    rec["metrics"]["speedup"] = 9.9        # the fault
    assert not verify_record(rec)


def test_digestless_record_never_verifies():
    assert not verify_record({"gate": "t", "verdict": "PASS"})


def test_env_drift_is_named_not_silent():
    rec = _probe_style_record()
    rec["env_lock"] = env_lock()
    assert check_env(rec) == []            # same env: re-runnable
    rec["env_lock"]["packages"]["torch"] = "0.0.1"   # the fault
    drift = check_env(rec)
    assert any("torch" in d for d in drift)


def test_missing_lock_is_itself_a_finding():
    assert check_env(_probe_style_record()) == [
        "record carries no env_lock"]


def test_fake_cache_progress_is_the_documented_fault():
    # the multimodal receipts: a static window masquerading as history
    # sends host glue down its continuation branch. The contract pin
    # lives with the family; this case documents it as a fault class.
    from flash_rt.structures.impls.decode_loop.whole_step import (
        _StaticHybridCache)

    c = _StaticHybridCache(2, [0], 1, 4, 16, "cpu")
    assert c.get_seq_length() == 0         # truth, not window size
    assert c.get_mask_sizes(1, 0) == (16, 0)
