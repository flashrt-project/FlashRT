"""Shared workspaces and the finalize tier.

The H3 lesson as tests: seat scratch must pool (one layer's worth, not
layers x tokens), a verified attachment must be able to release its
held originals, and a finalized attachment must refuse to detach.
"""

import pytest
import torch

from flash_rt.structures import workspace
from flash_rt.structures.swap import attach


def test_same_shape_and_tag_share_one_buffer():
    workspace.clear()
    a = workspace.lease((8, 16), torch.float32, "cpu", tag="qkv_stash")
    b = workspace.lease((8, 16), torch.float32, "cpu", tag="qkv_stash")
    c = workspace.lease((8, 32), torch.float32, "cpu", tag="qkv_stash")
    assert a.data_ptr() == b.data_ptr()
    assert c.data_ptr() != a.data_ptr()
    rep = workspace.report()
    assert rep["leases"]["qkv_stash"] == 3
    assert rep["held_bytes"] == (8 * 16 + 8 * 32) * 4
    workspace.clear()


def test_ones_fill_is_constant():
    workspace.clear()
    ones = workspace.lease((4,), torch.float32, "cpu",
                           tag="producer_ones", fill="ones")
    assert bool((ones == 1).all())
    workspace.clear()


def _tiny_host():
    host = torch.nn.Sequential(torch.nn.Linear(4, 4)).eval()
    replacement = torch.nn.Linear(4, 4)
    return host, replacement


def test_finalize_frees_originals_and_forbids_detach():
    host, repl = _tiny_host()
    original = host[0]
    handle = attach(host, {"0": repl})
    receipt = handle.finalize()
    assert receipt["freed_bytes"] > 0
    assert original.weight.is_meta or original.weight.numel() == 0
    with pytest.raises(RuntimeError):
        handle.detach()


def test_detach_still_works_before_finalize():
    host, repl = _tiny_host()
    original = host[0]
    handle = attach(host, {"0": repl})
    handle.detach()
    assert host[0] is original
    assert original.weight.numel() == 16


def test_pack_slots_never_collide():
    """The k-stash and v-stash of one pack share shape and lifetime
    window — they must never share storage (the 0.988-parity lesson)."""
    workspace.clear()
    k = workspace.lease((41, 1536), torch.float32, "cpu",
                        tag="qkv_stash1")
    v = workspace.lease((41, 1536), torch.float32, "cpu",
                        tag="qkv_stash2")
    assert k.data_ptr() != v.data_ptr()
    workspace.clear()
