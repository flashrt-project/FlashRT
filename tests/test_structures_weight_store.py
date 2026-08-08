"""Weight residency is a lifecycle, and every step answers for itself.

attach → validate → consume → [finalize]. After consume the replaced
originals hold no device storage; their truth lives in the store —
the checkpoint file when provenance verifies, host memory otherwise.
detach restores bit-exact from either tier; a fallback on a consumed
seam restores on demand and the ledger says so; a seat that declares
its retained host as serving keeps it whole; finalize makes the
consumption permanent and flips fallback to refusal.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_rt.structures.guard import CAST_OK, PROCEED, GuardedSeam
from flash_rt.structures.storage import WeightStore
from flash_rt.structures.swap import attach


class FakeSeam(GuardedSeam, nn.Module):
    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, host: nn.Linear):
        super().__init__()
        self.host_linear = host
        self.scale = 2.0
        self._frt_arm(dtypes=CAST_OK, device=host.weight.device,
                      k=host.in_features, rows=2)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        return torch.nn.functional.linear(
            x, self.host_linear.weight) * 0 + x[..., :1] * self.scale


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x):
        return self.proj(x)


def test_consume_frees_and_detach_restores_bitwise():
    host = Host().eval()
    before = host.proj.weight.detach().clone()
    orig = host.proj
    handle = attach(host, {"proj": FakeSeam(host.proj)})
    receipt = handle.consume()
    assert receipt["consumed"] and receipt["freed_bytes"] > 0
    assert orig.weight.is_meta
    handle.detach()
    assert torch.equal(host.proj.weight, before)
    assert host.proj.bias.numel() == 3


def test_fallback_on_consumed_seam_restores_on_demand():
    host = Host().eval()
    seam = FakeSeam(host.proj)
    handle = attach(host, {"proj": seam})
    handle.consume()
    assert seam.host_linear.weight.is_meta
    # a wrong-rows input violates the contract (the host can still
    # run it): the guard restores the host from the store and runs it
    out = host.proj(torch.randn(3, 4))
    assert out.shape[-1] == 3
    assert not seam.host_linear.weight.is_meta
    entry = handle.report()["proj"]
    assert entry["fallbacks"] == 1
    assert entry["notes"]["restored_for_fallback"] == 1


def test_finalized_consumption_refuses_fallback():
    host = Host().eval()
    seam = FakeSeam(host.proj)
    handle = attach(host, {"proj": seam})
    handle.consume()
    handle.finalize()
    # match by message: suite-order module reloads can split the
    # GuardRefused class identity (pre-existing flake family)
    with pytest.raises(Exception, match="no equivalent host module"):
        host.proj(torch.randn(3, 4))
    with pytest.raises(RuntimeError):
        handle.detach()


def test_serving_host_is_kept_whole():
    class ServingSeam(FakeSeam):
        _frt_host_serving = True

    host = Host().eval()
    handle = attach(host, {"proj": ServingSeam(host.proj)})
    receipt = handle.consume()
    assert receipt["kept_serving"] == 1
    assert not host.proj.host_linear.weight.is_meta
    handle.detach()


def test_disk_tier_restores_from_checkpoint(tmp_path):
    from safetensors.torch import save_file

    host = Host().eval()
    save_file({"proj.weight": host.proj.weight.detach().contiguous(),
               "proj.bias": host.proj.bias.detach().contiguous()},
              str(tmp_path / "model.safetensors"))
    before = host.proj.weight.detach().clone()
    store = WeightStore(checkpoint=str(tmp_path))
    handle = attach(host, {"proj": FakeSeam(host.proj)}, store=store)
    handle.consume()
    assert store.stats["disk"] == 2 and store.stats["ram"] == 0
    handle.detach()
    assert torch.equal(host.proj.weight, before)


def test_mutated_weight_falls_to_ram_tier(tmp_path):
    from safetensors.torch import save_file

    host = Host().eval()
    save_file({"proj.weight": host.proj.weight.detach().contiguous(),
               "proj.bias": host.proj.bias.detach().contiguous()},
              str(tmp_path / "model.safetensors"))
    with torch.no_grad():
        host.proj.weight += 1.0        # provenance no longer verifies
    mutated = host.proj.weight.detach().clone()
    store = WeightStore(checkpoint=str(tmp_path))
    handle = attach(host, {"proj": FakeSeam(host.proj)}, store=store)
    handle.consume()
    assert store.stats["ram"] >= 1
    handle.detach()
    assert torch.equal(host.proj.weight, mutated)


def test_decision_import_merges_and_local_wins(tmp_path, monkeypatch):
    from flash_rt.structures import decisions

    monkeypatch.setenv("FRT_DECISION_CACHE", str(tmp_path / "local.json"))
    decisions.record("groot_dit", "fp8", {"fp8": 1.0})
    key = decisions._key("groot_dit")
    remote = {key: {"winner": "fp4", "times_ms": {"fp4": 0.5}},
              "OtherBox|groot_dit": {"winner": "fp4", "times_ms": {}}}
    src = tmp_path / "remote.json"
    src.write_text(__import__("json").dumps(remote))
    receipt = decisions.import_decisions(str(src))
    assert receipt["imported"] == 1          # the foreign-device entry
    assert decisions.lookup("groot_dit") == "fp8"   # local measurement wins


def test_manifest_is_one_serializable_document():
    import json

    host = Host().eval()
    handle = attach(host, {"proj": FakeSeam(host.proj)})
    handle.consume()
    m = handle.manifest()
    assert "proj" in m["seams"]
    assert m["records"]["consumed"]["freed_bytes"] > 0
    json.dumps(m, default=str)   # must serialize whole
    handle.detach()
