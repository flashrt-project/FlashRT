"""Weight residency: the fallback store lives off the GPU.

The attachment used to keep every replaced module's parameters resident
in device memory so that per-call fallback and ``detach`` were instant.
That is a double weight bill paid exactly when memory is scarcest — at
bind time on a large host — and the only thing it bought was speed of a
path that production (captured, fixed-shape) never takes.

This module removes the resident tier. When an attachment consumes its
originals, each parameter's truth moves to one of two stores and the
device storage is freed:

``DISK``
    The parameter verifiably came from a checkpoint file: its
    safetensors shard and key are recorded, and a sampled-block
    comparison against the live tensor proved the mapping before
    anything was released. Restore reloads from the file and re-applies
    the load transform (a dtype cast). Zero extra copies held.

``HOST_RAM``
    No verifiable provenance (the host mutated the weight after
    loading, or the mapping could not be proven). The tensor is spilled
    to pinned CPU memory; restore is one host-to-device copy. Costs
    host RAM, never device memory.

Correctness never depends on the mapping heuristics: a provenance miss
costs RAM, not accuracy. Tied parameters (shared storage) are stashed
once and restored to every holder.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

_SAMPLE_BYTES = 65536


def _sample_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare shape, dtype and the first/last sample blocks."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    n = min(fa.numel(), _SAMPLE_BYTES // max(1, fa.element_size()))
    if n == 0:
        return True
    return (torch.equal(fa[:n].cpu(), fb[:n].cpu())
            and torch.equal(fa[-n:].cpu(), fb[-n:].cpu()))


@dataclass
class _Ticket:
    tier: str                     # "disk" | "ram"
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    shard: str | None = None      # disk tier
    key: str | None = None
    spill: torch.Tensor | None = None   # ram tier
    storage_key: int = 0          # data_ptr at stash time (tied weights)


@dataclass
class WeightStore:
    """Tiered off-device store for consumed host weights."""

    checkpoint: str | None = None
    _index: dict[str, str] | None = field(default=None, repr=False)
    _by_storage: dict[int, _Ticket] = field(default_factory=dict,
                                            repr=False)
    _stashed: list = field(default_factory=list, repr=False)
    stats: dict[str, Any] = field(default_factory=lambda: {
        "disk": 0, "ram": 0, "freed_bytes": 0, "restored": 0})

    # ---- provenance -------------------------------------------------

    def _load_index(self) -> dict[str, str]:
        if self._index is not None:
            return self._index
        index: dict[str, str] = {}
        if self.checkpoint and os.path.isdir(self.checkpoint):
            idx_path = os.path.join(
                self.checkpoint, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                with open(idx_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
                index = {k: os.path.join(self.checkpoint, v)
                         for k, v in weight_map.items()}
            else:
                for shard in glob.glob(
                        os.path.join(self.checkpoint, "*.safetensors")):
                    from safetensors import safe_open
                    with safe_open(shard, framework="pt") as f:
                        for k in f.keys():
                            index[k] = shard
        self._index = index
        return index

    def _disk_source(self, name: str,
                     tensor: torch.Tensor) -> tuple[str, str] | None:
        """A verified (shard, key) for this parameter, or None."""
        index = self._load_index()
        if not index:
            return None
        candidates = [name]
        # hosts commonly hang the checkpoint tree under one wrapper
        # attribute; try progressively stripped prefixes
        parts = name.split(".")
        for i in range(1, min(4, len(parts))):
            candidates.append(".".join(parts[i:]))
        from safetensors import safe_open
        for key in candidates:
            shard = index.get(key)
            if shard is None:
                continue
            with safe_open(shard, framework="pt") as f:
                disk = f.get_tensor(key)
            if _sample_equal(disk.to(tensor.dtype), tensor):
                return shard, key
        return None

    # ---- stash / restore -------------------------------------------

    @torch.no_grad()
    def stash_module(self, name: str, module: torch.nn.Module) -> int:
        """Move every parameter's truth off the device and free it.

        Returns the number of device bytes freed. Safe to call twice —
        already-emptied parameters are skipped. The module object stays
        whole (shapes, dtypes and attributes remain introspectable);
        only the storage leaves.
        """
        freed = 0
        for mpath, sub in module.named_modules():
            for leaf, par in list(sub._parameters.items()):
                if par is None or par.is_meta or par.numel() == 0:
                    continue
                pname = f"{mpath}.{leaf}" if mpath else leaf
                skey = par.data.data_ptr()
                ticket = self._by_storage.get(skey)
                if ticket is None:
                    full = f"{name}.{pname}" if name else pname
                    source = self._disk_source(full, par.data)
                    if source is not None:
                        ticket = _Ticket(
                            tier="disk", shape=par.shape, dtype=par.dtype,
                            device=par.device, shard=source[0],
                            key=source[1], storage_key=skey)
                        self.stats["disk"] += 1
                    else:
                        spill = torch.empty(
                            par.shape, dtype=par.dtype, device="cpu",
                            pin_memory=torch.cuda.is_available())
                        spill.copy_(par.data)
                        ticket = _Ticket(
                            tier="ram", shape=par.shape, dtype=par.dtype,
                            device=par.device, spill=spill, storage_key=skey)
                        self.stats["ram"] += 1
                    self._by_storage[skey] = ticket
                tickets = getattr(module, "_frt_tickets", None)
                if tickets is None:
                    tickets = {}
                    module._frt_tickets = tickets
                    module._frt_store = self
                    self._stashed.append(module)
                tickets[pname] = ticket
                freed += par.numel() * par.element_size()
                # release to a meta parameter of the SAME shape: the
                # storage is gone, but shape probes (an attention family
                # deriving head counts from a weight) still see the truth
                # — compute on it fails loudly and per-site, instead of
                # collapsing a whole discovery stage on a 1-D empty
                sub._parameters[leaf] = torch.nn.Parameter(
                    torch.empty(par.shape, dtype=par.dtype,
                                device="meta"), requires_grad=False)
        self.stats["freed_bytes"] += freed
        return freed

    @torch.no_grad()
    def restore_module(self, module: torch.nn.Module) -> bool:
        """Put every stashed parameter back on its device. Idempotent."""
        tickets = getattr(module, "_frt_tickets", None)
        if not tickets:
            return False
        subs = dict(module.named_modules())
        for pname, ticket in tickets.items():
            mpath, _, leaf = pname.rpartition(".")
            sub = subs.get(mpath)
            par = None if sub is None else sub._parameters.get(leaf)
            if par is None or not par.is_meta:
                continue
            if ticket.tier == "disk":
                from safetensors import safe_open
                with safe_open(ticket.shard, framework="pt") as f:
                    data = f.get_tensor(ticket.key)
                data = data.to(ticket.device, ticket.dtype)
            else:
                data = ticket.spill.to(ticket.device, non_blocking=False)
            if data.shape != ticket.shape:
                raise RuntimeError(
                    f"weight store: {pname} restored to {tuple(data.shape)}"
                    f", expected {tuple(ticket.shape)}")
            sub._parameters[leaf] = torch.nn.Parameter(
                data, requires_grad=False)
            self.stats["restored"] += 1
        del module._frt_tickets
        del module._frt_store
        return True

    def restore_all(self) -> int:
        """Abort path: put every stashed module's weights back.

        For a bind or attach that dies midway — the model returns to
        runnable, whatever the failure was.
        """
        return _restore_all_of(self)

    def drop_module(self, module: torch.nn.Module) -> None:
        """Forget a module's tickets — its consumption becomes final."""
        if hasattr(module, "_frt_tickets"):
            del module._frt_tickets
        if hasattr(module, "_frt_store"):
            del module._frt_store


def _restore_all_of(store: "WeightStore") -> int:
    count = 0
    for module in list(store._stashed):
        if store.restore_module(module):
            count += 1
    store._stashed.clear()
    return count


def restore_for_fallback(host: torch.nn.Module) -> bool:
    """Bring a consumed host module back for a live fallback.

    Called from the guard path when a seam outside its contract needs
    the host and the host's weights were consumed. Returns True when a
    restore happened. Consumption made final (tickets dropped) leaves
    nothing to restore — the caller refuses instead.
    """
    store = getattr(host, "_frt_store", None)
    if store is None:
        return False
    return store.restore_module(host)


__all__ = ["WeightStore", "restore_for_fallback"]
