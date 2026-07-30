#!/usr/bin/env python3
"""A bounded, streaming cache for routed-expert blocks.

The target device holds only a fraction of the experts, so the cache is the
runtime: what it can hold and how fast it can refill decide both token rate and
time to first token. Three properties follow from that and are enforced here
rather than left to convention.

**The budget is a hard limit, not a projection.** On a unified-memory device the
weights, the cache, the staging buffers and the operating system all draw on the
same physical memory, so a runtime that merely intends to stay small is not
measurable. Construction computes its own footprint and refuses to allocate if
it would exceed the budget.

**Reads bypass the page cache.** Streaming tens of GiB of blocks through the
page cache would make it compete with the resident weights for that same
memory. Reads use ``O_DIRECT``, which is why the bundle pads each block to a
4096-byte boundary.

**Misses are fetched concurrently.** A single reader leaves a large part of an
NVMe device idle; measured on one, four readers were worth 1.7x over one and
eight saturated it. ``get_many`` issues a layer's misses together.

A per-layer quota of at least ``num_experts_per_token`` also makes a class of
bug structurally impossible: the experts one token needs cannot evict each
other, so a caller may hold several pointers from the same ``get_many`` at once.
"""

from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class CacheConfig:
    """Sizing and placement. ``budget_bytes`` of 0 disables the check."""

    bundle: Path
    slots_per_layer: int
    staging_buffers: int = 4
    budget_bytes: int = 0
    reserve_bytes: int = 0
    resident_bytes: int = 0
    device: str = "cuda:0"
    experts_per_token: int = 8
    read_chunk: int = 1 << 28
    metadata: dict = field(default_factory=dict)


class CacheBudgetError(RuntimeError):
    """The requested cache does not fit the budget it was given."""


def _load_manifest(bundle: Path) -> dict:
    path = bundle / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"expert bundle is missing {path}")
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("block_bytes", "block_alignment", "num_layers",
                "num_experts", "block_sizes"):
        if key not in manifest:
            raise ValueError(f"{path} has no {key!r}")
    block_bytes = int(manifest["block_bytes"])
    alignment = int(manifest["block_alignment"])
    if block_bytes % alignment:
        raise ValueError(
            f"{path}: block_bytes {block_bytes} is not a multiple of "
            f"{alignment}, so direct reads of it are impossible")
    return manifest


class ExpertCache:
    """Per-layer LRU over fixed-size blocks read straight from storage."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.manifest = _load_manifest(config.bundle)
        self.block_bytes = int(self.manifest["block_bytes"])
        self.alignment = int(self.manifest["block_alignment"])
        self.num_layers = int(self.manifest["num_layers"])
        self.num_experts = int(self.manifest["num_experts"])

        if config.slots_per_layer < config.experts_per_token:
            raise ValueError(
                f"slots_per_layer={config.slots_per_layer} is below "
                f"experts_per_token={config.experts_per_token}; one token's "
                "experts would evict each other and a caller could not hold "
                "their pointers at once")

        self.footprint = self.plan(config, self.manifest)
        if config.budget_bytes:
            total = self.footprint["projected_bytes"]
            if total > config.budget_bytes:
                raise CacheBudgetError(
                    f"cache needs {total / 2**30:.3f} GiB "
                    f"(slots {self.footprint['slot_bytes'] / 2**30:.3f} + "
                    f"staging {self.footprint['staging_bytes'] / 2**30:.3f} + "
                    f"resident {config.resident_bytes / 2**30:.3f} + "
                    f"reserve {config.reserve_bytes / 2**30:.3f}) but the "
                    f"budget is {config.budget_bytes / 2**30:.3f} GiB. "
                    f"Reduce slots_per_layer below "
                    f"{self.max_slots_per_layer(config, self.manifest)}.")

        self._total_slots = config.slots_per_layer * self.num_layers
        self.slots = torch.empty(
            self._total_slots, self.block_bytes,
            dtype=torch.uint8, device=config.device)
        self._staging = [
            torch.empty(self.block_bytes, dtype=torch.uint8).pin_memory()
            for _ in range(config.staging_buffers)
        ]
        for buffer in self._staging:
            if buffer.data_ptr() % self.alignment:
                raise RuntimeError(
                    "a pinned staging buffer is not "
                    f"{self.alignment}-byte aligned, which direct reads "
                    "require")
        self._pool = ThreadPoolExecutor(max_workers=config.staging_buffers)

        # Per-layer LRU of expert -> slot index, and that layer's free slots.
        self._lru: list[OrderedDict[int, int]] = [
            OrderedDict() for _ in range(self.num_layers)]
        self._free: list[list[int]] = [
            list(range(layer * config.slots_per_layer,
                       (layer + 1) * config.slots_per_layer))
            for layer in range(self.num_layers)
        ]
        self._fds: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0

    # ── sizing, answerable before anything is allocated ──

    @staticmethod
    def plan(config: CacheConfig, manifest: dict) -> dict[str, int]:
        """What this configuration would occupy."""
        block_bytes = int(manifest["block_bytes"])
        slot_bytes = (
            config.slots_per_layer * int(manifest["num_layers"]) * block_bytes)
        staging_bytes = config.staging_buffers * block_bytes
        return {
            "block_bytes": block_bytes,
            "slots_per_layer": config.slots_per_layer,
            "slot_bytes": slot_bytes,
            "staging_bytes": staging_bytes,
            "resident_bytes": config.resident_bytes,
            "reserve_bytes": config.reserve_bytes,
            "projected_bytes": (
                slot_bytes + staging_bytes
                + config.resident_bytes + config.reserve_bytes),
        }

    @staticmethod
    def max_slots_per_layer(config: CacheConfig, manifest: dict) -> int:
        """Largest per-layer quota that fits ``config.budget_bytes``."""
        if not config.budget_bytes:
            return int(manifest["num_experts"])
        block_bytes = int(manifest["block_bytes"])
        available = (
            config.budget_bytes - config.resident_bytes
            - config.reserve_bytes - config.staging_buffers * block_bytes)
        if available <= 0:
            return 0
        return min(
            int(manifest["num_experts"]),
            available // (block_bytes * int(manifest["num_layers"])))

    # ── reading ──

    def _fd(self, layer: int) -> int:
        if layer not in self._fds:
            path = self.config.bundle / f"experts_layer_{layer:02d}.bin"
            expected = self.num_experts * self.block_bytes
            size = path.stat().st_size
            if size != expected:
                raise ValueError(
                    f"{path} is {size} bytes; expected {expected} "
                    f"({self.num_experts} x {self.block_bytes})")
            self._fds[layer] = os.open(
                path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        return self._fds[layer]

    def _fetch(self, layer: int, expert: int, slot: int, buffer: int) -> None:
        staging = self._staging[buffer]
        view = memoryview(staging.numpy())
        fd = self._fd(layer)
        base = expert * self.block_bytes
        offset = 0
        while offset < self.block_bytes:
            length = min(self.config.read_chunk, self.block_bytes - offset)
            read = os.preadv(fd, [view[offset:offset + length]], base + offset)
            if read <= 0:
                raise IOError(
                    f"short read of layer {layer} expert {expert} at "
                    f"{offset}/{self.block_bytes}")
            offset += read
        self.slots[slot].copy_(staging)
        self.bytes_read += self.block_bytes

    def _claim(self, layer: int, expert: int) -> int:
        """A slot for an expert not currently held, evicting if necessary."""
        free = self._free[layer]
        if free:
            slot = free.pop()
        else:
            slot = self._lru[layer].popitem(last=False)[1]
        self._lru[layer][expert] = slot
        return slot

    def get_many(self, layer: int, experts) -> list[int]:
        """Device pointers for several experts of one layer, misses in parallel.

        With ``slots_per_layer >= experts_per_token`` none of the returned
        pointers can be invalidated by the others.
        """
        wanted = list(dict.fromkeys(int(expert) for expert in experts))
        if len(wanted) > self.config.slots_per_layer:
            raise ValueError(
                f"asked for {len(wanted)} experts of layer {layer} but the "
                f"quota is {self.config.slots_per_layer}")
        pending = []
        for expert in wanted:
            slot = self._lru[layer].get(expert)
            if slot is not None:
                self._lru[layer].move_to_end(expert)
                self.hits += 1
                continue
            self.misses += 1
            pending.append((expert, self._claim(layer, expert)))
        if pending:
            futures = [
                self._pool.submit(
                    self._fetch, layer, expert, slot,
                    index % len(self._staging))
                for index, (expert, slot) in enumerate(pending)
            ]
            for future in futures:
                future.result()
            torch.cuda.synchronize(self.config.device)
        return [
            int(self.slots[self._lru[layer][expert]].data_ptr())
            for expert in wanted
        ]

    def get(self, layer: int, expert: int) -> int:
        return self.get_many(layer, (expert,))[0]

    # ── startup ──

    def warm(self, frequency: list[Counter]) -> int:
        """Preload each layer's most frequently selected experts.

        Entries stay evictable. Measured on held-out prompts, a set built from
        unrelated traffic removes about a quarter of decode misses and half of
        the cold prefill read at this quota; pinning it instead costs more
        adaptivity than it gains.
        """
        if len(frequency) != self.num_layers:
            raise ValueError(
                f"frequency has {len(frequency)} layers, expected "
                f"{self.num_layers}")
        loaded = 0
        for layer in range(self.num_layers):
            experts = [
                expert for expert, _ in
                frequency[layer].most_common(self.config.slots_per_layer)
            ]
            for start in range(0, len(experts), self.config.slots_per_layer):
                chunk = experts[start:start + self.config.slots_per_layer]
                self.get_many(layer, chunk)
                loaded += len(chunk)
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        return loaded

    # ── reporting ──

    def stats(self) -> dict[str, float]:
        requests = self.hits + self.misses
        report = dict(self.footprint)
        report.update({
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / requests if requests else 0.0,
            "bytes_read": self.bytes_read,
            "resident_experts": sum(len(lru) for lru in self._lru),
        })
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.config.device)
            report.update({
                "device_free_bytes": free,
                "device_total_bytes": total,
                "torch_allocated_bytes": torch.cuda.memory_allocated(
                    self.config.device),
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    self.config.device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(
                    self.config.device),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                    self.config.device),
            })
        return report

    def close(self) -> None:
        """Release everything, including the slots.

        The slot array is the largest single allocation a runtime makes, so a
        close that only dropped file descriptors would leak the entire cache
        on any reconfiguration -- on a device where the budget is the whole
        point, that is not a detail. After this the cache is unusable.
        """
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for lru in self._lru:
            lru.clear()
        self._free = [[] for _ in range(self.num_layers)]
        self.slots = None
        self._staging = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
