"""Buffered-pread safetensors reader for DGX Spark (mmap is a known loser here).

safetensors' safe_open(device='cpu') memory-maps the shard; on GB10 the
mmap fault path runs at <0.5 GB/s even with readahead tuned, and the page
cache churn fights the container's memory cgroup. Reading the same byte
ranges with plain pread() goes at NVMe speed (~6.6 GB/s single stream).

Parses each shard header once, then serves tensors via os.pread on a kept-open
fd. After finishing a shard pass, call `drop_cache(shard)` / `drop_all()` to
posix_fadvise(DONTNEED) so streamed bytes don't squeeze the cgroup.
"""

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import torch

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


class RawShardReader:
    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.device = device
        idx = json.load(
            open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self._fd = {}
        self._meta = {}      # shard -> {name: (dtype, shape, abs_start, abs_end)}
        self.bytes_read = 0

    def _open(self, shard: str):
        if shard in self._fd:
            return
        path = os.path.join(self.model_dir, shard)
        fd = os.open(path, os.O_RDONLY)
        hlen = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
        hdr = json.loads(os.pread(fd, hlen, 8))
        base = 8 + hlen
        meta = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            meta[name] = (_DTYPES[info["dtype"]], info["shape"],
                          base + s, base + e)
        self._fd[shard] = fd
        self._meta[shard] = meta

    _CHUNK = 1 << 28  # preadv chunk; Linux caps a single read at 2GB-4K

    def get(self, name: str, device: str | None = None) -> torch.Tensor:
        """preadv straight into a preallocated torch buffer: no bytes->
        bytearray copy, no GIL-held memcpy — workers parallelize cleanly."""
        shard = self.weight_map[name]
        self._open(shard)
        dtype, shape, s, e = self._meta[shard][name]
        n = e - s
        fd = self._fd[shard]
        t = torch.empty(shape, dtype=dtype)
        mv = memoryview(t.view(torch.uint8).numpy().reshape(-1))
        off = 0
        while off < n:
            r = os.preadv(fd, [mv[off:off + min(self._CHUNK, n - off)]],
                          s + off)
            if r <= 0:
                raise IOError(f"short read on {name} at {off}/{n}")
            off += r
        self.bytes_read += n
        dev = device if device is not None else self.device
        return t if dev == "cpu" else t.to(dev)

    def get_many(self, names, device: str = "cpu", workers: int = 16):
        """Parallel preadv of many tensors (order preserved). preadv
        releases the GIL and writes in place, so the pool drives the NVMe
        near its streaming limit. Returns CPU tensors by default; the
        caller copies into device-side buffers and frees."""
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda n: self.get(n, device), names))

    def drop_cache(self, shard: str):
        if shard in self._fd:
            try:
                os.posix_fadvise(self._fd[shard], 0, 0,
                                 os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def drop_all(self):
        for shard in list(self._fd):
            self.drop_cache(shard)
