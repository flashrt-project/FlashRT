"""Reader for Flint checkpoints in MLX affine-quantized safetensors form (e.g. ``RecursiveSI/flint-int4``).

The export stores every projection, the routed experts and the LM head as affine 4-bit codes with one
BF16 scale and one BF16 bias per group of 64 inputs (``w = scale * q + bias``), the token embedding as
affine 8-bit codes with the same group size, and the router, norms and attention sinks in BF16. Codes
are packed little-endian along the input dimension: for 4-bit, element ``2i`` is the low nibble of byte
``i``. This module only maps and validates the files (numpy, no framework); the hardware runtimes copy
the bytes verbatim into their weight buffers, so the numerics stay those of the export.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from flash_rt.models.flint.config import FlintConfig, FlintConfigError, config_from_json

GROUP_SIZE = 64
_DTYPES = {"U8": np.uint8, "I8": np.int8, "BF16": np.uint16, "F16": np.float16, "F32": np.float32,
           "U32": np.uint32, "I32": np.int32}


class SafetensorsFile:
    """Zero-copy view of one safetensors file (BF16 tensors are returned as raw uint16)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack("<Q", self._map[:8])
        header = json.loads(self._map[8:8 + n])
        header.pop("__metadata__", None)
        self._data = 8 + n
        self.entries: Dict[str, Tuple[str, Tuple[int, ...], int, int]] = {}
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            self.entries[name] = (entry["dtype"], tuple(entry["shape"]), self._data + begin, end - begin)

    def tensor(self, name: str) -> np.ndarray:
        dtype, shape, offset, nbytes = self.entries[name]
        if dtype not in _DTYPES:
            raise ValueError(f"{self.path.name}: unsupported dtype {dtype} for {name}")
        array = np.frombuffer(self._map, dtype=_DTYPES[dtype], count=nbytes // np.dtype(_DTYPES[dtype]).itemsize,
                              offset=offset)
        return array.reshape(shape)

    def close(self) -> None:
        self._map.close()
        self._file.close()


@dataclass(frozen=True)
class AffineMatrix:
    """Affine-quantized matrix ``[rows, k]`` (or ``[experts, rows, k]``) in export layout."""
    codes: np.ndarray    # uint8 [..., rows, k * bits / 8]
    scales: np.ndarray   # uint16 (BF16) [..., rows, k / 64]
    biases: np.ndarray   # uint16 (BF16) [..., rows, k / 64]
    bits: int

    @property
    def k(self) -> int:
        return self.scales.shape[-1] * GROUP_SIZE

    @property
    def rows(self) -> int:
        return self.scales.shape[-2]


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(value: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 bit pattern (finite inputs)."""
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def unpack_codes(m: AffineMatrix) -> np.ndarray:
    if m.bits == 8:
        return m.codes.astype(np.int32)
    if m.bits != 4:
        raise ValueError(f"{m.bits}-bit codes are not supported")
    lo = (m.codes & 0x0F).astype(np.int32)
    hi = (m.codes >> 4).astype(np.int32)
    return np.stack([lo, hi], axis=-1).reshape(*m.codes.shape[:-1], m.codes.shape[-1] * 2)


def dequantize(m: AffineMatrix) -> np.ndarray:
    """FP32 reference ``scale * q + bias`` (tests and probes; the kernels never materialize it)."""
    q = unpack_codes(m).astype(np.float32)
    shape = q.shape
    q = q.reshape(*shape[:-1], shape[-1] // GROUP_SIZE, GROUP_SIZE)
    s = bf16_to_f32(m.scales)[..., None]
    b = bf16_to_f32(m.biases)[..., None]
    return (q * s + b).reshape(shape)


class MLXCheckpoint:
    """A directory with ``config.json``, ``generation_config.json``, tokenizer files and
    ``model-*.safetensors`` shards of an MLX affine export of Flint."""

    def __init__(self, directory):
        self.directory = Path(directory)
        config_path = self.directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"{config_path} not found")
        self.raw_config = json.loads(config_path.read_text())
        gen_path = self.directory / "generation_config.json"
        self.generation = json.loads(gen_path.read_text()) if gen_path.is_file() else None
        self.config: FlintConfig = config_from_json(self.raw_config, self.generation)
        export = self.raw_config.get("mlx_export") or {}
        quant = self.raw_config.get("quantization") or {}
        if quant != {"group_size": GROUP_SIZE, "bits": 4}:
            raise FlintConfigError(f"expected group-64 4-bit affine quantization, found {quant}")
        bits = (export.get("bits"), export.get("routed_expert_bits"), export.get("embedding_bits"))
        if bits != (4, 4, 8):
            raise FlintConfigError(f"expected 4-bit projections/experts and 8-bit embedding, found {bits}")
        shards = sorted(self.directory.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model-*.safetensors in {self.directory}")
        self.files = [SafetensorsFile(p) for p in shards]
        self._index: Dict[str, SafetensorsFile] = {}
        for f in self.files:
            for name in f.entries:
                if name in self._index:
                    raise ValueError(f"tensor {name} appears in more than one shard")
                self._index[name] = f

    def names(self) -> Iterator[str]:
        return iter(self._index)

    def tensor(self, name: str) -> np.ndarray:
        try:
            return self._index[name].tensor(name)
        except KeyError:
            raise KeyError(f"checkpoint has no tensor {name}") from None

    def affine(self, prefix: str, bits: int = 4, k: Optional[int] = None) -> AffineMatrix:
        codes = self.tensor(prefix + ".weight")
        scales = self.tensor(prefix + ".scales")
        biases = self.tensor(prefix + ".biases")
        if codes.dtype != np.uint8 or scales.dtype != np.uint16 or biases.dtype != np.uint16:
            raise ValueError(f"{prefix}: expected uint8 codes and BF16 scales/biases")
        if scales.shape != biases.shape or codes.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"{prefix}: codes/scales/biases shapes disagree")
        kk = scales.shape[-1] * GROUP_SIZE
        if codes.shape[-1] * 8 != kk * bits or (k is not None and kk != k):
            raise ValueError(f"{prefix}: {codes.shape} codes do not hold {bits}-bit x {kk} inputs")
        return AffineMatrix(codes, scales, biases, bits)

    def close(self) -> None:
        for f in self.files:
            f.close()
