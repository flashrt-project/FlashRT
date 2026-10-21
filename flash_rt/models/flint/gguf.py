"""Independent GGUF v3 container reader for Flint.

GGUF is used here purely as a container: this module parses the header, the
key/value metadata and the tensor table, and exposes every tensor as a
zero-copy ``numpy.memmap`` byte view. It does not depend on llama.cpp or on the
``gguf`` Python package, and it never interprets tensor contents beyond the
storage geometry of the handful of ggml types Flint uses.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

GGUF_MAGIC = b"GGUF"

# ggml storage types used by the official Flint NVFP4 GGUF.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_NVFP4 = 40

# NVFP4 superblock: 64 values = 4 UE4M3 sub-block scales + 32 bytes of codes.
NVFP4_BLOCK_VALUES = 64
NVFP4_BLOCK_BYTES = 36

_TYPE_GEOMETRY = {
    # type: (values per block, bytes per block)
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_BF16: (1, 2),
    GGML_TYPE_NVFP4: (NVFP4_BLOCK_VALUES, NVFP4_BLOCK_BYTES),
}

_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_BF16: "BF16",
    GGML_TYPE_NVFP4: "NVFP4",
}

# GGUF metadata value types.
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_NUMPY_ARRAY_DTYPE = {
    0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
    5: np.int32, 6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64,
    12: np.float64,
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


class GGUFFormatError(ValueError):
    """The file is not a GGUF v3 container this reader can interpret."""


@dataclass(frozen=True)
class GGUFTensor:
    """One entry of the GGUF tensor table.

    ``shape`` follows ggml order (``ne[0]`` is the contiguous dimension), so a
    weight with ``shape == (K, N)`` stores ``N`` rows of ``K`` inputs, i.e. a
    row-major ``[N, K]`` matrix.
    """

    name: str
    ggml_type: int
    shape: Tuple[int, ...]
    offset: int  # absolute byte offset in the file
    nbytes: int

    @property
    def type_name(self) -> str:
        return _TYPE_NAMES.get(self.ggml_type, f"type{self.ggml_type}")

    @property
    def num_values(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def _tensor_nbytes(name: str, ggml_type: int, shape: Tuple[int, ...]) -> int:
    if ggml_type not in _TYPE_GEOMETRY:
        raise GGUFFormatError(
            f"tensor {name!r}: unsupported ggml type {ggml_type}")
    per_block, block_bytes = _TYPE_GEOMETRY[ggml_type]
    if shape[0] % per_block:
        raise GGUFFormatError(
            f"tensor {name!r}: ne[0]={shape[0]} is not a multiple of the "
            f"{_TYPE_NAMES[ggml_type]} block size {per_block}")
    rows = 1
    for d in shape[1:]:
        rows *= d
    return rows * (shape[0] // per_block) * block_bytes


class _Cursor:
    def __init__(self, buf: memoryview) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> memoryview:
        if self.pos + n > len(self.buf):
            raise GGUFFormatError("truncated GGUF header")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        return bytes(self.take(n)).decode("utf-8")

    def value(self, vtype: int) -> Any:
        if vtype in _SCALAR_FMT:
            return self.scalar(_SCALAR_FMT[vtype])
        if vtype == _GGUF_TYPE_STRING:
            return self.string()
        if vtype == _GGUF_TYPE_ARRAY:
            itype = self.scalar("<I")
            count = self.scalar("<Q")
            if itype in _NUMPY_ARRAY_DTYPE:
                dt = np.dtype(_NUMPY_ARRAY_DTYPE[itype])
                raw = self.take(count * dt.itemsize)
                return np.frombuffer(raw, dtype=dt).copy()
            return [self.value(itype) for _ in range(count)]
        raise GGUFFormatError(f"unknown GGUF metadata value type {vtype}")


class GGUFFile:
    """Read-only, memory-mapped view of a GGUF v3 file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "rb")
        self.file_size = os.fstat(self._fh.fileno()).st_size
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        cur = _Cursor(memoryview(self._mm))
        if bytes(cur.take(4)) != GGUF_MAGIC:
            raise GGUFFormatError(f"{path}: not a GGUF file")
        self.version = cur.scalar("<I")
        if self.version != 3:
            raise GGUFFormatError(f"{path}: GGUF version {self.version}, need 3")
        n_tensors = cur.scalar("<Q")
        n_kv = cur.scalar("<Q")

        self.metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = cur.string()
            self.metadata[key] = cur.value(cur.scalar("<I"))

        infos: List[Tuple[str, int, Tuple[int, ...], int]] = []
        for _ in range(n_tensors):
            name = cur.string()
            n_dims = cur.scalar("<I")
            shape = tuple(cur.scalar("<Q") for _ in range(n_dims))
            ggml_type = cur.scalar("<I")
            rel = cur.scalar("<Q")
            infos.append((name, ggml_type, shape, rel))

        align = int(self.metadata.get("general.alignment", 32))
        self.alignment = align
        self.data_offset = (cur.pos + align - 1) // align * align

        self.tensors: Dict[str, GGUFTensor] = {}
        for name, ggml_type, shape, rel in infos:
            if rel % align:
                raise GGUFFormatError(f"tensor {name!r}: misaligned offset {rel}")
            nbytes = _tensor_nbytes(name, ggml_type, shape)
            off = self.data_offset + rel
            if off + nbytes > self.file_size:
                raise GGUFFormatError(f"tensor {name!r}: extends past end of file")
            if name in self.tensors:
                raise GGUFFormatError(f"duplicate tensor {name!r}")
            self.tensors[name] = GGUFTensor(name, ggml_type, shape, off, nbytes)
        self._raw = np.frombuffer(self._mm, dtype=np.uint8)

    # -- metadata -----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.metadata:
            raise GGUFFormatError(f"{self.path}: missing metadata key {key!r}")
        return self.metadata[key]

    # -- tensors ------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __iter__(self) -> Iterator[GGUFTensor]:
        return iter(self.tensors.values())

    def tensor(self, name: str) -> GGUFTensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise GGUFFormatError(f"{self.path}: missing tensor {name!r}") from None

    def raw_bytes(self, name: str) -> np.ndarray:
        """Zero-copy uint8 view of a tensor's stored bytes."""
        t = self.tensor(name)
        return self._raw[t.offset:t.offset + t.nbytes]

    def array(self, name: str) -> np.ndarray:
        """Zero-copy typed view for F32/F16/BF16 tensors (BF16 as uint16 bits).

        The returned array is shaped in row-major order, i.e. ``shape`` reversed.
        """
        t = self.tensor(name)
        dt = {GGML_TYPE_F32: np.float32, GGML_TYPE_F16: np.float16,
              GGML_TYPE_BF16: np.uint16}.get(t.ggml_type)
        if dt is None:
            raise GGUFFormatError(f"tensor {name!r} is {t.type_name}, not a plain array")
        return self.raw_bytes(name).view(dt).reshape(tuple(reversed(t.shape)))

    def nvfp4_blocks(self, name: str) -> np.ndarray:
        """NVFP4 tensor as ``[rows, K/64, 36]`` uint8 superblocks (zero-copy)."""
        t = self.tensor(name)
        if t.ggml_type != GGML_TYPE_NVFP4:
            raise GGUFFormatError(f"tensor {name!r} is {t.type_name}, not NVFP4")
        rows = t.num_values // t.shape[0]
        return self.raw_bytes(name).reshape(rows, t.shape[0] // NVFP4_BLOCK_VALUES,
                                            NVFP4_BLOCK_BYTES)

    def close(self) -> None:
        self._raw = None
        try:
            self._mm.close()
        except BufferError:
            # Live numpy views still reference the map; the OS reclaims it on exit.
            pass
        self._fh.close()

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
