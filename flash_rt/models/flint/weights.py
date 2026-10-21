"""Lossless NVFP4 handling for the official Flint GGUF.

The GGUF stores every quantized matrix as ggml ``block_nvfp4`` superblocks of
64 values: four UE4M3 sub-block scale bytes (one per 16 values) followed by 32
bytes of E2M1 codes. Inside each 16-value sub-block, byte ``j`` holds value
``j`` in its low nibble and value ``j + 8`` in its high nibble ("split halves").
Every tensor ``X.weight`` has a companion f32 ``X.scale`` (one value, or one per
expert for stacked expert tensors) that multiplies the matmul result.

The runtime layout keeps exactly the same information, re-arranged for the
kernels:

* ``codes``: ``uint8 [rows, K/2]``, value ``2i`` in the low nibble and value
  ``2i + 1`` in the high nibble (the ModelOpt/FlashRT packing);
* ``scales``: ``uint8 [rows, K/16]`` UE4M3 bytes, copied verbatim;
* ``global_scale``: the f32 ``.scale`` value(s), applied to the fp32 accumulator.

Dequantization is ``w = e2m1(code) * e4m3(scale) * global_scale``; the first
product is exact in BF16 (at most 6 significant bits), which is what makes the
W4A16 kernels lossless with respect to the checkpoint. The functions here are
the numpy reference of that repack; the device loader produces byte-identical
buffers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# E2M1 magnitudes for codes 0..7; bit 3 is the sign.
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=np.float32)


def ue4m3_table() -> np.ndarray:
    """f32 value of every scale byte, matching the GGUF reference decode.

    Bit 7 is ignored and ``0x7F`` decodes to 0 (ggml ``ue4m3`` semantics); all
    other bytes are standard OCP E4M3 (bias 7, subnormals ``m * 2**-9``).
    """
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        x = b & 0x7F
        if x == 0 or x == 0x7F:
            continue
        e, m = (x >> 3) & 0xF, x & 7
        out[b] = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


_UE4M3 = ue4m3_table()


def split_nvfp4_blocks(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``[rows, K/64, 36]`` GGUF superblocks -> (codes ``[rows, K/2]``, scales ``[rows, K/16]``)."""
    if blocks.ndim != 3 or blocks.shape[2] != 36 or blocks.dtype != np.uint8:
        raise ValueError(f"expected uint8 [rows, K/64, 36] blocks, got {blocks.dtype} {blocks.shape}")
    rows, nb, _ = blocks.shape
    scales = np.ascontiguousarray(blocks[:, :, :4]).reshape(rows, nb * 4)
    qs = blocks[:, :, 4:].reshape(rows, nb, 4, 8)
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)        # natural order, [rows, nb, 4, 16]
    codes = (vals[..., 0::2] | (vals[..., 1::2] << 4)).astype(np.uint8)
    return codes.reshape(rows, nb * 32), scales


def dequantize(codes: np.ndarray, scales: np.ndarray, global_scale: float = 1.0) -> np.ndarray:
    """Reference f32 dequantization of a repacked ``[rows, K/2]`` / ``[rows, K/16]`` matrix."""
    rows, half_k = codes.shape
    k = half_k * 2
    vals = np.empty((rows, k), dtype=np.float32)
    vals[:, 0::2] = E2M1_VALUES[codes & 0x0F]
    vals[:, 1::2] = E2M1_VALUES[codes >> 4]
    s = _UE4M3[scales].repeat(16, axis=1)
    return vals * s * np.float32(global_scale)


def check_scale_bytes(scales: np.ndarray, name: str) -> None:
    """Reject scale bytes whose meaning differs between UE4M3 decoders.

    ``0x7F``/``0xFF`` decode to 0 in ggml but to NaN/480 in other E4M3
    decoders, and bit 7 is ignored by ggml. The official checkpoint contains
    none of them; refusing them keeps every consumer bit-identical.
    """
    bad = (scales & 0x80) | (scales == 0x7F)
    if np.any(bad):
        raise ValueError(f"{name}: scale bytes with bit 7 set or 0x7F are not supported")
