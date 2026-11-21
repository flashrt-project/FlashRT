"""Inkling 2 Mini (``inkling2_loop``) checkpoint: configuration and a zero-copy safetensors reader (numpy only).

The released checkpoint stores the text decoder under ``model.llm.``: BF16 attention, gates, embedding and LM head;
FP32 depth-attention queries and router biases; routed experts as NVFP4 (``U8`` packed E2M1 codes with ``F8_E4M3``
scales per 16 inputs and per-expert ``amax`` / ``outer_scale_denominator``) in most layers and MXFP8 (``F8_E4M3``
codes with ``U8`` E8M0 scales per 32 inputs) in the last ones. Tensors are returned in their stored bytes (BF16 as
uint16, FP8 / FP4 as uint8).
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PREFIX = "model.llm."

_DTYPES = {"U8": np.uint8, "I8": np.int8, "BF16": np.uint16, "F16": np.float16, "F32": np.float32,
           "U32": np.uint32, "I32": np.int32, "F8_E4M3": np.uint8}

# Geometry the Apple kernels are written for.
GEOMETRY = {
    "num_layers": 24, "hidden_size": 2304, "vocab_size": 201024, "unpadded_vocab_size": 200058,
    "num_heads": 24, "num_kv_heads": 8, "head_dim": 128, "d_rel": 16, "rel_extent": 1024,
    "sliding_window": 1024, "conv_kernel": 4, "layers_per_block": 6, "num_experts": 128,
    "num_experts_per_tok": 8, "expert_intermediate_size": 1152, "loop_start": 6, "loop_end": 18,
    "loop_passes": 3, "gate_rank": 64, "local_layers": (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                                                        18, 19, 20, 21, 22),
}


class Inkling2ConfigError(ValueError):
    """The checkpoint is not the Inkling 2 Mini text decoder the kernels are written for."""


@dataclass(frozen=True)
class Inkling2Config:
    num_layers: int
    hidden_size: int
    vocab_size: int
    unpadded_vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    conv_kernel: int
    layers_per_block: int
    num_experts: int
    num_experts_per_tok: int
    expert_intermediate_size: int
    loop_start: int
    loop_end: int
    loop_passes: int
    gate_rank: int
    local_layers: Tuple[int, ...]
    context_length: int
    eos_token_ids: Tuple[int, ...]
    temperature: float

    def is_global(self, layer: int) -> bool:
        return layer not in self.local_layers

    def is_looped(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def visits(self):
        """(layer, pass) in execution order: prelude, the looped layers once per pass, coda."""
        out = [(i, 0) for i in range(self.loop_start)]
        for p in range(self.loop_passes):
            out += [(i, p) for i in range(self.loop_start, self.loop_end)]
        out += [(i, 0) for i in range(self.loop_end, self.num_layers)]
        return out


def config_from_json(raw: dict, generation: dict | None = None) -> Inkling2Config:
    if raw.get("model_type") != "inkling2_loop":
        raise Inkling2ConfigError(f"model_type {raw.get('model_type')!r}, expected 'inkling2_loop'")
    t = raw.get("text_config") or {}
    loop = t.get("loop_config") or {}
    try:
        for a, b in (("num_attention_heads", "swa_num_attention_heads"),
                     ("num_key_value_heads", "swa_num_key_value_heads"), ("head_dim", "swa_head_dim")):
            if t[a] != t[b]:
                raise Inkling2ConfigError(f"{a} != {b}: sliding and global attention differ in shape")
        if t.get("n_shared_experts", 0) or t.get("q_bias") or t.get("o_bias") or loop.get("is_varloop"):
            raise Inkling2ConfigError("expected no shared expert, no attention biases and a fixed loop count")
        gen = generation or {}
        eos = gen.get("eos_token_id", raw.get("eos_token_id"))
        eos = tuple(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
        return Inkling2Config(
            num_layers=int(t["num_hidden_layers"]), hidden_size=int(t["hidden_size"]),
            vocab_size=int(t["vocab_size"]), unpadded_vocab_size=int(t["unpadded_vocab_size"]),
            num_heads=int(t["num_attention_heads"]), num_kv_heads=int(t["num_key_value_heads"]),
            head_dim=int(t["head_dim"]), d_rel=int(t["d_rel"]), rel_extent=int(t["rel_extent"]),
            sliding_window=int(t["sliding_window_size"]), conv_kernel=int(t["sconv_kernel_size"]),
            layers_per_block=int(t["n_layers_per_attn_res_block"]), num_experts=int(t["n_routed_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]), expert_intermediate_size=int(t["intermediate_size"]),
            loop_start=int(loop["core_start"]), loop_end=int(loop["core_end"]), loop_passes=int(loop["num_passes"]),
            gate_rank=int(loop["gate_rank"]), local_layers=tuple(int(i) for i in t["local_layer_ids"]),
            context_length=int(t["model_max_length"]), eos_token_ids=eos,
            temperature=float(gen.get("temperature", 1.0)))
    except KeyError as e:
        raise Inkling2ConfigError(f"config.json text_config lacks {e}") from None


def check_constants(raw: dict) -> None:
    """Scalar hyperparameters the kernels hard-code."""
    t = raw["text_config"]
    want = {"rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
            "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "route_scale": 8.0, "final_logit_softcapping": 30.0, "use_gate_bias": True,
            "use_value_norm_no_gamma": True, "attn_res_normalize_values": True}
    for key, value in want.items():
        if t.get(key) != value:
            raise Inkling2ConfigError(f"{key}={t.get(key)}; the Apple Inkling 2 kernels are built for {value}")


def check_geometry(cfg: Inkling2Config) -> None:
    for key, value in GEOMETRY.items():
        if getattr(cfg, key) != value:
            raise Inkling2ConfigError(f"{key}={getattr(cfg, key)}; the Apple Inkling 2 kernels are built for {value}")


class SafetensorsFile:
    """Zero-copy view of one safetensors file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack("<Q", self._map[:8])
        header = json.loads(self._map[8:8 + n])
        header.pop("__metadata__", None)
        self.entries: Dict[str, Tuple[str, Tuple[int, ...], int, int]] = {}
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            self.entries[name] = (entry["dtype"], tuple(entry["shape"]), 8 + n + begin, end - begin)

    def tensor(self, name: str) -> np.ndarray:
        dtype, shape, offset, nbytes = self.entries[name]
        if dtype not in _DTYPES:
            raise ValueError(f"{self.path.name}: unsupported dtype {dtype} for {name}")
        np_dtype = np.dtype(_DTYPES[dtype])
        return np.frombuffer(self._map, dtype=np_dtype, count=nbytes // np_dtype.itemsize,
                             offset=offset).reshape(shape)

    def dtype(self, name: str) -> str:
        return self.entries[name][0]

    def close(self) -> None:
        self._map.close()
        self._file.close()


class Inkling2Checkpoint:
    """A directory with ``config.json``, ``generation_config.json``, tokenizer files and ``model-*.safetensors``."""

    def __init__(self, directory):
        self.directory = Path(directory)
        config_path = self.directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"{config_path} not found")
        self.raw_config = json.loads(config_path.read_text())
        gen_path = self.directory / "generation_config.json"
        self.generation = json.loads(gen_path.read_text()) if gen_path.is_file() else None
        self.config = config_from_json(self.raw_config, self.generation)
        check_constants(self.raw_config)
        q = self.raw_config.get("quantization_config") or {}
        if q.get("quant_method") != "modelopt":
            raise Inkling2ConfigError(f"expected the released modelopt NVFP4 / MXFP8 checkpoint, found {q}")
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

    def has(self, name: str) -> bool:
        return PREFIX + name in self._index

    def tensor(self, name: str, dtype: str | None = None, shape=None) -> np.ndarray:
        """Text-decoder tensor ``name`` (without the ``model.llm.`` prefix), checked against the safetensors
        ``dtype`` and ``shape`` when given."""
        key = PREFIX + name
        if key not in self._index:
            raise Inkling2ConfigError(f"checkpoint has no tensor {key}")
        f = self._index[key]
        if dtype is not None and f.dtype(key) != dtype:
            raise Inkling2ConfigError(f"{key}: {f.dtype(key)}, expected {dtype}")
        t = f.tensor(key)
        if shape is not None and t.shape != tuple(shape):
            raise Inkling2ConfigError(f"{key}: shape {t.shape}, expected {tuple(shape)}")
        return t

    def expert_format(self, layer: int) -> str:
        """'nvfp4' or 'mxfp8' for the routed experts of ``layer``."""
        key = f"{PREFIX}layers.{layer}.mlp.experts.w13_weight"
        return "mxfp8" if self._index[key].dtype(key) == "F8_E4M3" else "nvfp4"

    def close(self) -> None:
        for f in self.files:
            f.close()


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
