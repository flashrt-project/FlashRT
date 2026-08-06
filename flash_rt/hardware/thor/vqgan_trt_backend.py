"""TensorRT VQ-GAN encoder backend for Jetson Thor.

Manages multiple fixed-shape TRT engines (one per resolution).
Lazily loads/deserializes engines on first use of each resolution.
Falls back gracefully if engines are unavailable.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    pass


class VQGANTRTBackend:
    ENGINE_DIR = Path.home() / ".flash_rt" / "trt_engines" / "vqgan"

    def __init__(self, engine_dir: Optional[Path] = None):
        self._engine_dir = Path(engine_dir) if engine_dir else self.ENGINE_DIR
        self._manifest = None
        self._engines = {}
        self._contexts = {}
        self._buffers = {}
        self._available = None
        self._trt_logger = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not _TRT_AVAILABLE:
            logger.warning("tensorrt not importable; TRT VQGAN disabled")
            self._available = False
            return False
        manifest_path = self._engine_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("No TRT VQGAN manifest at %s", manifest_path)
            self._available = False
            return False
        with open(manifest_path) as f:
            self._manifest = json.load(f)
        has_engines = len(self._manifest.get("engines", {})) > 0
        if has_engines:
            logger.info("TRT VQGAN backend: %d engines at %s (TRT %s)",
                        len(self._manifest["engines"]), self._engine_dir,
                        self._manifest.get("trt_version", "?"))
        self._available = has_engines
        return has_engines

    def supports_resolution(self, height: int, width: int) -> bool:
        if not self._available:
            return False
        key = f"{height}x{width}"
        return key in self._manifest.get("engines", {})

    def encode(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Run TRT VQGAN encoder.

        Args:
            image_tensor: [1, 3, H, W] float32 on CUDA, range [-1, 1]

        Returns:
            [1, H//16, W//16] int64 codebook indices on CUDA, or None on failure.
        """
        _, _, h, w = image_tensor.shape
        key = f"{h}x{w}"

        if key not in self._contexts:
            if not self._load_engine(h, w):
                return None

        ctx = self._contexts[key]
        inp_buf, out_buf, out_dtype_is_int32 = self._buffers[key]

        inp_buf.copy_(image_tensor)
        ctx.set_tensor_address("image", inp_buf.data_ptr())
        ctx.set_tensor_address("indices", out_buf.data_ptr())

        stream = torch.cuda.current_stream()
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            logger.error("TRT execute_async_v3 failed for resolution %s", key)
            return None

        if out_dtype_is_int32:
            return out_buf.to(torch.int64)
        return out_buf.clone()

    def encode_batch(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Run TRT VQGAN encoder on a multi-view batch.

        Args:
            image_tensor: [B, 3, H, W] float32 on CUDA, range [-1, 1]
                where B == engine batch (from per-engine manifest entry).

        Returns:
            [B, H//16, W//16] int64 codebook indices, or None on failure /
            batch mismatch.
        """
        b, _, h, w = image_tensor.shape
        key = f"{h}x{w}"

        if key not in self._contexts:
            if not self._load_engine(h, w):
                return None

        engine_batch = self._manifest["engines"][key]["input_shape"][0]
        if b != engine_batch:
            return None  # caller should fall back to per-view encode

        ctx = self._contexts[key]
        inp_buf, out_buf, out_dtype_is_int32 = self._buffers[key]

        inp_buf.copy_(image_tensor)
        ctx.set_tensor_address("image", inp_buf.data_ptr())
        ctx.set_tensor_address("indices", out_buf.data_ptr())

        stream = torch.cuda.current_stream()
        ok = ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            logger.error("TRT execute_async_v3 (batch=%d) failed at %s", b, key)
            return None

        if out_dtype_is_int32:
            return out_buf.to(torch.int64)
        return out_buf.clone()

    def _load_engine(self, height: int, width: int) -> bool:
        key = f"{height}x{width}"
        meta = self._manifest["engines"].get(key)
        if meta is None:
            return False

        engine_path = self._engine_dir / meta["file"]
        if not engine_path.exists():
            logger.error("Engine file missing: %s", engine_path)
            return False

        if self._trt_logger is None:
            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            if hasattr(trt, "init_libnvinfer_plugins"):
                trt.init_libnvinfer_plugins(self._trt_logger, "")

        runtime = trt.Runtime(self._trt_logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            logger.error("Failed to deserialize TRT engine: %s", engine_path)
            return False

        context = engine.create_execution_context()
        self._engines[key] = engine
        self._contexts[key] = context

        self._allocate_buffers(key, height, width)
        logger.info("TRT VQGAN engine loaded: %s (%s)", key, engine_path.name)
        return True

    def _allocate_buffers(self, key: str, height: int, width: int):
        # Per-engine batch (each res may have been built independently with
        # a different batch); the top-level manifest["batch"] is only the
        # last-built entry and unreliable when engines coexist.
        batch = self._manifest["engines"][key]["input_shape"][0]
        h_lat, w_lat = height // 16, width // 16
        device = torch.device("cuda")

        inp_buf = torch.empty(batch, 3, height, width, device=device, dtype=torch.float32)

        engine = self._engines[key]
        out_dtype_trt = engine.get_tensor_dtype("indices")
        out_dtype_is_int32 = (out_dtype_trt == trt.DataType.INT32)
        if out_dtype_is_int32:
            out_buf = torch.empty(batch, h_lat, w_lat, device=device, dtype=torch.int32)
        else:
            out_buf = torch.empty(batch, h_lat, w_lat, device=device, dtype=torch.int64)

        self._buffers[key] = (inp_buf, out_buf, out_dtype_is_int32)
