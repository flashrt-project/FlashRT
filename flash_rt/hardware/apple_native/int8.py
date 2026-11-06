"""Opt-in dynamic row/channel INT8 lowering of the existing native traversal.

Weights are packed once; activations and their scales update on the GPU each
replay. Optional SmoothQuant observes BF16 inputs during setup. Production
replay needs no calibration cache, host readback or allocation.
"""

import math

import numpy as np
import torch

from .plan import Plan, Tensor, Weight


def smoothquant_factors(activation_max, weight, alpha=0.5):
    """FP32 S[K] for X/S @ (S*W), using already folded W[N,K].

    SmoothQuant (Xiao et al., ICML 2023), arxiv.org/abs/2211.10438.
    Implements the paper's channel balancing equation; no upstream code used.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("SmoothQuant alpha must be in [0,1]")
    a = torch.as_tensor(np.asarray(activation_max, dtype=np.float32).copy())
    if a.shape != (weight.shape[1],) or not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError("SmoothQuant requires finite nonnegative channel maxima [K]")
    w = weight.abs().amax(dim=0).clamp_min(1e-5)
    s = (a.clamp_min(1e-5).pow(alpha) / w.pow(1 - alpha)).clamp_min(1e-5)
    if not torch.isfinite(s).all() or not torch.isfinite(s.reciprocal()).all():
        raise ValueError("Invalid SmoothQuant channel factors")
    return s


class INT8Plan(Plan):
    def __init__(
        self,
        library=None,
        *,
        stages=("decode",),
        observe=False,
        smooth_amax=None,
        smooth_stages=("prefix",),
        smooth_alpha=0.5,
        tune_tiles=True,
        fuse_gelu=True,
    ):
        self.stages = frozenset(stages)
        if not self.stages or not self.stages <= {"vision", "prefix", "decode"}:
            raise ValueError("INT8 stages must select vision, prefix and/or decode")
        self.smooth_stages = frozenset(smooth_stages)
        if not self.smooth_stages <= {"vision", "prefix", "decode"}:
            raise ValueError("Unknown SmoothQuant stage")
        if not math.isfinite(smooth_alpha) or not 0 <= smooth_alpha <= 1:
            raise ValueError("SmoothQuant alpha must be in [0,1]")
        if observe and smooth_amax is not None:
            raise ValueError("Observer cannot consume SmoothQuant calibration")
        self.observe = observe
        self.smooth_amax = (
            None
            if smooth_amax is None
            else {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in smooth_amax.items()
            }
        )
        self.smooth_alpha = smooth_alpha
        self.observers = {}
        self.smooth_inverse = {}
        self.packed_weights = {}
        self.weight_sources = {}
        self.scratch = {}
        self.linears = []
        self.tune_tiles = tune_tiles
        self.fuse_gelu = fuse_gelu
        self._pending_gelu = None
        super().__init__(library)

    def _flush_gelu(self):
        pending, self._pending_gelu = self._pending_gelu, None
        if pending is not None:
            stage, args, kwargs = pending
            current = self.stage
            try:
                self.stage = stage
                super().dispatch(*args, **kwargs)
            finally:
                self.stage = current

    def dispatch(
        self, name, tensors, writes, params, groups, threads=64, *, strided=()
    ):
        self._flush_gelu()
        # One-operation lookahead during plan construction only. Any intervening
        # dispatch/copy or finish flushes the original producer in semantic order.
        if (
            name == "tr_gelu_mul"
            and self.fuse_gelu
            and not self.observe
            and self.stage in self.stages & self.smooth_stages
            and self.smooth_amax is not None
            and params[0] > 16
            and params[1] >= 4096
        ):
            bound = [self.bind(t) for t in tensors]
            if bound[0].buffer != bound[1].buffer:
                self._pending_gelu = (
                    self.stage,
                    (name, bound, writes, list(params), groups, threads),
                    {"strided": strided},
                )
                return
        return super().dispatch(
            name, tensors, writes, params, groups, threads, strided=strided
        )

    def copy_buffer(self, source, destination):
        self._flush_gelu()
        return super().copy_buffer(source, destination)

    def finish(self):
        self._flush_gelu()
        return super().finish()

    def raw(self, array):
        array = np.ascontiguousarray(array)
        bid = self.lib.frt_apple_plan_buffer(self.handle, array.nbytes)
        if bid < 0:
            self.check(-1)
        self.check(
            self.lib.frt_apple_plan_write(
                self.handle, bid, 0, array.ctypes.data, array.nbytes
            )
        )
        return Tensor(self, array.shape, bid, itemsize=array.dtype.itemsize)

    def linear(self, out, x, w, bias=None):
        if not isinstance(w, Weight):
            raise TypeError("INT8 lowering requires a named setup-time weight")
        m, k, n = math.prod(x.shape[:-1]), x.shape[-1], w.shape[-1]
        eligible = self.stage in self.stages and min(k, n) >= 256 and out.contiguous()
        if not eligible:
            self._flush_gelu()
            self.linears.append(
                {"stage": self.stage, "shape": [m, n, k], "precision": "bf16"}
            )
            return super().linear(out, x, w, bias)
        if m <= 0 or w.shape != (k, n) or out.numel() != m * n or not x.contiguous():
            raise ValueError("INT8 GEMM requires contiguous A[M,K], W[K,N], C[M,N]")
        stride = (k + 127) // 128 * 128
        if max(m * stride, n * stride, m * n, k * 127 * 127) > np.iinfo(np.int32).max:
            raise ValueError("INT8 GEMM exceeds INT32 indexing or accumulator range")
        if bias is not None and bias.shape != (n,):
            raise ValueError("INT8 GEMM bias must have shape [N]")
        self.bind(x)
        self.bind(out)
        key = repr(
            (
                self.stage,
                w.name,
                w.tensor.storage_offset(),
                w.shape,
                tuple(w.tensor.stride()),
            )
        )
        source = w.tensor.untyped_storage().data_ptr()
        if key in self.weight_sources and self.weight_sources[key] != source:
            raise ValueError(f"Ambiguous INT8 weight identity: {key}")
        self.weight_sources[key] = source
        if self.observe:
            if key not in self.observers:
                self.observers[key] = self.raw(np.zeros(k, np.float32))
            self.dispatch(
                "int8_observe_channels",
                [x, self.observers[key]],
                {1},
                [m, k],
                ((k + 255) // 256,),
                256,
            )
            self.linears.append(
                {
                    "key": key,
                    "stage": self.stage,
                    "shape": [m, n, k],
                    "precision": "bf16_observed",
                }
            )
            return super().linear(out, x, w, bias)
        smooth = self.smooth_amax is not None and self.stage in self.smooth_stages
        if smooth and key not in self.smooth_amax:
            raise ValueError(f"Missing SmoothQuant activation maxima: {key}")
        if key not in self.packed_weights:
            weight = (
                w.tensor.detach().to(device="cpu", dtype=torch.float32).T.contiguous()
            )
            if not torch.isfinite(weight).all():
                raise ValueError(f"Nonfinite INT8 weight: {key}")
            if smooth:
                factors = smoothquant_factors(
                    self.smooth_amax[key], weight, self.smooth_alpha
                )
                weight = weight * factors[None, :]
                if not torch.isfinite(weight).all():
                    raise ValueError(f"Nonfinite SmoothQuant weight: {key}")
                self.smooth_inverse[key] = self.raw(factors.reciprocal().numpy())
                self.weight_bytes += factors.numel() * 4
            scales = weight.abs().amax(dim=1).div(127).clamp_min(1e-12)
            packed = np.zeros((n, stride), np.int8)
            packed[:, :k] = (
                (weight / scales[:, None])
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
                .numpy()
            )
            self.packed_weights[key] = self.raw(packed), self.raw(scales.numpy())
            self.weight_bytes += packed.nbytes + scales.numel() * 4
        weight, ws = self.packed_weights[key]
        if (m, stride) not in self.scratch:
            self.scratch[m, stride] = (
                self.raw(np.zeros((m, stride), np.int8)),
                self.raw(np.zeros(m, np.float32)),
            )
        qa, activation_scales = self.scratch[m, stride]
        small = m <= 16
        pending = self._pending_gelu
        fused = False
        if pending is not None and smooth and pending[0] == self.stage:
            producer_x, producer_out = pending[1][1]
            fused = (
                producer_out.buffer == x.buffer
                and producer_out.offset == x.offset
                and producer_out.shape == x.shape
                and producer_x.numel() == 2 * m * k
                and pending[1][3] == [m, k]
            )
        if fused:
            self._pending_gelu = None
            self.dispatch(
                "int8_gelu_mul_smooth",
                [producer_x, x, qa, activation_scales, self.smooth_inverse[key]],
                {1, 2, 3},
                [m, k, stride],
                (m,),
                512,
            )
        else:
            self.dispatch(
                ("int8_pack_small" if small else "int8_pack")
                + ("_smooth" if smooth else ""),
                [x, qa, activation_scales]
                + ([self.smooth_inverse[key]] if smooth else []),
                {1, 2},
                [m, k, stride],
                (m if small else (m + 3) // 4,),
                128,
            )
        tm, tn, threads, suffix = (
            (16, 128, 128, "_small") if small else (32, 64, 64, "")
        )
        # Measured large-row projection families; keep all other shapes on the
        # original tile. Selection uses geometry, never model/stage/weight names.
        if self.tune_tiles and 512 <= m <= 1024:
            if (k, n) in ((2048, 32768), (16384, 2048)):
                tm, tn, threads, suffix = 64, 128, 128, "_64x128"
            elif (k, n) == (2048, 2560):
                tm, tn, threads, suffix = 64, 64, 64, "_64x64"
        name = "int8_gemm" + suffix + ("_bias" if bias is not None else "")
        self.dispatch(
            name,
            [qa, weight, out, activation_scales, ws]
            + ([bias] if bias is not None else []),
            {2},
            [m, n, k, stride],
            ((n + tn - 1) // tn, (m + tm - 1) // tm),
            threads,
        )
        self.linears.append(
            {
                "key": key,
                "stage": self.stage,
                "shape": [m, n, k],
                "precision": "int8",
                "smoothquant": smooth,
                "fused_gelu": fused,
                "tile": [tm, tn, threads // 32],
            }
        )

    def activation_maxima(self):
        """Read setup observations; never called by production replay."""
        if not self.observe:
            raise ValueError("Only observer plans produce activation maxima")
        result = {}
        for key, tensor in self.observers.items():
            value = np.empty(tensor.shape, np.float32)
            self.check(
                self.lib.frt_apple_plan_read(
                    self.handle, tensor.buffer, 0, value.ctypes.data, value.nbytes
                )
            )
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite SmoothQuant calibration activation: {key}")
            result[key] = value
        return result
