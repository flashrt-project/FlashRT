"""BF16 native plan with geometry-selected GEMM tiles and fused GEMM epilogues.

Lowering stays the shared model traversal. During recording only, a GEMM is
held for one or two operations: when the following dispatches are its sole
elementwise consumers, they are emitted together.

* Large-row GEMMs (M > 16) fuse GELU multiply of the merged gate/up product,
  an in-place residual add, or an in-place GELU after bias into the epilogue.
* Few-row GEMMs (M <= 16, the denoising rows) are split along K into FP32
  partials so the weight stream keeps enough threadgroups in flight; the
  partials are summed by the consumer: RoPE with K/V placement, or a (gated)
  residual add optionally followed by its AdaRMS normalization. The merged
  gate/up product uses the GELU multiply epilogue directly.

With ``coprocessor`` set, a large-row gated FFN (merged gate/up GEMM, GELU multiply,
down GEMM, in-place residual) in the selected stages is split by hidden channels: the
GPU keeps the leading channels, and a Core ML program (Neural Engine) computes the
trailing channels' FP16 partial of the down projection in the same command buffer.
The GPU packs the FFN input to the program's channel-major FP16 layout, signals the
host, computes its share, waits for the program and adds the partial before the single
BF16 rounding of the down output. This changes numerics (FP16 on the coprocessor share)
and is therefore opt-in.

Every fused form keeps the BF16 rounding boundaries of the separate dispatches;
split-K sums its partials in a fixed order, so replay stays deterministic. Any
other operation, copy or finish first emits the held work unchanged, so the
recorded order is the semantic order.

``decode_weights="int8" | "int4"`` stores the weights of the few-row projections (and
only those) quantized: int8 with one FP32 scale per output channel, or unsigned 4-bit
affine codes with a BF16 scale and bias per 64 inputs of a channel. Activations,
accumulation (FP32) and every rounding boundary stay those of the BF16 lowering; the
kernels dequantize weight tiles to BF16 in threadgroup memory. Quantization uses the
weights alone and happens once at recording.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .dimensions import uniform_key
from .plan import Plan, Tensor, Weight

WEIGHT_BITS = {"int8": 8, "int4": 4}


def quantize_columns(weight, bits):
    """[K, N] weight -> (codes [N, ...] uint8/int8, scales, biases or None) per output channel."""
    w = weight.detach().to("cpu", torch.float32).T.contiguous()
    n, k = w.shape
    if not torch.isfinite(w).all():
        raise ValueError("Quantized weights must be finite")
    if bits == 8:
        scale = (w.abs().amax(1) / 127).clamp_min(1e-12)
        codes = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        return codes.numpy(), scale.numpy().astype(np.float32), None
    if k % 64:
        raise ValueError("int4 weights need K divisible by 64")
    g = w.view(n, k // 64, 64)
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / 15).clamp_min(1e-12).bfloat16().float()
    bias = lo.bfloat16().float()
    q = ((g - bias[..., None]) / scale[..., None]).round().clamp(0, 15).to(torch.uint8).view(n, k)
    codes = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    as_bits = lambda t: t.bfloat16().view(torch.int16).numpy().view(np.uint16)
    return codes.numpy(), as_bits(scale), as_bits(bias)

# (tile_m, tile_n, simdgroups) -> registry suffix. The production default tile is 64x64/2.
TILES = {(64, 64, 2): "", (32, 128, 2): "_32x128", (16, 64, 2): "_16x64"}
SPLITK_CHUNK = 256
FMA_ROWS, FMA_CHUNK = 10, 512   # exact-row scalar kernels for quantized few-row projections


def select_tile(m, n, k):
    """Measured M4 Pro tiles for M > 16 (geometry only, never model names)."""
    if n >= 8192 or k >= 4096:
        return 32, 128, 2
    return 64, 64, 2


def _same(a, b):
    return (a.buffer == b.buffer and uniform_key(a.offset) == uniform_key(b.offset)
            and tuple(map(uniform_key, a.shape)) == tuple(map(uniform_key, b.shape))
            and a.contiguous() and b.contiguous())


class FusedPlan(Plan):
    def __init__(self, library=None, *, fuse_epilogues=True, tune_tiles=True, coprocessor=None,
                 decode_weights=None):
        super().__init__(library)
        if decode_weights is not None and decode_weights not in WEIGHT_BITS:
            raise ValueError(f"decode_weights must be one of {sorted(WEIGHT_BITS)} or None")
        self.decode_bits = WEIGHT_BITS.get(decode_weights)
        self._quantized = {}
        self.fuse_epilogues = fuse_epilogues
        self.tune_tiles = tune_tiles
        # coprocessor: dict(hidden=channels per FFN on the Neural Engine, an int or {stage: int}
        #                   (default: 5/8 of the FFN rounded to 512, the measured M4 Pro balance
        #                   with the GPU share), stages=("prefix",),
        #                   spatial=4, cache_dir=None, compute_units="cpu_and_ne")
        self.coprocessor = None
        if coprocessor is not None:
            cfg = dict(hidden=None, stages=("prefix",), spatial=4, cache_dir=None, compute_units="cpu_and_ne")
            unknown = set(coprocessor) - set(cfg)
            if unknown:
                raise ValueError(f"Unknown coprocessor options: {sorted(unknown)}")
            cfg.update(coprocessor)
            sizes = cfg["hidden"].values() if isinstance(cfg["hidden"], dict) else [cfg["hidden"]]
            if any(h is not None and h <= 0 for h in sizes) or cfg["spatial"] <= 0:
                raise ValueError("Coprocessor hidden channels and spatial rows must be positive")
            self.coprocessor = cfg
        self.coprocessor_tasks = []
        self._pending = None
        # Repacked weights stay referenced so their storage identity cannot be reused.
        self._repacked = {}
        self._workspace = None
        self.fused = []

    # ------------------------------------------------------------------ recording
    def linear(self, out, x, w, bias=None):
        held = self._pending
        if (held is not None and "ffn" in held and "down" not in held
                and (bias is None or not held["ffn"]["gated"])
                and self.bind(out, allow_strided=True).contiguous()
                and _same(self.bind(x), held["ffn"]["hidden"]) and w.shape[-1] == held["ffn"]["gemm"]["k"]):
            x, out = self.bind(x), self.bind(out)
            m, k, n = math.prod(x.shape[:-1]), x.shape[-1], w.shape[-1]
            down = dict(out=out, x=x, w=w, bias=bias, m=m, n=n, k=k, stage=self.stage, small=False)
            self._pending = dict(held, down=down)
            return
        self._flush()
        m = math.prod(x.shape[:-1])
        small = m <= 16
        if (not self.fuse_epilogues and small) or not self.bind(out, allow_strided=True).contiguous():
            return super().linear(out, x, w, bias)
        x, out = self.bind(x), self.bind(out)
        k, n = x.shape[-1], w.shape[-1]
        if w.shape != (k, n) or out.numel() != m * n:
            raise ValueError("GEMM shape mismatch")
        if small and (bias is not None or min(k, n) < 256):
            return super().linear(out, x, w, bias)
        self._pending = dict(out=out, x=x, w=w, bias=bias, m=m, n=n, k=k, stage=self.stage, small=small)

    def dispatch(self, name, tensors, writes, params, groups, threads=64, *, strided=()):
        pending, self._pending = self._pending, None
        if pending is not None and "ffn" in pending:
            if "down" in pending and self._coprocess(pending, name, tensors, params):
                return
            self._emit_ffn_gpu(pending["ffn"])
            pending = pending.get("down")
        if pending is not None:
            if pending.get("residual") is not None:
                norm = self._residual_norm(pending, name, tensors)
                self._emit_splitk_residual(pending, norm)
                if norm is not None:
                    return
            elif self.fuse_epilogues and self._fuse(pending, name, tensors, params):
                return  # consumed (or held once more for a following AdaRMS)
            else:
                self._emit(pending)
        return super().dispatch(name, tensors, writes, params, groups, threads, strided=strided)

    def copy_buffer(self, source, destination):
        self._flush()
        return super().copy_buffer(source, destination)

    def finish(self):
        self._flush()
        return super().finish()

    def _flush(self):
        pending, self._pending = self._pending, None
        if pending is not None and "ffn" in pending:
            self._emit_ffn_gpu(pending["ffn"])
            pending = pending.get("down")
        if pending is not None and pending.get("residual") is not None:
            self._emit_splitk_residual(pending, None)
        elif pending is not None:
            self._emit(pending)

    # ------------------------------------------------------------------ lowering
    def _tile(self, m, n, k):
        return select_tile(m, n, k) if self.tune_tiles else (64, 64, 2)

    def _gemm(self, kind, p, out, weight, n_columns, extra=()):
        tm, tn, sg = self._tile(p["m"], n_columns, p["k"])
        suffix = TILES[(tm, tn, sg)]
        current = self.stage
        try:
            self.stage = p["stage"]
            Plan.dispatch(self, f"gemm{kind}_bf16{suffix}", [p["x"], weight, out, *extra],
                          {2}, [p["m"], n_columns, p["k"]],
                          ((n_columns + tn - 1) // tn, (p["m"] + tm - 1) // tm), sg * 32)
        finally:
            self.stage = current
        return tm, tn, sg

    def _emit(self, p):
        if p["small"]:
            current = self.stage
            try:
                self.stage = p["stage"]
                return super().linear(p["out"], p["x"], p["w"], p["bias"])
            finally:
                self.stage = current
        if p["bias"] is None:
            self._gemm("", p, p["out"], p["w"], p["n"])
        else:
            current = self.stage
            try:
                self.stage = p["stage"]
                tm, tn, sg = self._tile(p["m"], p["n"], p["k"])
                if (tm, tn, sg) == (64, 64, 2):
                    # Original bias entry: uniforms bound before bias.
                    return super().linear(p["out"], p["x"], p["w"], p["bias"])
                self._gemm("_bias", p, p["out"], p["w"], p["n"], (p["bias"],))
            finally:
                self.stage = current

    def _emit_ffn_gpu(self, ffn):
        if ffn["gated"]:
            self._emit_gelu_mul(ffn["gemm"], ffn["hidden"])
        else:
            p = ffn["gemm"]
            self._gemm("_bias_gelu", p, ffn["hidden"], p["w"], p["n"], (p["bias"],))
            self.fused.append(("bias_gelu", p["stage"], (p["m"], p["n"], p["k"])))

    def _emit_gelu_mul(self, p, out):
        tm, tn, sg = self._tile(p["m"], p["n"], p["k"])
        weight = self._packed_gelu_mul(p["w"], tn // 2)
        self._gemm("_gelu_mul", p, out, weight, p["n"])
        self.fused.append(("gelu_mul", p["stage"], (p["m"], p["n"], p["k"])))

    # ------------------------------------------------------------- coprocessor FFN
    def _coprocessor_share(self, p, *, gated):
        """(gpu_hidden, coprocessor_hidden) for an eligible FFN up-projection GEMM, else None."""
        cfg = self.coprocessor
        if cfg is None or p["stage"] not in cfg["stages"]:
            return None
        hidden = p["n"] // 2 if gated else p["n"]
        share = cfg["hidden"].get(p["stage"]) if isinstance(cfg["hidden"], dict) else cfg["hidden"]
        if share is None:
            share = hidden * 5 // 8 // 512 * 512
        share = min(share, hidden)
        gpu = hidden - share
        if gpu <= 0:
            return None
        if not gated:
            return gpu, share
        tm, tn, sg = self._tile(p["m"], 2 * gpu, p["k"])
        return (gpu, share) if gpu % (tn // 2) == 0 else None

    def _coprocess(self, pending, name, tensors, params):
        if not (name == "tr_residual" and len(params) == 3 and params[2] == 0):
            return False
        g, d, gated = pending["ffn"]["gemm"], pending["down"], pending["ffn"]["gated"]
        residual, update, out = (self.bind(t) for t in tensors[:3])
        if not (_same(update, d["out"]) and _same(out, residual)):
            return False
        from . import neural_engine

        cfg = self.coprocessor
        gpu, share = self._coprocessor_share(g, gated=gated)
        m, channels = g["m"], g["k"]
        hidden = g["n"] // 2 if gated else g["n"]
        spatial = cfg["spatial"]
        capacity = (int(m) + spatial - 1) // spatial * spatial
        w, wd = g["w"].tensor, d["w"].tensor
        if gated:
            model, input_name, output_name, shape = neural_engine.gated_ffn_share(
                w[:, gpu:hidden], w[:, hidden + gpu:], wd[gpu:], rows=capacity, spatial=spatial,
                cache_dir=cfg["cache_dir"])
        else:
            model, input_name, output_name, shape = neural_engine.gelu_ffn_share(
                w[:, gpu:], g["bias"].tensor[gpu:], wd[gpu:], rows=capacity, spatial=spatial,
                cache_dir=cfg["cache_dir"])
        current = self.stage
        try:
            self.stage = g["stage"]
            packed_in, partial = self.allocate((channels, capacity)), self.allocate((channels, capacity))
            task = self.coreml_task(model, packed_in, partial, shape, input_name=input_name,
                                    output_name=output_name,
                                    compute_units=neural_engine.COMPUTE_UNITS[cfg["compute_units"]])
            Plan.dispatch(self, "pack_channels_f16", [g["x"], packed_in], {1}, [m, channels, capacity],
                          ((capacity + 31) // 32, (channels + 31) // 32), 256)
            self.external(task, wait=False)
            # GPU share: leading hidden channels of the up projection(s), then K-range [0, gpu) of down.
            hidden_gpu = self.allocate((int(m), gpu))
            if gated:
                tm, tn, sg = self._tile(m, 2 * gpu, channels)
                key = ("ffn_share", w.untyped_storage().data_ptr(), w.storage_offset(), gpu, tn // 2)
                if key not in self._repacked:
                    block = tn // 2
                    source = w.detach().to("cpu")
                    gate, up = source[:, :gpu], source[:, hidden:hidden + gpu]
                    packed = torch.stack((gate.reshape(channels, -1, block), up.reshape(channels, -1, block)), 2)
                    self._repacked[key] = Weight(packed.reshape(channels, 2 * gpu).contiguous(),
                                                 g["w"].name + f".gelu_mul{block}.share{gpu}")
                self._gemm("_gelu_mul", dict(g, n=2 * gpu), hidden_gpu, self._repacked[key], 2 * gpu)
            else:
                up = Weight(w[:, :gpu].contiguous(), g["w"].name + f".share{gpu}")
                up_bias = Weight(g["bias"].tensor[:gpu].contiguous(), g["bias"].name + f".share{gpu}")
                self._repacked[("ffn_up", w.untyped_storage().data_ptr(), w.storage_offset(), gpu)] = (up, up_bias)
                self._gemm("_bias_gelu", dict(g, n=gpu), hidden_gpu, up, gpu, (up_bias,))
            # The local down product needs only the GPU share; it runs before waiting.
            down = Weight(wd[:gpu], d["w"].name + f".share{gpu}")
            self._repacked[("ffn_down", wd.untyped_storage().data_ptr(), wd.storage_offset(), gpu)] = down
            local = self.allocate((int(m), d["n"]), itemsize=4)
            self._gemm("_f32", dict(d, x=hidden_gpu, k=gpu), local, down, d["n"])
            self.external(task, wait=True)
            bias = d["bias"] if d["bias"] is not None else out
            Plan.dispatch(self, "residual_add_partials_bf16", [local, partial, out, bias], {2},
                          [m, d["n"], capacity, int(d["bias"] is not None)],
                          ((capacity + 31) // 32, (d["n"] + 31) // 32), 256)
        finally:
            self.stage = current
        self.coprocessor_tasks.append(dict(task=task, stage=g["stage"], gpu_hidden=gpu, coprocessor_hidden=share,
                                           rows=capacity, model=model))
        self.fused.append(("coprocessor_ffn", g["stage"], (m, g["n"], g["k"])))
        return True

    def _packed_gelu_mul(self, w, block):
        tensor = w.tensor
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
               tuple(tensor.stride()), block)
        if key not in self._repacked:
            k, n2 = tensor.shape
            source = tensor.detach().to("cpu")
            gate, up = source[:, : n2 // 2], source[:, n2 // 2:]
            packed = torch.stack((gate.reshape(k, -1, block), up.reshape(k, -1, block)), 2).reshape(k, n2)
            self._repacked[key] = Weight(packed.contiguous(), w.name + f".gelu_mul{block}")
        return self._repacked[key]

    # ------------------------------------------------------------- few-row split-K
    def raw(self, array):
        array = np.ascontiguousarray(array)
        bid = self.lib.frt_apple_plan_buffer(self.handle, array.nbytes)
        if bid < 0:
            self.check(-1)
        self.check(self.lib.frt_apple_plan_write(self.handle, bid, 0, array.ctypes.data, array.nbytes))
        self.weight_bytes += array.nbytes
        return Tensor(self, array.shape, bid, itemsize=array.dtype.itemsize)

    def _quantized_weight(self, w, packing=None, interleaved=False):
        """(codes, scales, biases) plan tensors of a few-row projection weight, packed once.

        interleaved=True stores uint32 code words input-block-major ([K/8][N] int4, [K/4][N] int8)
        and int4 scales/biases [K/64][N], the layout of the exact-row FMA kernels.
        """
        t = w.tensor
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), packing,
               interleaved)
        if key not in self._quantized:
            source = t
            if packing is not None:
                source = self._packed_gelu_mul(w, packing).tensor
            codes, scales, biases = quantize_columns(source, self.decode_bits)
            if interleaved:
                codes = np.ascontiguousarray(codes.view(np.uint32).T)
                if biases is not None:
                    scales, biases = np.ascontiguousarray(scales.T), np.ascontiguousarray(biases.T)
            self._quantized[key] = (self.raw(codes), self.raw(scales),
                                    self.raw(biases) if biases is not None else None, w)
        codes, scales, biases, _ = self._quantized[key]
        return codes, scales, biases if biases is not None else codes

    def _splitk(self, p):
        """FP32 partials [splits, M, N] of the held few-row GEMM in a shared workspace."""
        splits = (p["k"] + SPLITK_CHUNK - 1) // SPLITK_CHUNK
        need = splits * p["m"] * p["n"]
        if self._workspace is None or self._workspace.numel() < need:
            self._workspace = self.allocate((need,), itemsize=4)
        partial = self._workspace if self._workspace.numel() == need else self._workspace[:need]
        current = self.stage
        try:
            self.stage = p["stage"]
            grid = ((p["n"] + 63) // 64, (p["m"] + 15) // 16, splits)
            if self._fma_rows(p):
                splits = (p["k"] + FMA_CHUNK - 1) // FMA_CHUNK
                need = splits * p["m"] * p["n"]
                if self._workspace.numel() < need:
                    self._workspace = self.allocate((need,), itemsize=4)
                partial = self._workspace if self._workspace.numel() == need else self._workspace[:need]
                codes, scales, biases = self._quantized_weight(p["w"], interleaved=True)
                Plan.dispatch(self, f"gemm_rows{FMA_ROWS}_splitk_w{self.decode_bits}_f32_bf16",
                              [p["x"], codes, scales, biases, partial], {4},
                              [p["m"], p["n"], p["k"], FMA_CHUNK], (p["n"] // 64, 1, splits), 128)
            elif self.decode_bits:
                codes, scales, biases = self._quantized_weight(p["w"])
                Plan.dispatch(self, f"gemm_splitk_w{self.decode_bits}_f32_bf16",
                              [p["x"], codes, scales, biases, partial], {4},
                              [p["m"], p["n"], p["k"], SPLITK_CHUNK], grid, 64)
            else:
                Plan.dispatch(self, "gemm_splitk_f32_bf16", [p["x"], p["w"], partial], {2},
                              [p["m"], p["n"], p["k"], SPLITK_CHUNK], grid, 64)
        finally:
            self.stage = current
        return partial, splits

    def _fma_rows(self, p):
        return (self.decode_bits is not None and p["m"] == FMA_ROWS and p["n"] % 64 == 0
                and p["k"] % 64 == 0)

    def _residual_norm(self, p, name, tensors):
        """The AdaRMS dispatch immediately following a held split-K residual, if it reads that row."""
        if name != "tr_adarms":
            return None
        x, out, mod = (self.bind(t) for t in tensors)
        return (out, mod) if _same(x, p["residual"][0]) else None

    def _emit_splitk_residual(self, p, norm):
        partial, splits = self._splitk(p)
        x, gate, gated = p["residual"]
        out, mod = norm if norm is not None else (x, x)
        rows, dim = p["m"], p["n"]
        self.dispatch_raw("splitk_residual_adarms_bf16", [partial, x, gate, mod, out], {1, 4},
                          [rows, dim, splits, gated, int(norm is not None)], (rows,), 256, p["stage"])
        self.fused.append(("splitk_residual" + ("_adarms" if norm is not None else ""), p["stage"],
                           (p["m"], p["n"], p["k"])))

    def dispatch_raw(self, name, tensors, writes, params, groups, threads, stage):
        current = self.stage
        try:
            self.stage = stage
            Plan.dispatch(self, name, tensors, writes, params, groups, threads)
        finally:
            self.stage = current

    def _fuse_small(self, p, name, tensors, params):
        if name in ("tr_rope", "tr_rope_offsets"):
            qkv, q, k, v, cos, sin = (self.bind(t) for t in tensors)
            if not _same(qkv, p["out"]):
                return False
            partial, splits = self._splitk(p)
            offsets = list(params[4:6]) if name == "tr_rope_offsets" else [0, 0]
            rows, heads, dim = params[0], params[1], params[2]
            self.dispatch_raw("splitk_rope_bf16", [partial, q, k, v, cos, sin], {1, 2, 3},
                              [rows, heads, dim, params[3], *offsets, splits],
                              ((rows * (heads + 1) * (dim // 2) + 63) // 64,), 64, p["stage"])
            self.fused.append(("splitk_rope", p["stage"], (p["m"], p["n"], p["k"])))
            return True
        if name == "tr_residual" and len(params) == 3:
            residual, update, out, gate = (self.bind(t) for t in tensors)
            if not (_same(update, p["out"]) and _same(out, residual)):
                return False
            # Hold once more: a following AdaRMS of this row joins the reduction.
            self._pending = dict(p, residual=(residual, gate, int(params[2])))
            return True
        if name == "tr_gelu_mul" and p["n"] % 64 == 0:
            merged, out = (self.bind(t) for t in tensors)
            if not _same(merged, p["out"]):
                return False
            if self._fma_rows(p):
                partial, splits = self._splitk(p)
                rows, hidden = p["m"], p["n"] // 2
                self.dispatch_raw("splitk_gelu_mul_bf16", [partial, out], {1}, [rows, hidden, splits],
                                  ((rows * hidden + 63) // 64,), 64, p["stage"])
                self.fused.append(("splitk_gelu_mul", p["stage"], (p["m"], p["n"], p["k"])))
                return True
            current = self.stage
            try:
                self.stage = p["stage"]
                grid = (p["n"] // 64, (p["m"] + 15) // 16)
                if self.decode_bits:
                    codes, scales, biases = self._quantized_weight(p["w"], packing=32)
                    Plan.dispatch(self, f"gemm_gelu_mul_w{self.decode_bits}_bf16",
                                  [p["x"], codes, scales, biases, out], {4}, [p["m"], p["n"], p["k"], 0], grid, 64)
                else:
                    weight = self._packed_gelu_mul(p["w"], 32)
                    Plan.dispatch(self, "gemm_gelu_mul_bf16_16x64", [p["x"], weight, out], {2},
                                  [p["m"], p["n"], p["k"]], grid, 64)
            finally:
                self.stage = current
            self.fused.append(("gelu_mul", p["stage"], (p["m"], p["n"], p["k"])))
            return True
        return False

    def _fuse(self, p, name, tensors, params):
        if p["small"]:
            return self._fuse_small(p, name, tensors, params)
        if name == "tr_gelu_mul" and p["bias"] is None and p["n"] % 2 == 0:
            merged, out = (self.bind(t) for t in tensors)
            if not _same(merged, p["out"]):
                return False
            tm, tn, sg = self._tile(p["m"], p["n"], p["k"])
            if (p["n"] // 2) % (tn // 2):
                return False
            if self._coprocessor_share(p, gated=True) is not None:
                # Hold until the down projection and its residual show the whole FFN.
                self._pending = dict(ffn=dict(gemm=p, hidden=out, gated=True))
                return True
            self._emit_gelu_mul(p, out)
            return True
        if name == "tr_residual" and len(params) == 3 and params[2] == 0:
            residual, update, out = (self.bind(t) for t in tensors[:3])
            if not (_same(update, p["out"]) and _same(out, residual)):
                return False
            kind = "_bias_residual" if p["bias"] is not None else "_residual"
            extra = (p["bias"],) if p["bias"] is not None else ()
            self._gemm(kind, p, out, p["w"], p["n"], extra)
            self.fused.append((kind[1:], p["stage"], (p["m"], p["n"], p["k"])))
            return True
        if name == "gelu_bf16" and p["bias"] is not None:
            (target,) = (self.bind(t) for t in tensors)
            if not _same(target, p["out"]):
                return False
            if self._coprocessor_share(p, gated=False) is not None:
                self._pending = dict(ffn=dict(gemm=p, hidden=target, gated=False))
                return True
            self._gemm("_bias_gelu", p, target, p["w"], p["n"], (p["bias"],))
            self.fused.append(("bias_gelu", p["stage"], (p["m"], p["n"], p["k"])))
            return True
        return False
