"""Build fixed-shape native Metal plans from hardware operation providers.

Tensor views below are build-time descriptors, not eager tensor operations.
Replay updates input contents and invokes one prebuilt native command plan.
"""

from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import weakref
import numpy as np
import torch


class _PlanTiming(C.Structure):
    _fields_ = [
        (name, C.c_double)
        for name in (
            "encode_us",
            "gpu_us",
            "wall_us",
            "commit_us",
            "wait_us",
            "commit_to_gpu_us",
            "gpu_to_return_us",
            "kernel_us",
            "submit_host_s",
            "gpu_start_s",
            "gpu_end_s",
            "return_host_s",
        )
    ]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


def strides(shape):
    result = []
    stride = 1
    for n in reversed(shape):
        result.append(stride)
        stride *= n
    return tuple(reversed(result))


class Tensor:
    def __init__(self, plan, shape, buffer, offset=0, stride=None, itemsize=2):
        self.plan, self.shape, self.buffer, self.offset = (
            plan,
            tuple(shape),
            buffer,
            offset,
        )
        self.stride = stride or strides(self.shape)
        self.itemsize = itemsize

    def numel(self):
        return math.prod(self.shape)

    def contiguous(self):
        return all(
            n == 1 or s == t
            for n, s, t in zip(self.shape, self.stride, strides(self.shape))
        )

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            n = -math.prod(shape)
            shape = tuple(self.numel() // n if x == -1 else x for x in shape)
        if not self.contiguous() or math.prod(shape) != self.numel():
            raise ValueError("Invalid native view")
        return Tensor(
            self.plan, shape, self.buffer, self.offset, itemsize=self.itemsize
        )

    reshape = view

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if len(index) > len(self.shape):
            raise IndexError(index)
        shape = []
        stride = []
        off = self.offset
        for n, s, i in zip(
            self.shape,
            self.stride,
            index + (slice(None),) * (len(self.shape) - len(index)),
        ):
            if isinstance(i, int):
                i = i + n if i < 0 else i
                if not 0 <= i < n:
                    raise IndexError(i)
                off += i * s * self.itemsize
            elif isinstance(i, slice):
                a, b, step = i.indices(n)
                if step != 1 or b <= a:
                    raise ValueError("Native views require nonempty unit slices")
                off += a * s * self.itemsize
                shape.append(b - a)
                stride.append(s)
            else:
                raise TypeError(index)
        return Tensor(self.plan, shape, self.buffer, off, tuple(stride), self.itemsize)

    def copy_(self, source):
        # Validate ownership/lifetime even when matching buffer IDs suggest an
        # alias: buffer IDs are local to each plan.
        self.plan.bind(self)
        source = self.plan.bind(source)
        if (
            self.shape != source.shape
            or self.itemsize != 2
            or source.itemsize != 2
            or not self.contiguous()
            or not source.contiguous()
        ):
            raise ValueError("Native copy requires matching contiguous tensors")
        if (
            self.buffer == source.buffer
            and self.offset == source.offset
            and self.stride == source.stride
        ):
            return self
        self.plan.copy_buffer(source, self)
        return self

    def add_(self, update):
        self.plan.ops.residual(self, update, self)
        return self


class Weight:
    def __init__(self, tensor, name):
        self.tensor, self.name = tensor, name
        self.shape = tuple(tensor.shape)

    def __getitem__(self, index):
        return Weight(self.tensor[index], self.name)


class Plan:
    def __init__(self, library=None):
        path = Path(
            library
            or Path(__file__).resolve().parents[3]
            / ".cache/apple-native/libflashrt_apple_plan.dylib"
        )
        self.lib = C.CDLL(str(path))
        signatures = {
            "create": ([], C.c_void_p),
            "destroy": ([C.c_void_p], None),
            "error": ([], C.c_char_p),
            "buffer": ([C.c_void_p, C.c_size_t], C.c_int64),
            "write": (
                [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t],
                C.c_int,
            ),
            "read": (
                [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t],
                C.c_int,
            ),
            "data": ([C.c_void_p, C.c_uint32, C.c_size_t, C.c_size_t], C.c_void_p),
            "copy": ([C.c_void_p, C.c_uint32, C.c_size_t, C.c_uint32, C.c_size_t, C.c_size_t], C.c_int),
            "dispatch": (
                [
                    C.c_void_p,
                    C.c_char_p,
                    C.POINTER(C.c_uint32),
                    C.POINTER(C.c_size_t),
                    C.POINTER(C.c_uint32),
                    C.c_size_t,
                    C.POINTER(C.c_size_t),
                    C.c_size_t,
                    C.c_char_p,
                ],
                C.c_int,
            ),
            "finish": ([C.c_void_p], C.c_int),
            "residency": ([C.c_void_p, C.c_uint32], C.c_int),
            "run_timed": ([C.c_void_p, C.c_uint32, C.POINTER(_PlanTiming)], C.c_int),
            "profile": ([C.c_void_p, C.POINTER(_PlanTiming), C.c_size_t], C.c_int),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, "frt_apple_plan_" + name)
            f.argtypes = args
            f.restype = result
        self.handle = self.lib.frt_apple_plan_create()
        if not self.handle:
            self.check(-1)
        self._finalizer = weakref.finalize(
            self, self.lib.frt_apple_plan_destroy, self.handle
        )
        self.weights = {}
        self.dispatch_count = 0
        self.copy_count = 0
        self.dispatches = []
        self.operations = []
        self.stage = "setup"
        self.weight_bytes = 0
        self.ops = Ops(self)
        self.uniforms = {}
        self._uniform_buffer = None
        self._uniform_offset = 65536
        self._mappings = {}

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.frt_apple_plan_error().decode())

    def close(self):
        self._mappings.clear()
        self._finalizer()
        self.handle = None

    def _host_view(self, t):
        # Private borrowed view: never return it to callers. The owning model must
        # serialize input writes, execution, output conversion and close.
        if self.handle is None:
            raise RuntimeError("Native plan is closed")
        if t.plan is not self or not t.contiguous() or t.itemsize != 2:
            raise ValueError("Expected a contiguous tensor owned by this plan")
        key = (t.buffer, t.offset, t.numel())
        if key not in self._mappings:
            address = self.lib.frt_apple_plan_data(
                self.handle, t.buffer, t.offset, t.numel() * 2
            )
            if not address:
                self.check(-1)
            self._mappings[key] = np.ctypeslib.as_array(
                (C.c_uint16 * t.numel()).from_address(address)
            )
        return self._mappings[key].reshape(t.shape)

    def allocate(self, shape, *, itemsize=2):
        if itemsize not in (2, 4):
            raise ValueError("Expected BF16 or FP32 storage")
        n = math.prod(shape) * itemsize
        bid = self.lib.frt_apple_plan_buffer(self.handle, n)
        if bid < 0:
            self.check(-1)
        return Tensor(self, shape, bid, itemsize=itemsize)

    def write(self, t, value):
        if isinstance(value, torch.Tensor):
            v = (
                value.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                .view(torch.int16)
                .numpy()
                .view(np.uint16)
            )
        else:
            v = np.asarray(value)
            if v.dtype != np.uint16:
                f = np.ascontiguousarray(v, dtype=np.float32)
                bits = f.view(np.uint32)
                v = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        v = np.ascontiguousarray(v)
        if v.shape != t.shape or not t.contiguous():
            raise ValueError("Native upload shape mismatch")
        np.copyto(self._host_view(t), v)

    def read(self, t):
        if not t.contiguous():
            raise ValueError("Native download must be contiguous")
        v = self._host_view(t)
        # Conversion owns its memory, so subsequent replay/close cannot mutate it.
        return (v.astype(np.uint32) << 16).view(np.float32)

    def bind(self, t, *, allow_strided=False):
        if self.handle is None:
            raise RuntimeError("Native plan is closed")
        if isinstance(t, Weight):
            v = t.tensor
            k = (
                v.untyped_storage().data_ptr(),
                v.storage_offset(),
                tuple(v.shape),
                tuple(v.stride()),
            )
            if k not in self.weights:
                target = self.allocate(v.shape)
                self.write(target, v)
                self.weights[k] = target
                self.weight_bytes += v.numel() * 2
            t = self.weights[k]
        if not isinstance(t, Tensor):
            raise ValueError("Expected a native Tensor kernel binding")
        # Buffer IDs are local to a plan, so C++ cannot recover this ownership
        # after the Python descriptor has been reduced to an integer ID.
        if t.plan is not self:
            raise ValueError("Kernel tensor belongs to another plan")
        if not allow_strided and not t.contiguous():
            raise ValueError("Noncontiguous kernel binding")
        return t

    def dispatch(self, name, tensors, writes, params, groups, threads=64, *, strided=()):
        bound = [self.bind(t, allow_strided=i in strided) for i, t in enumerate(tensors)]
        # Uniform buffers are raw uint32, padded for uint3 alignment.
        values = np.asarray(list(params) + [0] * ((-len(params)) % 4), dtype=np.uint32)
        key = tuple(values)
        if key not in self.uniforms:
            if self._uniform_offset + values.nbytes > 65536:
                self._uniform_buffer = self.lib.frt_apple_plan_buffer(
                    self.handle, 65536
                )
                if self._uniform_buffer < 0:
                    self.check(-1)
                self._uniform_offset = 0
            bid, parameter_offset = self._uniform_buffer, self._uniform_offset
            self.check(
                self.lib.frt_apple_plan_write(
                    self.handle,
                    bid,
                    parameter_offset,
                    values.ctypes.data,
                    values.nbytes,
                )
            )
            self.uniforms[key] = (bid, parameter_offset)
            self._uniform_offset += values.nbytes
        bid, parameter_offset = self.uniforms[key]
        n = len(bound) + 1
        buffer_ids = [t.buffer for t in bound] + [bid]
        byte_offsets = [t.offset for t in bound] + [parameter_offset]
        accesses = [2 if i in writes else 0 for i in range(len(bound))] + [0]
        if name in ("gemm_bias_bf16", "gemm_small_bias_bf16", "gemm_patch_bf16"):
            for values_list in (buffer_ids, byte_offsets, accesses):
                values_list.insert(3, values_list.pop())
        ids = (C.c_uint32 * n)(*buffer_ids)
        offsets = (C.c_size_t * n)(*byte_offsets)
        access = (C.c_uint32 * n)(*accesses)
        groups = tuple(groups) + (1,) * (3 - len(groups))
        self.check(
            self.lib.frt_apple_plan_dispatch(
                self.handle,
                name.encode(),
                ids,
                offsets,
                access,
                n,
                (C.c_size_t * 3)(*groups),
                threads,
                f"{self.dispatch_count}.{name}".encode(),
            )
        )
        self.dispatch_count += 1
        meta = dict(
            kind="dispatch",
            kernel=name,
            stage=self.stage,
            params=list(params),
            groups=list(groups),
            threads=threads,
        )
        self.dispatches.append(meta)
        self.operations.append(meta)

    def copy_buffer(self, source, destination):
        source, destination = self.bind(source), self.bind(destination)
        if source.shape != destination.shape or source.itemsize != destination.itemsize:
            raise ValueError("Native buffer copy requires matching storage")
        size = source.numel() * source.itemsize
        self.check(self.lib.frt_apple_plan_copy(
            self.handle, source.buffer, source.offset,
            destination.buffer, destination.offset, size,
        ))
        self.copy_count += 1
        self.operations.append(dict(kind="copy", stage=self.stage, bytes=size))

    def linear(self, out, x, w, bias=None):
        out = self.bind(out, allow_strided=True)
        x, w = map(self.bind, (x, w))
        m = math.prod(x.shape[:-1])
        k = x.shape[-1]
        n = w.shape[-1]
        if w.shape != (k, n) or out.numel() != m * n:
            raise ValueError("GEMM shape mismatch")
        small = m <= 16
        name = "gemm_small" if small else "gemm"
        params = [m, n, k]
        strided = not out.contiguous()
        if strided:
            if (bias is not None or out.shape != (m, n)
                    or out.stride[1] != 1 or out.stride[0] < n):
                raise ValueError("Strided GEMM requires a row-major 2D output without bias")
            name += "_strided"
            params.append(out.stride[0])
        # The fused epilogue still rounds to BF16 before adding bias.
        tensors = [x, w, out] + ([bias] if bias is not None else [])
        self.dispatch(
            name + ("_bias" if bias is not None else "") + "_bf16",
            tensors,
            {2},
            params,
            ((n + 63) // 64, (m + (15 if small else 63)) // (16 if small else 64)),
            strided={2} if strided else (),
        )

    def linear_residual(self, *args, **kwargs):
        return False

    def patch(self, out, images, weight, bias, position):
        if images.shape[1:] != (224, 224, 3) or out.shape != (
            images.shape[0],
            256,
            1152,
        ):
            raise ValueError("Native patch requires 224x224 RGB and SigLIP width 1152")
        rows = images.shape[0] * 256
        packed = self.allocate((rows, 588))
        patch = self.allocate(out.shape)
        # OIHW -> K,N in exactly the channel/y/x order used by tr_patch_pack.
        w = weight.reshape(1152, 588).T.contiguous()
        self.dispatch(
            "tr_patch_pack",
            [images, packed],
            {1},
            [images.shape[0]],
            ((packed.numel() + 63) // 64,),
        )
        self.dispatch(
            "gemm_patch_bf16",
            [packed, Weight(w, "patch.w"), patch, Weight(bias, "patch.bias")],
            {2},
            [rows, 1152, 588],
            (18, (rows + 63) // 64),
        )
        self.dispatch(
            "tr_patch_position",
            [patch, Weight(position, "patch.position"), out],
            {2},
            [rows, 1152],
            ((out.numel() + 63) // 64,),
        )
        return patch

    def finish(self):
        self.check(self.lib.frt_apple_plan_finish(self.handle))

    def set_resident(self, enabled):
        self.check(self.lib.frt_apple_plan_residency(self.handle, int(enabled)))

    def profile(self):
        timings = (_PlanTiming * len(self.operations))()
        self.check(self.lib.frt_apple_plan_profile(self.handle, timings, len(timings)))
        return [
            dict(**meta, **t.as_dict()) for meta, t in zip(self.operations, timings)
        ]

    def run(self, execution="icb"):
        if execution not in ("icb", "direct"):
            raise ValueError("Expected icb/direct")
        t = _PlanTiming()
        self.check(
            self.lib.frt_apple_plan_run_timed(
                self.handle, int(execution == "icb"), C.byref(t)
            )
        )
        return t.as_dict()


class Ops:
    fused_rope = True
    merged_encoder_ffn = True
    merged_decoder_ffn = True

    def __init__(self, plan):
        self.p = plan

    def layer_norm(self, out, x, w, b, eps=1e-6):
        if eps != 1e-6:
            raise ValueError("Unsupported epsilon")
        rows = math.prod(x.shape[:-1])
        d = x.shape[-1]
        self.p.dispatch(
            "norm_bf16",
            [x, out, w, b],
            {1},
            [1, rows, d, 0, 1, 1],
            ((rows + 3) // 4,),
            128,
        )

    def rms_norm(self, out, x, weight=None, eps=1e-6):
        if weight is not None or eps != 1e-6:
            raise ValueError("Unsupported RMSNorm")
        rows = math.prod(x.shape[:-1])
        self.p.dispatch("tr_rms", [x, out], {1}, [rows, x.shape[-1]], (rows,), 32)

    def adarms(self, out, x, cond, weight, bias, eps=1e-6, modulation=None):
        if cond is not None or modulation is None or eps != 1e-6:
            raise ValueError("Precomputed AdaRMS required")
        rows = math.prod(x.shape[:-1])
        d = x.shape[-1]
        mod = self.p.bind(modulation)
        self.p.dispatch("tr_adarms", [x, out, mod], {1}, [rows, d], (rows,), 32)
        return mod[:, 2 * d :]

    def residual(self, out, update, residual, gate=None):
        self.p.dispatch(
            "tr_residual",
            [residual, update, out, gate if gate is not None else residual],
            {2},
            [out.numel(), out.shape[-1], int(gate is not None)],
            ((out.numel() + 63) // 64,),
        )

    def residual_rms(self, out_sum, out_norm, update, residual, eps=1e-6):
        self.residual(out_sum, update, residual)
        self.rms_norm(out_norm, out_sum, eps=eps)

    def residual_adarms(
        self,
        out_sum,
        out_norm,
        update,
        residual,
        gate,
        cond,
        w,
        b,
        eps=1e-6,
        modulation=None,
    ):
        self.residual(out_sum, update, residual, gate)
        return self.adarms(out_norm, out_sum, cond, w, b, eps, modulation)

    def gelu(self, out, x):
        if out is not x:
            out.copy_(x)
        self.p.dispatch(
            "gelu_bf16",
            [out],
            {0},
            [out.numel(), out.shape[-1]],
            ((out.numel() + 63) // 64,),
        )

    def gelu_mul_merged(self, out, x):
        self.p.dispatch(
            "tr_gelu_mul",
            [x, out],
            {1},
            [math.prod(out.shape[:-1]), out.shape[-1]],
            ((out.numel() + 63) // 64,),
        )

    def gelu_mul(self, out, gate, up):
        # The shared traversal passes adjacent column slices of one allocation.
        # Both GEMMs write directly into it; reuse the same activation kernel.
        rows, dim = out.shape
        if (gate.shape != out.shape or up.shape != out.shape
                or gate.buffer != up.buffer or gate.plan is not self.p or up.plan is not self.p
                or gate.itemsize != 2 or up.itemsize != 2
                or gate.stride != (2 * dim, 1) or up.stride != gate.stride
                or up.offset != gate.offset + dim * 2):
            raise ValueError("Native GELU multiply requires adjacent gate/up column views")
        merged = Tensor(self.p, (rows, 2 * dim), gate.buffer, gate.offset)
        self.gelu_mul_merged(out, merged)

    def qkv_rope(self, q, k, v, qkv, cos, sin, position):
        rows, heads, dim = q.shape
        self.p.dispatch(
            "tr_rope",
            [qkv, q, k, v, cos, sin],
            {1, 2, 3},
            [rows, heads, dim, position],
            ((rows * (heads + 1) * (dim // 2) + 63) // 64,),
        )


class Attention:
    def __init__(self, plan, *, split_decode=True):
        self.p = plan
        self.split_decode = split_decode
        self.workspaces = {}

    def _bf16_tensor(self, tensor, operation, role):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"{operation} {role} requires an explicit native BF16 tensor")
        tensor = self.p.bind(tensor)
        if tensor.itemsize != 2 or not tensor.shape or min(tensor.shape) <= 0:
            raise ValueError(
                f"{operation} {role} requires nonempty BF16 storage; "
                f"shape={tensor.shape}, itemsize={tensor.itemsize}"
            )
        return tensor

    def vision(self, qkv, out=None):
        qkv = self._bf16_tensor(qkv, "Native vision attention", "qkv")
        out = self._bf16_tensor(out, "Native vision attention", "out")
        if len(qkv.shape) != 3 or qkv.shape[1:] != (256, 3456):
            raise ValueError(
                f"Native vision attention requires qkv=[views,256,3456]; got {qkv.shape}"
            )
        views, seq, channels = qkv.shape
        dim = channels // 3
        if out.shape != (views, seq, dim):
            raise ValueError(
                f"Native vision attention requires out={(views, seq, dim)}; got {out.shape}"
            )
        self.p.dispatch(
            "attention_d72_bf16",
            [qkv, out],
            {1},
            [views, seq, dim, 0, 16, 0],
            ((seq + 15) // 16, 16, views),
            128,
        )
        return out

    def gqa(self, q, k, v, *, valid_prefix, prefix_capacity, out=None):
        q, k, v, out = (
            self._bf16_tensor(tensor, "Native GQA", role)
            for tensor, role in ((q, "q"), (k, "k"), (v, "v"), (out, "out"))
        )
        if len(q.shape) != 3:
            raise ValueError(f"Native GQA requires q=[rows,heads,256]; got {q.shape}")
        rows, heads, dim = q.shape
        if (
            dim != 256
            or k.shape[1:] != (1, 256)
            or k.shape != v.shape
            or not 0 <= valid_prefix <= prefix_capacity
            or k.shape[0] not in (valid_prefix, valid_prefix + rows)
        ):
            raise ValueError(
                f"Native GQA requires compact unmasked D256 attention; "
                f"q={q.shape}, k={k.shape}, v={v.shape}, "
                f"prefix={valid_prefix}, capacity={prefix_capacity}"
            )
        if out.shape not in ((rows, heads * dim), (rows, heads, dim)):
            raise ValueError(
                f"Native GQA requires out={(rows, heads * dim)} or {q.shape}; got {out.shape}"
            )
        if self.split_decode and rows <= 16 and k.shape[0] > 128:
            keys = k.shape[0]
            splits = (keys + 127) // 128
            shape = (splits, rows, heads)
            if shape not in self.workspaces:
                self.workspaces[shape] = (
                    self.p.allocate((*shape, dim), itemsize=4),
                    self.p.allocate((*shape, 2), itemsize=4),
                )
            partial, stats = self.workspaces[shape]
            self.p.dispatch(
                "tr_gqa_split",
                [q, k, v, partial, stats],
                {3, 4},
                [rows, keys, heads, dim],
                (1, heads, splits),
                128,
            )
            self.p.dispatch(
                "tr_gqa_merge",
                [partial, stats, out],
                {2},
                [rows, keys, heads, dim],
                (rows * heads,),
                32,
            )
            return out
        self.p.dispatch(
            "tr_gqa",
            [q, k, v, out],
            {3},
            [rows, k.shape[0], heads, dim],
            ((rows + 15) // 16, heads),
            128,
        )
        return out
