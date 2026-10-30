"""Host side of the MolmoAct2 action-expert persistent layer kernel.

Builds the static tables of ``molmoact2_action_mk_sm120`` once, after the
real-data calibration: fragment-major FP8 copies of the qkv / cross-query /
up / gate weights, the per-layer pointer table (with the L2 prefetch
regions), the per-(step, layer) static scales and one parameter block per
flow step. The ctypes layouts mirror ``molmoact2_action_mk_sm120.cuh`` and
are checked against the compiled sizes.
"""
from __future__ import annotations

import ctypes

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer

_NPF = 12
_LAYER_PTRS = ("qkv_w", "qkv_b", "so_w", "so_sf", "so_b", "cq_w", "cq_b",
               "co_w", "co_sf", "co_b", "up_w", "gate_w", "up_b", "gate_b",
               "dn_w", "dn_sf", "dn_b", "ctx_k", "ctx_v")
_SCALES = ("qkv_qs", "qkv_alpha", "cq_qs", "cq_alpha", "ffn_qs", "up_alpha",
           "gate_alpha", "so_ags", "so_alpha", "co_ags", "co_alpha", "dn_ags",
           "dn_alpha", "pad0", "pad1", "pad2")
_PARAM_PTRS = ("layers", "scales", "mod_self", "mod_cross", "mod_ffn",
               "rope_cos", "rope_sin", "x", "qkv", "attn_fp4", "attn_sf", "act_fp4",
               "act_sf", "xo_part", "xo_lse", "counters", "bar", "next_first")


class _Layer(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_void_p) for n in _LAYER_PTRS]
                + [("pf_ptr", ctypes.c_void_p * _NPF),
                   ("pf_bytes", ctypes.c_ulonglong * _NPF),
                   ("pf_count", ctypes.c_int)])


class _Scales(ctypes.Structure):
    _fields_ = [(n, ctypes.c_float) for n in _SCALES]


class _Params(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_void_p) for n in _PARAM_PTRS]
                + [("num_layers", ctypes.c_int), ("rows", ctypes.c_int),
                   ("ctx_len", ctypes.c_int), ("ctx_rows", ctypes.c_int),
                   ("eps", ctypes.c_float), ("prefetch", ctypes.c_int)])


def _f32(buf_ptr: int) -> float:
    out = np.zeros(1, np.float32)
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaDeviceSynchronize()
    rc = cudart.cudaMemcpy(ctypes.c_void_p(out.ctypes.data),
                           ctypes.c_void_p(buf_ptr), 4, 2)
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed ({rc})")
    return float(out[0])


class ActionLayerKernel:
    """Static state of the persistent action-expert kernel for one pipeline."""

    PREFETCH_CTAS = 1

    def __init__(self, pipe, *, layers: int, d: int, hid: int, eps: float,
                 ctx_stride: int):
        self.fvk = pipe.fvk
        sizes = tuple(self.fvk.molmo_action_mk_struct_sizes())
        want = (ctypes.sizeof(_Layer), ctypes.sizeof(_Scales),
                ctypes.sizeof(_Params))
        if sizes[:3] != want:
            raise RuntimeError(f"action kernel struct mismatch {sizes} vs {want}")
        W = pipe.weights
        fp8, nk, sk, bias = W["fp8"], W["fp8_nk"], W["skinny"], W["bias"]
        self._keep: list[CudaBuffer] = []

        def pack(ptr: int, n: int, k: int, kn: int) -> int:
            out = CudaBuffer.device_empty(n * k, np.uint8)
            self._keep.append(out)
            rc = self.fvk.molmo_action_mk_pack_fp8(ptr, out.ptr.value, n, k, kn, 0)
            if rc != 0:
                raise RuntimeError(f"molmo_action_mk_pack_fp8 rc={rc}")
            return out.ptr.value

        table = (_Layer * layers)()
        heads = d // 96
        # head-major context: [layer][head][kv_pad][ctx_stride] bf16
        ctx_layer = heads * pipe.kv_pad * ctx_stride * 2
        for i in range(layers):
            ly = table[i]
            ly.qkv_w = pack(fp8[f"act_qkv_{i}"][0], 3 * d, d, 1)
            ly.qkv_b = bias[f"act_qkv_{i}"]
            ly.cq_w = pack(fp8[f"act_cq_{i}"][0], d, d, 1)
            ly.cq_b = bias[f"act_cq_{i}"]
            ly.up_w = pack(nk[f"act_up_{i}"][0], hid, d, 0)
            ly.up_b = bias[f"act_up_{i}"]
            ly.gate_w = pack(nk[f"act_gate_{i}"][0], hid, d, 0)
            ly.gate_b = bias[f"act_gate_{i}"]
            for short, site in (("so", "so"), ("co", "co"), ("dn", "down")):
                name = f"act_{site}_{i}"
                w_pack, sf_pack = sk[name][0], sk[name][1]
                setattr(ly, f"{short}_w", w_pack)
                setattr(ly, f"{short}_sf", sf_pack)
                setattr(ly, f"{short}_b", bias[name])
            ly.ctx_k = pipe.bufs["ctxK_hm"].ptr.value + i * ctx_layer
            ly.ctx_v = pipe.bufs["ctxV_hm"].ptr.value + i * ctx_layer
            regions = [
                (ly.qkv_w, 3 * d * d), (ly.so_w, d * d // 2),
                (ly.so_sf, d * d // 16), (ly.cq_w, d * d),
                (ly.co_w, d * d // 2), (ly.co_sf, d * d // 16),
                (ly.up_w, hid * d), (ly.gate_w, hid * d),
                (ly.dn_w, hid * d // 2), (ly.dn_sf, hid * d // 16),
                (ly.ctx_k, ctx_layer), (ly.ctx_v, ctx_layer)]
            for j, (ptr, nbytes) in enumerate(regions):
                ly.pf_ptr[j] = ptr
                ly.pf_bytes[j] = nbytes
            ly.pf_count = len(regions)
        self.layers_buf = CudaBuffer.from_numpy(
            np.frombuffer(bytes(table), np.uint8))

        # static scales, frozen from the calibration (same expressions as the
        # multi-launch route: alpha = activation scale * weight scale)
        host = pipe.fp8_scale_host
        wsc = {}
        for i in range(layers):
            wsc[("qkv", i)] = _f32(fp8[f"act_qkv_{i}"][1])
            wsc[("cq", i)] = _f32(fp8[f"act_cq_{i}"][1])
            for site in ("so", "co", "down"):
                wsc[(site, i)] = _f32(sk[f"act_{site}_{i}"][2])
        self.scale_bufs = []
        for step in range(pipe.num_steps):
            arr = (_Scales * layers)()
            for i in range(layers):
                def g(name):
                    return host[f"{name}__s{step}"]
                a = arr[i]
                a.qkv_qs = g(f"act_qkv_{i}")
                a.qkv_alpha = float(np.float32(a.qkv_qs) * np.float32(wsc[("qkv", i)]))
                a.cq_qs = g(f"act_cq_{i}")
                a.cq_alpha = float(np.float32(a.cq_qs) * np.float32(wsc[("cq", i)]))
                a.ffn_qs = g(f"act_up_{i}")
                a.up_alpha = g(f"act_up_{i}") * nk[f"act_up_{i}"][2]
                a.gate_alpha = g(f"act_gate_{i}") * nk[f"act_gate_{i}"][2]
                for short, site in (("so", "so"), ("co", "co"), ("dn", "down")):
                    ags = np.float32(float(np.float32(g(f"act_{site}_{i}"))) / 6.0)
                    setattr(a, f"{short}_ags", float(ags))
                    setattr(a, f"{short}_alpha",
                            float(ags * np.float32(wsc[(site, i)])))
            self.scale_bufs.append(CudaBuffer.from_numpy(
                np.frombuffer(bytes(arr), np.uint8)))

        rows = pipe.horizon
        self.scratch = {
            "qkv": CudaBuffer.device_zeros(32 * 3 * d, np.float16),
            "attn_fp4": CudaBuffer.device_zeros(32 * d // 2, np.uint8),
            "attn_sf": CudaBuffer.device_zeros(32 * d // 16, np.uint8),
            "act_fp4": CudaBuffer.device_zeros(32 * hid // 2, np.uint8),
            "act_sf": CudaBuffer.device_zeros(32 * hid // 16, np.uint8),
            "xo_part": CudaBuffer.device_zeros(8 * 4 * 32 * 96, np.float32),
            "xo_lse": CudaBuffer.device_zeros(8 * 4 * 32, np.float32),
            "counters": CudaBuffer.device_zeros(16, np.uint32),
            "bar": CudaBuffer.device_zeros(64, np.uint32),
        }
        sms = ctypes.c_int()
        ctypes.CDLL("libcudart.so").cudaDeviceGetAttribute(
            ctypes.byref(sms), 16, 0)   # cudaDevAttrMultiProcessorCount
        self.grid = int(sms.value)
        per_step = layers * 3 * d * 2
        self.params: list[bytes] = []
        for step in range(pipe.num_steps):
            p = _Params()
            p.layers = self.layers_buf.ptr.value
            p.scales = self.scale_bufs[step].ptr.value
            for name in ("mod_self", "mod_cross", "mod_ffn"):
                setattr(p, name, pipe.bufs[name].ptr.value + step * per_step)
            p.rope_cos = pipe.bufs["act_cos"].ptr.value
            p.rope_sin = pipe.bufs["act_sin"].ptr.value
            p.x = pipe.bufs["ax"].ptr.value
            for name in ("qkv", "attn_fp4", "attn_sf", "act_fp4", "act_sf", "xo_part",
                         "xo_lse", "counters", "bar"):
                setattr(p, name, self.scratch[name].ptr.value)
            # after the last layer, start streaming layer 0 of the next step
            p.next_first = self.layers_buf.ptr.value
            p.num_layers = layers
            p.rows = rows
            p.ctx_len = pipe.S
            p.ctx_rows = pipe.kv_pad
            p.eps = eps
            p.prefetch = self.PREFETCH_CTAS
            self.params.append(bytes(p))

    def launch(self, step: int, stream: int) -> None:
        rc = self.fvk.molmo_action_mk_layers(self.params[step], self.grid, stream)
        if rc != 0:
            raise RuntimeError(f"molmo_action_mk_layers rc={rc}")
