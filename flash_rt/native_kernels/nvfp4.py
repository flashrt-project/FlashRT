"""Tensor-facing W4A16 native artifact, ABI 1.

This package owns packing, device capabilities and launch validation. The
Structures consumer owns discovery, guards, weight residency and activation.
All weights arrive already quantized; no precision conversion is performed.
Native source provenance: the frozen Flint SM120 implementation. Existing
raw-pointer exports and their numerical behavior remain unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_EXTENSION = None
# W4A8 (int8 activations, per-32 fp32 scales; int8 tensor cores) for rows >= _A8_MIN_M.
# Opt-in: FLASHRT_NATIVE_A8=1. The M == 1 decode path always keeps the exact W4A16 GEMV.
_A8 = os.environ.get("FLASHRT_NATIVE_A8", "0") == "1"
_A8_MIN_M = int(os.environ.get("FLASHRT_NATIVE_A8_MIN_M", "2"))
_A8_GROUPED_GU = os.environ.get("FLASHRT_NATIVE_A8_GROUPED_GU", "0") == "1"
# Dense projections switch to int8 only where the graph-captured sweep showed a win (M >= 192);
# the vocabulary projection and the routed down projection switch at _A8_MIN_M.
_A8_DENSE_MIN_M = int(os.environ.get("FLASHRT_NATIVE_A8_DENSE_MIN_M", "192"))
_A8_PER_ROW = os.environ.get("FLASHRT_NATIVE_A8_PER_ROW", "0") == "1"  # quality experiment: one scale per row
# int8 routed gate/up for chunks of at least this many rows (prefill); off by default until qualified
_A8_GU_MIN_M = int(os.environ.get("FLASHRT_NATIVE_A8_GU_MIN_M", "1000000000"))
_GROUPED_128_MIN_MEAN_ROWS = int(os.environ.get("FLASHRT_NATIVE_GROUPED_128_MIN_MEAN_ROWS", "96"))
# prefill-sized MoE on the sm120 block-scaled FP4 tensor cores (vLLM's CUTLASS grouped GEMM): W4A4 for the two
# expert GEMMs only (quality gate AR-023), dense projections and the shared expert stay on int8
_FP4_MOE_MIN_M = int(os.environ.get("FLASHRT_NATIVE_FP4_MOE_MIN_M", "1000000000"))
_FP4_SF_CACHE: dict = {}


def _fp4_sf(scales: torch.Tensor, n: int, kb: int) -> torch.Tensor:
    """Expert block scales [E, n, kb] E4M3 in the CUTLASS swizzled layout (cached per weight tensor)."""
    key = (scales.data_ptr(), n, kb)
    t = _FP4_SF_CACHE.get(key)
    if t is None:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
        e = scales.numel() // (n * kb)
        t = swizzle_blockscale(scales.reshape(e, n, kb).view(torch.float8_e4m3fn))
        if os.environ.get("FLASHRT_NATIVE_FP4_SF_CACHE", "1") == "1":
            _FP4_SF_CACHE[key] = t
    return t


@triton.jit
def _silu_mul_il_kernel(c_ptr, out_ptr, n_rows, N: tl.constexpr, BLOCK: tl.constexpr):
    # rows of [N/8 groups][gate8 | up8] bf16 -> [N] bf16: silu(gate) * up, group-interleaved layout; 512 outputs per step
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < n_rows
    j = tl.arange(0, 512)
    g = j // 8; k = j % 8
    for c0 in tl.static_range(0, N, 512):
        gate = tl.load(c_ptr + r[:, None] * (2 * N) + (2 * c0 + g * 16 + k)[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        up = tl.load(c_ptr + r[:, None] * (2 * N) + (2 * c0 + g * 16 + 8 + k)[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        y = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + r[:, None] * N + (c0 + j)[None, :], y.to(tl.bfloat16), mask=m[:, None])


def _silu_mul_interleaved(c1: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    rows = c1.shape[0]; n = c1.shape[1] // 2
    if out is None or not out.is_contiguous() or out.shape != (rows, n):
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=c1.device)
    _silu_mul_il_kernel[(triton.cdiv(rows, 16),)](c1, out, rows, N=n, BLOCK=16, num_warps=4)
    return out


def _fp4_moe_prefill(x, ids, tw, egc, egs, egg, edc, eds, edg, shared, out, m):
    """x [m, 2048] bf16 -> out = sum_topk tw * expert(x) + shared on the CUTLASS NVFP4 grouped GEMMs, with the
    fused gather/quant, silu*up/quant and un-shuffle/combine kernels of fp4_moe.py."""
    import vllm._custom_ops as ops
    from .fp4_moe import act_amax, combine_topk, gather_quant_nvfp4, row_experts, silu_mul_quant_nvfp4
    dev = x.device
    E = egg.numel(); topk = ids.shape[1]          # expert banks are flattened [E * rows, K / 2]
    n_int, k_hid = 512, 2048
    i32 = torch.int32
    M = m * topk
    expert_offsets = torch.empty(E + 1, dtype=i32, device=dev); bs_offsets = torch.empty(E + 1, dtype=i32, device=dev)
    ps1 = torch.empty((E, 3), dtype=i32, device=dev); ps2 = torch.empty((E, 3), dtype=i32, device=dev)
    a_map = torch.empty(M, dtype=i32, device=dev); c_map = torch.empty(M, dtype=i32, device=dev)
    ops.get_cutlass_moe_mm_data(ids.to(i32), expert_offsets, ps1, ps2, a_map, c_map, E, n_int, k_hid, bs_offsets, is_gated=True)
    rexp = row_experts(expert_offsets, M, E)
    sf_rows = M + E * 128                        # upper bound of the padded scale rows (no host sync)
    g1 = (448.0 * 6.0 / torch.linalg.vector_norm(x, float("inf")).float().clamp(min=1e-6)).reshape(1)
    a_fp4, a_bs = gather_quant_nvfp4(x, a_map, rexp, expert_offsets, bs_offsets, g1, sf_rows)
    c1 = torch.empty((M, 2 * n_int), dtype=torch.bfloat16, device=dev)
    ops.cutlass_fp4_moe_mm(c1, a_fp4, egc.reshape(E, 2 * n_int, k_hid // 2), a_bs, _fp4_sf(egs, 2 * n_int, k_hid // 16),
                           (egg.float() / g1).contiguous(), ps1, expert_offsets[:-1], bs_offsets[:-1])
    # silu(gate) * up on the group-interleaved (gate8 | up8) columns, quantized in the same pass; the global
    # scale needs the activation's own range (a bound from c1 saturates blocks: KL 0.009 at 4k), one read-only pass
    g2 = (448.0 * 6.0 / act_amax(c1).clamp(min=1e-6)).reshape(1)
    act_fp4, act_bs = silu_mul_quant_nvfp4(c1, rexp, expert_offsets, bs_offsets, g2, sf_rows)
    c3 = torch.empty((M, k_hid), dtype=torch.bfloat16, device=dev)
    ops.cutlass_fp4_moe_mm(c3, act_fp4, edc.reshape(E, k_hid, n_int // 2), act_bs, _fp4_sf(eds, k_hid, n_int // 16),
                           (edg.float() / g2).contiguous(), ps2, expert_offsets[:-1], bs_offsets[:-1])
    combine_topk(c3, c_map, tw[:m].contiguous(), shared[:m], out)
# Few-row exact GEMV (weight-streaming) for 2..N rows; the tile GEMM takes over above it.
_GEMV_ROWS_MAX = int(os.environ.get("FLASHRT_NATIVE_GEMV_ROWS_MAX", "0"))
# Tensor-core few-row GEMV (weights streamed once, exact W4A16) for 2..16 rows.
_GEMV_MMA_MAX = int(os.environ.get("FLASHRT_NATIVE_GEMV_MMA_MAX", "16"))
# Weight-streaming few-row GEMM (16-column strips, whole strip in flight, 4-deep activation pipeline) for
# 2.._STREAM_MAX_M rows; the vocabulary projection and the 12288-wide dense gate|up stay on gemv_mma.
_STREAM_MAX_M = int(os.environ.get("FLASHRT_NATIVE_STREAM_MAX_M", "0"))  # off: faster alone, loses in situ (steals DRAM from the MoE critical path; see PROGRESS 10-23 marginal analysis)
# Resident-activation f16 GEMM (K-slice of the rows in smem, strips streamed, partials drained) for 9..32 rows on
# the shapes where it beats the register GEMV / tile GEMM (local graph-timed sweep): (n, k, epilogue) ->
# {row bucket: KS}. Buckets: 16 = 9..16 rows, 32 = 17..32 rows. FLASHRT_NATIVE_RESIDENT=0 disables.
_RESIDENT = int(os.environ.get("FLASHRT_NATIVE_RESIDENT", "1"))
# In situ (interleaved A/B x2): C16 +1.6% (6.96 vs 7.10 ms); the 17..32 bucket was -1% and is not used.
_RESIDENT_TABLE = {
    (5120, 2048, 0): {16: 256},             # qkv 8.8/7.9 -> 7.6
    (2048, 4096, 0): {16: 256},             # o_proj 10.0/8.2 -> 6.9
    (2048, 2048, 2): {16: 512},             # shared gate|up 8.0/6.1 -> 5.1
    (2048, 6144, 0): {16: 256},             # dense-layer down 12.7 -> 10.5
}
# optional shape filter for A/B: comma-separated "n:k" pairs (empty = all eligible shapes)
_STREAM_SHAPES = {tuple(int(v) for v in p.split(":")) for p in os.environ.get("FLASHRT_NATIVE_STREAM_SHAPES", "").split(",") if p}
# Tensor-core router (identical selections to the cluster router) for 2..32 rows.
_ROUTER_SMALL_MAX = int(os.environ.get("FLASHRT_NATIVE_ROUTER_SMALL_MAX", "64"))
# Dense split-K through global fp32 partials (independent K slices + fixed-order drain) for the one-wave
# shapes where the cluster/DSMEM reduction loses: (N, K, epilogue) -> {row bucket: (bm, stages, kb, split)}.
# Buckets: 64 covers 17..64 rows, 128 covers 65..128 rows (RTX 5090 graph-timed sweep, splitk_test.py).
_SPLITK = int(os.environ.get("FLASHRT_NATIVE_SPLITK", "0"))  # neutral in situ (C64 dense 1858 vs 1851 us); kept for study
_SPLITK_TABLE = {
    (2048, 4096, 0): {64: (32, 3, 2, 4), 128: (32, 2, 4, 2)},     # o_proj
    (2048, 1024, 0): {64: (16, 4, 2, 2), 128: (32, 2, 4, 2)},     # shared down
    (2048, 6144, 0): {64: (32, 2, 4, 4), 128: (32, 2, 4, 2)},     # dense-layer down
    (12288, 2048, 2): {64: (32, 2, 4, 2)},                        # dense-layer gate|up
    (2048, 2048, 2): {128: (32, 2, 4, 2)},                        # shared gate|up
}
# Shared expert on a side stream, overlapped with router/scan/routed experts (as vLLM does).
_SIDE_STREAM = os.environ.get("FLASHRT_NATIVE_SIDE_STREAM", "1") == "1"
_side_streams = {}


def _side_stream(device):
    st = _side_streams.get(device)
    if st is None:
        st = _side_streams[device] = torch.cuda.Stream(device=device)
    return st



def load():
    """Resolve and qualify the installed artifact once, before binding."""
    global _EXTENSION
    if _EXTENSION is None:
        try:
            from flash_rt import flash_rt_flint
        except ImportError as exc:
            raise ValueError(
                "native NVFP4 artifact unavailable; build flash_rt_flint with "
                "FLASHRT_ENABLE_FLINT=ON (see docs/native_nvfp4_api.md)"
            ) from exc
        metadata = getattr(flash_rt_flint, "TENSOR_API_METADATA", {})
        if metadata.get("abi") != 1:
            raise ValueError("native NVFP4 artifact has no compatible Tensor API ABI 1")
        if not torch.cuda.is_available():
            raise ValueError("native NVFP4 artifact requires a CUDA device")
        device = torch.cuda.get_device_capability()
        if tuple(device) not in tuple(map(tuple, metadata["capabilities"])):
            raise ValueError(
                f"native artifact does not declare device capability {device}"
            )
        _EXTENSION = flash_rt_flint
        _ov = os.environ.get("FLASHRT_NATIVE_S8_OVERRIDE")   # "dbm,dst,dkb,dsplit,gbm,gst,gkb,gsplit" (study knob)
        if _ov and hasattr(flash_rt_flint, "set_gemm_s8_override"):
            flash_rt_flint.set_gemm_s8_override(*[int(v) for v in _ov.split(",")])
    return _EXTENSION


def _tensor(tensor, dtype, shape, device, name):
    if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name}: expected {dtype} {tuple(shape)}, got {tensor.dtype} {tuple(tensor.shape)}"
        )
    if tensor.device != device or device.type != "cuda" or not tensor.is_contiguous():
        raise ValueError(f"{name}: expected contiguous tensor on {device}")


@dataclass(frozen=True)
class PackedMatrix:
    """Row-major [N,K/2] adjacent-nibble codes and [N,K/16] E4M3 bytes.

    SwiGLU gate/up uses groups of eight rows, gate then up, interleaved
    within each 16-row group. Ordinary projections use their original rows.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: float

    @property
    def n(self):
        return self.codes.shape[-2]

    @property
    def k(self):
        return self.codes.shape[-1] * 2

    def validate(self):
        if (
            self.codes.ndim != 2
            or self.k <= 0
            or self.n <= 0
            or self.k % 64
            or self.n % 128
        ):
            raise ValueError(
                "packed matrix requires rank 2, K divisible by 64 and N by 128"
            )
        _tensor(
            self.codes, torch.uint8, (self.n, self.k // 2), self.codes.device, "codes"
        )
        _tensor(
            self.scales,
            torch.uint8,
            (self.n, self.k // 16),
            self.codes.device,
            "scales",
        )
        if not math.isfinite(self.global_scale) or self.global_scale <= 0:
            raise ValueError("global scale must be finite and positive")
        return self


def _activation(x, width):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != width:
        raise ValueError(f"activation requires nonempty [M,{width}]")
    _tensor(x, torch.bfloat16, x.shape, x.device, "activation")


def _quant_a8(x):
    """bf16 [M, K] -> (int8 [M, K] in the kernel's block-permuted order, fp32 [M, K/32])."""
    ext = _EXTENSION
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, k // 32), dtype=torch.float32, device=x.device)
    (ext.quant_rows_s8_perrow if _A8_PER_ROW else ext.quant_rows_s8)(
        x.data_ptr(), m, k, x.stride(0), q.data_ptr(), s.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream)
    return q, s


def _gemm_a8(q, s, codes, scales, gs, out, epilogue):
    ext = _EXTENSION
    m, k = q.shape
    ext.gemm_w4a8(q.data_ptr(), s.data_ptr(), m, codes.data_ptr(), scales.data_ptr(), codes.shape[0], k, gs,
                  out.data_ptr(), out.shape[-1], epilogue, torch.cuda.current_stream(q.device).cuda_stream)


_FAKE_A4_MIN_M = int(os.environ.get("FLASHRT_NATIVE_FAKE_A4_MIN_M", "0"))   # study: NVFP4 (per-16 E4M3) activations
_FAKE_A4_SCOPE = set(os.environ.get("FLASHRT_NATIVE_FAKE_A4_SCOPE", "dense,gu,down").split(","))
_FAKE_A4_SMOOTH = float(os.environ.get("FLASHRT_NATIVE_FAKE_A4_SMOOTH", "0"))     # alpha; 0 = off
_FAKE_DENSE_FP8_MIN_M = int(os.environ.get("FLASHRT_NATIVE_FAKE_DENSE_FP8_MIN_M", "0"))   # study: W8A8 E4M3 dense
# prefill-size dense projections on E4M3 x E4M3 (CUTLASS scaled GEMM; quality gate AR-031)
_DENSE_FP8_MIN_M = int(os.environ.get("FLASHRT_NATIVE_DENSE_FP8_MIN_M", "1000000000"))


def _fake_e4m3_rows(t: torch.Tensor) -> torch.Tensor:
    """per-row (token / output channel) E4M3 with absmax / 448 scales, and back."""
    tf = t.float()
    sc = (tf.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-12)
    return ((tf / sc).to(torch.float8_e4m3fn).float() * sc).to(t.dtype)


def _fake_dense_fp8(x, codes, scales, gs, out, epilogue):
    """study: dense GEMM as E4M3 x E4M3 (weights from the production kernel, folded per output row; activations
    per token), bf16 matmul on the fake operands."""
    k = x.shape[1]; n = codes.shape[0]
    eye = torch.eye(k, dtype=torch.bfloat16, device=x.device)
    wt = torch.empty((k, n), dtype=torch.bfloat16, device=x.device)
    _gemm_a8(*_quant_a8(eye), codes, scales, gs, wt, 0)
    w8 = _fake_e4m3_rows(wt.t().contiguous())           # [N, K]
    if os.environ.get("FLASHRT_FAKE_FP8_DEQ_CHECK") == "1" and not getattr(_fake_dense_fp8, "_chk", False) and n == 5120:
        _fake_dense_fp8._chk = True
        from vllm.models.flint.gguf_nvfp4 import dequantize_nvfp4
        wx = wt.t().float()
        for gsv, tag in ((float(gs), "gs"), (0.5 * float(gs), "0.5gs")):
            wd = dequantize_nvfp4(codes, scales.view(torch.float8_e4m3fn), torch.tensor(gsv), torch.float32)
            print("DEQ_CHECK %s: rel err vs extracted %.3e (ratio mean %.4f)" % (tag, ((wd - wx).abs().mean() / wx.abs().mean()).item(), (wd.abs().mean() / wx.abs().mean()).item()), flush=True)
    del wt, eye
    y = _fake_e4m3_rows(x) @ w8.t()
    if epilogue == 2:
        nn = y.shape[1] // 2
        yv = y.view(y.shape[0], nn // 8, 2, 8)
        y = (torch.nn.functional.silu(yv[:, :, 0, :].float()) * yv[:, :, 1, :].float()).reshape(y.shape[0], nn)
    out.copy_(y.to(out.dtype))
_SMOOTH_CACHE: dict = {}


def _fake_dense_smooth(x, codes, scales, gs, out, epilogue):
    key = codes.data_ptr()
    # W^T extracted through the production kernel itself (identity rows), so packing / scale conventions cannot drift
    k = x.shape[1]; n = codes.shape[0]
    eye = torch.eye(k, dtype=torch.bfloat16, device=x.device)
    wt = torch.empty((k, n), dtype=torch.bfloat16, device=x.device)
    _gemm_a8(*_quant_a8(eye), codes, scales, gs, wt, 0)
    w = wt.t().float()                                                # [N, K]
    sm = _SMOOTH_CACHE.get(key)
    if sm is None:
        try:
            from vllm.forward_context import get_forward_context
            real = get_forward_context().attn_metadata is not None
        except Exception:
            real = True
        if not real:   # vLLM's profiling run (dummy activations): no calibration, plain fake A4
            del w, wt, eye
            xq = _fake_nvfp4(x)
            _gemm_a8(*_quant_a8(xq), codes, scales, gs, out, epilogue)
            return
        y_ref = (x.float() @ w.t()); y_k = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        _gemm_a8(*_quant_a8(x), codes, scales, gs, y_k, 0)
        _LOG_SMOOTH = (y_k.float() - y_ref).abs().mean() / y_ref.abs().mean()
        print("FAKE_SMOOTH extracted W check: rel err %.3e (n %d k %d) nan x %d w %d y_k %d y_ref %d" % (
            _LOG_SMOOTH.item(), n, k, torch.isnan(x).sum().item(), torch.isnan(w).sum().item(),
            torch.isnan(y_k).sum().item(), torch.isnan(y_ref).sum().item()), flush=True)
    if sm is None:
        ax = x.float().abs().amax(0).clamp(min=1e-5)                 # per input channel, from this chunk
        aw = w.abs().amax(0).clamp(min=1e-5)
        a = _FAKE_A4_SMOOTH
        sm = (ax ** a) / (aw ** (1 - a))
        sm = (sm / sm.mean()).to(torch.bfloat16)
        _SMOOTH_CACHE[key] = sm
    w_eff = _fake_nvfp4((w * sm.float()[None, :]).to(torch.bfloat16))   # re-quantized smoothed weight (per call: screen only)
    del wt, eye
    xq = _fake_nvfp4((x.float() / sm.float()[None, :]).to(torch.bfloat16))
    y = xq @ w_eff.t()                                               # [m, N] bf16
    if not getattr(_fake_dense_smooth, "_logged", False) and _SMOOTH_CACHE.get(key) is not None and x.shape[0] >= 1024:
        y_exact = x.float() @ w.t()
        dump = os.environ.get("FLASHRT_FAKE_SMOOTH_DUMP")
        if dump and n == 5120:
            torch.save({"x": x.clone(), "w": w.to(torch.bfloat16).clone(), "sm": sm.clone(), "y": y.clone(), "xq": xq.clone(), "w_eff": w_eff.clone(), "gs": float(gs)}, dump)
        print("FAKE_SMOOTH smoothed-A4 rel err %.3e (n %d k %d, sm min %.3e max %.3e)" % (
            ((y.float() - y_exact).abs().mean() / y_exact.abs().mean().clamp(min=1e-9)).item(), n, k,
            sm.float().min().item(), sm.float().max().item()), flush=True)
        if n == 2048 and k == 4096:
            _fake_dense_smooth._logged = True
    del w
    if epilogue == 2:   # swiglu on the group-interleaved (gate8 | up8) rows of interleave_gate_up
        n = y.shape[1] // 2
        yv = y.view(y.shape[0], n // 8, 2, 8)
        y = (torch.nn.functional.silu(yv[:, :, 0, :].float()) * yv[:, :, 1, :].float()).reshape(y.shape[0], n)
    out.copy_(y.to(out.dtype))
_E2M1_MID = None


def _fake_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """Round x to NVFP4 (E2M1 codes with per-16 E4M3 scales, absmax / 6) and back: the quality pre-screen for a
    W4A4 prefill path on the block-scaled tensor cores."""
    global _E2M1_MID
    if _E2M1_MID is None:
        _E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
        _E2M1_LVL = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
        _fake_nvfp4.lvl = _E2M1_LVL
    m, k = x.shape
    xf = x.float().reshape(m, k // 16, 16)
    # two-level scaling as in NVFP4: fp32 global scale g = 448 * 6 / absmax keeps the E4M3 block scales in range
    g = (448.0 * 6.0 / xf.abs().amax().clamp(min=1e-12))
    s = (g * xf.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    inv = torch.where(s > 0, g / s, torch.zeros_like(s))
    q = xf * inv
    lvl = torch.bucketize(q.abs(), _E2M1_MID)
    y = _fake_nvfp4.lvl[lvl] * torch.sign(q) * (s / g)
    return y.reshape(m, k).to(x.dtype)


def _gemm(x, codes, scales, gs, out, epilogue):
    """Internal launch only: resolution and metadata checks happened at bind."""
    ext = _EXTENSION
    if ext is None:
        raise RuntimeError("native artifact was not loaded before execution")
    m, k = x.shape
    if _DENSE_FP8_MIN_M and m >= _DENSE_FP8_MIN_M and epilogue in (0, 2):
        from .dense_fp8 import dense_fp8_gemm
        y = dense_fp8_gemm(x, codes, scales, gs, out if epilogue == 0 else None)
        if os.environ.get("FLASHRT_NATIVE_DENSE_FP8_CHECK") == "1" and not getattr(_gemm, "_fp8_checked", False):
            _gemm._fp8_checked = True
            yk = torch.empty((m, codes.shape[0]), dtype=torch.bfloat16, device=x.device)
            _gemm_a8(*_quant_a8(x), codes, scales, gs, yk, 0)
            print("DENSE_FP8 check vs int8 kernel: rel err %.3e (n %d k %d)" % (
                ((y.float() - yk.float()).abs().mean() / yk.float().abs().mean()).item(), codes.shape[0], k), flush=True)
        if epilogue == 2:
            _silu_mul_interleaved(y, out)
        elif y is not out:
            out.copy_(y)
        return
    if _FAKE_DENSE_FP8_MIN_M and m >= _FAKE_DENSE_FP8_MIN_M:
        _fake_dense_fp8(x, codes, scales, gs, out, epilogue)
        return
    if _FAKE_A4_MIN_M and m >= _FAKE_A4_MIN_M and "dense" in _FAKE_A4_SCOPE:
        if _FAKE_A4_SMOOTH > 0:
            # study: SmoothQuant-style per-input-channel factor s_k = max|x_k|^a / max|w_k|^(1-a), folded into a
            # re-quantized NVFP4 weight (self-calibrated on the first prefill chunk of each layer); the GEMM runs
            # in bf16 on the fake-quantized operands
            _fake_dense_smooth(x, codes, scales, gs, out, epilogue)
            return
        x = _fake_nvfp4(x)
    n = codes.shape[0]
    stream = torch.cuda.current_stream(x.device).cuda_stream
    if _A8 and m >= max(_A8_MIN_M, _A8_DENSE_MIN_M):
        q, s = _quant_a8(x)
        _gemm_a8(q, s, codes, scales, gs, out, epilogue)
        return
    if (2 <= m <= _STREAM_MAX_M and x.is_contiguous() and k % 512 == 0 and n % 16 == 0 and epilogue in (0, 1, 2)
            and n * k <= 64 * 1024 * 1024 and not (epilogue == 2 and n >= 12288 and m > 4)
            and (not _STREAM_SHAPES or (n, k) in _STREAM_SHAPES)):
        ext.gemm_stream_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), m, out.data_ptr(), n, k, gs,
                              epilogue, stream, 0)
        return
    if _RESIDENT and 9 <= m <= 32 and x.is_contiguous():
        ks_ = _RESIDENT_TABLE.get((n, k, epilogue), {}).get(16 if m <= 16 else 32)
        if ks_ is not None:
            part = x.new_empty(((k // ks_) * m, n), dtype=torch.float32)
            ext.gemm_resident_w4a16(x.data_ptr(), m, k, codes.data_ptr(), scales.data_ptr(), n, ks_, gs,
                                    part.data_ptr(), out.data_ptr(), out.shape[-1], epilogue, stream)
            return
    if 2 <= m <= _GEMV_MMA_MAX and x.is_contiguous() and k in (1024, 2048, 4096, 6144) and epilogue in (0, 1, 2):
        ext.gemv_mma_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), m, out.data_ptr(), n, k, gs, epilogue,
                           stream)
        return
    if _SPLITK and 16 < m <= 128 and x.is_contiguous():
        cfg = _SPLITK_TABLE.get((n, k, epilogue), {}).get(64 if m <= 64 else 128)
        if cfg is not None:
            bm, stages, kb, split = cfg
            part = x.new_empty((split * ((m + bm - 1) // bm) * bm, n), dtype=torch.float32)
            ext.gemm_w4a16_splitk(x.data_ptr(), m, k, codes.data_ptr(), scales.data_ptr(), n, k, gs, part.data_ptr(),
                                  out.data_ptr(), out.shape[-1], epilogue, bm, stages, split, kb, stream)
            return
    if 2 <= m <= _GEMV_ROWS_MAX and x.is_contiguous() and (
        (epilogue == 0 and k in (1024, 2048, 4096, 6144)) or (k == 2048 and epilogue in (1, 2))
    ):
        ext.gemv_w4a16_rows(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), m, out.data_ptr(), n, k, gs,
                            epilogue, stream)
        return
    # GEMV supports a narrower set of contraction widths than GEMM.
    if m == 1 and (
        (epilogue == 0 and k in (1024, 2048, 4096, 6144))
        or (k == 2048 and epilogue in (1, 2))
    ):
        ext.gemv_w4a16(
            codes.data_ptr(),
            scales.data_ptr(),
            x.data_ptr(),
            out.data_ptr(),
            n,
            k,
            gs,
            epilogue,
            stream,
        )
    else:
        ext.gemm_w4a16(
            x.data_ptr(),
            m,
            k,
            codes.data_ptr(),
            scales.data_ptr(),
            n,
            k,
            gs,
            out.data_ptr(),
            out.shape[-1],
            epilogue,
            stream,
        )


@torch.library.custom_op("flashrt_native::nvfp4_linear", mutates_args=())
def _linear(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, global_scale: float
) -> torch.Tensor:
    _activation(x, codes.shape[1] * 2)
    if codes.device != x.device or scales.device != x.device:
        raise ValueError("linear: weights and activation must share a device")
    out = x.new_empty((x.shape[0], codes.shape[0]))
    with torch.cuda.device(x.device):
        _gemm(x, codes, scales, global_scale, out, 0)
    return out


@_linear.register_fake
def _(x, codes, scales, global_scale):
    return x.new_empty((x.shape[0], codes.shape[0]))


@torch.library.custom_op("flashrt_native::nvfp4_dense_ffn", mutates_args=())
def _dense_ffn(
    x: torch.Tensor,
    gu_codes: torch.Tensor,
    gu_scales: torch.Tensor,
    gu_global: float,
    down_codes: torch.Tensor,
    down_scales: torch.Tensor,
    down_global: float,
) -> torch.Tensor:
    _activation(x, gu_codes.shape[1] * 2)
    intermediate = x.new_empty((x.shape[0], gu_codes.shape[0] // 2))
    out = x.new_empty((x.shape[0], down_codes.shape[0]))
    with torch.cuda.device(x.device):
        _gemm(x, gu_codes, gu_scales, gu_global, intermediate, 2)
        _gemm(intermediate, down_codes, down_scales, down_global, out, 0)
    return out


@_dense_ffn.register_fake
def _(x, gu_codes, gu_scales, gu_global, down_codes, down_scales, down_global):
    return x.new_empty((x.shape[0], down_codes.shape[0]))


@torch.library.custom_op("flashrt_native::nvfp4_lm_head", mutates_args=())
def _lm_head(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 logits = x @ W^T of an NVFP4 vocabulary projection (global scale left to the caller)."""
    _activation(x, codes.shape[1] * 2)
    out = torch.empty((x.shape[0], codes.shape[0]), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        if x.shape[0] == 1:
            _EXTENSION.gemv_fused_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), 0, 0, 0, 0, 0.0,
                                        out.data_ptr(), codes.shape[0], codes.shape[1] * 2, 1.0, 1, _stream(x))
        elif _A8 and x.shape[0] >= _A8_MIN_M:
            q, s = _quant_a8(x)
            _gemm_a8(q, s, codes, scales, 1.0, out, 1)
        else:
            _gemm(x, codes, scales, 1.0, out, 1)
    return out


@_lm_head.register_fake
def _(x, codes, scales):
    return torch.empty((x.shape[0], codes.shape[0]), dtype=torch.float32, device=x.device)


class LmHead:
    """Prepared NVFP4 vocabulary projection returning fp32 logits (M == 1: exact W4A16 GEMV)."""

    def __init__(self, matrix: PackedMatrix):
        load()
        self.matrix = matrix.validate()

    def __call__(self, x):
        m = self.matrix
        return _lm_head(x, m.codes, m.scales)


_ATTN_TARGET_CTAS = int(os.environ.get("FLASHRT_NATIVE_ATTN_CTAS", "512"))
_ATTN_CHUNK = int(os.environ.get("FLASHRT_NATIVE_ATTN_CHUNK", "128"))
# partial-sum traffic budget: splits are capped so that the fp32 partials (16 KB per split and head) stay below
# 1/_ATTN_PB of the KV bytes read (fp8: 1 KB per token and head)
_ATTN_PB = int(os.environ.get("FLASHRT_NATIVE_ATTN_PB", "8"))


def _attn_splits(n_seq, window, max_len=0):
    """KV splits per (sequence, kv head): enough CTAs for the GPU (target _ATTN_TARGET_CTAS), never shorter than
    _ATTN_CHUNK tokens per split. max_len 0 = unknown length -> the small-batch presets."""
    if max_len > 0:
        eff = min(max_len, window) if window > 0 else max_len
        chunks = max(1, (eff + _ATTN_CHUNK - 1) // _ATTN_CHUNK)
        want = max(1, (_ATTN_TARGET_CTAS + 4 * n_seq - 1) // (4 * n_seq))
        by_traffic = max(1, (eff * 1024) // (16384 * _ATTN_PB))
        return max(1, min(want, chunks, by_traffic))
    if window > 0:
        return 8 if n_seq <= 2 else (4 if n_seq <= 8 else 1)
    return 16 if n_seq <= 2 else (4 if n_seq <= 8 else (2 if n_seq <= 32 else 1))


@torch.library.custom_op("flashrt_native::nvfp4_decode_attention", mutates_args=())
def _decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
    scale: float,
    max_len: int = 0,
) -> torch.Tensor:
    """Paged-KV decode attention for q [n, 32*128] bf16 (one token per sequence) -> [n, 32*128] bf16."""
    ext = _EXTENSION
    n = q.shape[0]
    n_split = _attn_splits(n, window, max_len)
    out = torch.empty_like(q)
    o_part = torch.empty((n * n_split * 32 * 128,), dtype=torch.float32, device=q.device)
    ml_part = torch.empty((n * n_split * 32 * 2,), dtype=torch.float32, device=q.device)
    with torch.cuda.device(q.device):
        ext.decode_attention_paged_bf16(
            q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), k_cache.stride(0), k_cache.stride(1),
            k_cache.stride(2), k_cache.shape[1], block_tables.data_ptr(), block_tables.stride(0),
            seq_lens.data_ptr(), n, window, sinks.data_ptr(), scale, n_split, o_part.data_ptr(),
            ml_part.data_ptr(), out.data_ptr(), torch.cuda.current_stream(q.device).cuda_stream,
            1 if k_cache.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8) else 0, 0, 0, 0, 0, 0, 0, 0, 0,
        )
    return out


@_decode_attention.register_fake
def _(q, k_cache, v_cache, block_tables, seq_lens, sinks, window, scale, max_len=0):
    return torch.empty_like(q)

@torch.library.custom_op("flashrt_native::nvfp4_decode_attention_v4", mutates_args=())
def _decode_attention_v4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
    scale: float,
    max_len: int,
    v4c: torch.Tensor,
    v4s: torch.Tensor,
) -> torch.Tensor:
    """Paged-KV decode attention for q [n, 32*128] bf16 (one token per sequence) -> [n, 32*128] bf16."""
    ext = _EXTENSION
    n = q.shape[0]
    n_split = _attn_splits(n, window, max_len)
    out = torch.empty_like(q)
    o_part = torch.empty((n * n_split * 32 * 128,), dtype=torch.float32, device=q.device)
    ml_part = torch.empty((n * n_split * 32 * 2,), dtype=torch.float32, device=q.device)
    with torch.cuda.device(q.device):
        ext.decode_attention_paged_bf16(
            q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), k_cache.stride(0), k_cache.stride(1),
            k_cache.stride(2), k_cache.shape[1], block_tables.data_ptr(), block_tables.stride(0),
            seq_lens.data_ptr(), n, window, sinks.data_ptr(), scale, n_split, o_part.data_ptr(),
            ml_part.data_ptr(), out.data_ptr(), torch.cuda.current_stream(q.device).cuda_stream,
            2, v4c.data_ptr(), v4s.data_ptr(), v4c.stride(0), v4c.stride(1), v4c.stride(2), v4s.stride(0), v4s.stride(1), v4s.stride(2),
        )
    return out


@_decode_attention_v4.register_fake
def _(q, k_cache, v_cache, block_tables, seq_lens, sinks, window, scale, max_len, v4c, v4s):
    return torch.empty_like(q)


# ---- single-token (M == 1) fused decode pieces: the frozen frontend's PDL chain ----
def _stream(t):
    return torch.cuda.current_stream(t.device).cuda_stream


def fused_m1_attn_in(matrix, hidden, residual, norm_w32, eps):
    """residual_out = hidden (+ residual); qkv = gemv(rmsnorm(residual_out) * w). Returns (qkv, residual_out)."""
    ext = _EXTENSION
    res_out = torch.empty_like(hidden)
    qkv = hidden.new_empty((1, matrix.n))
    ext.gemv_fused_w4a16(matrix.codes.data_ptr(), matrix.scales.data_ptr(), hidden.data_ptr(),
                         0 if residual is None else residual.data_ptr(), res_out.data_ptr(), norm_w32.data_ptr(),
                         1 if residual is None else 0, eps, qkv.data_ptr(), matrix.n, matrix.k, matrix.global_scale,
                         0, _stream(hidden))
    return qkv, res_out


def rows_norm(hidden, residual, norm_w32, eps):
    """Batched residual add + RMSNorm (our PDL kernel, early trigger): returns (normed, residual_out) with
    residual_out = hidden (+ residual), like vLLM's fused_add_rms_norm."""
    ext = _EXTENSION
    m = hidden.shape[0]
    res_out = torch.empty_like(hidden)
    xn = torch.empty_like(hidden)
    ext.add_rmsnorm_rows_bf16(hidden.data_ptr(), 0 if residual is None else residual.data_ptr(), norm_w32.data_ptr(),
                              eps, res_out.data_ptr(), xn.data_ptr(), m, _stream(hidden))
    return xn, res_out


def fused_m1_linear(matrix, x, epilogue=0):
    ext = _EXTENSION
    n = matrix.n // 2 if epilogue == 2 else matrix.n
    out = torch.empty((1, n), dtype=torch.float32 if epilogue == 1 else torch.bfloat16, device=x.device)
    ext.gemv_fused_w4a16(matrix.codes.data_ptr(), matrix.scales.data_ptr(), x.data_ptr(), 0, 0, 0, 0, 0.0,
                         out.data_ptr(), matrix.n, matrix.k, matrix.global_scale, epilogue, _stream(x))
    return out


def fused_m1_moe(args, hidden, residual, norm_w32, eps):
    """Post-attention norm + router + top-6 + routed/shared experts for one token (PDL chain)."""
    ext = _EXTENSION
    (router, egc, egs, egg, edc, eds, edg, sgc, sgs, sgg, sdc, sds, sdg, scratch, _cap) = args
    ids, tw = scratch[0], scratch[1]
    logits, counter, one_act = scratch[11], scratch[12], scratch[13]
    res_out = torch.empty_like(hidden)
    xn = torch.empty_like(hidden)
    out = torch.empty_like(hidden)
    st = _stream(hidden)
    ext.router_norm_bf16(hidden.data_ptr(), residual.data_ptr(), norm_w32.data_ptr(), eps, res_out.data_ptr(),
                         xn.data_ptr(), router.data_ptr(), logits.data_ptr(), counter.data_ptr(), ids.data_ptr(),
                         tw.data_ptr(), st)
    ext.moe_gate_up_pdl_bf16(xn.data_ptr(), ids.data_ptr(), egc.data_ptr(), egs.data_ptr(), egg.data_ptr(),
                             sgc.data_ptr(), sgs.data_ptr(), sgg, one_act.data_ptr(), st)
    ext.moe_down_combine_pdl_bf16(one_act.data_ptr(), ids.data_ptr(), tw.data_ptr(), edc.data_ptr(), eds.data_ptr(),
                                  edg.data_ptr(), sdc.data_ptr(), sds.data_ptr(), sdg, out.data_ptr(), st)
    return out, res_out


def fused_m1_dense(ffn, hidden, residual, norm_w32, eps):
    """Post-attention norm fused into the dense gate|up GEMV, then the down GEMV (PDL)."""
    ext = _EXTENSION
    g, d = ffn.gate_up, ffn.down
    res_out = torch.empty_like(hidden)
    act = hidden.new_empty((1, g.n // 2))
    ext.gemv_fused_w4a16(g.codes.data_ptr(), g.scales.data_ptr(), hidden.data_ptr(), residual.data_ptr(),
                         res_out.data_ptr(), norm_w32.data_ptr(), 0, eps, act.data_ptr(), g.n, g.k, g.global_scale, 2,
                         _stream(hidden))
    return fused_m1_linear(d, act, 0), res_out


@torch.library.custom_op("flashrt_native::nvfp4_decode_attention_qkv", mutates_args=("k_cache", "v_cache"))
def _decode_attention_qkv(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
    has_rope: bool,
    eps: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
    scale: float,
    max_len: int = 0,
) -> torch.Tensor:
    """Q/K RMSNorm + RoPE + current-token KV write + paged decode attention for qkv [n, 5120] bf16 -> [n, 4096]."""
    ext = _EXTENSION
    n = qkv.shape[0]
    n_split = _attn_splits(n, window, max_len)
    out = torch.empty((n, 32 * 128), dtype=torch.bfloat16, device=qkv.device)
    o_part = torch.empty((n * n_split * 32 * 128,), dtype=torch.float32, device=qkv.device)
    ml_part = torch.empty((n * n_split * 32 * 2,), dtype=torch.float32, device=qkv.device)
    with torch.cuda.device(qkv.device):
        ext.decode_attention_paged_qkv_bf16(
            qkv.data_ptr(), qkv.stride(0), positions.data_ptr(), q_w.data_ptr(), k_w.data_ptr(),
            cos_sin.data_ptr() if has_rope else 0, cos_sin.stride(0) if has_rope else 0, eps,
            k_cache.data_ptr(), v_cache.data_ptr(), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            k_cache.shape[1], block_tables.data_ptr(), block_tables.stride(0), seq_lens.data_ptr(), n, window,
            sinks.data_ptr(), scale, n_split, o_part.data_ptr(), ml_part.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream(qkv.device).cuda_stream, 1 if k_cache.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8) else 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
    return out


@_decode_attention_qkv.register_fake
def _(qkv, positions, q_w, k_w, cos_sin, has_rope, eps, k_cache, v_cache, block_tables, seq_lens, sinks, window, scale, max_len=0):
    return torch.empty((qkv.shape[0], 32 * 128), dtype=torch.bfloat16, device=qkv.device)

@torch.library.custom_op("flashrt_native::nvfp4_decode_attention_qkv_v4", mutates_args=("k_cache", "v_cache", "v4c", "v4s"))
def _decode_attention_qkv_v4(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
    has_rope: bool,
    eps: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
    scale: float,
    max_len: int,
    v4c: torch.Tensor,
    v4s: torch.Tensor,
) -> torch.Tensor:
    """Q/K RMSNorm + RoPE + current-token KV write + paged decode attention for qkv [n, 5120] bf16 -> [n, 4096]."""
    ext = _EXTENSION
    n = qkv.shape[0]
    n_split = _attn_splits(n, window, max_len)
    out = torch.empty((n, 32 * 128), dtype=torch.bfloat16, device=qkv.device)
    o_part = torch.empty((n * n_split * 32 * 128,), dtype=torch.float32, device=qkv.device)
    ml_part = torch.empty((n * n_split * 32 * 2,), dtype=torch.float32, device=qkv.device)
    with torch.cuda.device(qkv.device):
        ext.decode_attention_paged_qkv_bf16(
            qkv.data_ptr(), qkv.stride(0), positions.data_ptr(), q_w.data_ptr(), k_w.data_ptr(),
            cos_sin.data_ptr() if has_rope else 0, cos_sin.stride(0) if has_rope else 0, eps,
            k_cache.data_ptr(), v_cache.data_ptr(), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            k_cache.shape[1], block_tables.data_ptr(), block_tables.stride(0), seq_lens.data_ptr(), n, window,
            sinks.data_ptr(), scale, n_split, o_part.data_ptr(), ml_part.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream(qkv.device).cuda_stream, 2,
            v4c.data_ptr(), v4s.data_ptr(), v4c.stride(0), v4c.stride(1), v4c.stride(2), v4s.stride(0), v4s.stride(1), v4s.stride(2),
        )
    return out


@_decode_attention_qkv_v4.register_fake
def _(qkv, positions, q_w, k_w, cos_sin, has_rope, eps, k_cache, v_cache, block_tables, seq_lens, sinks, window, scale, max_len, v4c, v4s):
    return torch.empty((qkv.shape[0], 32 * 128), dtype=torch.bfloat16, device=qkv.device)


def workspace_shapes(capacity):
    """Scratch only; returned model outputs and KV state never join this pool."""
    return [
        ((capacity, 6), torch.int32),
        ((capacity, 6), torch.float32),
        ((256,), torch.int32),
        ((capacity * 6,), torch.int32),
        ((capacity * 6,), torch.int32),
        ((3 * (capacity * 6 // 16 + 256),), torch.int32),
        ((1,), torch.int32),
        ((capacity * 6, 512), torch.bfloat16),
        ((capacity * 6, 2048), torch.bfloat16),
        ((capacity, 1024), torch.bfloat16),
        ((capacity, 2048), torch.bfloat16),
        ((256,), torch.float32),
        ((1,), torch.int32),
        ((4096,), torch.bfloat16),
    ]


@torch.library.custom_op(
    "flashrt_native::nvfp4_routed_shared_ffn", mutates_args=("scratch",)
)
def _routed_shared_ffn(
    x: torch.Tensor,
    router: torch.Tensor,
    egc: torch.Tensor,
    egs: torch.Tensor,
    egg: torch.Tensor,
    edc: torch.Tensor,
    eds: torch.Tensor,
    edg: torch.Tensor,
    sgc: torch.Tensor,
    sgs: torch.Tensor,
    sgg: float,
    sdc: torch.Tensor,
    sds: torch.Tensor,
    sdg: float,
    scratch: list[torch.Tensor],
    capacity: int,
) -> torch.Tensor:
    _activation(x, 2048)
    m = x.shape[0]
    if m > capacity:
        raise ValueError(f"routed/shared FFN rows {m} exceed capacity {capacity}")
    if len(scratch) != 14:
        raise ValueError("routed/shared FFN needs its declared scratch layout")
    ext = _EXTENSION
    if ext is None:
        raise RuntimeError("native artifact was not loaded before execution")
    (
        ids,
        tw,
        counts,
        sorted_tok,
        inv,
        tiles,
        ntiles,
        act,
        y,
        sh_act,
        shared,
        logits,
        counter,
        one_act,
    ) = scratch
    out = x.new_empty(x.shape)
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if m == 1 and not (_A8 and _A8_MIN_M <= 1):
            ext.router_topk_bf16(
                x.data_ptr(),
                router.data_ptr(),
                logits.data_ptr(),
                counter.data_ptr(),
                ids.data_ptr(),
                tw.data_ptr(),
                stream,
            )
            ext.moe_gate_up_bf16(
                x.data_ptr(),
                ids.data_ptr(),
                egc.data_ptr(),
                egs.data_ptr(),
                egg.data_ptr(),
                sgc.data_ptr(),
                sgs.data_ptr(),
                sgg,
                one_act.data_ptr(),
                stream,
            )
            ext.moe_down_combine_bf16(
                one_act.data_ptr(),
                ids.data_ptr(),
                tw.data_ptr(),
                edc.data_ptr(),
                eds.data_ptr(),
                edg.data_ptr(),
                sdc.data_ptr(),
                sds.data_ptr(),
                sdg,
                out.data_ptr(),
                stream,
            )
        else:
            mean_rows = (m * 6 + 255) // 256
            tile_rows = 16 if mean_rows <= 12 else 32 if mean_rows <= 24 else 64
            a8 = _A8 and m >= _A8_MIN_M
            if a8 and m >= _A8_GU_MIN_M and mean_rows >= _GROUPED_128_MIN_MEAN_ROWS:
                tile_rows = 128   # prefill: both grouped GEMMs on the 128-row int8 tiles
            max_tiles = m * 6 // tile_rows + 256
            side = _side_stream(x.device) if _SIDE_STREAM else None
            if side is not None:
                main = torch.cuda.current_stream(x.device)
                fork = torch.cuda.Event()
                fork.record(main)
                side.wait_event(fork)
                with torch.cuda.stream(side):
                    _gemm(x, sgc, sgs, sgg, sh_act[:m], 2)
                    _gemm(sh_act[:m], sdc, sds, sdg, shared[:m], 0)
                join = torch.cuda.Event()
                join.record(side)
            else:
                _gemm(x, sgc, sgs, sgg, sh_act[:m], 2)
                _gemm(sh_act[:m], sdc, sds, sdg, shared[:m], 0)
            if m <= _ROUTER_SMALL_MAX:
                lg = x.new_empty((4 * m, 256), dtype=torch.float32)  # k-quarter partial logits
                ext.router_topk_small_bf16(x.data_ptr(), m, router.data_ptr(), lg.data_ptr(), ids.data_ptr(),
                                           tw.data_ptr(), counts.data_ptr(), stream)
            else:
                ext.router_topk_rows_bf16(
                    x.data_ptr(), m, router.data_ptr(), ids.data_ptr(), tw.data_ptr(), counts.data_ptr(), 0, 0, stream,
                )
            ext.moe_scan_place(
                ids.data_ptr(),
                m,
                tile_rows,
                counts.data_ptr(),
                sorted_tok.data_ptr(),
                inv.data_ptr(),
                tiles.data_ptr(),
                ntiles.data_ptr(),
                0,
                0,
                stream,
            )
            if _FAKE_A4_MIN_M and m >= _FAKE_A4_MIN_M and "gu" in _FAKE_A4_SCOPE:
                x = _fake_nvfp4(x)
            if m >= _FP4_MOE_MIN_M:
                if side is not None:
                    torch.cuda.current_stream(x.device).wait_event(join)
                _fp4_moe_prefill(x, ids[:m], tw, egc, egs, egg, edc, eds, edg, shared, out, m)
                return out
            if a8 and m >= _A8_GU_MIN_M:
                # prefill-sized chunks: int8 tensor cores for gate/up too (compute-bound regime)
                xq, xs = _quant_a8(x)
                ext.gemm_w4a8_grouped(
                    xq.data_ptr(), xs.data_ptr(), sorted_tok.data_ptr(), tiles.data_ptr(), ntiles.data_ptr(),
                    max_tiles, egc.data_ptr(), egs.data_ptr(), 1024, 2048, egg.data_ptr(), act.data_ptr(), 512, 3,
                    tile_rows, mean_rows, stream,
                )
                if _FAKE_A4_MIN_M and m >= _FAKE_A4_MIN_M and "down" in _FAKE_A4_SCOPE:
                    act[: m * 6].copy_(_fake_nvfp4(act[: m * 6]))
                aq, as_ = _quant_a8(act[: m * 6])
                ext.gemm_w4a8_grouped(
                    aq.data_ptr(), as_.data_ptr(), 0, tiles.data_ptr(), ntiles.data_ptr(), max_tiles,
                    edc.data_ptr(), eds.data_ptr(), 2048, 512, edg.data_ptr(), y.data_ptr(), 2048, 0,
                    tile_rows, mean_rows, stream,
                )
            elif a8:
                aq = x.new_empty((m * 6, 512), dtype=torch.int8)
                as_ = x.new_empty((m * 6, 16), dtype=torch.float32)
                ext.gemm_w4a16_grouped_q8(
                    x.data_ptr(), sorted_tok.data_ptr(), 2048, tiles.data_ptr(), ntiles.data_ptr(), max_tiles,
                    egc.data_ptr(), egs.data_ptr(), 1024, 2048, egg.data_ptr(), aq.data_ptr(), 512, as_.data_ptr(),
                    tile_rows, mean_rows, stream,
                )
                ext.gemm_w4a8_grouped(
                    aq.data_ptr(), as_.data_ptr(), 0, tiles.data_ptr(), ntiles.data_ptr(), max_tiles,
                    edc.data_ptr(), eds.data_ptr(), 2048, 512, edg.data_ptr(), y.data_ptr(), 2048, 0,
                    tile_rows, mean_rows, stream,
                )
            else:
                ext.gemm_w4a16_grouped(
                    x.data_ptr(),
                    sorted_tok.data_ptr(),
                    2048,
                    tiles.data_ptr(),
                    ntiles.data_ptr(),
                    max_tiles,
                    egc.data_ptr(),
                    egs.data_ptr(),
                    1024,
                    2048,
                    egg.data_ptr(),
                    act.data_ptr(),
                    512,
                    3,
                    tile_rows,
                    mean_rows,
                    stream,
                )
                ext.gemm_w4a16_grouped(
                    act.data_ptr(),
                    0,
                    512,
                    tiles.data_ptr(),
                    ntiles.data_ptr(),
                    max_tiles,
                    edc.data_ptr(),
                    eds.data_ptr(),
                    2048,
                    512,
                    edg.data_ptr(),
                    y.data_ptr(),
                    2048,
                    0,
                    tile_rows,
                    mean_rows,
                    stream,
                )
            if side is not None:
                torch.cuda.current_stream(x.device).wait_event(join)
            ext.moe_combine_rows_bf16(
                y.data_ptr(),
                inv.data_ptr(),
                tw.data_ptr(),
                shared.data_ptr(),
                out.data_ptr(),
                m,
                stream,
            )
    return out


@_routed_shared_ffn.register_fake
def _(
    x,
    router,
    egc,
    egs,
    egg,
    edc,
    eds,
    edg,
    sgc,
    sgs,
    sgg,
    sdc,
    sds,
    sdg,
    scratch,
    capacity,
):
    return torch.empty_like(x)


def interleave_gate_up(codes, scales):
    """Lossless [E,gate;up,K/2] -> [E,group,gate8,up8,K/2], at bind time.

    Dense gate/up accepts a rank-two input and returns rank two. Expert banks
    accept rank three and flatten E into their physical row dimension.
    """
    if codes.ndim not in (2, 3) or codes.shape[-2] % 16:
        raise ValueError(
            "gate/up requires rank 2/3 and an intermediate width divisible by eight"
        )
    e = codes.shape[0] if codes.ndim == 3 else 1
    i = codes.shape[-2] // 2

    def convert(t):
        return (
            t.reshape(e, 2, i // 8, 8, t.shape[-1])
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(-1, t.shape[-1])
        )

    return convert(codes), convert(scales)


class Linear:
    """Prepared projection. Weights are immutable for this object's lifetime."""

    def __init__(self, matrix: PackedMatrix):
        load()
        self.matrix = matrix.validate()

    def __call__(self, x):
        m = self.matrix
        return _linear(x, m.codes, m.scales, m.global_scale)


class DenseFfn:
    """Prepared precise-SiLU FFN with interleaved gate/up weights."""

    def __init__(self, gate_up: PackedMatrix, down: PackedMatrix):
        load()
        self.gate_up, self.down = gate_up.validate(), down.validate()
        if gate_up.n != 2 * down.k or down.n != gate_up.k:
            raise ValueError("inconsistent SwiGLU gate/up and down dimensions")
        if gate_up.codes.device != down.codes.device:
            raise ValueError("FFN weights must share a device")

    def __call__(self, x):
        g, d = self.gate_up, self.down
        return _dense_ffn(
            x, g.codes, g.scales, g.global_scale, d.codes, d.scales, d.global_scale
        )


class RoutedSharedFfn:
    """Prepared sigmoid/top6 FFN for the artifact's declared profile.

    This boundary includes router, routed experts and one ungated shared
    expert. It must not be called an external-routing expert bank. All
    scratch is call-transient and may be shared only by sequential calls on
    one execution lane. State, outputs and weights never use that scratch.
    """

    def __init__(
        self,
        router,
        expert_gu: PackedMatrix,
        expert_down: PackedMatrix,
        gu_global,
        down_global,
        shared_gu: PackedMatrix,
        shared_down: PackedMatrix,
        scratch,
        capacity,
    ):
        ext = load()
        profile = tuple(ext.TENSOR_API_METADATA["routed_shared_profile"])
        if profile != (2048, 256, 6, 512, 1024):
            raise ValueError("unrecognized routed/shared artifact profile")
        if type(capacity) is not int or not 1 <= capacity <= 8192:
            raise ValueError(
                "routed/shared row capacity must be an integer in [1,8192]"
            )
        device = router.device
        for matrix, n, k in [
            (expert_gu, 256 * 1024, 2048),
            (expert_down, 256 * 2048, 512),
            (shared_gu, 2048, 2048),
            (shared_down, 2048, 1024),
        ]:
            matrix.validate()
            if (matrix.n, matrix.k) != (n, k) or matrix.codes.device != device:
                raise ValueError(
                    "routed/shared weight geometry outside artifact profile"
                )
        _tensor(router, torch.bfloat16, (256, 2048), device, "router")
        for t in [gu_global, down_global]:
            _tensor(t, torch.float32, (256,), device, "per-expert global scales")
            if not torch.isfinite(t).all().item() or not (t > 0).all().item():
                raise ValueError("per-expert global scales must be finite and positive")
        shapes = workspace_shapes(capacity)
        if len(scratch) != len(shapes):
            raise ValueError("routed/shared scratch layout mismatch")
        for index, (tensor, (shape, dtype)) in enumerate(zip(scratch, shapes)):
            _tensor(tensor, dtype, shape, device, f"scratch[{index}]")
        self.capacity = capacity
        self.args = (
            router,
            expert_gu.codes,
            expert_gu.scales,
            gu_global,
            expert_down.codes,
            expert_down.scales,
            down_global,
            shared_gu.codes,
            shared_gu.scales,
            shared_gu.global_scale,
            shared_down.codes,
            shared_down.scales,
            shared_down.global_scale,
            list(scratch),
            capacity,
        )

    def __call__(self, x):
        return _routed_shared_ffn(x, *self.args)
