"""FlashRT -- Nex-N2-mini (qwen3_5_moe) kernelized NVFP4 forward.

Production prefill forward (S>1) that drives the gated fvk kernels off the
pre-quantized :class:`WeightHandles` produced by ``extract_weights_nexn2_nvfp4``.
The ``sm120`` in several kernel names is where each was written, not where it
runs: this path also serves Qwen3.6 on Jetson AGX Thor, and which of several
interchangeable kernels each step calls is a :class:`KernelPolicy` below.
Every heavy op runs on a FlashRT kernel -- no ``torch`` matmul, no
``F.scaled_dot_product_attention``, no host-side sync in the hot path -- so the
prefill is fully on-device and bit-reproducible (it seeds the decode state).

Compute path:
  * Dense projections (full-attn q/k/v/o, GDN in/out_proj, router, shared
    gate/up/down, lm_head): the deterministic ``w16a16_gemm_sm120`` (BF16
    weight x BF16 act, FP32 register accumulate, single pass over K). Matches
    the fp32 argmax, bit-identical run-to-run -- so it can seed decode. The
    non-red-line projections may instead take NVFP4 W4A16 under
    ``quant_scope='full'`` (``fp4_w4a16_gemm_sm120``).
  * Full-attn: vendored FA2 causal (``flash_rt_fa2.fwd_bf16_causal``), native
    GQA (KV stays at 2 heads, no repeat_interleave). Winner of the prefill
    attention meta-test (cos 1.0; beats flash_attn pip ~4%, Sage rejects
    HD=256) -- see nexn2_dev/tests/phase_attn_metatest.py.
  * GDN linear attn: WY chunked delta-rule (``linear_attn_gdn_wy_*``) +
    fused gating / gated-norm / causal-conv1d / partial-RoPE kernels.
  * MoE: NVFP4 block-tile mma (``moe_blocktile_mma_sm120``) over sync-free
    tiles + deterministic unpermute; router softmax/topk on torch CUDA ops.

All fvk pointer args bind to named tensors first -- an inline
``x.to(bf16).contiguous().data_ptr()`` temporary is GC'd before the
kernel launches and reads freed memory (validated regression: 0.479 vs
1.0). See feedback_ctypes_temp_tensor_gc.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _cs():
    """Current CUDA stream handle.

    Inside a graph capture this is the capture stream; eager, the default one.
    A hard-coded 0 is not the same thing: during capture it names a stream that
    is not being captured, which is an illegal access at capture_end. This
    forward was written for prefill, which is never captured -- a speculative
    verify block is.
    """
    return torch.cuda.current_stream().cuda_stream


from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes

# Static Nex-N2-mini dims (config.json:text_config). Kept module-local so
# the forward reads like the validation script it was lifted from.
HID = 2048
NK, NV, HK, HV, KS = 16, 32, 128, 128, 4      # GDN: 16 K-heads / 32 V-heads
KD, VD = NK * HK, NV * HV                       # 2048 / 4096
CONV = KD + KD + VD                             # 8192 in_proj_qkv conv channels
NQ, NKV, HD, ROPE = 16, 2, 256, 64              # full-attn GQA + partial rope
INTER, TOPK = 512, 8


def build_rope_tables(seq_len, theta, rope_dim, device):
    """(cos, sin) each (S, rope_dim) bf16 -- HF cat([freqs, freqs]) layout."""
    inv = 1.0 / (theta ** (
        torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    ang = torch.arange(seq_len, device=device).float()[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _rms(x, w, eps):
    """Plain (already (1+w)-folded) RMSNorm in fp32, bf16 out (torch ref)."""
    v = x.float().pow(2).mean(-1, keepdim=True)
    return ((x.float() * torch.rsqrt(v + eps)) * w.float()).to(torch.bfloat16)


def _rms_k(x, w, fvk, device, eps):
    """RMSNorm via the fused fvk kernel (fp32 internal, bf16 out) -- the
    kernelized prefill replacement for _rms. w is the (1+w)-folded weight; the
    kernel normalises each row over the last dim. Bit-equivalent to _rms."""
    shp = x.shape
    dim = shp[-1]
    x2 = x.reshape(-1, dim).contiguous()
    out = torch.empty(x2.shape[0], dim, dtype=torch.bfloat16, device=device)
    fvk.rms_norm(x2.data_ptr(), w.data_ptr(), out.data_ptr(),
                 x2.shape[0], dim, eps, _cs())
    return out.reshape(shp)


def _add_rms_k(h, x, w, fvk, device, eps):
    """h += x in place, and return rmsnorm(h, w).

    The residual add and the norm that always follows it were a tensor add and
    a separate kernel: two passes over (S, HID) and two launches per half
    layer, 160 of each per forward. Same weight and eps convention as the
    plain norm, so this is the two of them and not a third behaviour.

    Both must be bf16 and contiguous -- the kernel writes through `h`.
    """
    dim = h.shape[-1]
    h2 = h.reshape(-1, dim)
    out = torch.empty(h2.shape[0], dim, dtype=torch.bfloat16, device=device)
    fvk.residual_add_rms_norm(
        h2.data_ptr(), x.reshape(-1, dim).data_ptr(), w.data_ptr(),
        out.data_ptr(), h2.shape[0], dim, eps, _cs())
    return out.reshape(h.shape)


def _proj(x2d, ld, base, n, fvk, device):
    """y = x @ w.T for one projection, dispatching on the loader's scope.

    NVFP4 site (``<base>_packed`` present) -> fp4 W4A16 GEMM. Otherwise the
    weight was kept BF16 (``<base>_w_t``, quant_scope='experts') -> cuBLAS
    matmul with fp32 accumulate. ``n`` is ignored on the BF16 path (taken
    from the weight).
    """
    if ld.get(base + '_packed') is not None:
        return _nvfp4_gemm(x2d, ld[base + '_packed'], ld[base + '_sf'],
                           ld[base + '_alpha'], n, fvk, device)
    w = ld[base + '_w_t']
    # A speculative verify block is a handful of rows, so the M>=64 heuristic
    # below would send it to the BF16 GEMM -- four times the weight bytes, on
    # the pass whose whole purpose is to read the weights once. It wants the
    # 4-bit weight for the same reason decode does: at this M the cost is
    # traffic, not throughput.
    if (_SPEC_VERIFY and x2d.shape[0] <= 8
            and (w.shape[1] % 16) == 0):
        return _w4a16_mrows(x2d, w, ld, base + '_w_t', fvk, device)
    if (_DENSE_W4A16 and x2d.shape[0] >= 64
            and (w.shape[0] % 64) == 0 and (x2d.shape[1] % 64) == 0):
        return _gemm_w4a16(x2d, w, ld, base + '_w_t', fvk, device)
    if (_DENSE_W16A16 and x2d.shape[0] >= _DENSE_BF16_MIN_M
            and (x2d.shape[1] % 64) == 0):
        return _gemm_w16a16(x2d, w, fvk, device)
    if _DENSE_FP4:
        return _gemm_fp4(x2d, w, ld, base + '_w_t', fvk, device)
    if _DENSE_BF16 and x2d.shape[0] >= _DENSE_BF16_MIN_M:
        # BF16 cuBLAS matmul. Faster than the hand-tuned kernel but its split-K
        # accumulation is non-deterministic (flips near-tie argmaxes run-to-run,
        # breaking the token-exact red line) -- kept only for A/B, not default.
        return (x2d.to(torch.bfloat16) @ w.t()).to(torch.bfloat16)
    return (x2d.float() @ w.float().T).to(torch.bfloat16)


# cuBLASLt for the same product, where the build has it. It is 4-8x the
# hand-written kernel at every shape prefill issues -- 140 ms of a 1024-token
# prefill against 22.5 -- and it is a drop-in in the strongest sense: bitwise
# identical output at every shape checked, and bit-reproducible across repeated
# launches.
#
# The determinism caveat elsewhere in this file is about torch.matmul, whose
# split-K reduction order can vary and flip a near-tie argmax. It does not apply
# to this entry point, which was measured rather than assumed. Set False to
# force the hand-written kernel.
import os as _os_early


class KernelPolicy:
    """Which implementation each interchangeable step of this model calls.

    The forward and decode paths have, at several steps, more than one kernel
    that computes the same thing: a fused form and the chain it replaces, a
    warp-per-row form and a block-per-row one, an "edge" shared-memory layout
    and the original. Each pair was checked against the other with
    ``torch.equal`` -- not a tolerance -- so which one runs decides speed and
    cannot decide output.

    They are gathered here rather than decided at each call site by asking the
    module what symbols it happens to export. A kernel appearing in a build is
    not a reason to change what an already-validated model path does; a caller
    saying so is. The frontend owns one of these and can hand a different one
    down, and the environment variables that predate it remain the defaults, so
    an existing configuration behaves exactly as it did.

    Fields are read at call time. A policy must therefore not be changed
    between a CUDA graph capture and its replay -- the replay repeats whichever
    branch the capture took, so the two would disagree.
    """

    __slots__ = ('dense_cublaslt', 'cublaslt_max_algos', 'wy_gdn',
                 'edge_w4a16', 'route_kernel', 'fused_shared_combine',
                 'warp_router_topk', 'gdn_recurrent_edge', 'verify_k_rows')

    def __init__(self, *,
                 dense_cublaslt=None,
                 cublaslt_max_algos=1,
                 wy_gdn=None,
                 edge_w4a16=None,
                 route_kernel=None,
                 fused_shared_combine=True,
                 warp_router_topk=True,
                 gdn_recurrent_edge=True,
                 verify_k_rows=None):
        env = _os_early.environ.get

        def _flag(value, name, default='1'):
            if value is not None:
                return bool(value)
            return env(name, default) != '0'

        # cuBLASLt for the dense bf16 GEMMs; the in-house kernel is also
        # deterministic but 66% slower at 2048 (693 against 418 ms).
        self.dense_cublaslt = _flag(dense_cublaslt, 'NEXN2_DENSE_CUBLASLT')
        # How many cuBLASLt candidates the first call for a shape times. 1
        # takes the heuristic's own pick; see _gemm_w16a16.
        self.cublaslt_max_algos = int(cublaslt_max_algos)
        # WY chunked gated-delta scan for the GDN prefill instead of the
        # sequential scan (11x at S=2048).
        self.wy_gdn = _flag(wy_gdn, 'NEXN2_WY_GDN')
        # The "edge" shared-memory layout of the two weight-only 4-bit GEMVs.
        self.edge_w4a16 = _flag(
            edge_w4a16, 'FLASHRT_QWEN35MOE_W4A16_EDGE')
        # The five-kernel routing producer instead of the tensor chain.
        self.route_kernel = _flag(route_kernel, 'NEXN2_ROUTE_KERNEL')
        # Decode-side fusions, each bit-identical to the chain it replaces.
        self.fused_shared_combine = bool(fused_shared_combine)
        self.warp_router_topk = bool(warp_router_topk)
        self.gdn_recurrent_edge = bool(gdn_recurrent_edge)
        # Run a speculative verify window through the decode kernels at k+1
        # rows rather than through the prefill forward. Two names, as the rest
        # of this model's variables have: the generic one and the one it
        # shipped under.
        if verify_k_rows is not None:
            self.verify_k_rows = bool(verify_k_rows)
        else:
            self.verify_k_rows = env(
                'FLASHRT_QWEN35MOE_VERIFY_K_ROWS',
                env('FLASHRT_NEXN2_VERIFY_K_ROWS', '1')) != '0'

    def __repr__(self) -> str:                              # pragma: no cover
        fields = ', '.join(
            f'{name}={getattr(self, name)!r}' for name in self.__slots__)
        return f'KernelPolicy({fields})'


_POLICY = KernelPolicy()


def kernel_policy():
    """The policy the forward and decode paths are currently reading."""
    return _POLICY


def set_kernel_policy(policy):
    """Install ``policy``; returns the one it replaced.

    Not to be called between a CUDA graph capture and its replay.
    """
    global _POLICY
    if not isinstance(policy, KernelPolicy):
        raise TypeError(
            f'expected a KernelPolicy, got {type(policy).__name__}')
    previous, _POLICY = _POLICY, policy
    return previous


# The cuBLASLt wrapper picks its algorithm by *timing* eight candidates at
# first use. Timing is noisy, so different processes pick different algorithms,
# and different algorithms reduce in different orders -- which makes the model
# itself non-deterministic across processes. Measured on one binary: the golden
# prefix came out 16/16 three times and 14/16 three times, flipping between
# exactly two token streams. Asking for one candidate takes the heuristic's own
# choice instead, and five of five processes then agree.
#
# It is not a speed trade worth making either way round: at 1024 tokens the
# timed pick is worth 1.6% warm (213.6 against 217.1 ms) and costs 25% of the
# cold time (about 1020 against 770 ms), because the timing loop runs inside
# the first call. Determinism and a faster first token for 1.6% of the warm
# path.
#
# Requested per call through KernelPolicy.cublaslt_max_algos, not by setting
# the kernel's environment variable: that
# variable is process-global and shared with every other frontend, so setting
# it here would decide the algorithm for a model loaded later in the same
# process that never asked. The kernel caches its plan per
# (M, N, K, max_algos), so this choice stays with these call sites.

# Set once, on the first call: a build predating the max_algos argument still
# links and still runs, one autotune behaviour older.
_CUBLASLT_TAKES_ALGOS = None


def _gemm_w16a16(x2d, w, fvk, device):
    """y = x @ w.T via the deterministic bf16-act x bf16-weight tensor-core
    GEMM (fp32 register accumulate). Matches the fp32 path's argmax (cos 1.0)
    and is bit-identical run-to-run, at ~1.75x the fp32/TF32 op."""
    global _CUBLASLT_TAKES_ALGOS
    m, k = x2d.shape
    n = w.shape[0]
    xc = x2d.contiguous()
    wc = w.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    policy = kernel_policy()
    if policy.dense_cublaslt and hasattr(fvk, 'bf16_matmul_cublaslt_bf16'):
        algos = policy.cublaslt_max_algos
        if _CUBLASLT_TAKES_ALGOS is None:
            try:
                fvk.bf16_matmul_cublaslt_bf16(
                    xc.data_ptr(), wc.data_ptr(), y.data_ptr(), m, n, k,
                    _cs(), algos)
                _CUBLASLT_TAKES_ALGOS = True
                return y
            except TypeError:
                _CUBLASLT_TAKES_ALGOS = False
        if _CUBLASLT_TAKES_ALGOS:
            fvk.bf16_matmul_cublaslt_bf16(
                xc.data_ptr(), wc.data_ptr(), y.data_ptr(), m, n, k, _cs(),
                algos)
        else:
            fvk.bf16_matmul_cublaslt_bf16(xc.data_ptr(), wc.data_ptr(),
                                          y.data_ptr(), m, n, k, _cs())
        return y
    fvk.w16a16_gemm_sm120_bf16(xc.data_ptr(), wc.data_ptr(), y.data_ptr(),
                               m, n, k, 1.0, _cs())
    return y


# True W4A16 (bf16 activation x fp4 weight) GEMM for the large prefill dense
# projections: 2.18x the fp32 path. BUT the cos cost of the dense proj is the
# fp4 *weight* (not the activation), so W4A16 lands at the same ~0.987 as W4A4
# while being slower than the CUTLASS W4A4 -- dominated. Default OFF; the 0.994
# path needs a bf16-*weight* GEMM (repurpose this kernel's 2.18x structure).
#
# Retested on a part with no W4A4 at all, where it might have been expected to
# win on traffic: at S=1024 it moves TTFT 1335.0 -> 1346.6 ms, i.e. nothing.
# The 178 ms those GEMMs cost is not what bounds this prefill.
_DENSE_W4A16 = False

# Set only around a speculative verify block; see _proj and the lm_head below.
#
# Off, on evidence, twice over. Routing the verify block through _gemm_w4a16
# loses whether or not the window is captured -- 36.15 to 33.38 eager, and
# 39.77 to 20.84 captured -- and it also reintroduces the divergence from plain
# greedy that a BF16 verify does not have.
#
# Both readings say the same thing: this is the wrong path, not the wrong idea.
# _gemm_w4a16 quantises the weight through its own helper into its own cache,
# so it is neither the tensor the decode GEMV reads nor a kernel shaped for
# three rows. What the verify wants is a small-M GEMM over the *same* packed
# weights the decode path already caches -- then it reads a quarter of the
# bytes and differs from decode only by reduction order. Until that exists,
# BF16 is both faster and the one that agrees with plain greedy token for
# token.
# On: the verify runs the dense projections through the M-row form of the
# decode GEMV, over the tensor the decode path caches. Off, it reads the same
# weights at BF16 -- four times the bytes, and a different answer from the step
# it is verifying (measured logit cosine 0.988 against decode, which is what
# made the emitted text diverge from plain greedy).
_SPEC_VERIFY_W4A16 = _os_early.environ.get(
    'NEXN2_SPEC_VERIFY_W4A16', '1') != '0'
_SPEC_VERIFY = False


def set_spec_verify(on: bool) -> None:
    global _SPEC_VERIFY
    _SPEC_VERIFY = bool(on) and _SPEC_VERIFY_W4A16

# BF16 tensor-core dense projections (vs the default fp32/TF32 matmul). The
# experts-scope q/k/v/o/out/shared/router projections dominate the prefill
# profile as fp32 GEMMs; bf16 inputs with fp32 accumulate roughly halve that
# bucket at near-identical cos (no fp4 weight rounding -> stays ~0.994).
# w16a16 masks partial M tiles (every prompt's last tile already exercises
# this), so the prefill forward kernelizes every M; the fp32 fallback below is
# only reachable for non-sm120 / K-not-multiple-of-64. (Decode M=1 has its own
# weight-bound GEMV path in _nexn2_rtx_decode.) _DENSE_BF16 (cuBLAS bf16) is an
# A/B knob only -- nondeterministic split-K, so off by default.
_DENSE_BF16 = False
_DENSE_BF16_MIN_M = 1

# Deterministic hand-tuned bf16-act x bf16-weight tensor-core GEMM (fp32 reg
# accumulate, single pass over K). Matches the fp32 path's argmax (cos 1.0),
# bit-identical run-to-run, ~1.75x the fp32/TF32 op -- so it replaces the fp32
# dense projections at prefill M with no precision or determinism cost. Default
# ON; decode (M=1) stays on its own GEMV (weight-bound -> fp4). Set False to
# fall back to the fp32 path. (K must be a multiple of 64.)
_DENSE_W16A16 = True


def _w4a16_mrows(x2d, w, ld, key, fvk, device):
    """A few rows against the 4-bit weight the *decode* path caches.

    This is what makes a speculative verify the same function as the step it
    verifies. It reads the identical packed tensor under the identical cache
    keys the decode GEMV uses -- not a second copy quantised by a different
    helper -- and runs the M-row form of that GEMV, whose per-row accumulation
    order is the GEMV's. So a verified row equals the decode row it stands in
    for, bit for bit, and the window reads the weight once rather than once per
    token and at a quarter of the bytes the BF16 path reads.
    """
    n, k = w.shape
    pk = key + '_w4a16_p'
    if pk not in ld:
        packed = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
        sf = torch.zeros(_sf_swz_bytes(n, k), dtype=torch.uint8, device=device)
        scr = torch.zeros(1, dtype=torch.float32, device=device)
        og = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.bf16_weight_to_nvfp4_swizzled(
            w.contiguous().data_ptr(), packed.data_ptr(), sf.data_ptr(),
            scr.data_ptr(), og.data_ptr(), n, k, _cs())
        torch.cuda.synchronize()
        ld[pk] = packed
        ld[key + '_w4a16_sf'] = sf
        ld[key + '_w4a16_a'] = float(og.item())
    m = x2d.shape[0]
    xc = x2d.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    rc = fvk.w4a16_mrows_edge_sm120_bf16(
        xc.data_ptr(), ld[pk].data_ptr(), ld[key + '_w4a16_sf'].data_ptr(),
        y.data_ptr(), m, n, k, ld[key + '_w4a16_a'], _cs())
    if rc:
        raise RuntimeError(f'M-row W4A16 failed with {rc} at M={m}')
    return y


def _gemm_w4a16(x2d, w, ld, key, fvk, device):
    """y = x @ w.T via the bf16-act x fp4-weight tensor-core GEMM. Weight
    quantised to NVFP4 once (cached); activation stays BF16 (precise)."""
    m, k = x2d.shape
    p, s, a = _wquant(w, ld, key, fvk, device)
    n = p.shape[0]
    xc = x2d.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.w4a16_gemm_sm120_bf16(xc.data_ptr(), p.data_ptr(), s.data_ptr(),
                              y.data_ptr(), m, n, k, a, _cs())
    return y


# A/B option (off by default): route the non-red-line dense projections
# (full-attn q/k/v/o, GDN out_proj, shared expert) through the fp4 W4A4 GEMM
# (weight quantised once). It crosses the llama.cpp prefill target but the fp4
# *activation* drops cos to 0.987 (tight against the 0.984 red line), so the
# shipped default is the BF16-weight w16a16 GEMM below (_DENSE_W16A16) -- same
# argmax as fp32, deterministic, ~1.75x. _DENSE_FP4 is kept only for bisection.
_DENSE_FP4 = False


def _wquant(w, ld, key, fvk, device):
    """Quantise a bf16 weight to swizzled NVFP4 once, cached on ld[key+'_w4*']."""
    pk = key + '_w4p'
    if pk not in ld:
        nn, kk = w.shape
        p = torch.empty(nn, kk // 2, dtype=torch.uint8, device=device)
        s = torch.zeros(_sf_swz_bytes(nn, kk), dtype=torch.uint8, device=device)
        scr = torch.zeros(1, dtype=torch.float32, device=device)
        og = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.bf16_weight_to_nvfp4_swizzled(
            w.contiguous().data_ptr(), p.data_ptr(), s.data_ptr(),
            scr.data_ptr(), og.data_ptr(), nn, kk, _cs())
        torch.cuda.synchronize()
        ld[pk] = p
        ld[key + '_w4s'] = s
        ld[key + '_w4a'] = float(og.item())
    return ld[pk], ld[key + '_w4s'], ld[key + '_w4a']


def _gemm_fp4(x2d, w, ld, key, fvk, device, xp=None, xsf=None):
    """y = x @ w.T via the fp4 block-scaled GEMM (W4A4, bf16 out). 6-10x the
    fp32 path at large M. Weight quantised once (cached); activation quantised
    per call unless (xp, xsf) are supplied (shared across same-input projs)."""
    m, k = x2d.shape
    p, s, a = _wquant(w, ld, key, fvk, device)
    n = ld[key + '_w4p'].shape[0]
    if xp is None:
        xp, xsf = _quant_act(x2d, fvk, device)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        xp.data_ptr(), p.data_ptr(), y.data_ptr(), m, n, k,
        xsf.data_ptr(), s.data_ptr(), a, _cs())
    return y


def _quant_act(x2d, fvk, device, stream=0):
    """Quantise one (M,K) bf16 activation to NVFP4 swizzled; return (xp, xsf).

    Split out so a shared activation (e.g. the M=1 decode token routed to
    several experts) is quantised once and reused across GEMMs.
    """
    m, kk = x2d.shape
    xc = x2d.contiguous()
    xp = torch.empty(m, kk // 2, dtype=torch.uint8, device=device)
    xsf = torch.zeros(_sf_swz_bytes(m, kk), dtype=torch.uint8, device=device)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        xc.data_ptr(), xp.data_ptr(), xsf.data_ptr(), m, kk, stream)
    return xp, xsf


def _nvfp4_gemm_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, m, n, k, fvk, device,
                     stream=0, out=None):
    """y = x @ w.T from a pre-quantised activation (xp, xsf).

    ``out`` lets a caller point the result at a slice of a buffer it already
    owns, which is what the per-expert loop wants: it writes 256 blocks into
    one matrix, and allocating each of them separately costs more in Python
    than the GEMM costs on the device.
    """
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device) if out is None \
        else out
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        xp.data_ptr(), wp_ptr, y.data_ptr(), m, n, k,
        xsf.data_ptr(), wsf_ptr, alpha, stream)
    return y


def _nvfp4_gemm(x2d, wp_ptr, wsf_ptr, alpha, n, fvk, device, stream=0):
    """y = x @ w.T via NVFP4. x2d is (M,K) bf16; weight given by ptrs+alpha.

    Activation quantised per call (swizzled). All ptr args are bound to
    named tensors that outlive the launch.
    """
    m, kk = x2d.shape
    xp, xsf = _quant_act(x2d, fvk, device, stream)
    return _nvfp4_gemm_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, m, n, kk,
                            fvk, device, stream)


def _silu_mul(g, u, fvk, device):
    """out = silu(g) * u via one fused kernel (was 4 torch ops). g, u bf16."""
    n = g.numel()
    gc = g.reshape(-1).contiguous()
    uc = u.reshape(-1).contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    fvk.silu_mul_sm120_bf16(gc.data_ptr(), uc.data_ptr(), out.data_ptr(), n, _cs())
    return out.reshape(g.shape)


# WY chunked gated-delta-rule scan for the GDN prefill. The seq-scan kernel is
# O(S) sequential per head with only NV=32 blocks (19% occupancy on the 5090),
# so it is the prefill wall (~9.4 ms/layer at S=2048). The WY/UT chunked form
# (FLA delta rule) runs the intra-chunk work as tensor-core matmuls and only
# the inter-chunk state recurrence is sequential -> 11x faster at S=2048,
# bit-exact (out cos 0.99998, state cos 0.99997 vs the seq-scan). Default on.
import os as _os
_WY_MIN_S = 64        # below this the seq-scan's lower fixed overhead wins


def _gdn_wy_chunk(qb, kb, v, g, beta, fvk, device, init_state=None):
    """WY chunked scan. qb/kb (S,32,128) raw post-conv with q/k already
    GQA-broadcast across the 32 v-head slots, v (S,32,128), g/beta (S,32).
    Returns core (S,32,128) + final state (32,128,128).

    ``init_state`` (NV,HK,HV) is the recurrent state to continue from -- the
    chunk_h kernel reads it as h0[0] and writes the post-block state back, so a
    chunked prefill carries it across blocks (probe-verified bit-exact: whole
    vs two state-carried halves match at cos 1.0). Defaults to zeros.

    Pipeline (FLA chunked delta rule, kernels throughout): norm+pack_q+cumsum
    -> kkt -> solve_tril(+pack) -> recompute_wu -> chunk_h (inter-chunk state)
    -> pack_v -> output_o.
    """
    S = qb.shape[0]
    chunks = (S + 63) // 64
    CH, QKG = 64, NV // NK

    # l2norm of q and k, the GQA broadcast of q into the 32 v-head slots, the
    # chunk-major packing of q, and the per-chunk gate cumulative sum, in one
    # kernel. The broadcast never materialises and q is normalised straight
    # into its packed slots, so the only q traffic is the packed write.
    k_l2 = torch.empty(S, NK, HK, dtype=torch.bfloat16, device=device)
    q_pack = torch.empty(chunks, NV, CH, HK, dtype=torch.bfloat16,
                         device=device)
    gc = torch.empty(S, NV, dtype=torch.bfloat16, device=device)
    fvk.gdn_wy_norm_pack_q_cumsum_edge_bf16(
        qb.data_ptr(), kb.data_ptr(), g.data_ptr(), k_l2.data_ptr(),
        q_pack.data_ptr(), gc.data_ptr(), S, NK, NV, HK, QKG, _cs())

    k_pack = torch.empty(chunks, NK, CH, HK, dtype=torch.bfloat16, device=device)
    kkt_base = torch.empty(chunks, NK, CH, CH, dtype=torch.float32, device=device)
    A = torch.empty(chunks, NV, CH, CH, dtype=torch.float32, device=device)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        k_l2.data_ptr(), beta.data_ptr(), gc.data_ptr(), k_pack.data_ptr(),
        kkt_base.data_ptr(), A.data_ptr(), S, NK, NV, HK, QKG, _cs())

    Ai = torch.empty(chunks, NV, CH, CH, dtype=torch.float32, device=device)
    Ai_pack = torch.empty(chunks, NV, CH, CH, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_parallel_pack(
        A.data_ptr(), Ai.data_ptr(), Ai_pack.data_ptr(), S, NV, _cs())

    w_pack = torch.empty(chunks, NV, CH, HV, dtype=torch.bfloat16, device=device)
    u_pack = torch.empty(chunks, NV, CH, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla(
        k_l2.data_ptr(), v.data_ptr(), beta.data_ptr(), gc.data_ptr(),
        Ai_pack.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(),
        S, NK, NV, HK, QKG, _cs())

    state = (init_state.clone() if init_state is not None
             else torch.zeros(NV, HK, HV, dtype=torch.bfloat16, device=device))
    h0 = torch.empty(chunks, NV, HK, HV, dtype=torch.bfloat16, device=device)
    v_new = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla(
        k_l2.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(), gc.data_ptr(),
        state.data_ptr(), h0.data_ptr(), v_new.data_ptr(), 0, 0,
        S, NK, NV, HK, QKG, _cs())

    # v is the only side still needing a packed copy; the raw-K output_o does
    # the GQA expansion of k in-kernel, so k never gets a 32-head buffer.
    v_pack = torch.empty(chunks, NV, CH, HV, dtype=torch.bfloat16,
                         device=device)
    fvk.gdn_wy_pack_v_edge_bf16(
        v_new.data_ptr(), v_pack.data_ptr(), S, NV, HV, _cs())
    core = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_output_o_b64_bf16_mma_fla_rawk(
        q_pack.data_ptr(), k_l2.data_ptr(), v_pack.data_ptr(),
        h0.data_ptr(), gc.data_ptr(), core.data_ptr(),
        S, NK, NV, HV, QKG, float(HV ** -0.5), _cs())
    return core, state


def _gdn_layer(h, ld, fvk, device, eps, cap=None, rank=None,
              init_state=None, conv_hist=None):
    """Gated DeltaNet (linear_attention) layer. h (1,S,HID) -> (1,S,HID).

    When ``cap`` is given (a Nexn2DecodeState), the final recurrent state and
    the last KS-1 conv inputs are written into its decode buffers so a batched
    prefill leaves exactly the state the per-token decode path would.

    ``init_state`` (NV,HK,HV) continues the recurrent scan from a previous block
    and ``conv_hist`` (1,CONV,KS-1) supplies the conv's causal history -- both
    for chunked prefill. With both None (batched / first block) the behaviour is
    identical to the single-pass prefill.
    """
    B, S, _ = h.shape
    Wqkv = ld['in_proj_qkv_w_t']
    Wz = ld['in_proj_z_w_t']
    Wb, Wa = ld['in_proj_b_w_t'], ld['in_proj_a_w_t']
    convw = ld['conv1d_w_t']
    A_log, dtb = ld['A_log_t'].float(), ld['dt_bias_t'].float()
    nw = ld['gdn_norm_w_t']

    # in_proj must NOT be quantized (red line: fp4 weight/act collapses GDN).
    # The big mixed (N=CONV) and z (N=NV*HV) projections route through the
    # deterministic bf16-weight w16a16 GEMM -- bf16 weight + bf16 act + fp32
    # accumulate is the same precision as the fp32 path (no quantization), just
    # on bf16 tensor cores. The tiny b/a (N=NV=32) stay on the fp32 matmul.
    h2 = h.reshape(B * S, HID)
    if _DENSE_W16A16 and (B * S) >= _DENSE_BF16_MIN_M:
        mixed = _gemm_w16a16(h2, Wqkv, fvk, device).reshape(B, S, -1)
        z = _gemm_w16a16(h2, Wz, fvk, device).reshape(B, S, NV, HV)
        b = _gemm_w16a16(h2, Wb, fvk, device).reshape(B, S, NV)
        a = _gemm_w16a16(h2, Wa, fvk, device).reshape(B, S, NV)
    else:
        mixed = (h.float() @ Wqkv.float().T).to(torch.bfloat16)
        z = (h.float() @ Wz.float().T).reshape(B, S, NV, HV)
        b = (h.float() @ Wb.float().T).to(torch.bfloat16)
        a = (h.float() @ Wa.float().T).to(torch.bfloat16)

    # causal depthwise conv1d + silu via the fused kernel (was F.conv1d glue).
    # Same (B, S, conv_dim) layout the decode update kernel uses; no bias. For a
    # chunked block, prepend the previous block's last KS-1 inputs (conv_hist)
    # so the block's first outputs see the right history, then drop them.
    # The row-blocked entry walks several tokens per thread with the window in
    # registers, so each input is read once instead of once per output that
    # needs it. Bit-identical to the per-token entry (probe: exact at every
    # length measured) and 3.3x, which puts it within 1.3x of what its traffic
    # implies rather than 4.2x off. It also lifts the gridDim.y ceiling that
    # capped a single launch at 65535 tokens.
    _conv = getattr(fvk, 'causal_conv1d_qwen36_rows_bf16',
                    fvk.causal_conv1d_qwen36_bf16)
    convw_k = convw.reshape(CONV, KS).contiguous()
    xc = torch.empty(B, S, CONV, dtype=torch.bfloat16, device=device)
    _hist_conv = getattr(fvk, 'causal_conv1d_qwen36_rows_hist_bf16', None)
    if conv_hist is not None and _hist_conv is not None:
        # The conv reads the previous block's trailing inputs where it needs
        # them. Prepending them to the activations instead meant concatenating
        # and then slicing the whole block back off -- two copies of it per
        # layer, 691 ms of a 32768-token prefill, to supply three tokens.
        # conv_hist is already (1, CONV, KS-1), newest last, which is the
        # layout the kernel reads, so the transpose goes with the copy.
        _hist_conv(mixed.contiguous().data_ptr(), convw_k.data_ptr(), 0,
                   conv_hist.contiguous().data_ptr(), xc.data_ptr(),
                   B, S, CONV, KS, True, _cs())
    elif conv_hist is not None:
        hist = conv_hist[0].transpose(0, 1).reshape(1, KS - 1, CONV)
        mixed_ext = torch.cat(
            [hist.to(mixed.dtype), mixed], dim=1).contiguous()
        Se = mixed_ext.shape[1]
        xc_ext = torch.empty(B, Se, CONV, dtype=torch.bfloat16, device=device)
        _conv(mixed_ext.data_ptr(), convw_k.data_ptr(), 0,
              xc_ext.data_ptr(), B, Se, CONV, KS, True, _cs())
        xc = xc_ext[:, KS - 1:, :].contiguous()
    else:
        _conv(mixed.contiguous().data_ptr(), convw_k.data_ptr(), 0,
              xc.data_ptr(), B, S, CONV, KS, True, _cs())
    # split conv output + broadcast q/k 16 -> 32 heads in one fvk kernel.
    xc_bf = xc.reshape(B * S, CONV).contiguous()
    qb = torch.empty(B, S, NV, HK, dtype=torch.bfloat16, device=device)
    kb = torch.empty(B, S, NV, HK, dtype=torch.bfloat16, device=device)
    vb = torch.empty(B, S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_lin_split_qkv_broadcast_bf16(
        xc_bf.data_ptr(), qb.data_ptr(), kb.data_ptr(), vb.data_ptr(),
        B * S, _cs())

    neg = (-A_log.exp()).float().contiguous()
    dtb_c = dtb.contiguous()
    a_bf = a.to(torch.bfloat16).contiguous()
    b_bf = b.to(torch.bfloat16).contiguous()
    g_out = torch.empty(B, S, NV, dtype=torch.bfloat16, device=device)
    bo = torch.empty(B, S, NV, dtype=torch.bfloat16, device=device)
    fvk.qwen36_gdn_gating_bf16(
        a_bf.data_ptr(), b_bf.data_ptr(), neg.data_ptr(), dtb_c.data_ptr(),
        g_out.data_ptr(), bo.data_ptr(), B * S, NV, _cs())

    if kernel_policy().wy_gdn and S >= _WY_MIN_S:
        # WY chunked delta-rule scan: 11x faster than the seq-scan at S=2048,
        # bit-exact. qb/kb carry the 16->32 broadcast heads (src_h = h//2); the
        # front kernel reads the group leaders and re-expands where it packs,
        # so no strided slice is taken here.
        core, state = _gdn_wy_chunk(
            qb.reshape(S, NV, HK), kb.reshape(S, NV, HK),
            vb.reshape(S, NV, HV), g_out.reshape(S, NV),
            bo.reshape(S, NV), fvk, device, init_state=init_state)
        core = core.reshape(B, S, NV, HV)
    else:
        # Sequential scan over the whole prompt in ONE launch (state stays in
        # registers across all S timesteps -> no per-token state HBM round-trip
        # / S kernel launches). Bit-equivalent to the per-token recurrent loop
        # (out cos 0.99999); the short-prompt fallback below _WY_MIN_S.
        state = (init_state.clone() if init_state is not None
                 else torch.zeros(NV, HK, HV, dtype=torch.bfloat16,
                                  device=device))
        core = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
        fvk.gdn_recurrent_seq_sm120_bf16(
            qb.reshape(S, NV, HK).contiguous().data_ptr(),
            kb.reshape(S, NV, HK).contiguous().data_ptr(),
            vb.reshape(S, NV, HV).contiguous().data_ptr(),
            g_out.reshape(S, NV).contiguous().data_ptr(),
            bo.reshape(S, NV).contiguous().data_ptr(),
            state.data_ptr(), core.data_ptr(), S, NV, HK, True, _cs())
        core = core.reshape(B, S, NV, HV)

    if cap is not None:
        # The per-step snapshots go FIRST. `init_state` and `conv_hist` are the
        # very tensors the two copies below overwrite -- the caller passes
        # cap.lin_state[rank] / cap.lin_conv_state[rank] in directly -- so
        # replaying the scan after the copies would start it from the state the
        # block ended at, and every rewind would restore a fabricated one.
        if getattr(cap, 'spec_capture', False):
            _capture_per_token_state(cap, rank, S, init_state, conv_hist,
                                     mixed, qb, kb, vb, g_out, bo, fvk, device)
        # GDN recurrent final state = `state` after the S-step scan; conv state
        # = the last KS-1 `mixed` inputs (channel-major, newest at index -1),
        # matching the causal_conv1d_update rolling buffer (1, CONV, KS-1).
        cap.lin_state[rank].copy_(state)
        cs = mixed[0, S - (KS - 1):S, :].transpose(0, 1).contiguous()
        cap.lin_conv_state[rank].copy_(cs.unsqueeze(0))

    cf = core.reshape(-1, HV).contiguous()
    zf = z.reshape(-1, HV).to(torch.bfloat16).contiguous()
    nf = torch.empty_like(cf)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        cf.data_ptr(), zf.data_ptr(), nw.data_ptr(), nf.data_ptr(),
        cf.shape[0], HV, eps, _cs())
    out = _proj(nf.reshape(B * S, VD), ld, 'out_proj', HID, fvk, device)
    return out.reshape(B, S, HID)


# Vendored FA2 causal kernel for the prefill full-attn. The attention meta-test
# (nexn2_dev/tests/phase_attn_metatest.py) over the Nex-N2 full-attn shape
# (S, 16Q/2KV, HD=256, causal, bf16) ranks fwd_bf16_causal first at every S
# (cos 1.0; beats flash_attn pip by ~4%, Sage rejects HD=256, the cublas mha
# materialises O(S^2) scores). It also takes native GQA, so the KV no longer
# needs repeat_interleave to 16 heads (was the SDPA path). The kernel lives in
# the pre-existing flash_rt_fa2.so (already a hard dep of the decode backend),
# so this adds no new csrc.
_FA2_MOD = None
_FA2_USABLE = None
_NUM_SMS = None


def _get_fa2():
    """The vendored FA2 module, or None where the target does not build it.

    Thor is such a target: its arch list omits FA2 because it uses FA4. Absence
    is a fallback, not an error -- ``_sdpa_causal_attn`` computes the same
    thing -- so this returns None rather than raising, the way the decode
    attention backend already treats it.
    """
    global _FA2_MOD
    if _FA2_MOD is None:
        import importlib

        try:
            _m = importlib.import_module("flash_rt.flash_rt_fa2")
        except ModuleNotFoundError as exc:
            if exc.name != "flash_rt.flash_rt_fa2":
                raise
            _FA2_MOD = False
        else:
            _FA2_MOD = _m
    return _FA2_MOD or None


def _fa2_usable(device):
    """Does the vendored kernel actually compute here?

    Importing it and finding its symbols proves neither: its arch handling can
    leave a build that links, loads, prints a complaint and returns without
    writing the output, which downstream looks like wrong attention rather than
    a failure. The decode backend probes for the same reason. One launch, once.
    """
    global _FA2_USABLE
    if _FA2_USABLE is not None:
        return _FA2_USABLE
    if _get_fa2() is None:
        _FA2_USABLE = False
        return False
    g = torch.Generator(device=device).manual_seed(1)
    q = torch.randn(1, 8, NQ, HD, generator=g, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(1, 8, NKV, HD, generator=g, device=device,
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    try:
        produced = _fa2_causal_attn(q, k, v, device, _probe=True).float()
        torch.cuda.synchronize(device)
    except Exception as exc:                                 # noqa: BLE001
        import warnings

        warnings.warn(
            "Qwen3.6 FA2 causal probe failed; falling back to bottom-right "
            f"SDPA attention: {exc!r}", RuntimeWarning, stacklevel=2)
        _FA2_USABLE = False
        return False
    expected = _sdpa_causal_attn(q, k, v, device).float()
    _FA2_USABLE = bool(
        torch.isfinite(produced).all()
        and ((produced - expected).norm()
             / expected.norm().clamp_min(1e-6)).item() < 0.05)
    if not _FA2_USABLE:
        import warnings

        warnings.warn(
            "Qwen3.6 FA2 causal probe produced invalid or mismatched output; "
            "falling back to bottom-right SDPA attention.",
            RuntimeWarning, stacklevel=2)
    return _FA2_USABLE


_FLEX_CACHE = {}


def _flex_causal(sq, sk, device):
    """A bottom-right-causal flex_attention closure for this block shape.

    Block masks are built per (Sq, Sk) and cached, because a chunked prefill
    revisits the same shapes as it walks the prompt. Returns None where flex is
    unavailable, leaving the explicit-mask path to handle it.
    """
    key = (sq, sk, str(device))
    got = _FLEX_CACHE.get(key)
    if got is not None:
        return got
    if key in _FLEX_CACHE:
        return None
    try:
        from torch.nn.attention.flex_attention import (
            create_block_mask, flex_attention,
        )

        off = sk - sq

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx + off

        block_mask = create_block_mask(mask_mod, 1, NQ, sq, sk, device=device)

        def run(q, k, v):
            return flex_attention(q, k, v, block_mask=block_mask,
                                  scale=float(HD) ** -0.5, enable_gqa=True)

        _FLEX_CACHE[key] = run
        return run
    except Exception:                                        # noqa: BLE001
        _FLEX_CACHE[key] = None
        return None


def _sdpa_causal_attn(qf, kf, vf, device):
    """Reference causal GQA attention, for a build without the FA2 kernel.

    FA2 causal aligns bottom-right -- query i attends to keys [0, Sk-Sq+i] --
    which is exactly a chunked block's absolute causal window. torch's
    ``is_causal=True`` aligns top-left, and the two only agree when Sq == Sk,
    so the mask is built explicitly rather than left to a flag whose convention
    differs where it matters.
    """
    import torch.nn.functional as F

    Sq, Sk = qf.shape[1], kf.shape[1]
    q = qf.transpose(1, 2)                               # (1, NQ, Sq, HD)
    k, v = kf.transpose(1, 2), vf.transpose(1, 2)        # (1, NKV, Sk, HD)
    # An explicit mask forces the math backend, which materialises the scores
    # and runs them through SIMT fp32 GEMMs -- 131.7 ms of a 2048-token prefill
    # in twenty launches, growing as S^2. When the block is square the two
    # causal conventions coincide, so say is_causal and let the fused backend
    # take it; the mask is only needed when Sq < Sk, which is chunked prefill.
    if Sq == Sk:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=float(HD) ** -0.5, enable_gqa=True
        ).transpose(1, 2).contiguous()
    # A chunked block's window is bottom-right causal, which is_causal does not
    # mean (measured: cos 0.24 against this mask, i.e. it silently truncates the
    # history) and which a boolean mask only expresses by materialising the
    # scores. flex_attention states it as a predicate and skips fully-masked
    # blocks -- numerically right, cos 0.999997 -- but it compiles per shape,
    # and a chunked prefill hands it a new (Sq, Sk) for every chunk: measured
    # 10240 tokens 4071 -> 4330 ms, 16384 tokens 13858. Left out on that
    # evidence; it becomes the right answer once the chunk shapes are fixed and
    # warmed, which is where the long-context work goes next.
    qi = torch.arange(Sk - Sq, Sk, device=device).unsqueeze(1)
    mask = torch.arange(Sk, device=device).unsqueeze(0) <= qi
    try:
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=float(HD) ** -0.5, enable_gqa=True)
    except TypeError:                       # torch without native GQA
        groups = NQ // NKV
        o = F.scaled_dot_product_attention(
            q, k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1),
            attn_mask=mask, scale=float(HD) ** -0.5)
    return o.transpose(1, 2).contiguous()


def _num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _NUM_SMS


def _fa2_causal_attn(qf, kf, vf, device, *, _probe=False):
    """Causal GQA attention via the vendored FA2 kernel (bf16, native GQA -- no
    KV repeat). qf (1,Sq,NQ,HD), kf/vf (1,Sk,NKV,HD). Returns (1,Sq,NQ,HD).
    Sk may exceed Sq (chunked prefill: a block of Sq queries against the Sk
    accumulated KV); FA2 causal uses bottom-right alignment, so query i attends
    to keys [0, Sk-Sq+i] -- exactly the block's absolute causal window. splitkv
    off (large-q parallelism).

    Falls back to the reference where the kernel is absent or refuses. ``_probe``
    forces the kernel, since the probe is what decides that question."""
    if not _probe and not _fa2_usable(device):
        return _sdpa_causal_attn(qf, kf, vf, device)
    Sq = qf.shape[1]
    Sk = kf.shape[1]
    qc, kc, vc = qf.contiguous(), kf.contiguous(), vf.contiguous()
    o = torch.empty(1, Sq, NQ, HD, dtype=torch.bfloat16, device=device)
    lse = torch.empty(1, NQ, Sq, dtype=torch.float32, device=device)
    _get_fa2().fwd_bf16_causal(
        Q=qc.data_ptr(), K=kc.data_ptr(), V=vc.data_ptr(), O=o.data_ptr(),
        softmax_lse=lse.data_ptr(), softmax_lse_accum=0, o_accum=0,
        batch=1, seqlen_q=Sq, seqlen_k=Sk, num_heads_q=NQ, num_heads_kv=NKV,
        head_dim=HD, q_strides=qc.stride()[:3], k_strides=kc.stride()[:3],
        v_strides=vc.stride()[:3], o_strides=o.stride()[:3],
        softmax_scale=float(HD) ** -0.5, num_sms=_num_sms(), stream=_cs())
    return o


def _full_attn_layer(h, ld, ct, st, fvk, device, eps, cap=None, rank=None,
                     pos_offset=0):
    """Full GQA attention layer with output gate + partial RoPE.

    When ``cap`` is given, the RoPE'd K and V for the S block positions are
    written into the decode KV cache at [pos_offset, pos_offset+S). With
    pos_offset>0 (chunked prefill) the block's queries attend to the whole
    accumulated cache [0, pos_offset+S); with pos_offset=0 (batched) it is the
    block's own K/V, identical to the previous single-pass prefill.
    """
    B, S, _ = h.shape
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']      # already (1+w)-folded
    x2 = h.reshape(B * S, HID)

    qg = _proj(x2, ld, 'q_proj', NQ * 2 * HD, fvk, device).contiguous()
    # split interleaved [q_pre(256), gate(256)] per head via fvk kernel.
    q_pre = torch.empty(B * S, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(B * S, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), B * S, _cs())
    q = q_pre.view(B, S, NQ, HD)
    gate = gate.view(B, S, NQ * HD)
    q = _rms_k(q.to(torch.bfloat16), qnw, fvk, device, eps)
    k = _proj(x2, ld, 'k_proj', NKV * HD, fvk, device).view(B, S, NKV, HD)
    k = _rms_k(k, knw, fvk, device, eps)
    v = _proj(x2, ld, 'v_proj', NKV * HD, fvk, device).view(B, S, NKV, HD)

    qo = torch.empty(S, NQ, HD, dtype=torch.bfloat16, device=device)
    ko = torch.empty(S, NKV, HD, dtype=torch.bfloat16, device=device)
    qin = q.reshape(S, NQ, HD).contiguous()
    kin = k.reshape(S, NKV, HD).contiguous()
    ctc, stc = ct.contiguous(), st.contiguous()
    fvk.qwen36_partial_rope_qk_bf16(
        qin.data_ptr(), kin.data_ptr(), ctc.data_ptr(), stc.data_ptr(),
        qo.data_ptr(), ko.data_ptr(), S, NQ, NKV, HD, ROPE, _cs())

    # Causal GQA attention via the vendored FA2 kernel (native GQA: KV stays at
    # NKV=2, no repeat_interleave; layout is FA2's (B,S,H,HD), no transpose).
    if cap is not None:
        # Write the block's RoPE'd K + raw V into the decode KV cache at its
        # absolute slots; a chunked block then attends to all KV seen so far.
        end = pos_offset + S
        cap.attn.K_cache[rank, pos_offset:end].copy_(ko.reshape(S, NKV, HD))
        cap.attn.V_cache[rank, pos_offset:end].copy_(v.reshape(S, NKV, HD))
        if pos_offset > 0:
            kf = cap.attn.K_cache[rank, :end].reshape(1, end, NKV, HD)
            vf = cap.attn.V_cache[rank, :end].reshape(1, end, NKV, HD)
        else:
            kf = ko.reshape(1, S, NKV, HD)
            vf = v.reshape(1, S, NKV, HD)
    else:
        kf = ko.reshape(1, S, NKV, HD)
        vf = v.reshape(1, S, NKV, HD)
    at = _fa2_causal_attn(
        qo.reshape(1, S, NQ, HD), kf, vf, device).reshape(B, S, NQ * HD)
    # output gate: at * sigmoid(gate) via the fused kernel (was torch glue).
    atc = at.reshape(-1).to(torch.bfloat16).contiguous()
    gc = gate.reshape(-1).contiguous()
    ato = torch.empty_like(atc)
    fvk.sigmoid_mul_sm120_bf16(atc.data_ptr(), gc.data_ptr(),
                               ato.data_ptr(), atc.numel(), _cs())
    at = ato.reshape(B * S, NQ * HD)
    return _proj(at, ld, 'o_proj', HID, fvk, device).reshape(B, S, HID)


# The two weight-only 4-bit GEMVs each have an "edge" variant that is bitwise
# identical -- it differs only in a shared-memory layout and in decoding the
# UE4M3 scale byte arithmetically rather than through a constant-memory lookup
# whose index differs per lane. Because the outputs are identical to the bit,
# choosing between them is purely a performance decision and cannot move a
# token, so preferring the variant needs no accuracy argument. Set
# KernelPolicy.edge_w4a16 to False (or FLASHRT_QWEN35MOE_W4A16_EDGE=0) to force
# the original.


def w4a16_matvec(fvk):
    """The dense 4-bit GEMV entry point this build should call."""
    if kernel_policy().edge_w4a16:
        fn = getattr(fvk, 'w4a16_matvec_edge_sm120_bf16', None)
        if fn is not None:
            return fn
    return fvk.w4a16_matvec_sm120_bf16


def moe_grouped_w4a16(fvk):
    """The grouped per-slot 4-bit GEMV entry point this build should call."""
    if kernel_policy().edge_w4a16:
        fn = getattr(fvk, 'moe_grouped_w4a16_edge_sm120_bf16', None)
        if fn is not None:
            return fn
    return fvk.moe_grouped_w4a16_sm120_bf16


# Grouped MoE for prefill (on by default); set False to use the per-expert loop.
_USE_GROUPED_MOE = True
# One GEMM per expert, for a build without the block-scaled MMA tiles. Reads
# each expert's weight once instead of once per token that routed to it.
#
# It only pays once the tokens per expert are worth a launch. Each expert costs
# about five launches (quantise, two GEMMs, the gate), so with 256 of them a
# layer that is thousands of launches whichever way; below the threshold the
# grouped GEMV's two launches win even though it re-reads the weight. Measured
# at S=256 -- eight tokens an expert -- the per-expert path takes TTFT from 585
# to 923 ms, while at S=1024 it takes it from 2237 to 1354.
_USE_PER_EXPERT_GEMM = True
_PER_EXPERT_MIN_M = 16              # mean tokens per expert, = S * TOPK / 256
# M=16 tensor-core mma MoE: tokens are sorted into 16-row expert tiles and the
# SM120 block-scaled mma runs each expert once at full M-utilisation -- ~5.6x
# the SIMT grouped W4A16 at large S (the compute wall). W4A4 (FP4 activation),
# so cos is a touch lower; used for S >= _M16_MIN_S (small S keeps W4A16).
_USE_M16_MOE = True
_M16_MIN_S = 64
_N_EXPERTS = 256
# Multi-warp block-tile (BM=BN=64, 4 warps) W4A4 GEMM: 4.0x the M16 tile (the
# activation + weight loaded once into smem and shared across warps). Default on
# for S >= _M16_MIN_S; set False to fall back to the M16 tile.
_USE_BT_MOE = True


def _capture_per_token_state(cap, rank, S, init_state, conv_hist, mixed,
                             qb, kb, vb, g_out, bo, fvk, device):
    """Record what the recurrent state would be after each token of a block.

    A verified speculative window is accepted up to some prefix, and the layer
    that has to be rewound is this one: the KV cache is a cursor, but the
    recurrent and conv states are not -- they have already absorbed every token
    of the block, including the rejected tail.

    Rather than re-deriving them afterwards, run the scan a token at a time and
    keep each intermediate. The block is a handful of tokens, so this is a few
    extra launches and a few MB of state copies; recovering the state any other
    way means either re-running the block's projections or reconstructing the
    recurrence from saved inputs, both of which cost more than they save at
    this length.
    """
    state = (init_state.clone() if init_state is not None
             else torch.zeros(NV, HK, HV, dtype=torch.bfloat16, device=device))
    q3 = qb.reshape(S, NV, HK).contiguous()
    k3 = kb.reshape(S, NV, HK).contiguous()
    v3 = vb.reshape(S, NV, HV).contiguous()
    g2 = g_out.reshape(S, NV).contiguous()
    b2 = bo.reshape(S, NV).contiguous()
    core1 = torch.empty(1, NV, HV, dtype=torch.bfloat16, device=device)
    for t in range(S):
        fvk.gdn_recurrent_seq_sm120_bf16(
            q3[t:t + 1].data_ptr(), k3[t:t + 1].data_ptr(),
            v3[t:t + 1].data_ptr(), g2[t:t + 1].data_ptr(),
            b2[t:t + 1].data_ptr(), state.data_ptr(), core1.data_ptr(),
            1, NV, HK, True, _cs())
        cap.spec_states[rank][t].copy_(state)

    # Conv state after t+1 tokens: the last KS-1 entries of the block's inputs
    # preceded by whatever history the block started from.
    prev = (conv_hist[0] if conv_hist is not None
            else torch.zeros(mixed.shape[-1], KS - 1,
                             dtype=mixed.dtype, device=device))
    hist = torch.cat([prev, mixed[0].transpose(0, 1)], dim=1)
    for t in range(S):
        cap.spec_conv[rank][t].copy_(hist[:, t + 1:t + KS].unsqueeze(0))


def _moe_experts_m16(x, ti, tw, ld, fvk, device):
    """Routed experts via the M=16 tensor-core block-scaled mma. Sort the
    S*TOPK assignments by expert, pack into zero-padded 16-row tiles, quant once
    and run the mma (16 real tokens/tile -> full tensor-core M, each expert
    weight once). gate_up + silu(bf16) + down + scatter."""
    S = x.shape[0]
    E = _N_EXPERTS
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    # Stable, so equal-expert ties keep token order and the rows packed into
    # each quantisation tile are the same run to run.
    order = exp_flat.argsort(stable=True)
    se = exp_flat[order].long()
    stok = tok_flat[order]
    counts = torch.bincount(se, minlength=E)
    tile_counts = (counts + 15) // 16
    tile_off = torch.cumsum(tile_counts, 0) - tile_counts
    total_tiles = int(tile_counts.sum().item())
    tile_expert = torch.repeat_interleave(
        torch.arange(E, device=device), tile_counts).to(torch.int32)
    cumcount = torch.cumsum(counts, 0) - counts
    pos = torch.arange(S * TOPK, device=device) - cumcount[se]
    tiled_row = (tile_off[se] + pos // 16) * 16 + (pos % 16)

    A_t = torch.zeros(total_tiles * 16, HID, dtype=torch.bfloat16, device=device)
    A_t[tiled_row] = x[stok]
    ap, asf = _quant_act(A_t, fvk, device)
    d_gu = torch.empty(total_tiles * 16, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_m16_mma_sm120_bf16(
        ap.data_ptr(), gu_p.data_ptr(), asf.data_ptr(), gu_s.data_ptr(),
        d_gu.data_ptr(), gu_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_gu, HID, 0, gu_p[0].numel(), gu_s[0].numel(), _cs())
    inter = _silu_mul(d_gu[:, :INTER], d_gu[:, INTER:], fvk, device).contiguous()
    ip, isf = _quant_act(inter, fvk, device)
    d_dn = torch.empty(total_tiles * 16, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_m16_mma_sm120_bf16(
        ip.data_ptr(), dn_p.data_ptr(), isf.data_ptr(), dn_s.data_ptr(),
        d_dn.data_ptr(), dn_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_dn, INTER, 0, dn_p[0].numel(), dn_s[0].numel(), _cs())
    # Deterministic unpermute, as the grouped-GEMM path does: one kernel sums
    # each token's TOPK rows in a fixed order. Slot i of the token-major
    # routing sits at sorted position inv[i], whose output row is tiled_row of
    # that position.
    inv = torch.empty(S * TOPK, dtype=torch.long, device=device)
    inv[order] = torch.arange(S * TOPK, device=device)
    rows = tiled_row[inv].to(torch.int32).contiguous()
    twc = tw.contiguous()
    out = torch.empty(S, HID, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), rows.data_ptr(), twc.data_ptr(),
        out.data_ptr(), S, TOPK, n_dn, n_dn, _cs())
    return out


def _moe_experts_bt(x, ti, tw, ld, fvk, device):
    """Routed experts via the multi-warp block-tile (BM=BN=64, 4 warps) W4A4
    block-scaled GEMM. 64-row expert tiles; the activation rows and weight cols
    are loaded once into smem and shared across the 4 warps, so traffic ~ (1/64
    + 1/64) -- 4.0x the M16 tile / 1.96x the M64 tile at 1024 rows, cos identical
    (same FP4 mma). The sm120 hand-tuned equivalent of a DeepGEMM/SGLang tile."""
    S = x.shape[0]
    E = _N_EXPERTS
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    # Stable sort: equal-expert ties keep token order, so the tokens packed into
    # each fp4-quant tile (and thus the per-block scale factors / rounding) are
    # deterministic run-to-run. Prefill seeds the decode state, so this keeps the
    # token-exact red line (an unstable sort jitters the MoE output ~1e-3 cos).
    order = exp_flat.argsort(stable=True)
    se = exp_flat[order].long()
    stok = tok_flat[order]
    counts = torch.bincount(se, minlength=E)
    tile_counts = (counts + 63) // 64
    tcum = torch.cumsum(tile_counts, 0)               # inclusive prefix (E,)
    tile_off = tcum - tile_counts                     # each expert's start tile
    total_tiles = tcum[-1]                            # device scalar (no .item())
    # The exact tile count is data-dependent, so the old code read it to the
    # host (.sum().item()) and built tile_expert via repeat_interleave (whose
    # output size also forces a sync) -- two host stalls every MoE layer. The
    # worst case is host-known from S (each expert rounds up by <1 tile, so
    # total_tiles <= S*TOPK//64 + E), so size the grid + buffers to that fixed
    # bound and mark the unused tail tiles e=-1 (they early-exit in the kernel:
    # one load + return, no over-compute). Fully sync-free.
    MAX_TILES = (S * TOPK) // 64 + E
    tidx = torch.arange(MAX_TILES, device=device)
    # tile t belongs to the smallest expert e with tcum[e] > t (searchsorted
    # right); tiles past total_tiles get the sentinel -1.
    tile_expert = torch.searchsorted(tcum, tidx, right=True).to(torch.int32)
    tile_expert = torch.where(tidx < total_tiles, tile_expert,
                              torch.full_like(tile_expert, -1))
    cumcount = torch.cumsum(counts, 0) - counts
    pos = torch.arange(S * TOPK, device=device) - cumcount[se]
    tiled_row = (tile_off[se] + pos // 64) * 64 + (pos % 64)

    A_t = torch.zeros(MAX_TILES * 64, HID, dtype=torch.bfloat16, device=device)
    A_t[tiled_row] = x[stok]
    ap, asf = _quant_act(A_t, fvk, device)
    # d_gu/d_dn rows for the e=-1 tail tiles are never written (kernel exits)
    # and never gathered (tiled_row only indexes real slots), so their
    # uninitialised contents can't reach the output -- empty is safe.
    d_gu = torch.empty(MAX_TILES * 64, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_blocktile_mma_sm120_bf16(
        ap.data_ptr(), gu_p.data_ptr(), asf.data_ptr(), gu_s.data_ptr(),
        d_gu.data_ptr(), gu_a.data_ptr(), tile_expert.data_ptr(),
        MAX_TILES, n_gu, HID, 0, gu_p[0].numel(), gu_s[0].numel(), _cs())
    inter = _silu_mul(d_gu[:, :INTER], d_gu[:, INTER:], fvk, device).contiguous()
    ip, isf = _quant_act(inter, fvk, device)
    d_dn = torch.empty(MAX_TILES * 64, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_blocktile_mma_sm120_bf16(
        ip.data_ptr(), dn_p.data_ptr(), isf.data_ptr(), dn_s.data_ptr(),
        d_dn.data_ptr(), dn_a.data_ptr(), tile_expert.data_ptr(),
        MAX_TILES, n_dn, INTER, 0, dn_p[0].numel(), dn_s[0].numel(), _cs())
    # Deterministic unpermute via the fused gather-weighted-sum kernel: invert
    # the routing permutation (inv: orig slot -> sorted position, a 131 KB int
    # scatter) to get each token's TOPK d_dn rows, then one kernel computes
    # out[t] = sum_k tw[t,k] * d_dn[rows[t,k]] in fixed k-order. No
    # (S, TOPK, HID) intermediate (it was 4 GB at S=32k -> the long-context
    # wall) and no atomics -> bit-reproducible.
    inv = torch.empty(S * TOPK, dtype=torch.long, device=device)
    inv[order] = torch.arange(S * TOPK, device=device)
    rows = tiled_row[inv].to(torch.int32).contiguous()
    twc = tw.reshape(S, TOPK).contiguous()
    out = torch.empty(S, HID, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), rows.data_ptr(), twc.data_ptr(), out.data_ptr(),
        S, TOPK, n_dn, n_dn, _cs())
    return out


_GROUPED_SCRATCH = {}
_ROUTE_CONST = {}
_ROUTE_BUF = {}
# Off puts the routing back on the tensor chain, which is how the kernel's
# output is A/B'd against it end to end rather than only in a probe.


def _route_constants(S, device):
    """The parts of the routing permutation that depend only on the shape.

    Each layer routes differently, but the token index per slot and the slot
    index itself do not change -- they are a function of S alone, and every
    layer was rebuilding both. Forty layers of arange + repeat_interleave is
    launches and traffic spent to recompute a constant.
    """
    key = (S, str(device))
    got = _ROUTE_CONST.get(key)
    if got is None:
        tok_flat = torch.arange(
            S, device=device).repeat_interleave(TOPK).contiguous()
        slot_ix = torch.arange(S * TOPK, device=device)
        got = (tok_flat, slot_ix)
        _ROUTE_CONST[key] = got
    return got


def _route_buffers(S, fvk, device):
    """The routing kernel's outputs, allocated once per prompt length.

    Their sizes depend only on S, so the forty layers of a prefill write
    through the same buffers at the same addresses -- which is what lets the
    call sit inside a captured region, and incidentally saves forty rounds of
    allocation per forward.
    """
    key = (S, str(device))
    got = _ROUTE_BUF.get(key)
    if got is None:
        slots = S * TOPK
        ws_bytes = int(fvk.moe_route_prefill_workspace_bytes(
            S, TOPK, _N_EXPERTS))
        got = {
            'ti': torch.empty(S, TOPK, dtype=torch.int32, device=device),
            'tw': torch.empty(S, TOPK, dtype=torch.float32, device=device),
            'se': torch.empty(slots, dtype=torch.int32, device=device),
            # int64: the activation quantiser reads this gather index
            # as a long, and int32 there is an illegal access.
            'stok': torch.empty(slots, dtype=torch.int64, device=device),
            'inv': torch.empty(slots, dtype=torch.int32, device=device),
            'group_off': torch.empty(_N_EXPERTS + 1, dtype=torch.int32,
                                     device=device),
            'ws': torch.empty(ws_bytes, dtype=torch.uint8, device=device),
            'ws_bytes': ws_bytes,
            'sfa_off': {},
        }
        _ROUTE_BUF[key] = got
    return got


def _route_prefill(logits, fvk, device):
    """Softmax, top-k, and the permutation the grouped GEMM reads, in kernels.

    Replaces softmax + top-k + a renormalising divide + a stable argsort + two
    gathers + a bincount + a cumulative sum + a scatter: ten tensor ops a
    layer, of which the top-k alone was 25 ms of a 2048-token prefill.

    Returns None where the kernel is absent, so the tensor chain stays the
    fallback rather than this being a hard dependency.
    """
    if (not kernel_policy().route_kernel
            or not hasattr(fvk, 'moe_route_prefill_bf16')):
        return None
    S = logits.shape[0]
    b = _route_buffers(S, fvk, device)
    rc = fvk.moe_route_prefill_bf16(
        logits.data_ptr(), b['ti'].data_ptr(), b['tw'].data_ptr(),
        b['se'].data_ptr(), b['stok'].data_ptr(), b['inv'].data_ptr(),
        b['group_off'].data_ptr(), b['ws'].data_ptr(), b['ws_bytes'],
        S, _N_EXPERTS, TOPK, _cs())
    if rc:
        raise RuntimeError(f'prefill routing failed with {rc}')
    return b


def _route_sfa_off(route, k, fvk, device):
    """Per-expert scale-factor byte offsets for one projection's K."""
    n_col = ((k // 16) + 3) // 4
    off = route['sfa_off'].get(k)
    if off is None:
        off = torch.empty(_N_EXPERTS, dtype=torch.int32, device=device)
        route['sfa_off'][k] = off
    fvk.moe_route_sfa_offsets(
        route['group_off'].data_ptr(), off.data_ptr(), _N_EXPERTS, n_col,
        _cs())
    return off, n_col


def _grouped_scratch(fvk, device):
    """One scratch buffer per device for the grouped GEMM's descriptor arrays.

    Its size depends only on the expert count, not on how the routing falls, so
    it is allocated once and never resized -- which is also what lets the call
    sit inside a captured region.
    """
    key = str(device)
    got = _GROUPED_SCRATCH.get(key)
    if got is None:
        nbytes = int(fvk.moe_grouped_gemm_nvfp4_sm100_scratch_bytes(_N_EXPERTS))
        got = (torch.empty(nbytes, dtype=torch.uint8, device=device), nbytes)
        _GROUPED_SCRATCH[key] = got
    return got


def _sf_layout(counts, k, device):
    """Per-group scale-factor byte offsets, and a host-known bound on the total.

    The block-scaled layout blocks rows by 128, so a group of c rows needs
    ceil(c/128) super-blocks. Summing that needs the counts, which live on the
    device -- but the sum is bounded by (experts + slots/128) super-blocks
    whatever the routing does, and that bound follows from the shapes alone.
    Sizing the buffer from the bound rather than the sum is what keeps this free
    of a host read.
    """
    n_col = ((k // 16) + 3) // 4
    per_group = ((counts + 127) // 128) * (n_col * 512)
    off = torch.zeros(_N_EXPERTS, dtype=torch.int32, device=device)
    off[1:] = per_group.cumsum(0)[:-1].to(torch.int32)
    return off, n_col


def _moe_experts_grouped_gemm(x, ti, tw, ld, fvk, device, route=None):
    """Every routed expert of the layer in two GEMM launches.

    The per-expert loop below reads each weight once, which is the right amount,
    but pays a launch and a host iteration per expert -- and the host iteration
    is fatal twice over: it dominated the time at S=1024, and it makes the layer
    impossible to capture. A grouped GEMM takes the per-group shapes from device
    memory, so the launch geometry depends only on the expert count and the
    routing never reaches the host.

    Measured against the loop it replaces, at the shapes prefill issues: 6.0x on
    gate_up and 14.5x on down at S=1024, 512 launches down to 2, output bitwise
    identical.
    """
    S = x.shape[0]
    slots = S * TOPK
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']
    scratch, scratch_bytes = _grouped_scratch(fvk, device)

    if route is not None:
        se, stok, group_off, order, counts = (
            route['se'], route['stok'], route['group_off'], None, None)
    else:
        tok_flat, slot_ix = _route_constants(S, device)
        exp_flat = ti.reshape(-1).to(torch.int32)
        order = exp_flat.argsort(stable=True)
        se = exp_flat[order].contiguous()
        stok = tok_flat[order]

        counts = torch.bincount(se, minlength=_N_EXPERTS)
        group_off = torch.zeros(_N_EXPERTS + 1, dtype=torch.int32,
                                device=device)
        group_off[1:] = counts.cumsum(0).to(torch.int32)

    def project(A, k, n, w_p, w_s, alpha, out, gate=False, perm=None):
        if route is not None:
            sfa_off, n_col = _route_sfa_off(route, k, fvk, device)
        else:
            sfa_off, n_col = _sf_layout(counts, k, device)
        bound = (_N_EXPERTS + slots // 128 + 1) * n_col * 512
        packed = torch.empty(slots, k // 2, dtype=torch.uint8, device=device)
        sfa = torch.empty(bound, dtype=torch.uint8, device=device)
        if gate:
            # A is the merged (slots, 2k) gate/up output: gate it and quantise
            # in one pass rather than slicing two strided halves out of it,
            # copying both, gating into a third buffer and reading that back.
            # Warp-per-row: a lane owns one scale-factor group and keeps
            # it in registers, so there is no shared memory and no barrier.
            # Byte-identical to the block-per-row form and 2.7x at the shape
            # prefill issues, which puts it at 1.04x of its traffic bound.
            _sq = getattr(fvk, 'moe_grouped_silu_quant_nvfp4_warp_bf16',
                          fvk.moe_grouped_silu_quant_nvfp4_bf16)
            rc = _sq(
                A.data_ptr(), se.data_ptr(), group_off.data_ptr(),
                sfa_off.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
                slots, k, _cs())
        else:
            rc = fvk.moe_grouped_quant_nvfp4_bf16(
                A.data_ptr(), se.data_ptr(), group_off.data_ptr(),
                sfa_off.data_ptr(), 0 if perm is None else perm.data_ptr(),
                packed.data_ptr(), sfa.data_ptr(), slots, k, _cs())
        if rc:
            raise RuntimeError(f'grouped activation quant failed with {rc}')
        rc = fvk.moe_grouped_gemm_nvfp4_sm100_bf16out(
            packed.data_ptr(), sfa.data_ptr(), w_p.data_ptr(),
            w_s.data_ptr(), alpha.data_ptr(), out.data_ptr(),
            group_off.data_ptr(), sfa_off.data_ptr(),
            _N_EXPERTS, n, k, w_p[0].numel(), w_s[0].numel(),
            scratch.data_ptr(), scratch_bytes, _cs())
        if rc:
            raise RuntimeError(f'grouped MoE GEMM failed with {rc}')

    d_gu = torch.empty(slots, n_gu, dtype=torch.bfloat16, device=device)
    project(x, HID, n_gu, gu_p, gu_s, gu_a, d_gu, perm=stok)
    d_dn = torch.empty(slots, n_dn, dtype=torch.bfloat16, device=device)
    project(d_gu, INTER, n_dn, dn_p, dn_s, dn_a, d_dn, gate=True)

    # Deterministic unpermute, the same one the block-tile path uses: invert the
    # routing permutation and let one kernel sum each token's TOPK rows in fixed
    # order. index_add_ was 37.8 ms of a 1024-token prefill and reduces through
    # atomics, so its order varies -- which prefill cannot afford, since it
    # seeds a decode that has to be reproducible.
    # rows[i] is which sorted row holds slot i, which is exactly the inverse
    # permutation -- gathering arange through it, as the tiled path has to,
    # would just reproduce it.
    if route is not None:
        inv, twc = route['inv'], route['tw']
    else:
        inv = torch.empty(slots, dtype=torch.int32, device=device)
        inv[order] = slot_ix.to(torch.int32)
        twc = tw.contiguous()
    out = torch.empty(S, HID, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), inv.data_ptr(), twc.data_ptr(),
        out.data_ptr(), S, TOPK, n_dn, n_dn, _cs())
    return out


def _moe_experts_per_expert_gemm(x, ti, tw, ld, fvk, device):
    """Routed experts as one GEMM per expert over the tokens that chose it.

    The grouped GEMV below issues one GEMV per (token, expert) slot, so an
    expert's weight is re-read once per token that routed to it -- 8192 reads
    of a 1.18 MB weight per layer at S=1024, which is 9.7 GB of traffic a layer
    even before the down projection. Sorting by expert makes those reads hit L2
    rather than DRAM, which is why it works at all, but it is still bounded by
    L2 bandwidth and it dominates prefill: measured 74.6% of a 1024-token
    prefill, 1762 ms of 2361.

    Grouping the tokens instead turns each expert into a single M-row GEMM that
    reads its weight once. The block-scaled 4-bit MMA tile does the same thing
    and better, but it is a build tier that is not present everywhere; this path
    needs only the NVFP4 W4A16 GEMM, which is.

    The count per expert is data-dependent, so this reads it to the host -- one
    sync per layer, which prefill can afford and a captured decode could not.
    """
    S = x.shape[0]
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_list' not in ld:
        ld['experts_gate_up_alpha_list'] = ld['experts_gate_up_alpha_t'].tolist()
        ld['experts_down_alpha_list'] = ld['experts_down_alpha_t'].tolist()
    gu_a = ld['experts_gate_up_alpha_list']
    dn_a = ld['experts_down_alpha_list']

    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    # Stable, so equal-expert ties keep token order and the rows packed into
    # each quantisation tile are the same run to run.
    order = exp_flat.argsort(stable=True)
    se = exp_flat[order]
    stok = tok_flat[order]

    counts = torch.bincount(se, minlength=_N_EXPERTS).tolist()
    slots = S * TOPK
    A = x[stok].contiguous()                          # (slots, HID) bf16
    # One buffer per projection, written in place by each expert's GEMM. The
    # activation is a slot-major matrix throughout, so the gate is one launch
    # over all of it rather than one per expert -- 256 launches a layer and two
    # slice copies each, for an op that does not care where the rows came from.
    d_gu = torch.empty(slots, n_gu, dtype=torch.bfloat16, device=device)
    d_dn = torch.empty(slots, n_dn, dtype=torch.bfloat16, device=device)

    off = 0
    bounds = []
    for e, cnt in enumerate(counts):
        if cnt == 0:
            continue
        bounds.append((e, off, cnt))
        xp, xsf = _quant_act(A[off:off + cnt], fvk, device, _cs())
        _nvfp4_gemm_preq(xp, xsf, gu_p[e].data_ptr(), gu_s[e].data_ptr(),
                         gu_a[e], cnt, n_gu, HID, fvk, device, _cs(),
                         out=d_gu[off:off + cnt])
        off += cnt

    inter = _silu_mul(d_gu[:, :INTER], d_gu[:, INTER:], fvk, device)
    for e, off_e, cnt in bounds:
        xp, xsf = _quant_act(inter[off_e:off_e + cnt].contiguous(), fvk,
                             device, _cs())
        _nvfp4_gemm_preq(xp, xsf, dn_p[e].data_ptr(), dn_s[e].data_ptr(),
                         dn_a[e], cnt, n_dn, INTER, fvk, device, _cs(),
                         out=d_dn[off_e:off_e + cnt])

    # Deterministic unpermute, as the grouped-GEMM path does; index_add_
    # reduces through atomics, so its order varies run to run.
    inv = torch.empty(S * TOPK, dtype=torch.int32, device=device)
    inv[order] = torch.arange(S * TOPK, dtype=torch.int32, device=device)
    twc = tw.contiguous()
    out = torch.empty(S, HID, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), inv.data_ptr(), twc.data_ptr(),
        out.data_ptr(), S, TOPK, n_dn, n_dn, _cs())
    return out


def _moe_experts_grouped(x, ti, tw, ld, fvk, device):
    """Routed experts via the grouped W4A16 GEMV. Flatten the S*TOPK
    (token, expert) assignments, sort by expert so consecutive slots share a
    weight (L2-amortised -> each expert weight read ~once), and run one grouped
    GEMV per gate_up / down (BF16 activation, no per-expert launch/quant). The
    Python expert loop's ~5000 tiny-M GEMMs collapse to 2 kernel launches."""
    S = x.shape[0]
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    slots = S * TOPK
    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    # Stable, so equal-expert ties keep token order run to run.
    order = exp_flat.argsort(stable=True)
    se = exp_flat[order].contiguous()
    stok = tok_flat[order]

    A = x[stok].contiguous()                          # (slots, HID) bf16
    d_gu = torch.empty(slots, n_gu, dtype=torch.bfloat16, device=device)
    moe_grouped_w4a16(fvk)(
        A.data_ptr(), gu_p.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(),
        se.data_ptr(), d_gu.data_ptr(), slots, n_gu, HID,
        HID, gu_p[0].numel(), gu_s[0].numel(), _cs())
    g, u = d_gu[:, :INTER], d_gu[:, INTER:]
    inter = _silu_mul(g, u, fvk, device).contiguous()
    d_dn = torch.empty(slots, n_dn, dtype=torch.bfloat16, device=device)
    moe_grouped_w4a16(fvk)(
        inter.data_ptr(), dn_p.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(),
        se.data_ptr(), d_dn.data_ptr(), slots, n_dn, INTER,
        INTER, dn_p[0].numel(), dn_s[0].numel(), _cs())
    # Deterministic unpermute, as the grouped-GEMM path does: invert the
    # routing permutation and let one kernel sum each token's TOPK rows in a
    # fixed order. index_add_ reduces through atomics, so eight fp32 addends
    # land in whatever order the blocks retire -- and a prefill cannot afford
    # that, because it seeds a decode that has to be reproducible.
    inv = torch.empty(slots, dtype=torch.int32, device=device)
    inv[order] = torch.arange(slots, dtype=torch.int32, device=device)
    twc = tw.contiguous()
    out = torch.empty(S, HID, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), inv.data_ptr(), twc.data_ptr(),
        out.data_ptr(), S, TOPK, n_dn, n_dn, _cs())
    return out


def _moe_layer(h, ld, fvk, device):
    """Fine-grained MoE FFN: 256 experts top-8 routed + 1 shared expert."""
    B, S, _ = h.shape
    x = h.reshape(-1, HID)
    rw = ld['router_w_t']
    gu_p, gu_s, gu_a = (ld['experts_gate_up_packed_t'],
                        ld['experts_gate_up_sf_t'], ld['experts_gate_up_alpha_t'])
    dn_p, dn_s, dn_a = (ld['experts_down_packed_t'],
                        ld['experts_down_sf_t'], ld['experts_down_alpha_t'])
    n_gu = gu_p.shape[1]          # 2 * inter
    n_dn = dn_p.shape[1]          # hidden

    # Router GEMM via the deterministic w16a16 kernel (bf16 weight, fp32
    # accumulate) instead of the fp32 upcast matmul. bf16 logits match the
    # bf16 reference router.
    lg = _gemm_w16a16(x, rw, fvk, device)

    # The block-scaled 4-bit MMA tiles are a build tier, not a given: a target
    # whose toolchain has no block-scaled mma builds the weight-only tier
    # instead. Ask the module what it has rather than assuming, so the tile
    # choice degrades to the grouped GEMV instead of raising mid-prefill.
    big = x.shape[0] >= _M16_MIN_S
    use_bt = (_USE_BT_MOE and big
              and hasattr(fvk, 'moe_blocktile_mma_sm120_bf16'))
    use_m16 = (not use_bt and _USE_M16_MOE and big
               and hasattr(fvk, 'moe_m16_mma_sm120_bf16'))
    grouped = (not use_bt and not use_m16 and big
               and hasattr(fvk, 'moe_grouped_gemm_nvfp4_sm100_bf16out'))
    # Only the grouped path reads the kernel's permutation, and the tiled
    # paths index with the tensor top-k's own indices, so the routing is not
    # computed twice for a path that will not use it.
    route = _route_prefill(lg, fvk, device) if grouped else None
    if route is not None:
        ti, tw = route['ti'], route['tw']
    else:
        logit = F.softmax(lg.float(), -1)
        tw, ti = torch.topk(logit, TOPK, -1)
        tw = tw / tw.sum(-1, keepdim=True)

    if use_bt:
        out = _moe_experts_bt(x, ti, tw, ld, fvk, device)
    elif use_m16:
        out = _moe_experts_m16(x, ti, tw, ld, fvk, device)
    elif grouped:
        # No threshold: the grouped path wins at every prefill length measured,
        # because it does not pay per expert for anything.
        out = _moe_experts_grouped_gemm(x, ti, tw, ld, fvk, device, route)
    elif (big and _USE_PER_EXPERT_GEMM
            and x.shape[0] * TOPK >= _PER_EXPERT_MIN_M * _N_EXPERTS
            and hasattr(fvk, 'fp4_w4a16_gemm_sm120_bf16out')):
        out = _moe_experts_per_expert_gemm(x, ti, tw, ld, fvk, device)
    elif _USE_GROUPED_MOE:
        out = _moe_experts_grouped(x, ti, tw, ld, fvk, device)
    else:
        out = torch.zeros(x.shape[0], HID, device=device)
        gu_a_l = gu_a.tolist()
        dn_a_l = dn_a.tolist()
        for e in torch.unique(ti).tolist():
            m = (ti == e)
            tok = m.any(-1).nonzero(as_tuple=True)[0]
            w = (tw * m)[tok].sum(-1)
            gu_e_p, gu_e_s = gu_p[e], gu_s[e]
            gu = _nvfp4_gemm(x[tok].contiguous(), gu_e_p.data_ptr(),
                             gu_e_s.data_ptr(), gu_a_l[e], n_gu,
                             fvk, device)
            g, u = gu.chunk(2, -1)
            inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
            dn_e_p, dn_e_s = dn_p[e], dn_s[e]
            dpj = _nvfp4_gemm(inter, dn_e_p.data_ptr(), dn_e_s.data_ptr(),
                              dn_a_l[e], n_dn, fvk, device)
            out[tok] += dpj.float() * w.unsqueeze(-1)

    sg = _proj(x, ld, 'shared_gate_proj', INTER, fvk, device)
    su = _proj(x, ld, 'shared_up_proj', INTER, fvk, device)
    si = _silu_mul(sg, su, fvk, device)
    shared = _proj(si, ld, 'shared_down_proj', HID, fvk, device)
    # shared-expert scalar gate: GEMM (N=1) via w16a16. The sigmoid, the
    # broadcast multiply, the add onto the routed sum and the cast are one
    # kernel; the routed sum stays fp32 until the single rounding at its store.
    glog = _gemm_w16a16(x, ld['shared_gate_w_t'], fvk, device)
    if hasattr(fvk, 'moe_shared_gate_combine_edge_bf16'):
        comb = torch.empty(x.shape[0], HID, dtype=torch.bfloat16,
                           device=device)
        fvk.moe_shared_gate_combine_edge_bf16(
            out.data_ptr(), shared.data_ptr(), glog.data_ptr(),
            comb.data_ptr(), x.shape[0], HID, _cs())
        return comb.reshape(B, S, HID)
    sgate = torch.sigmoid(glog.float())
    return (out + shared.float() * sgate).reshape(B, S, HID).to(torch.bfloat16)


def nexn2_forward_nvfp4(handles, input_ids, fvk, device, cap=None,
                        return_hidden=False, last_logits_only=False,
                        pos_offset=0, compute_logits=True):
    """Full kernelized NVFP4 prefill forward: token ids -> logits.

    Args:
        handles: WeightHandles from extract_weights_nexn2_nvfp4.
        input_ids: (1, S) long on device.
        fvk: flash_rt_kernels module.
        device: cuda device string.
        cap: optional Nexn2DecodeState; when given, the GDN recurrent/conv
            state and the full-attn KV cache are seeded so a subsequent decode
            continues from position pos_offset+S.
        last_logits_only: when True compute the lm_head for only the final
            position, returning logits (1, vocab). The all-position logits are
            (S, vocab) -- 4 GB at S=8192 -- and only the last row seeds decode,
            so this is what the decode-seeding path uses to keep long-context
            prefill within memory (KV stays bf16 and small).
        pos_offset: absolute position of input_ids[0] (chunked prefill). >0
            continues every GDN layer from cap's carried recurrent/conv state
            and attends each full-attn block to cap's accumulated KV; bounds the
            per-layer activation memory to the block size. 0 == single-pass.
        compute_logits: False skips the final norm + lm_head (intermediate
            chunks of a chunked prefill, which only need to advance the state).

    Returns:
        logits: (S, vocab) bf16, or (1, vocab) when last_logits_only, or None
        when compute_logits is False.
    """
    p = handles.ptrs
    eps = float(p['rms_norm_eps'])
    theta = float(p['rope_theta'])
    rope_dim = int(p['head_dim'] * p['partial_rotary_factor'])
    types = p['layer_types']
    layers = p['layers']

    h = F.embedding(input_ids, p['embed_w_t'])
    S = h.shape[1]
    # RoPE tables for this block's absolute positions [pos_offset, pos_offset+S).
    ct_full, st_full = build_rope_tables(pos_offset + S, theta, rope_dim, device)
    ct, st = ct_full[pos_offset:], st_full[pos_offset:]
    chunked = pos_offset > 0
    lin_rank = full_rank = 0
    # Every residual add is immediately followed by the norm of what it
    # produced, so the two run as one kernel that updates the residual stream
    # in place -- which means the loop carries the *normed* tensor across each
    # boundary and takes the first norm before entering it.
    h = h.contiguous()
    n = _rms_k(h, layers[0]['input_norm_w_t'], fvk, device, eps)
    for L in range(p['num_layers']):
        ld = layers[L]
        if types[L] == 'linear_attention':
            init_s = cap.lin_state[lin_rank] if chunked else None
            conv_h = cap.lin_conv_state[lin_rank] if chunked else None
            attn = _gdn_layer(n, ld, fvk, device, eps, cap, lin_rank,
                              init_state=init_s, conv_hist=conv_h)
            lin_rank += 1
        else:
            attn = _full_attn_layer(n, ld, ct, st, fvk, device, eps,
                                    cap, full_rank, pos_offset=pos_offset)
            full_rank += 1
        n = _add_rms_k(h, attn, ld['post_norm_w_t'], fvk, device, eps)
        moe = _moe_layer(n, ld, fvk, device)
        # The norm after the last layer's residual is the final norm, and
        # between layers it is the next layer's input norm -- one call either
        # way, so the boundary is a choice of weight rather than a branch.
        nxt = (layers[L + 1]['input_norm_w_t'] if L + 1 < p['num_layers']
               else p['final_norm_w_t'])
        n = _add_rms_k(h, moe, nxt, fvk, device, eps)

    hidden = h[0]                       # (S, HID) residual stream, pre-final-norm
    if not compute_logits:
        return (None, hidden) if return_hidden else None
    h = n                               # already the final norm, see above
    # lm_head via w16a16 (bf16 weight, fp32 accumulate): reads the ~1GB weight
    # as bf16 (no fp32 widen), same argmax. logits returned bf16. Slice to the
    # last position first when only the seeding logit is needed (avoids the
    # (S, vocab) materialisation that dominates long-context prefill memory).
    h_lm = h[0][-1:].contiguous() if last_logits_only else h[0]
    if _SPEC_VERIFY:
        # The lm_head is the single largest weight; at BF16 it is a gigabyte a
        # verify, which on its own outweighs what the window saves.
        logits = _gemm_w4a16(h_lm, p['lm_head_w_t'], p, 'lm_head_w_t',
                             fvk, device)
    else:
        logits = _gemm_w16a16(h_lm, p['lm_head_w_t'], fvk, device)
    if return_hidden:
        return logits, hidden
    return logits
