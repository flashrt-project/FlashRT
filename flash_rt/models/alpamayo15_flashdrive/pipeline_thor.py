"""FlashDrive's Alpamayo 1.5 rollout on Jetson AGX Thor (SM110) with FlashRT execution.

FlashDrive (https://github.com/z-lab/flashdrive) accelerates Alpamayo 1.5 by
changing the inference algorithm: streaming windows over a pre-RoPE static KV
cache, DFlash block speculation, a W4A8 (ParoQuant) language model, and
adaptive action caching with a fine-tuned action expert. This module keeps that
orchestration -- FlashDrive's own ``sample_trajectories_streaming``,
``_dflash_generate`` and ``_dflash_decode_loop`` drive every window -- and
replaces the execution of its primitives with FlashRT kernels and CUDA graphs.
Each group of primitives is selected independently (``backends``), so every
replacement can be validated and measured against FlashDrive's own execution.

expert   ``_action``: the Euler loop with cached velocities (``euler_with_cache``:
         fresh expert evaluations on the non-cached steps, the previous velocity
         reused on ``cache_steps``) as one CUDA graph on FlashRT kernels (FP32
         residual stream, fused head RMSNorm + FP32 RoPE, FA4 attention with a
         device-side key length) and FlashDrive's fused q|k|v and gate|up weights.
vision   ``_encode``: the Qwen3-VL vision tower of the FlashRT Alpamayo 1.5 engine
         (one GEMM patch embedding, FP32 RoPE kernel, batched FA4 attention), one
         CUDA graph per frame grid.
prefill  ``_dflash_prefill`` (streaming windows), ``_shift_streaming_kv_cache``
         and the cache bookkeeping of ``_first_prefill``: the 36-layer language
         model on FlashRT kernels with FlashDrive's W4A8 projections, FA4 causal
         attention by cache position, one CUDA graph per stream.
decode   ``_dflash_block_step`` (draft proposal, 8-row verification, sampling and
         acceptance decisions in one CUDA graph) and ``_dflash_traj_forward``.
         Requires ``prefill="flashrt"``.

Cache contract (kept from FlashDrive): FlashDrive's ``StaticCache`` stays the
store of record -- keys *un-rotated* and values, ``[1, Hk, S, 128]`` per layer,
fixed position table per stream. FlashDrive rotates the whole cache on every
forward; the backend instead keeps attention copies that always equal those
rotations: ``kvlm`` (VLM view, BF16 tables, each product and sum rounded to
BF16) for every cache row, updated wherever a row is written, and the expert's
view (FP32 tables, one BF16 rounding) rebuilt once per window. Values are read
in place through strided views. Masks become FA4 lengths:

- streaming prefill: FlashDrive's block mask equals causal attention by cache
  position for Alpamayo 1.5 (extended view ranges); each contiguous run of
  window rows attends ``[0, its last row]`` (one varlen call, shared key prefix);
- verification block: causal over ``[0, cur + 8)``;
- trajectory-start forward (one row, FlashDrive passes no mask): every cache row;
- expert: rows ``[0, action_start)`` plus its 64 action rows (contiguous).

Other semantics kept from FlashDrive under ``autocast(bfloat16)``: the expert's
Euler state, time grid and noise are BF16 (noise drawn with ``normal_`` at
FlashDrive's draw point); draft proposals are greedy, posterior sampling uses
FlashDrive's ``sample_tokens``; every cache row FlashDrive writes (including
rejected draft rows and the expert's action rows) is written with the same
contents, so later reads of stale rows see what FlashDrive would.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
DEV = "cuda"
logger = logging.getLogger(__name__)

BACKEND_CHOICES = ("flashrt", "flashdrive")

# Kernels of the FlashDrive backend (csrc/kernels/alpamayo15_flashdrive_kernels.cuh and the W4A8 GEMM in
# csrc/gemm/alpamayo15_flashdrive_w4a8_sm100.cu), built with
# -DFLASHRT_ENABLE_ALPAMAYO15=ON next to the Alpamayo 1.5 engine kernels.
FLASHDRIVE_KERNEL_SYMBOLS = ("flashdrive_qk_prep_bf16", "flashdrive_rope_rows", "flashdrive_w4a8_act_prep",
                             "flashdrive_rms_f16delta_bf16", "flashdrive_silu_mul_f16in_bf16",
                             "flashdrive_w4a8_gemm_sm100", "flashdrive_w4a8_gemm_sm100_sizes")
W4A8_BACKENDS = ("flashdrive", "fused")
W4A8_GEMMS = ("tcgen05", "marlin")
TCGEN05_MIN_ROWS = 256      # W4A8 GEMMs with fewer rows (verification, trajectory start) stay on Marlin


def euler_cache_schedule(inference_step: int, cache_steps) -> list[bool]:
    """Per Euler step: True when the step reuses the cached velocity (FlashDrive ``euler_with_cache``)."""
    steps = set(int(s) for s in cache_steps)
    if 0 in steps:
        raise ValueError("cache_steps cannot include step 0: no velocity is cached yet.")
    if any(s < 0 or s >= inference_step for s in steps):
        raise ValueError(f"cache_steps {sorted(steps)} outside [1, {inference_step})")
    return [i in steps for i in range(inference_step)]


def contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
    """Split strictly increasing cache rows into contiguous ``(start, end)`` runs (end exclusive)."""
    runs: list[tuple[int, int]] = []
    for r in rows:
        if runs and r == runs[-1][1]:
            runs[-1] = (runs[-1][0], r + 1)
        elif runs and r < runs[-1][1]:
            raise ValueError("cache rows must be strictly increasing")
        else:
            runs.append((r, r + 1))
    return runs


def rotate_keys_fp32(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """FlashDrive's expert-side rotation of cached (un-rotated) BF16 keys with FP32 tables.

    ``(k * cos) + (rotate_half(k) * sin)`` with BF16 ``k`` and FP32 ``cos``/``sin``
    promotes to FP32 (each product and the sum rounded to FP32); attention under
    autocast then casts the result to BF16 once. ``k`` is ``[..., S, H, D]`` and
    the tables ``[S, 1, D]`` (broadcast over heads). Returns FP32. Reference for
    the ``flashdrive_rope_rows`` kernel (``fp32_rule=1``).
    """
    half = k.shape[-1] // 2
    rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return k * cos + rot * sin


def reset_flashdrive_stream(model) -> None:
    """End FlashDrive's current stream so its next window starts a fresh one.

    FlashDrive treats ``_past_key_values is None`` as a new stream but keeps
    per-stream state elsewhere: the vision and language-model shape caches of
    the last (4-frame) streaming window, the streaming attention mask, and its
    compiled-step registry (input buffers sized for the old stream). All are
    cleared, as a fresh process would start; FlashDrive-executed compiled
    primitives are therefore compiled again for the next stream.
    """
    model._past_key_values = None
    model.vlm.model.visual.reset_shape_caches()
    model.vlm.model.language_model.reset_shape_caches()
    model._cached_streaming_attention_mask = None
    model._compiled_step_registry.clear()


def _require_autocast() -> None:
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != BF16:
        raise ValueError("FlashDriveThor reproduces FlashDrive's rollout under torch.autocast('cuda', bfloat16); "
                         "call it inside that context (as FlashDrive's scripts/infer.py does)")


def load_flashdrive(base: str, paro: str, dflash: str, *, torch_compile: str | None = None) -> torch.nn.Module:
    """Load the full FlashDrive stack from explicit checkpoint locations.

    Mirrors ``flashdrive.from_pretrained`` for an optimized checkpoint (setup
    order: rollout, compile, streaming, ParoQuant, expert fusion, DFlash) but
    takes the W4A8 and draft companions explicitly instead of deriving them from
    a ``z-lab/`` path prefix, so local directories load the optimized stack too.
    ``torch_compile`` is FlashDrive's compile mode for the primitives FlashRT
    does not replace (``None`` = eager). The model is placed on the current CUDA device.
    """
    try:
        import flashdrive
        from flashdrive import FlashDriveMixin, resolve_model_class
    except ImportError as exc:
        raise ImportError("the FlashDrive backend needs the flashdrive package "
                          "(https://github.com/z-lab/flashdrive)") from exc
    model_cls = resolve_model_class(base)
    extended = getattr(flashdrive, "_EXTENDED_VISION_RANGES", {}).get(model_cls)
    if extended is None:
        raise ValueError(f"FlashDrive does not define streaming ranges for {model_cls.__name__}")
    model_cls._supports_sdpa = True
    config = model_cls.config_class.from_pretrained(base)
    config.attn_implementation = config._attn_implementation = "sdpa"
    model = model_cls.from_pretrained(base, config=config, dtype=BF16).to(DEV).eval()
    model.__class__ = type(f"FlashDrive{model_cls.__name__}", (FlashDriveMixin, model_cls), {})
    model.setup_rollout()
    model.setup_torch_compile(torch_compile)
    model.setup_streaming(extended_vision_ranges=extended)
    model.setup_paroquant(paro)
    model.setup_expert_fusion()
    model.setup_dflash(dflash)
    return model


def marlin_w4a8_to_int8(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int, N: int,
                        maps: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a Marlin W4A8 linear (UINT4 + zero points, group 128, INT8-activation integer scales) into
    ``(w8, s)``: ``w8 [N, K]`` int8 = ``w - z`` and ``s [K/128, N]`` int32, the integers Marlin multiplies.
    The permutations are inverted through vLLM's own definitions of the Marlin layout (index maps)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_permute_weights
    G = K // 128
    key = (K, N)
    if maps is None or key not in maps:
        P = marlin_permute_weights(torch.arange(K * N, dtype=torch.int32).reshape(K, N), K, N,
                                   get_weight_perm(4, is_a_8bit=True), is_a_8bit=True)
        scale_perm, scale_perm_single = get_scale_perms()
        grid = torch.arange(G * N, dtype=torch.int32)
        Ps = grid.reshape(-1, len(scale_perm_single))[:, torch.tensor(scale_perm_single)].reshape(-1)
        Pz = grid.reshape(-1, len(scale_perm))[:, torch.tensor(scale_perm)].reshape(-1)
        m = (P.reshape(-1).to(DEV, torch.long), Ps.to(DEV, torch.long), Pz.to(DEV, torch.long))
        if maps is not None:
            maps[key] = m
    else:
        m = maps[key]

    def nibbles(t):                      # [R, C] int32 -> [R * 8C], nibble i of word j at 8j + i
        return torch.stack([(t >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(-1)

    w = torch.empty(K * N, dtype=torch.int16, device=DEV)
    w[m[0]] = nibbles(qweight).to(torch.int16)
    z = torch.empty(G * N, dtype=torch.int16, device=DEV)
    z[m[2]] = nibbles(qzeros).to(torch.int16)
    s = torch.empty(G * N, dtype=torch.int32, device=DEV)
    s[m[1]] = scales.view(torch.int16).reshape(-1).to(torch.int32)
    if bool((s < 0).any()) or bool((s > 4096).any()):       # marlin_act_int8_process_scales: integers 0..4096
        raise ValueError("unexpected Marlin INT8-activation group scales (expected integers in [0, 4096])")
    w8 = (w.view(K, N) - z.view(G, N).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
    return w8, s.view(G, N)


class _W4A8Group:
    """FlashDrive ParoQuant linears that share one input (q|k|v, o, gate|up, down) executed as one fused
    activation pass (``flashdrive_w4a8_act_prep``: FP16 cast, pairwise rotation, per-token INT8 quantization,
    bit-exact with paroquant + vLLM) and the W4A8 GEMM of each linear: vLLM's Marlin kernel, or (``gemm="tcgen05"``,
    groups of at least ``TCGEN05_MIN_ROWS`` rows) FlashRT's INT8 tensor-core GEMM on the unpacked weights, which
    computes Marlin's integer group-scaled sums exactly. Returns the FP16 GEMM outputs; FlashDrive's cast back to
    BF16 is applied by the consumer kernels on load."""

    def __init__(self, wrappers, scratch: dict, gemm: str = "marlin", maps: dict | None = None) -> None:
        from flashdrive.quantization import BF16RotateLinearWrapper
        from vllm.scalar_type import scalar_types
        mods = []
        for w in wrappers:
            if not isinstance(w, BF16RotateLinearWrapper):
                raise TypeError(f"expected FlashDrive's BF16RotateLinearWrapper, got {type(w).__name__}")
            mods.append(w.rotate_linear)
        self.K = mods[0].in_features
        for m in mods:
            q, r = m.qlinear, m.rotation
            if m.in_features != self.K or self.K % 128:
                raise ValueError("fused W4A8 group needs one input width, a multiple of 128")
            if (r.theta.shape != (8, self.K // 2) or r.pairs.shape != (8, self.K) or r.channel_scales.numel() != self.K
                    or r.theta.dtype != torch.float16 or r.channel_scales.dtype != torch.float16
                    or r.pairs.dtype != torch.int16):
                raise ValueError("unexpected ParoQuant rotation layout (8 layers, group 128, fp16 angles/scales)")
            if q.quant_type != scalar_types.uint4 or q.bias is not None or q.g_idx.numel() or q.scales.dtype != torch.float16:
                raise ValueError("unexpected Marlin W4A8 layout (uint4 + zero points, fp16 group scales, no act-order/bias)")
        self.mods = mods
        self.N = [m.out_features for m in mods]
        self._scratch = scratch     # shared by every group: consumed by this group's GEMMs before the next group runs
        self.int8 = None
        if gemm == "tcgen05":       # (w - z) int8 [N, K] and integer group scales [K/128, N] per linear
            self.int8 = [marlin_w4a8_to_int8(m.qlinear.qweight, m.qlinear.scales, m.qlinear.qzeros, self.K, N, maps)
                         for m, N in zip(mods, self.N)]

    def __call__(self, fvk, x: torch.Tensor) -> list[torch.Tensor]:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import should_use_atomic_add_reduce
        M = x.shape[0]
        n = len(self.mods)
        b = self._scratch.get((M, self.K, n))
        if b is None:
            b = {"xr": [torch.empty(M, self.K, device=DEV, dtype=torch.float16) for _ in range(n)],
                 "part": [torch.empty(M, self.K // 128, device=DEV, dtype=torch.float32) for _ in range(n)],
                 "xq": [torch.empty(M, self.K, device=DEV, dtype=torch.int8) for _ in range(n)],
                 "as": [torch.empty(M, 1, device=DEV, dtype=torch.float32) for _ in range(n)]}
            self._scratch[(M, self.K, n)] = b
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        fvk.flashdrive_w4a8_act_prep(
            x.data_ptr(), x.stride(0), M, self.K, ptrs(m.rotation.theta for m in self.mods),
            ptrs(m.rotation.pairs for m in self.mods), ptrs(m.rotation.channel_scales for m in self.mods),
            ptrs(m.qlinear.input_global_scale for m in self.mods), ptrs(b["xr"]), ptrs(b["part"]), ptrs(b["xq"]),
            ptrs(b["as"]), torch.cuda.current_stream().cuda_stream)
        outs = []
        if self.int8 is not None and M >= TCGEN05_MIN_ROWS:
            for (w8, sg), xq, a_s, N in zip(self.int8, b["xq"], b["as"], self.N):
                outs.append(self._gemm_int8(fvk, xq, w8, sg, a_s, M, N))
            return outs
        for m, xq, a_s, N in zip(self.mods, b["xq"], b["as"], self.N):
            q = m.qlinear
            out = ops.marlin_gemm(xq, None, q.qweight, q.bias, q.scales, a_s, None, q.qzeros, q.g_idx,
                                  q.g_idx_sort_indices, q.workspace, q.quant_type, size_m=M, size_n=N, size_k=self.K,
                                  use_atomic_add=should_use_atomic_add_reduce(M, N, self.K, x.device, torch.float16),
                                  use_fp32_reduce=True, is_zp_float=False)
            outs.append(out)
        return outs

    def _gemm_int8(self, fvk, xq, w8, sg, a_s, M, N) -> torch.Tensor:
        key = ("int8", M, self.K, N)
        ws = self._scratch.get(key)
        if ws is None:
            nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, self.K)
            ws = (torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device=DEV),     # padded: the last token tile
                  torch.ones(int(n_ones) + 1024, dtype=torch.int32, device=DEV))      # may read past the layout
            self._scratch[key] = ws
        out = torch.empty(M, N, dtype=torch.float16, device=DEV)
        fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ws[1].data_ptr(), a_s.data_ptr(),
                                       out.data_ptr(), M, N, self.K, ws[0].data_ptr(), ws[0].numel(),
                                       torch.cuda.current_stream().cuda_stream)
        return out

    def check_against_marlin(self, fvk, rows: int = TCGEN05_MIN_ROWS) -> float:
        """Run both GEMMs of every linear on random INT8 activations; returns the smallest fraction of identical
        FP16 outputs. Raises on a gross disagreement (unpacking or layout mismatch): an output off by more than one
        FP16 ulp plus Marlin's FP32 slice-sum error bound, or fewer than 99% identical outputs."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import should_use_atomic_add_reduce
        g = torch.Generator(device=DEV).manual_seed(0)
        xq = torch.randint(-127, 128, (rows, self.K), dtype=torch.int8, device=DEV, generator=g)
        u = torch.rand(rows, 1, device=DEV, generator=g) * 2e-2 + 1e-3     # per-token absmax / 127, as act prep
        worst = 1.0
        for m, (w8, sg), N in zip(self.mods, self.int8, self.N):
            q = m.qlinear
            a_s = (u * q.input_global_scale).contiguous()
            ref = ops.marlin_gemm(xq, None, q.qweight, q.bias, q.scales, a_s, None, q.qzeros, q.g_idx,
                                  q.g_idx_sort_indices, q.workspace, q.quant_type, size_m=rows, size_n=N, size_k=self.K,
                                  use_atomic_add=should_use_atomic_add_reduce(rows, N, self.K, xq.device, torch.float16),
                                  use_fp32_reduce=True, is_zp_float=False)
            out = self._gemm_int8(fvk, xq, w8, sg, a_s, rows, N)
            r = ref.float()
            tol = (r.abs() * 2.0 ** -10).clamp_min(2.0 ** -24) + r.abs().amax(dim=1, keepdim=True) * 2.0 ** -20
            same = float((out == ref).float().mean())
            if bool(((out.float() - r).abs() > tol).any()) or same < 0.99:
                raise RuntimeError("tcgen05 W4A8 GEMM disagrees with Marlin (weight layout?)")
            worst = min(worst, same)
        return worst


def _capture(fn, warmups: int = 2) -> torch.cuda.CUDAGraph:
    """Warm ``fn`` up on a side stream, then capture it (BF16 autocast, no autograd). The autocast cast cache is
    disabled: a cached cast would be freed when the caller's autocast region ends while the graph keeps its address."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad(), torch.autocast("cuda", dtype=BF16, cache_enabled=False):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=BF16, cache_enabled=False), torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class FlashDriveThor:
    """FlashRT execution backend installed on a FlashDrive Alpamayo 1.5 model.

    Args:
        model: a FlashDrive model (``FlashDriveMixin``) after all setups, on CUDA.
        expert, vision, prefill, decode: ``"flashrt"`` (default) or
            ``"flashdrive"`` per primitive group (see the module docstring);
            ``decode="flashrt"`` needs ``prefill="flashrt"``.
        w4a8: execution of the W4A8 linears inside the FlashRT language model:
            ``"fused"`` (default with ``prefill="flashrt"``; one activation pass
            per shared input and Marlin outputs consumed by the next kernel) or
            ``"flashdrive"`` (FlashDrive's ParoQuant modules).
        w4a8_gemm: GEMM of fused W4A8 groups with at least ``TCGEN05_MIN_ROWS``
            rows (streaming-window prefill): ``"tcgen05"`` (default; INT8 tensor
            cores on unpacked ``w - z`` weights, about 7 GB more memory) or
            ``"marlin"``. Smaller groups always use Marlin. Only with
            ``w4a8="fused"``.
        expert_splits: FA4 KV splits for expert attention.

    The backend serves one trajectory sample per window (``num_traj_samples=1``,
    ``num_traj_sets=1``) and FlashDrive's ``euler_with_cache`` integrator.
    """

    # The vision tower of the FlashRT Alpamayo 1.5 engine (same Qwen3-VL tower and numerics).
    from flash_rt.models.alpamayo15.pipeline_thor import Alpamayo15Thor as _Engine
    _vision_tables = _Engine._vision_tables
    _vision = _Engine._vision
    del _Engine

    def __init__(self, model, *, expert: str = "flashrt", vision: str = "flashrt", prefill: str = "flashrt",
                 decode: str = "flashrt", w4a8: str | None = None, w4a8_gemm: str | None = None,
                 expert_splits: int = 1) -> None:
        backends = {"expert": expert, "vision": vision, "prefill": prefill, "decode": decode}
        if w4a8 is None:
            w4a8 = "fused" if prefill == "flashrt" else "flashdrive"
        for name, choice in backends.items():
            if choice not in BACKEND_CHOICES:
                raise ValueError(f"unknown {name} backend {choice!r}; choose from {BACKEND_CHOICES}")
        if w4a8 not in W4A8_BACKENDS:
            raise ValueError(f"unknown w4a8 execution {w4a8!r}; choose from {W4A8_BACKENDS}")
        if w4a8 == "fused" and prefill != "flashrt":
            raise ValueError("w4a8='fused' applies to the FlashRT language model (prefill='flashrt')")
        if w4a8_gemm is not None and w4a8 != "fused":
            raise ValueError("w4a8_gemm applies to w4a8='fused'")
        if w4a8 == "fused" and w4a8_gemm is None:
            w4a8_gemm = "tcgen05"
        if w4a8 == "fused" and w4a8_gemm not in W4A8_GEMMS:
            raise ValueError(f"unknown w4a8_gemm {w4a8_gemm!r}; choose from {W4A8_GEMMS}")
        backends["w4a8"] = w4a8
        if w4a8 == "fused":
            backends["w4a8_gemm"] = w4a8_gemm
        if decode == "flashrt" and prefill != "flashrt":
            raise ValueError("decode='flashrt' needs prefill='flashrt' (the rotated cache copy is kept by FlashRT writes)")
        try:
            from flashdrive import FlashDriveMixin
        except ImportError as exc:
            raise ImportError("the FlashDrive backend needs the flashdrive package") from exc
        if not isinstance(model, FlashDriveMixin):
            raise TypeError("FlashDriveThor needs a FlashDrive model (see load_flashdrive)")
        if model.training:
            raise ValueError("FlashDriveThor requires an eval-mode model")
        self.m = model
        self.backends = backends
        self.expert_splits = int(expert_splits)
        self._st = None                     # per-stream state (FlashDrive cache, tables, graphs)
        self.record_steps = False           # expert parity mode: keep per-step velocities
        self._steps_rec: dict = {}
        if "flashrt" not in backends.values():
            return
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.hardware.thor import fa4_backend
        from flash_rt.models.alpamayo15.pipeline_thor import require_alpamayo15_kernels
        require_alpamayo15_kernels(fvk)
        missing = [n for n in FLASHDRIVE_KERNEL_SYMBOLS if not hasattr(fvk, n)]
        if missing:
            raise RuntimeError(f"FlashRT was built without the FlashDrive backend kernels ({', '.join(missing)}); "
                               "rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_ALPAMAYO15=ON")
        self._fvk = fvk
        self._fa4 = fa4_backend.fa4_fwd()
        if self._fa4 is None:
            raise RuntimeError(f"FA4 unavailable: {fa4_backend.status()}")
        self._init_geometry()
        if expert == "flashrt":
            self._init_expert()
            model._action = self._action
        if vision == "flashrt":
            self.vlm = model.vlm
            self.vision_attention = "fa4"
            self._vis_tables: dict = {}
            self._vis_graphs: dict = {}
            model._encode = self._encode
        if w4a8 == "fused":
            self._w4, scratch, maps = [], {}, {}
            grp = lambda mods: _W4A8Group(mods, scratch, w4a8_gemm, maps)
            for layer in model.vlm.model.language_model.layers:
                a, mlp = layer.self_attn, layer.mlp
                self._w4.append({"qkv": grp([a.q_proj, a.k_proj, a.v_proj]), "o": grp([a.o_proj]),
                                 "gu": grp([mlp.gate_proj, mlp.up_proj]), "down": grp([mlp.down_proj])})
            del maps, grp
            torch.cuda.empty_cache()        # index maps of the weight unpacking (about 1 GB, transient)
            if w4a8_gemm == "tcgen05":      # unpacked layout checked against Marlin itself, once per shape
                self.w4a8_gemm_agreement = min(g.check_against_marlin(fvk) for g in self._w4[0].values())
        if prefill == "flashrt" and expert == "flashdrive":
            fd_action = model._action

            def action_then_sync(**kw):     # FlashDrive's expert writes action-row K/V; mirror them into kvlm
                out = fd_action(**kw)
                st = self._stream()
                slot = kw["cache_position"][:1]
                for li in range(self.n_layers):
                    self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, self.n_act, slot=slot)
                return out
            model._action = action_then_sync
        if prefill == "flashrt":
            self._orig_first_prefill = model._first_prefill
            model._first_prefill = self._first_prefill
            model._dflash_prefill = self._dflash_prefill
            model._shift_streaming_kv_cache = self._shift
        if decode == "flashrt":
            model._dflash_block_step = self._block_step
            model._dflash_traj_forward = self._traj_forward

    # ------------------------------------------------------------------
    # geometry and per-stream state
    # ------------------------------------------------------------------
    def _init_geometry(self) -> None:
        m = self.m
        tcfg = m.vlm.config.text_config
        ecfg = m.expert.config
        if ecfg.num_hidden_layers != tcfg.num_hidden_layers or ecfg.num_key_value_heads != tcfg.num_key_value_heads:
            raise ValueError("expert attention geometry must match the VLM cache")
        if tcfg.head_dim != 128 or ecfg.head_dim != 128:
            raise ValueError("Alpamayo 1.5 kernels require head_dim 128")
        self.n_layers = tcfg.num_hidden_layers
        self.n_kv = tcfg.num_key_value_heads
        self.head_dim = 128
        self.n_heads = tcfg.num_attention_heads
        self.hidden = tcfg.hidden_size
        self.ffn = tcfg.intermediate_size
        self.eps = tcfg.rms_norm_eps
        self.e_heads = ecfg.num_attention_heads
        self.e_hidden = ecfg.hidden_size
        self.e_ffn = ecfg.intermediate_size
        self.e_eps = ecfg.rms_norm_eps
        self.n_act, self.act_dim = m.action_space.get_action_space_dims()
        self.scale = self.head_dim ** -0.5

    def _stream(self) -> dict:
        """State bound to FlashDrive's current stream: its cache object and fixed position table."""
        m = self.m
        pkv = m._past_key_values
        pos = m.streaming_position_ids
        st = self._st
        if st is not None and st["pkv"] is pkv and st["pos"] is pos:
            return st
        if pkv is None or pos is None:
            raise RuntimeError("no active FlashDrive stream")
        S = pkv.max_cache_len
        for layer in pkv.layers:
            if not layer.is_initialized or layer.keys.shape[0] != 1 or layer.keys.shape[2] != S:
                raise ValueError("FlashDriveThor expects an initialized single-sample FlashDrive cache")
        st = {"pkv": pkv, "pos": pos, "S": S, "graphs": {}, "keep": {},
              "k": [layer.keys for layer in pkv.layers], "v": [layer.values for layer in pkv.layers]}
        if self.backends["prefill"] == "flashrt":
            # VLM rotary tables over the stream's fixed position table (BF16, as FlashDrive's BF16 forward computes).
            cos, sin = m.vlm.model.language_model.rotary_emb(torch.zeros(1, device=DEV, dtype=BF16), pos)
            st["cos"], st["sin"] = cos[0].contiguous(), sin[0].contiguous()
            st["kvlm"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
        if self.backends["expert"] == "flashrt":
            cos, sin = m.expert.rotary_emb(torch.zeros(1, device=DEV, dtype=torch.float32), pos)
            st["e_cos_tab"], st["e_sin_tab"] = cos[0].contiguous(), sin[0].contiguous()
            st["ke"] = torch.empty(self.n_layers, S, self.n_kv, self.head_dim, device=DEV, dtype=BF16)
        self._st = st
        if "kvlm" in st:
            self._sync_kvlm_all(st)          # rotated view of whatever FlashDrive's cache already holds
        return st

    def _v_view(self, st: dict, li: int) -> torch.Tensor:
        """FlashDrive's value cache of layer ``li`` as an attention view [1, S, Hk, D] (strided, no copy)."""
        return st["v"][li].permute(0, 2, 1, 3)

    def _rope_rows(self, st: dict, li: int, dst: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, fp32_rule: int,
                   rows: int, row_map: torch.Tensor | None = None, row0: int = 0, slot: torch.Tensor | None = None,
                   n_dev: torch.Tensor | None = None) -> None:
        S, D = st["S"], self.head_dim
        self._fvk.flashdrive_rope_rows(
            st["k"][li].data_ptr(), D, S * D, dst.data_ptr(), self.n_kv * D, D, cos.data_ptr(), sin.data_ptr(),
            fp32_rule, row_map.data_ptr() if row_map is not None else 0, row0,
            slot.data_ptr() if slot is not None else 0, n_dev.data_ptr() if n_dev is not None else 0,
            rows, self.n_kv, torch.cuda.current_stream().cuda_stream)

    def _sync_kvlm_all(self, st: dict) -> None:
        """Rotate every cache row of every layer into the VLM view (after FlashDrive wrote the cache)."""
        for li in range(self.n_layers):
            self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, st["S"])

    # ------------------------------------------------------------------
    # expert
    # ------------------------------------------------------------------
    def _init_expert(self) -> None:
        m = self.m
        if not all(getattr(layer.self_attn, "fuse_qkv", False) and getattr(layer.mlp, "fuse_gate_up", False)
                   for layer in m.expert.layers):
            raise ValueError("FlashDriveThor expects FlashDrive's fused expert projections (setup_expert_fusion)")
        if not m.config.expert_non_causal_attention:
            raise ValueError("FlashDriveThor's expert attention is non-causal (expert_non_causal_attention=True)")
        if getattr(m.expert.config, "hidden_act", None) != "silu":
            raise ValueError(f"FlashDriveThor's expert MLP is SiLU-gated, got {getattr(m.expert.config, 'hidden_act', None)!r}")
        for li, layer in enumerate(m.expert.layers):
            for name, lin in (("qkv_proj", layer.self_attn.qkv_proj), ("o_proj", layer.self_attn.o_proj),
                              ("gate_up_proj", layer.mlp.gate_up_proj), ("down_proj", layer.mlp.down_proj)):
                if getattr(lin, "bias", None) is not None:
                    raise ValueError(f"expert layer {li} {name} has a bias; FlashDriveThor's expert GEMMs have none")
        A, E, D = self.n_act, self.e_hidden, self.head_dim
        mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
        self._eb = {"res": mk(A, E, dt=torch.float32), "hn": mk(A, E),
                    "qkv": mk(A, (self.e_heads + 2 * self.n_kv) * D), "q": mk(A, self.e_heads * D),
                    "o": mk(A, E), "d": mk(A, E), "e_cos": mk(A, D, dt=torch.float32), "e_sin": mk(A, D, dt=torch.float32)}
        self.noise = mk(1, A, self.act_dim)            # FlashDrive's BF16 action noise buffer
        self.x_out = mk(1, A, self.act_dim)
        self.v_cache = mk(1, A, self.act_dim)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)    # action_start (first action cache row)
        self.d_seqk = torch.zeros(1, dtype=torch.int32, device=DEV)   # action_start + n_act keys attended
        self.d_ke_rows = torch.zeros(1, dtype=torch.int32, device=DEV)  # rows [0, action_start) to rotate

    def _velocity(self, x: torch.Tensor, t: torch.Tensor, st: dict, rec: list | None) -> torch.Tensor:
        """One FlashDrive expert evaluation (``step_fn``) on FlashRT kernels; returns v [1, A, act_dim] bf16."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        A, E, D, HQ, KV, F_, S = self.n_act, self.e_hidden, self.head_dim, self.e_heads, self.n_kv, self.e_ffn, st["S"]
        b = self._eb
        res, hn, qkv, q, o, d = b["res"], b["hn"], b["qkv"], b["q"], b["o"], b["d"]
        eps = self.e_eps
        layers = m.expert.layers
        emb = m.action_in_proj(x, t)
        res.copy_(emb.reshape(A, E))                                             # FP32 residual stream
        fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(),
                                        hn.data_ptr(), A, E, eps, s)
        ld = qkv.shape[1]
        for li, layer in enumerate(layers):
            a = layer.self_attn
            torch.mm(hn, a.qkv_proj.weight.t(), out=qkv)
            # q rotated (FP32 rule); action-row K (normalized, un-rotated) and V written into FlashDrive's cache
            # at action_start, as FlashDrive's expert forward does, and K's rotation into the expert view.
            fvk.flashdrive_qk_prep_bf16(
                qkv.data_ptr(), qkv[:, HQ * D:].data_ptr(), qkv[:, (HQ + KV) * D:].data_ptr(), ld, ld, ld,
                a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), b["e_cos"].data_ptr(), b["e_sin"].data_ptr(), 1,
                q.data_ptr(), st["k"][li].data_ptr(), D, S * D, st["ke"][li].data_ptr(), KV * D, D,
                st["v"][li].data_ptr(), D, S * D, 0, self.d_slot.data_ptr(), A, HQ, KV, eps, s)
            out, _ = self._fa4(q.view(1, A, HQ, D), st["ke"][li].unsqueeze(0), self._v_view(st, li),
                               seqused_k=self.d_seqk, softmax_scale=self.scale, causal=False, pack_gqa=True,
                               num_splits=self.expert_splits)
            torch.mm(out.view(A, HQ * D), a.o_proj.weight.t(), out=o)
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), o.data_ptr(), layer.post_attention_layernorm.weight.data_ptr(),
                                            hn.data_ptr(), A, E, eps, s)
            gu = torch.mm(hn, layer.mlp.gate_up_proj.weight.t())
            act = F.silu(gu[:, :F_]) * gu[:, F_:]
            torch.mm(act, layer.mlp.down_proj.weight.t(), out=d)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else m.expert.norm.weight
            fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), d.data_ptr(), nxt.data_ptr(), hn.data_ptr(), A, E, eps, s)
        v = F.linear(hn, m.action_out_proj.weight, m.action_out_proj.bias).view(1, A, self.act_dim)
        if rec is not None:
            rec.append(v.clone())
        return v

    def _denoise(self, st: dict, schedule: list[bool], time_steps: torch.Tensor, rec: dict | None) -> None:
        # Expert view of every cached prompt/reasoning row (FP32 rule); the action rows are written per step.
        for li in range(self.n_layers):
            self._rope_rows(st, li, st["ke"][li], st["e_cos_tab"], st["e_sin_tab"], 1, st["S"], n_dev=self.d_ke_rows)
        b = self._eb
        idx = torch.arange(self.n_act, device=DEV) + self.d_slot
        torch.index_select(st["e_cos_tab"], 0, idx, out=b["e_cos"])
        torch.index_select(st["e_sin_tab"], 0, idx, out=b["e_sin"])
        x = self.noise
        for i, cached in enumerate(schedule):
            dt = time_steps[i + 1] - time_steps[i]
            t = time_steps[i].view(1, 1, 1).expand(1, 1, 1)
            if cached:
                v = self.v_cache
            else:
                v = self._velocity(x, t, st, None if rec is None else rec["v"])
                self.v_cache.copy_(v)
            x = x + dt * v
            if rec is not None:
                rec["x"].append(x.clone())
        self.x_out.copy_(x)
        if "kvlm" in st:  # keep the VLM view of the action rows FlashDrive's cache now holds
            for li in range(self.n_layers):
                self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, self.n_act, slot=self.d_slot)

    @torch.no_grad()
    def _action(self, total_samples: int, position_ids: torch.Tensor, attention_mask: torch.Tensor,
                cache_position: torch.Tensor, diffusion_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Drop-in for FlashDrive's ``_action`` (same arguments, same BF16 [1, A, act_dim] result)."""
        kw = dict(diffusion_kwargs or {})
        if kw.get("int_method") != "euler_with_cache":
            raise ValueError("FlashDriveThor's expert graph implements FlashDrive's euler_with_cache integrator; "
                             f"got int_method={kw.get('int_method')!r}")
        if total_samples != 1:
            raise ValueError(f"FlashDriveThor serves one trajectory sample per window, got {total_samples}")
        _require_autocast()
        n_steps = int(kw.get("inference_step", self.m.diffusion.num_inference_steps))
        schedule = tuple(euler_cache_schedule(n_steps, kw["cache_steps"]))
        st = self._stream()
        if position_ids is not st["pos"]:
            raise ValueError("expert position table differs from the stream's")
        action_start = int(cache_position[0])
        if action_start + self.n_act > st["S"]:
            raise ValueError(f"action rows [{action_start}, {action_start + self.n_act}) exceed the cache ({st['S']})")
        key = ("expert", schedule, n_steps, self.record_steps)
        g = st["graphs"].get(key)
        if g is None:
            time_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=DEV, dtype=BF16)   # FlashDrive's BF16 grid
            st["keep"][key] = time_steps                   # captured by address: keep alive with the graph
            rec = {"v": [], "x": []} if self.record_steps else None
            # Warm up / capture on scratch rows past every live row; the touched cache rows are restored.
            scratch = st["S"] - self.n_act
            saved = [t.clone() for t in (self.noise, self.d_slot, self.d_seqk, self.d_ke_rows, self.v_cache)]
            rows = [(k[0, :, scratch:].clone(), v[0, :, scratch:].clone()) for k, v in zip(st["k"], st["v"])]
            kvlm_rows = st["kvlm"][:, scratch:].clone() if "kvlm" in st else None
            self.d_slot.fill_(scratch); self.d_seqk.fill_(st["S"]); self.d_ke_rows.fill_(scratch)
            self.noise.zero_()
            g = _capture(lambda: self._denoise(st, list(schedule), time_steps, rec))
            for t, sv in zip((self.noise, self.d_slot, self.d_seqk, self.d_ke_rows, self.v_cache), saved):
                t.copy_(sv)
            for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), rows):
                k[0, :, scratch:].copy_(ks); v[0, :, scratch:].copy_(vs)
            if kvlm_rows is not None:
                st["kvlm"][:, scratch:].copy_(kvlm_rows)
            if rec is not None:          # keep the captured pass only (the two warm-ups ran on scratch rows)
                for k_ in rec:
                    rec[k_][:] = rec[k_][len(rec[k_]) * 2 // 3:]
            st["keep"][("rec",) + key] = rec or {}
            st["graphs"][key] = g
        self._steps_rec = st["keep"][("rec",) + key]
        self.d_slot.fill_(action_start)
        self.d_seqk.fill_(action_start + self.n_act)
        self.d_ke_rows.fill_(action_start)
        self.noise.normal_()             # FlashDrive's draw: BF16 noise, sampled before the graph
        g.replay()
        return self.x_out.clone()

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """Drop-in for FlashDrive's ``_encode``: (image embeddings, DeepStack features) of the window's frames."""
        _require_autocast()
        key = (tuple(pixel_values.shape), tuple(image_grid_thw.flatten().tolist()))
        ent = self._vis_graphs.get(key)
        if ent is None:
            px = torch.empty_like(pixel_values)
            px.copy_(pixel_values)
            with torch.autocast("cuda", dtype=BF16):
                self._vision(px, image_grid_thw)            # builds the per-grid tables outside capture
            grid_host = image_grid_thw.cpu()                # table lookup key only: no device sync while capturing
            tables = self._vis_tables[tuple(grid_host.flatten().tolist())]   # captured by address: keep alive
            out = {}

            def run():
                out["img"], out["deep"] = self._vision(px, grid_host)
            ent = {"px": px, "out": out, "tables": tables, "graph": _capture(run)}
            self._vis_graphs[key] = ent
        ent["px"].copy_(pixel_values)
        ent["graph"].replay()
        return ent["out"]["img"], ent["out"]["deep"]

    # ------------------------------------------------------------------
    # language model on FlashRT kernels (FlashDrive's W4A8 projections)
    # ------------------------------------------------------------------
    def _lm_buffers(self, st: dict, n: int, ctx_rows: int) -> dict:
        key = ("lmbuf", n, ctx_rows)
        b = st["keep"].get(key)
        if b is None:
            H, D, NH, F_ = self.hidden, self.head_dim, self.n_heads, self.ffn
            mk = lambda *shape, dt=BF16: torch.empty(*shape, device=DEV, dtype=dt)
            ncap = len(self.m.vlm.model.language_model._capture_layer_ids)
            b = {"x": mk(n, H), "h": mk(n, H), "q": mk(n, NH * D), "attn": mk(n, NH * D), "act": mk(n, F_),
                 "cos": mk(n, D), "sin": mk(n, D), "rows": torch.zeros(n, dtype=torch.long, device=DEV),
                 "ctx": mk(ctx_rows, ncap * H)}
            st["keep"][key] = b
        return b

    def _lm_layers(self, st: dict, b: dict, n: int, attend, row_map: torch.Tensor | None, slot: torch.Tensor | None,
                   deep: list | None = None, vis_idx: torch.Tensor | None = None, cap_from: int = 0) -> None:
        """36 decoder layers over ``b['x']`` [n, H]; leaves the final-norm output in ``b['h']`` and the
        DFlash context (captured layers, rows ``cap_from:``) in ``b['ctx']``."""
        m, fvk = self.m, self._fvk
        s = torch.cuda.current_stream().cuda_stream
        lm = m.vlm.model.language_model
        layers = lm.layers
        H, D, NH, KV, S, F_, eps = self.hidden, self.head_dim, self.n_heads, self.n_kv, st["S"], self.ffn, self.eps
        x, h, q, act = b["x"], b["h"], b["q"], b["act"]
        cap_ids = sorted(lm._capture_layer_ids)
        fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, layers[0].input_layernorm.weight.data_ptr(), h.data_ptr(), n, H, eps, s)
        for li, layer in enumerate(layers):
            a = layer.self_attn
            # fused: FP16 GEMM outputs consumed directly (cast to BF16 on load, as FlashDrive's wrapper casts)
            w4 = self._w4[li] if self.backends["w4a8"] == "fused" else None
            if w4 is not None:
                qp, kp, vp = w4["qkv"](fvk, h)
            else:
                qp, kp, vp = a.q_proj(h), a.k_proj(h), a.v_proj(h)    # FlashDrive's W4A8 (ParoQuant) projections
            fvk.flashdrive_qk_prep_bf16(
                qp.data_ptr(), kp.data_ptr(), vp.data_ptr(), qp.shape[1], kp.shape[1], vp.shape[1],
                a.q_norm.weight.data_ptr(), a.k_norm.weight.data_ptr(), b["cos"].data_ptr(), b["sin"].data_ptr(), 0,
                q.data_ptr(), st["k"][li].data_ptr(), D, S * D, st["kvlm"][li].data_ptr(), KV * D, D,
                st["v"][li].data_ptr(), D, S * D, row_map.data_ptr() if row_map is not None else 0,
                slot.data_ptr() if slot is not None else 0, n, NH, KV, eps, s, f16_in=int(w4 is not None))
            attend(st, li, q, b["attn"])
            post_w = layer.post_attention_layernorm.weight.data_ptr()
            if w4 is not None:
                fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), w4["o"](fvk, b["attn"])[0].data_ptr(), post_w,
                                                 h.data_ptr(), n, H, eps, s)
            else:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), a.o_proj(b["attn"]).data_ptr(), post_w, h.data_ptr(), n, H, eps, s)
            mlp = layer.mlp
            if w4 is not None:
                g, u = w4["gu"](fvk, h)
                fvk.flashdrive_silu_mul_f16in_bf16(g.data_ptr(), u.data_ptr(), act.data_ptr(), n * F_, s)
                d = w4["down"](fvk, act)[0]
            else:
                g, u = mlp.gate_proj(h), mlp.up_proj(h)
                fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), act.data_ptr(), n * F_, s)
                d = mlp.down_proj(act)
            nxt = layers[li + 1].input_layernorm.weight if li + 1 < len(layers) else lm.norm.weight
            if deep is not None and li < len(deep):
                if w4 is not None:
                    d = d.to(BF16)
                fvk.residual_add(x.data_ptr(), d.data_ptr(), n * H, s)
                x.index_add_(0, vis_idx, deep[li])                      # FlashDrive's DeepStack injection
                fvk.alpamayo15_rms_bf16(x.data_ptr(), 0, nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            elif w4 is not None:
                fvk.flashdrive_rms_f16delta_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            else:
                fvk.alpamayo15_rms_bf16(x.data_ptr(), d.data_ptr(), nxt.data_ptr(), h.data_ptr(), n, H, eps, s)
            if li in cap_ids:                                           # decoder-layer output, as FlashDrive captures
                j = cap_ids.index(li)
                b["ctx"][:, j * H:(j + 1) * H].copy_(x[cap_from:])

    # streaming prefill ---------------------------------------------------
    def _first_prefill(self, input_ids, attention_mask, pixel_values, image_grid_thw) -> None:
        """FlashDrive's full first-window prefill; the stream state created next rotates every row it wrote."""
        self._orig_first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw)
        self._stream()

    def _attend_prefill(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        p = st["keep"]["prefill_meta"]
        o, _ = self._fa4(q.view(-1, self.n_heads, self.head_dim), st["kvlm"][li], self._v_view(st, li)[0],
                         cu_seqlens_q=p["cu_q"], cu_seqlens_k=p["cu_k"], seqused_k=p["used"],
                         max_seqlen_q=p["max_q"], max_seqlen_k=st["S"], softmax_scale=self.scale, causal=True,
                         pack_gqa=True)
        out.copy_(o.view(out.shape))

    @torch.no_grad()
    def _dflash_prefill(self, inputs_embeds, position_ids, cache_position, visual_pos_masks, deepstack_image_embeds,
                        streaming_attention_mask):
        """Drop-in for FlashDrive's ``_dflash_prefill``: (last-row logits [1, V], DFlash context [1, ctx, C])."""
        _require_autocast()
        m = self.m
        st = self._stream()
        n = inputs_embeds.shape[1]
        g = st["graphs"].get("prefill")
        if g is not None and n != st["keep"]["prefill_n"]:
            raise ValueError(f"streaming window has {n} rows; this stream's prefill graph was captured for "
                             f"{st['keep']['prefill_n']}")
        b = self._lm_buffers(st, n, m._dflash_context_len)
        if g is None:
            st["keep"]["prefill_n"] = n
            rows = cache_position.tolist()
            if not m._extended_vision_ranges:
                raise ValueError("causal-by-position streaming attention holds for extended vision ranges (Alpamayo 1.5)")
            runs = contiguous_runs(rows)
            lens = [e - s for s, e in runs]
            st["keep"]["prefill_meta"] = {
                "cu_q": torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV),
                "cu_k": torch.zeros(len(runs) + 1, dtype=torch.int32, device=DEV),
                "used": torch.tensor([e for _, e in runs], dtype=torch.int32, device=DEV), "max_q": max(lens)}
            b["rows"].copy_(cache_position)
            st["keep"]["prefill_rows_obj"] = cache_position   # FlashDrive's fixed per-stream tensor
            b["emb"] = torch.empty_like(inputs_embeds[0])
            b["vis"] = visual_pos_masks[0].nonzero().squeeze(1)
            b["deep"] = [torch.empty_like(t) for t in deepstack_image_embeds]
            b["ctx_len"] = m._dflash_context_len
            b["emb"].copy_(inputs_embeds[0])
            for dst, src in zip(b["deep"], deepstack_image_embeds):
                dst.copy_(src)
            torch.index_select(st["cos"], 0, b["rows"], out=b["cos"])
            torch.index_select(st["sin"], 0, b["rows"], out=b["sin"])
            saved = [(k.clone(), v.clone()) for k, v in zip(st["k"], st["v"])]   # capture writes the window rows
            kvlm = st["kvlm"].clone()
            out = {}

            def run():
                b["x"].copy_(b["emb"])
                self._lm_layers(st, b, n, self._attend_prefill, b["rows"], None, b["deep"], b["vis"],
                                cap_from=n - b["ctx_len"])
                out["logits"] = m.vlm.lm_head(b["h"][n - 1:n])
            g = _capture(run)
            for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), saved):
                k.copy_(ks); v.copy_(vs)
            st["kvlm"].copy_(kvlm)
            del saved, kvlm
            st["graphs"]["prefill"] = g
            st["keep"]["prefill_out"] = out
        elif cache_position is not st["keep"]["prefill_rows_obj"]:
            raise ValueError("streaming cache positions changed within a stream")
        b["emb"].copy_(inputs_embeds[0])
        for dst, src in zip(b["deep"], deepstack_image_embeds):
            dst.copy_(src)
        # Image-token rows are fixed per stream (FlashDrive caches them at the first streaming window too).
        g.replay()
        return st["keep"]["prefill_out"]["logits"], b["ctx"].unsqueeze(0)

    @torch.no_grad()
    def _shift(self) -> None:
        """Drop-in for FlashDrive's ``_shift_streaming_kv_cache`` (frame blocks f -> f-1 of every view),
        plus the VLM view of the moved rows."""
        m = self.m
        st = self._stream()
        g = st["graphs"].get("shift")
        if g is None:
            src, dst = m._shift_src_index, m._shift_dst_index
            st["keep"]["shift"] = (src, dst)
            if not torch.equal(dst.sort().values, dst) or not torch.equal(src.sort().values, src):
                raise ValueError("FlashDrive shift indices must be increasing")

            def run():
                for li in range(self.n_layers):
                    for t in (st["k"][li], st["v"][li]):
                        t.index_copy_(2, dst, t.index_select(2, src))
                    self._rope_rows(st, li, st["kvlm"][li], st["cos"], st["sin"], 0, dst.numel(), row_map=dst)
            # The warm-up shifts the live cache once; it is restored from the snapshot before the real replay.
            saved = [(k.clone(), v.clone()) for k, v in zip(st["k"], st["v"])]
            kvlm = st["kvlm"].clone()
            g = _capture(run, warmups=1)
            for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), saved):
                k.copy_(ks); v.copy_(vs)
            st["kvlm"].copy_(kvlm)
            del saved, kvlm
            st["graphs"]["shift"] = g
        g.replay()

    # decode ----------------------------------------------------------------
    def _attend_verify(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        n = q.shape[0]
        o, _ = self._fa4(q.view(1, n, self.n_heads, self.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                         seqused_k=st["keep"]["dec"]["used"], softmax_scale=self.scale, causal=True, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _attend_all_rows(self, st: dict, li: int, q: torch.Tensor, out: torch.Tensor) -> None:
        # FlashDrive's one-row forward passes no mask: SDPA attends every cache row.
        o, _ = self._fa4(q.view(1, 1, self.n_heads, self.head_dim), st["kvlm"][li].unsqueeze(0), self._v_view(st, li),
                         softmax_scale=self.scale, causal=False, pack_gqa=True)
        out.copy_(o.view(out.shape))

    def _dec_state(self, st: dict) -> dict:
        d = st["keep"].get("dec")
        if d is None:
            m = self.m
            B = m.dflash_block_size
            d = {"slot": torch.zeros(1, dtype=torch.long, device=DEV), "used": torch.zeros(1, dtype=torch.int32, device=DEV),
                 "block_ids": torch.zeros(1, B, dtype=torch.long, device=DEV),
                 "th": torch.zeros(1, m._dflash_context_len, len(m._draft_model.target_layer_ids) * self.hidden,
                                   device=DEV, dtype=BF16),
                 "traj_id": torch.full((1,), m.traj_start_token_id, dtype=torch.long, device=DEV),
                 "neg1": torch.full((), -1, dtype=torch.long, device=DEV),
                 "draft_pos": torch.arange(m._dflash_context_len + B, device=DEV).unsqueeze(0)}
            st["keep"]["dec"] = d
        return d

    @torch.no_grad()
    def _block_step(self, block_output_ids, target_hidden, position_ids, cache_position, temperature, top_p):
        """Drop-in for FlashDrive's ``_dflash_block_step``: (block, posterior, context, stats) in one graph."""
        _require_autocast()
        from flashdrive.dflash import sample_tokens
        m = self.m
        st = self._stream()
        d = self._dec_state(st)
        B = m.dflash_block_size
        key = ("block", float(temperature), float(top_p))
        g = st["graphs"].get(key)
        if g is None:
            b = self._lm_buffers(st, B, B)
            out = {}
            stop_id, period_id = m._dflash_stop_token_id, m._dflash_period_token_id

            def run():
                lm = m.vlm.model.language_model
                noise = lm.embed_tokens(d["block_ids"])
                hidden = m._draft_model(target_hidden=d["th"], noise_embedding=noise, position_ids=d["draft_pos"],
                                        is_causal=False)
                draft_logits = m.vlm.lm_head(hidden[:, 1:, :])
                proposed = sample_tokens(draft_logits, m._dflash_traj_mask, 0.0)
                follows_stop = F.pad(proposed[:, :-1] == stop_id, (1, 0))
                proposed = torch.where(follows_stop, d["traj_id"], proposed)
                block = torch.cat([d["block_ids"][:, :1], proposed], dim=1)
                torch.index_select(lm.embed_tokens.weight, 0, block[0], out=b["x"])
                rows = torch.arange(B, device=DEV) + d["slot"]
                torch.index_select(st["cos"], 0, rows, out=b["cos"])
                torch.index_select(st["sin"], 0, rows, out=b["sin"])
                self._lm_layers(st, b, B, self._attend_verify, None, d["slot"], cap_from=0)
                verify_logits = m.vlm.lm_head(b["h"]).unsqueeze(0)
                posterior = sample_tokens(verify_logits, m._dflash_traj_mask, temperature, top_p)
                # FlashDrive's round decisions (dflash.py, block_step_fn), verbatim.
                matches = block[:, 1:] == posterior[:, :-1]
                acceptance = matches.cumprod(dim=1).sum(dim=1)[0]
                slot = acceptance.unsqueeze(0)
                bonus = posterior[0].gather(0, slot)
                emission = torch.cat([block[0], bonus]).scatter(0, slot + 1, bonus)
                slots = torch.arange(emission.shape[0], device=DEV)
                stop_hits = (emission == stop_id) & (slots <= acceptance + 1)
                first_stop = torch.where(stop_hits.any(), stop_hits.int().argmax(), d["neg1"])
                period_hit = (bonus[0] == period_id).long()
                out["block"], out["posterior"] = block, posterior
                out["ctx"] = b["ctx"].unsqueeze(0)
                out["stats"] = torch.stack([acceptance, first_stop, period_hit])
            saved = self._snapshot_rows(st, cache_position)
            d["block_ids"].copy_(block_output_ids); d["th"].copy_(target_hidden)
            d["slot"].copy_(cache_position[:1]); d["used"].copy_((cache_position[:1] + B).int())
            gen = torch.cuda.default_generators[torch.cuda.current_device()]
            rng = gen.get_state()
            g = _capture(run)
            gen.set_state(rng)                      # warm-up / capture draws must not consume the caller's stream
            self._restore_rows(st, cache_position, saved)
            st["graphs"][key] = g
            st["keep"][("block_out",) + key] = out
        out = st["keep"][("block_out",) + key]
        d["block_ids"].copy_(block_output_ids)
        d["th"].copy_(target_hidden)
        d["slot"].copy_(cache_position[:1])
        d["used"].copy_((cache_position[:1] + B).int())
        g.replay()
        return out["block"], out["posterior"], out["ctx"], out["stats"]

    @torch.no_grad()
    def _traj_forward(self, input_ids, position_ids, cache_position) -> None:
        """Drop-in for FlashDrive's ``_dflash_traj_forward``: the one-row trajectory-start forward (K/V only)."""
        _require_autocast()
        m = self.m
        st = self._stream()
        d = self._dec_state(st)
        g = st["graphs"].get("traj")
        if g is None:
            b = self._lm_buffers(st, 1, 1)
            ids = torch.zeros(1, dtype=torch.long, device=DEV)
            st["keep"]["traj_ids"] = ids

            def run():
                lm = m.vlm.model.language_model
                torch.index_select(lm.embed_tokens.weight, 0, ids, out=b["x"])
                torch.index_select(st["cos"], 0, d["slot"], out=b["cos"])
                torch.index_select(st["sin"], 0, d["slot"], out=b["sin"])
                self._lm_layers(st, b, 1, self._attend_all_rows, None, d["slot"], cap_from=0)
            saved = self._snapshot_rows(st, cache_position)
            ids.copy_(input_ids.view(-1)); d["slot"].copy_(cache_position[:1])
            g = _capture(run)
            self._restore_rows(st, cache_position, saved)
            st["graphs"]["traj"] = g
        st["keep"]["traj_ids"].copy_(input_ids.view(-1))
        d["slot"].copy_(cache_position[:1])
        g.replay()
        # FlashDrive warms this primitive once (first trajectory window after a fresh registry) and skips it after.
        m._compiled_step_registry.setdefault("dflash_traj_forward", None)

    def _snapshot_rows(self, st: dict, cache_position: torch.Tensor) -> tuple:
        """Cache rows a captured decode graph writes during warm-up/capture (restored afterwards)."""
        r0 = int(cache_position[0])
        r1 = min(st["S"], r0 + self.m.dflash_block_size)
        return r0, r1, [(k[0, :, r0:r1].clone(), v[0, :, r0:r1].clone()) for k, v in zip(st["k"], st["v"])], \
            st["kvlm"][:, r0:r1].clone()

    def _restore_rows(self, st: dict, cache_position: torch.Tensor, saved: tuple) -> None:
        r0, r1, rows, kvlm = saved
        for (k, v), (ks, vs) in zip(zip(st["k"], st["v"]), rows):
            k[0, :, r0:r1].copy_(ks); v[0, :, r0:r1].copy_(vs)
        st["kvlm"][:, r0:r1].copy_(kvlm)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start a new stream (see ``reset_flashdrive_stream``); stream-bound buffers and graphs are
        rebuilt for it."""
        reset_flashdrive_stream(self.m)
        self._st = None
