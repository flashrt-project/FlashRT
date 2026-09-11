"""LTX-2.5 FFN swap: upstream NVFP4Linear pair -> FlashRT W4A4 CUTLASS chain.

Upstream FeedForward runs six launches per call (activation quantize +
cuBLASLt + bias, torch GELU, activation quantize + cuBLASLt + bias). The
FlashRT chain runs three:

    quantize_bf16_to_nvfp4_swizzled(x)
    fp4_w4a16_gemm_bias_gelu_fp4out_sm120   (up proj + bias + tanh-GELU + fp4)
    fp4_w4a16_gemm_sm120_bf16out            (down proj)  [+ add_bias]

Weights are repacked once per build from the checkpoint's NVFP4 layout
(dequantize to bf16, requantize into the fvk swizzled layout); the original
parameters are dropped right after so resident memory stays flat. Measured on
block-0 weights at the model's real M sizes this is 1.25-1.28x the upstream
pair at equivalent accuracy against a bf16 golden reference.

All launches take raw pointers on preallocated buffers -- safe to record into
a CUDA graph.
"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

try:
    fvk = importlib.import_module("flash_rt.flash_rt_kernels")
except ModuleNotFoundError as exc:
    # Same rule, and the same reason for going through importlib, as the
    # attention swap: absent is a fallback, broken is a bug, and the
    # package's ``__getattr__`` would make both arrive as ImportError.
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    fvk = None

_REQUIRED = (
    "quantize_bf16_to_nvfp4_swizzled",
    "fp4_w4a16_gemm_bias_gelu_fp4out_sm120",
    "fp4_w4a16_gemm_sm120_bf16out",
    "add_bias_bf16",
)


def fvk_ffn_available() -> bool:
    return fvk is not None and all(hasattr(fvk, s) for s in _REQUIRED)


def rows_are_swappable(rows: int) -> bool:
    """Whether the fused chain will accept a call of this row count.

    The CUTLASS chain reports a validation failure for row counts that are
    not 128-aligned and returns *without writing its output*, so a call that
    reached it anyway would leave the caller reading whatever the buffer
    held. The predicate lives here, named, because two places have to agree
    on it: the forward that routes the call and the installer that decides
    whether upstream weights can be freed.
    """
    return rows % 128 == 0


def _swizzled_sf_bytes(rows: int, cols: int) -> int:
    # CUTLASS blockscaled atom layout: one e4m3 byte per 16-element block,
    # rows padded to 128 and SF columns padded to 4 (measured write footprint
    # of quantize_bf16_to_nvfp4_swizzled; the motus-era x16 formula
    # over-allocated).
    n_blocks = cols // 16
    return ((rows + 127) // 128 * 128) * ((n_blocks + 3) // 4 * 4)


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _quantize_bf16_to_swz(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    packed = torch.empty(rows, cols // 2, dtype=torch.uint8, device=x.device)
    sf = torch.zeros(_swizzled_sf_bytes(rows, cols), dtype=torch.uint8,
                     device=x.device)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        int(x.data_ptr()), int(packed.data_ptr()), int(sf.data_ptr()),
        rows, cols, _stream())
    return packed, sf


def _repack_nvfp4_linear(lin: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Upstream NVFP4Linear -> (fvk packed weight, fvk swizzled SF, alpha).

    Dequantizes with the exact upstream reference kernel, requantizes into the
    fvk layout, then frees the original weight storage. alpha stays 1.0: the
    fvk per-16 E4M3 block scales absorb the dynamic range that upstream
    splits into block scales x weight_scale_2.
    """
    from ltx_kernels import nvfp4 as ltx_nvfp4

    w_bf16 = ltx_nvfp4.dequantize_nvfp4(
        lin.weight, lin.weight_scale_2,
        lin.weight_scale.view(torch.float8_e4m3fn),
    ).to(torch.bfloat16)
    packed, sf = _quantize_bf16_to_swz(w_bf16)
    del w_bf16
    return packed, sf, 1.0


class _FfnBuffers:
    """Per-(device, M, inner) intermediate buffers, grown on demand."""

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def get(self, device: torch.device, m: int, dim: int, inner: int):
        key = (device, m, dim, inner)
        bufs = self._cache.get(key)
        if bufs is None:
            xq = torch.empty(m, dim // 2, dtype=torch.uint8, device=device)
            xsf = torch.zeros(_swizzled_sf_bytes(m, dim), dtype=torch.uint8,
                              device=device)
            h4 = torch.empty(m, inner // 2, dtype=torch.uint8, device=device)
            h4sf = torch.zeros(_swizzled_sf_bytes(m, inner), dtype=torch.uint8,
                               device=device)
            y = torch.empty(m, dim, dtype=torch.bfloat16, device=device)
            bufs = (xq, xsf, h4, h4sf, y)
            self._cache[key] = bufs
        return bufs


def _make_ffn_forward(ff: torch.nn.Module, buffers: _FfnBuffers,
                      free_upstream: bool = False):
    up = ff.net[0].proj
    down = ff.net[2]
    dim, inner = up.in_features, up.out_features
    # Survive re-installs across stage rebuilds: always chain to the true
    # original forward, not a previous swap closure.
    upstream_forward = getattr(ff, "_flash_rt_ffn_orig", None)
    if upstream_forward is None:
        upstream_forward = ff.forward
        ff._flash_rt_ffn_orig = upstream_forward

    # The builder reuses module objects across stage rebuilds and reloads the
    # same checkpoint weights into them, so the repacked tensors stay valid --
    # cache them on the module instead of repacking per build.
    pack = getattr(ff, "_flash_rt_ffn_pack", None)
    if pack is None:
        up_p, up_sf, _ = _repack_nvfp4_linear(up)
        dn_p, dn_sf, _ = _repack_nvfp4_linear(down)
        ff._flash_rt_ffn_pack = (up_p, up_sf, dn_p, dn_sf)
    else:
        up_p, up_sf, dn_p, dn_sf = pack
    up_bias = up.bias.detach() if up.bias is not None else torch.zeros(
        inner, dtype=torch.bfloat16, device=up_p.device)
    dn_bias = down.bias.detach() if down.bias is not None else None

    freed = False
    if free_upstream and dim >= 4096:
        # Resident mode never reloads the checkpoint, so the upstream fp4
        # storage is dead weight (~3.6GB across the video FFNs). Audio FFNs
        # keep theirs -- their unaligned-M calls fall back to upstream.
        up.weight.data = up.weight.data.new_empty(0)
        up.weight_scale.data = up.weight_scale.data.new_empty(0)
        down.weight.data = down.weight.data.new_empty(0)
        down.weight_scale.data = down.weight_scale.data.new_empty(0)
        freed = True

    # The upstream fp4 params stay in place: the builder reuses the module
    # across stage rebuilds and reloads the state dict into them. The
    # repacked copies add ~4.8GB resident for the 48-block model, which fits
    # alongside the 23GB inference peak on a 32GB part.
    keep = (up_p, up_sf, dn_p, dn_sf, up_bias, dn_bias)

    def forward(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        m = x2.shape[0]
        if not rows_are_swappable(m):
            # The chain rejects unaligned M (can_implement status 11)
            # without writing the output; e.g. the ~126-token audio branch.
            # Those calls are negligible work -- keep them on upstream.
            if freed:
                raise RuntimeError(
                    f"ltx25 FFN: M={m} is not 128-aligned but the upstream "
                    "weights were freed (resident mode). Use dimensions whose "
                    "token counts are 128-aligned, or disable capture mode.")
            return upstream_forward(x)
        x2 = x2 if x2.is_contiguous() else x2.contiguous()
        stream = _stream()
        xq, xsf, h4, h4sf, y = buffers.get(x2.device, m, dim, inner)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            int(x2.data_ptr()), int(xq.data_ptr()), int(xsf.data_ptr()),
            m, dim, stream)
        fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
            int(xq.data_ptr()), int(up_p.data_ptr()),
            int(xsf.data_ptr()), int(up_sf.data_ptr()),
            int(up_bias.data_ptr()), int(h4.data_ptr()), int(h4sf.data_ptr()),
            m, inner, dim, 1.0, stream)
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            int(h4.data_ptr()), int(dn_p.data_ptr()), int(y.data_ptr()),
            m, dim, inner,
            int(h4sf.data_ptr()), int(dn_sf.data_ptr()), 1.0, stream)
        if dn_bias is not None:
            fvk.add_bias_bf16(
                int(y.data_ptr()), int(dn_bias.data_ptr()), m, dim, stream)
        return y.view(*shape[:-1], dim)

    # Raw-pointer launches are opaque to dynamo; keep them eager under
    # torch.compile (CUDA graph capture still records the kernels).
    forward = torch._dynamo.disable(forward)
    forward._flash_rt_keep = keep
    return forward


def install_nvfp4_ffn(model: torch.nn.Module, *,
                      free_upstream: bool = False) -> int:
    """Swap every NVFP4 FeedForward on ``model`` to the fvk W4A4 chain.

    Returns the number of FeedForward modules swapped. Modules whose linears
    are not upstream NVFP4Linear (e.g. a bf16 build) are left untouched.
    """
    if not fvk_ffn_available():
        logger.info("[ltx25] fvk FFN chain unavailable; keeping upstream FFN")
        return 0
    from ltx_core.quantization.nvfp4.linear import NVFP4Linear

    buffers = _FfnBuffers()
    count = 0
    for module in model.modules():
        net = getattr(module, "net", None)
        if net is None or len(net) != 3:
            continue
        proj = getattr(net[0], "proj", None)
        if not isinstance(proj, NVFP4Linear) or not isinstance(net[2], NVFP4Linear):
            continue
        if proj.in_features % 16 or proj.out_features % 16:
            continue
        module.forward = _make_ffn_forward(module, buffers,
                                           free_upstream=free_upstream)
        count += 1
    logger.info("[ltx25] swapped %d FeedForward modules to fvk W4A4 chain",
                count)
    return count


def uninstall_nvfp4_ffn(model: torch.nn.Module) -> int:
    """Undo :func:`install_nvfp4_ffn`. Returns the number of modules restored.

    Deleting the instance attribute restores the class's own ``forward`` and
    drops the closure holding the repacked FP4 weights and the shared buffer
    pool -- which is the only way that memory comes back, because the builder
    caches model shells by structure and reuses them across builds. Freeing
    the loaded weights (``dispose``) does not touch anything a swap attached
    to the shell; this does.
    """
    count = 0
    for module in model.modules():
        forward = module.__dict__.get("forward")
        if forward is not None and hasattr(forward, "_flash_rt_keep"):
            del module.forward
            count += 1
    if count:
        logger.info("[ltx25] restored %d upstream FeedForward modules", count)
    return count


class SwapInstallingBuilder:
    """Model-builder wrapper that installs FlashRT swaps after each build.

    Wraps an upstream ``ModelBuilderProtocol``; ``build`` delegates and then
    applies the requested swap installers to the loaded model. The functional
    ``with_*`` combinators re-wrap so the wrapper survives
    ``DiffusionStage._prepared_builder``.
    """

    def __init__(self, inner, installers) -> None:
        self._inner = inner
        self._installers = tuple(installers)

    def build(self, **kwargs):
        model = self._inner.build(**kwargs)
        for install in self._installers:
            install(model)
        return model

    def _rewrap(self, inner):
        return SwapInstallingBuilder(inner, self._installers)

    def with_module_ops(self, ops):
        return self._rewrap(self._inner.with_module_ops(ops))

    def with_sd_ops(self, ops):
        return self._rewrap(self._inner.with_sd_ops(ops))

    def with_loras(self, loras):
        return self._rewrap(self._inner.with_loras(loras))

    def with_fuse_rule(self, rule):
        return self._rewrap(self._inner.with_fuse_rule(rule))

    def __getattr__(self, item):
        return getattr(self._inner, item)
