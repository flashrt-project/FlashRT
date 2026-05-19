"""G5 prep — Graph-safe replacement for Wan's 3D RoPE on video Q/K.

Upstream ``wan.modules.model.rope_apply`` is the 3D factorized
complex-multiply RoPE applied to video Q/K every attention layer.
Two ops in it are forbidden inside torch.cuda.graph capture:

    1. torch.unique(grid_sizes, dim=0, return_inverse=True)
       merge_sort dispatches a cudaStreamSynchronize on host.
    2. uniq.tolist() inside the for-g_idx loop
       Forces a GPU->CPU readback per call.

For Motus at B=1 with a fixed video shape (single grid_size = [T_l,
H', W']) the entire unique/tolist path collapses to "always one
grid". We can precompute the per-token freq_grid tensor once at
set_prompt time and use a vectorized complex-multiply that doesn't
touch CPU side.

Numerical contract: produces output identical to upstream rope_apply
(verified by isolated test in G5 debug). Cos drift expected zero —
the math is exactly the same x_complex * freq_grid_complex multiply,
just without the unique/loop scaffolding.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Cached freq_grid (set per pipeline / per (T, H, W) tuple).
# Module-level singleton because Wan's rope_apply is also module-level.
_FREQ_GRID: Optional[torch.Tensor] = None  # complex64, shape [seq_len, 1, c_complex]
_GRID_KEY: Optional[tuple] = None          # (T, H, W) for invalidation

# G7.15: separate real/imag fp32 views of _FREQ_GRID, layout
# (seq_len, c_complex) contiguous. Set alongside _FREQ_GRID in
# precompute_freq_grid; consumed by the fused fvk.rope_apply_bf16_to_fp32.
_FREQ_GRID_RE_FP32: Optional[torch.Tensor] = None
_FREQ_GRID_IM_FP32: Optional[torch.Tensor] = None


def _build_freq_grid(freqs: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    """Replicate upstream _make_freq_grid for a single (T, H, W).

    `freqs` shape: [1024, c // num_heads // 2] complex64
    Returns: [T*H*W, 1, c_complex_total] complex64
    """
    c = freqs.shape[1]
    c_f = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3
    fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)
    fi = torch.cat([
        fpart[:T].view(T, 1, 1, -1).expand(T, H, W, -1),
        hpart[:H].view(1, H, 1, -1).expand(T, H, W, -1),
        wpart[:W].view(1, 1, W, -1).expand(T, H, W, -1),
    ], dim=-1).reshape(T * H * W, 1, -1)
    return fi.contiguous()


def precompute_freq_grid(model, grid_sizes: torch.Tensor) -> None:
    """Build the per-token freq_grid for the given (T, H, W) grid.

    Args:
        model: the loaded Motus model (we read wan_model.freqs).
        grid_sizes: [B, 3] int tensor with rows (T, H, W). For Motus
                    B=1 and all rows equal — we read row 0.
    """
    global _FREQ_GRID, _GRID_KEY, _UNPATCH_GRID
    global _FREQ_GRID_RE_FP32, _FREQ_GRID_IM_FP32
    freqs = model.video_model.wan_model.freqs
    if not freqs.is_cuda:
        # Frontend should have moved this; defensive.
        freqs = freqs.cuda()
    gsz = grid_sizes.to(torch.long)
    T = int(gsz[0, 0].item())
    H = int(gsz[0, 1].item())
    W = int(gsz[0, 2].item())
    key = (T, H, W)
    if _GRID_KEY == key and _FREQ_GRID is not None:
        return
    _FREQ_GRID = _build_freq_grid(freqs, T, H, W)
    _GRID_KEY = key
    _UNPATCH_GRID = (T, H, W)
    # G7.15: pre-cache real/imag fp32 contiguous tensors for the fused
    # rope_apply kernel. _FREQ_GRID has shape [seq_len, 1, c_complex];
    # squeeze and split.
    fg = _FREQ_GRID.squeeze(1)  # [seq_len, c_complex] complex64
    _FREQ_GRID_RE_FP32 = fg.real.to(torch.float32).contiguous()
    _FREQ_GRID_IM_FP32 = fg.imag.to(torch.float32).contiguous()
    logger.info(
        f"[g5/rope] freq_grid cached for (T={T}, H={H}, W={W}); "
        f"shape={tuple(_FREQ_GRID.shape)}, dtype={_FREQ_GRID.dtype}")


def graph_safe_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,  # ignored; cached grid used
    freqs: torch.Tensor,        # ignored; cached
) -> torch.Tensor:
    """Drop-in replacement for wan.modules.model.rope_apply.

    Assumes precompute_freq_grid was called for the active grid size.
    Same math as upstream (complex multiply per token), zero CPU sync.

    x shape: [B, T_total, num_heads, head_dim]   bf16/fp32
    Returns: same shape, fp32 (matches upstream's `.float()` return).
    """
    if _FREQ_GRID is None:
        # Fall back to upstream — only happens if precompute wasn't run.
        from wan.modules.model import rope_apply_original
        return rope_apply_original(x, grid_sizes, freqs)

    B, T, N, head_dim = x.shape
    seq_len = _FREQ_GRID.shape[0]

    # G7.15/G7.16 fused kernel path.
    # G7.16 default: bf16 output (keeps cat in bf16, FA2 fast path).
    # FLASH_RT_MOTUS_NO_G7_16=1 → fp32 output (G7.15 legacy semantics).
    if (x.dtype == torch.bfloat16 and x.is_cuda
            and _FREQ_GRID_RE_FP32 is not None):
        import os
        import flash_rt.flash_rt_kernels as fvk
        x_c = x if x.is_contiguous() else x.contiguous()
        bf16_out = (os.environ.get('FLASH_RT_MOTUS_NO_G7_16', '0') != '1'
                    and hasattr(fvk, 'rope_apply_bf16_to_bf16'))
        out_dtype = torch.bfloat16 if bf16_out else torch.float32
        out = torch.empty(B, T, N, head_dim, dtype=out_dtype, device=x.device)
        if bf16_out:
            fvk.rope_apply_bf16_to_bf16(
                int(x_c.data_ptr()),
                int(_FREQ_GRID_RE_FP32.data_ptr()),
                int(_FREQ_GRID_IM_FP32.data_ptr()),
                int(out.data_ptr()),
                int(B), int(T), int(N), int(head_dim), int(seq_len),
                torch.cuda.current_stream().cuda_stream)
        else:
            fvk.rope_apply_bf16_to_fp32(
                int(x_c.data_ptr()),
                int(_FREQ_GRID_RE_FP32.data_ptr()),
                int(_FREQ_GRID_IM_FP32.data_ptr()),
                int(out.data_ptr()),
                int(B), int(T), int(N), int(head_dim), int(seq_len),
                torch.cuda.current_stream().cuda_stream)
        return out

    # Legacy fallback (matches upstream view_as_complex round-trip).
    x_c = torch.view_as_complex(
        x.to(torch.float64).reshape(B, T, N, -1, 2)).contiguous()
    if T == seq_len:
        y_c = x_c * _FREQ_GRID.unsqueeze(0)
    else:
        y_c = x_c.clone()
        y_c[:, :seq_len] = x_c[:, :seq_len] * _FREQ_GRID.unsqueeze(0)
    return torch.view_as_real(y_c).reshape(B, T, N, -1).float()


_INSTALLED = False

# Cached unpatchify grid (T, H, W) — set when rope precompute runs.
_UNPATCH_GRID: Optional[tuple] = None


def _make_graph_safe_unpatchify(wan_model_self):
    """Build a closure replacing WanModel.unpatchify for B=1 fixed grid.

    Upstream uses ``for u, v in zip(x, grid_sizes.tolist()): ...``
    where ``grid_sizes.tolist()`` is a CPU sync forbidden in capture.
    We bake (T, H, W) into the closure (set by precompute_freq_grid).
    """
    import math
    c = wan_model_self.out_dim
    patch_size = wan_model_self.patch_size

    def graph_safe_unpatchify(x, grid_sizes):
        if _UNPATCH_GRID is None:
            # Fall back; only happens pre-precompute.
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
                out.append(u)
            return out

        T, H, W = _UNPATCH_GRID
        seq_len = T * H * W
        # x is a list of [L, C_out * prod(patch_size)] tensors at B=1.
        out = []
        for u in x:
            u = u[:seq_len].view(T, H, W, *patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(
                c, T * patch_size[0], H * patch_size[1], W * patch_size[2])
            out.append(u)
        return out

    return graph_safe_unpatchify


def install_graph_safe_rope(model=None) -> None:
    """Monkey-patch wan.modules.model.rope_apply (and WanModel.unpatchify
    if model passed) with graph-safe variants. Idempotent.
    """
    global _INSTALLED
    if not _INSTALLED:
        import wan.modules.model as _wan_model  # noqa: WPS433
        _wan_model.rope_apply = graph_safe_rope_apply
        _INSTALLED = True
        logger.info("[g5/rope] monkey-patched wan.modules.model.rope_apply")

    if model is not None:
        wan_model = model.video_model.wan_model
        if not getattr(wan_model, "_g5_unpatchify_installed", False):
            new_unpatch = _make_graph_safe_unpatchify(wan_model)
            # Bind as instance method (closure captures self via factory).
            import types
            wan_model.unpatchify = types.MethodType(
                lambda self, x, grid_sizes: new_unpatch(x, grid_sizes),
                wan_model)
            wan_model._g5_unpatchify_installed = True
            logger.info("[g5/rope] patched WanModel.unpatchify (B=1 fixed grid)")
