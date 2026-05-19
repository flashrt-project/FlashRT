"""G3b — fvk BF16 Linear (GEMM) substitution for Motus.

Replaces ``nn.Linear.forward`` of every Linear under hot-path scopes
with a ``GemmRunner.bf16_nn`` / ``bf16_nn_bias`` invocation. Same
monkey-patch pattern as G3a norms; pure data path; no source changes.

Layout note:
    nn.Linear stores weight as [out, in] = [N, K].
    fvk.GemmRunner.bf16_nn does ``D = A @ B`` where A is [M, K] and
    B is [K, N], i.e. NN (no transpose). To replace ``X @ W^T`` we
    pre-transpose the weight to [K, N] and pin the contiguous tensor
    in the closure cell (same closure-pinning rule as G3a — failing
    to pin causes a dangling pointer / NaN, learned in G3a debug).

Scope (hot path = per-step, per-layer, per-expert):
    * video_model.wan_model.blocks.{i}.{q,k,v,o}            (24*30=720 calls/step)
    * video_model.wan_model.blocks.{i}.ffn.{0,2}            (60*30 calls/step? actually 2 per layer per step = 60)
    * video_model.wan_model.blocks.{i}.cross_attn.{q,k,v,o}
    * video_model.wan_model.text_embedding (one-shot per call)
    * video_model.wan_model.time_embedding[.0,.2]
    * video_model.wan_model.time_projection[.1]
    * video_model.wan_model.head
    * action_expert.{wan_action_o, ffn.0, ffn.2, decoder.action_head}
    * action_expert.{time_embedding[.0,.2], time_projection[.1]}
    * action_expert.input_encoder.action_encoder.* (mlp3x_silu)
    * und_expert.{wan_und_o, ffn.0, ffn.2}
    * und_expert.vlm_adapter.* (mlp3x_silu)

Skipped (NOT replaced):
    * vlm_model.*           Qwen3-VL-2B frozen, runs once outside hot loop
    * video_model.vae.*     Wan2.2 VAE encode/decode, one-shot per call
    * action_expert.wan_action_qkv  nn.Parameter (no .forward) — torch.einsum
                                     in motus.py:235; lives with attention
                                     dispatch, defer to G3c
    * und_expert.wan_und_qkv         same as above

Numerical contract (G3b vs G3a):
    * cuBLASLt BF16 GEMM produces output within ~1-2 ULPs of PyTorch
      F.linear (fp32 accumulator either way). Cumulative drift over
      900+ GEMMs/step × 10 steps puts cos floor at ~0.997-0.999.
    * Cos must stay >= 0.999 per MOTUS_EXECUTION_PLAN.md §G3.5.
"""

from __future__ import annotations

import logging
import os

import torch

from flash_rt.models.motus._stream import cs
import torch.nn as nn

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)


# Set to '1' to log every replacement (verbose).
_TRACE = os.environ.get('FLASH_RT_MOTUS_LINEAR_TRACE', '0') == '1'


# Module name prefixes that are in the hot path (every one of their
# nn.Linear children gets replaced). Anything outside this set stays
# on the upstream PyTorch path.
_HOT_PATH_PREFIXES = (
    'video_model.wan_model.blocks.',
    'video_model.wan_model.text_embedding',
    'video_model.wan_model.time_embedding',
    'video_model.wan_model.time_projection',
    'video_model.wan_model.head',
    'video_module',          # video_module.* itself (no Linears, but be safe)
    'action_expert.',
    'action_module',
    'und_expert.',
    'und_module',
)

# Explicit deny prefixes — these are NEVER replaced even if they
# happen to fall under a hot-path prefix above.
_SKIP_PREFIXES = (
    'video_model.vae',                   # VAE encode/decode
    'video_model.wan_model.patch_embedding',  # Conv3D, not Linear; just in case
)


# ──────────────────────────────────────────────────────────────────
# Forward factory
# ──────────────────────────────────────────────────────────────────

def _make_linear_forward(
    weight: torch.Tensor,                # [N, K] from nn.Linear
    bias: torch.Tensor | None,           # [N] or None
    gemm: fvk.GemmRunner,
    label: str = 'linear',
    bias_skip_flag: list = None,         # G7.12: 1-elem mutable list, flip post-install
):
    if bias_skip_flag is None:
        bias_skip_flag = [False]
    """Return a closure that replaces nn.Linear.forward.

    nn.Linear.forward computes  ``D = X @ W^T + b``  with W shape
    [N, K]. We pre-transpose to [K, N] and call fvk.GemmRunner.bf16_nn
    which is a plain NN GEMM (D = A @ B, no transpose).

    Bias handling: ``bf16_nn_bias`` (cuBLASLt BIAS epilogue) returns
    CUBLAS_STATUS_NOT_SUPPORTED for some BF16 shapes on sm_120 in
    current cuBLAS — caught at G3b debug. We fall back to plain
    ``bf16_nn`` + a torch ``+ bias`` (broadcast) which is a tiny
    elementwise op and does not move us off the fvk hot path for the
    GEMM itself.
    """
    N, K = int(weight.shape[0]), int(weight.shape[1])
    # IN-PLACE pre-transpose: swap the weight tensor's underlying storage
    # to the contiguous transpose. This avoids keeping TWO copies (the
    # original [N,K] + a transposed [K,N]); on Motus the +12 GB second
    # copy across 493 Linears OOMs the 32 GB 5090 during VAE decode.
    # Caveat: after this, ``module.weight.shape`` returns [K, N]. The
    # upstream Motus / Wan code only ever invokes Linears via
    # ``self.q(x)`` etc. (no direct .weight reads in the hot path), so
    # this shape mutation is observed only by our own wrapper.
    w_t = weight.detach().t().contiguous()              # [K, N]
    weight.data = w_t                                   # release [N,K] storage
    w_t_ptr = int(weight.data_ptr())

    if bias is not None:
        # Bias is just [N], cheap; keep contiguous, no transpose.
        if not bias.is_contiguous():
            bias.data = bias.data.contiguous()
        b_pinned = bias
    else:
        b_pinned = None

    def forward(x: torch.Tensor) -> torch.Tensor:
        # Keep transposed weight + bias alive via closure refs.
        _ = (w_t, b_pinned)

        # Remember the upstream's expected output dtype. Upstream may
        # call this Linear inside torch.amp.autocast('cuda', fp32) (e.g.
        # Wan's get_time_embedding asserts fp32 output) — we must
        # preserve that contract.
        in_dtype = x.dtype

        # fvk bf16_nn reads input bytes as bf16; fp32 silently corrupts.
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K)
        M = flat.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=flat.device)

        gemm.bf16_nn(
            int(flat.data_ptr()), w_t_ptr, int(out.data_ptr()),
            M, N, K, cs(),
        )

        # Bias add — in-place to avoid 493 fresh allocs/step churn.
        # Done in torch to dodge the cuBLASLt BIAS-epilogue NOT_SUPPORTED
        # path on sm_120. Broadcast over the M dimension.
        # G7.12: skip if a downstream kernel will fold bias into the
        # residual / gated_residual op. bias_skip_flag is a 1-element
        # list so callers can flip it post-install.
        if b_pinned is not None and not bias_skip_flag[0]:
            out.add_(b_pinned)

        # Cast back to upstream's expected dtype (fp32 in autocast(fp32)
        # paths; bf16 in the regular hot path).
        if in_dtype != torch.bfloat16:
            out = out.to(in_dtype)

        return out.view(*in_shape[:-1], N)

    return forward


# ──────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────

def install_fvk_linears(model, gemm: fvk.GemmRunner | None = None) -> dict:
    """Walk ``model`` and replace nn.Linear.forward under hot-path
    scopes with fvk.GemmRunner.bf16_nn[_bias].

    Args:
        model: the loaded Motus nn.Module.
        gemm:  shared GemmRunner instance. If None, allocates one
               here and stores it on the model as ``_g3b_gemm`` so
               subsequent calls reuse it.

    Returns:
        Stats dict {'replaced', 'skipped_scope', 'skipped_dtype'}.
    """
    if gemm is None:
        gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()
        model._g3b_gemm = gemm  # pin so it lives as long as the model

    counts = {'replaced': 0, 'skipped_scope': 0, 'skipped_dtype': 0}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Scope filter
        in_hot = any(name.startswith(p) for p in _HOT_PATH_PREFIXES)
        in_skip = any(name.startswith(p) for p in _SKIP_PREFIXES)
        if (not in_hot) or in_skip:
            counts['skipped_scope'] += 1
            if _TRACE:
                logger.info(f"[g3b] SKIP scope: {name}")
            continue

        # Dtype check — weight must be bf16 (frontend already moved to cuda+bf16)
        if module.weight.dtype != torch.bfloat16 or not module.weight.is_cuda:
            logger.warning(
                f"[g3b] SKIP {name}: weight dtype/device "
                f"{module.weight.dtype}/{module.weight.device}")
            counts['skipped_dtype'] += 1
            continue
        if module.bias is not None and (
            module.bias.dtype != torch.bfloat16 or not module.bias.is_cuda
        ):
            logger.warning(
                f"[g3b] SKIP {name}: bias dtype/device "
                f"{module.bias.dtype}/{module.bias.device}")
            counts['skipped_dtype'] += 1
            continue

        bias_skip_flag = [False]
        module.forward = _make_linear_forward(
            module.weight, module.bias, gemm, label=name,
            bias_skip_flag=bias_skip_flag)
        # G7.12: pin the flag + bias so downstream fusion swaps
        # (e.g. wan_action_o bias_skip) can find them.
        module._bf16_bias_skip_flag = bias_skip_flag
        module._bf16_bias = module.bias
        counts['replaced'] += 1
        if _TRACE:
            logger.info(
                f"[g3b] linear {name}: in={module.in_features}, "
                f"out={module.out_features}, bias={module.bias is not None}")

    logger.info(
        f"[g3b] linear swap: replaced={counts['replaced']}, "
        f"skipped_scope={counts['skipped_scope']}, "
        f"skipped_dtype={counts['skipped_dtype']}")
    return counts
