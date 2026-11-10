# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA token-generation attention for head_dim=256, over nkilib's tuned attention_tkg.

A thin shim over the vendored attention_tkg kernel. The QK^T / online-softmax / P.V compute path is
byte-for-byte AWS code; only the head_dim-on-partition layout sites are tiled into
D_TILES = ceil(d_head/128) partition tiles, so d_head=256 fits the PE array. The banner in
vendored/attention_tkg.py marks the exact patch sites.

Input contract (single TP shard; GQA: q_head query heads share 1 kv head):
  * Q arrives PRE-SCALED by 1/sqrt(d_head) and already RoPE'd / qk-normed. fuse_rope is False here,
    so the caller owns scaling, RoPE and norm. K is RoPE'd / normed but NOT scaled.
  * The caller supplies the full attention mask (1=keep, 0=mask); use_pos_id is False, so the
    kernel generates no causality itself.
  * curr_sprior (== full KV length L) must be a multiple of 128, and of 256 when s_prior is sharded
    across 2 cores. The active tokens occupy the LAST s_active slots of the L-length KV.

Precision: bf16 IO, fp32 matmul/softmax accumulate.

Tensor layouts and the head_dim tiling: specs/gqa_tkg.md.
"""

import nki.language as nl

from ..vendored.attention_tkg import _D_HEAD_TILE, _d_tiles, attention_tkg
from ..vendored.attention_tkg_utils import AttnTKGConfig

try:
    from nkilib.core.utils.allocator import SbufManager
except ImportError:  # pragma: no cover - nkilib is expected to be installed
    SbufManager = None

# Auto-alloc budget. With use_auto_alloc the compiler places SBUF; this bound is
# only the manager's bookkeeping ceiling, so a generous value is safe.
SBM_BUDGET_BYTES = 24 * 1024 * 1024


def kernel_assert(condition, msg):
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {msg}"
    )


def build_attention_tkg_config(
    bs,
    q_head,
    s_active,
    curr_sprior,
    head_dim,
    full_sprior=None,
    v_in_sb=False,
):
    """Build the AttnTKGConfig for the head_dim=256 GQA decode path.

    Pins our flags: flat KV, no fp8, no FA s_prior tiling, no fused RoPE, no in-kernel mask gen,
    q/k pre-loaded in SBUF and output kept there. v_in_sb also takes the active V from SBUF.
    LNC2 sharding is decided inside the kernel from the SPMD grid size.
    """
    full_sprior = curr_sprior if full_sprior is None else full_sprior
    return AttnTKGConfig(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=curr_sprior,
        full_sprior=full_sprior,
        d_head=head_dim,
        block_len=0,
        tp_k_prior=False,
        strided_mm1=False,
        use_pos_id=False,
        fuse_rope=False,
        use_gpsimd_sb2sb=True,
        qk_in_sb=True,
        k_out_in_sb=False,
        out_in_sb=True,
        v_in_sb=v_in_sb,
        enable_fa_s_prior_tiling=False,
    )


def gqa_attention_d256(
    q_sb,
    k_active_sb,
    k_prior,
    v_prior,
    v_active,
    mask,
    out_sb,
    bs,
    q_head,
    s_active,
    curr_sprior,
    head_dim,
    full_sprior=None,
    sbm=None,
    v_in_sb=False,
    name_prefix="",
):
    """Head_dim=256 GQA decode attention via the vendored attention_tkg.

    SBUF-in (q_sb, k_active_sb), SBUF-out; the KV cache is streamed from HBM by the vendored
    kernel's tuned DMA path. Writes out_sb in place and returns it.

    Args:
        q_sb, k_active_sb: head_dim-tiled SBUF query / active key.
        k_prior, v_prior:  HBM KV cache tensors.
        v_active:          HBM [B, 1, s_active, d_head], or SBUF [s_active, d_head] when v_in_sb.
        mask:              HBM uint8 attention mask, 1=keep.
        out_sb:            SBUF output, written in place.
        full_sprior:       KV buffer capacity; defaults to curr_sprior.
        sbm:               optional SbufManager; a megakernel may pass its own.
        v_in_sb:           take the active V from SBUF instead of HBM.
    """
    d_tiles = _d_tiles(head_dim)
    kernel_assert(head_dim % _D_HEAD_TILE == 0, "head_dim must be a multiple of 128")
    kernel_assert(
        q_sb.shape[0] == _D_HEAD_TILE and q_sb.shape[1] == d_tiles,
        f"q_sb must be [128, {d_tiles}, B*H*s_active], got {q_sb.shape}",
    )
    kernel_assert(
        out_sb.shape[0] == _D_HEAD_TILE and out_sb.shape[1] == d_tiles,
        f"out_sb must be [128, {d_tiles}, B*H*s_active], got {out_sb.shape}",
    )
    kernel_assert(
        curr_sprior % _D_HEAD_TILE == 0, "curr_sprior (L) must be a multiple of 128"
    )

    cfg = build_attention_tkg_config(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=curr_sprior,
        head_dim=head_dim,
        full_sprior=full_sprior,
        v_in_sb=v_in_sb,
    )

    own_sbm = sbm is None
    if own_sbm:
        kernel_assert(SbufManager is not None, "nkilib SbufManager unavailable")
        sbm = SbufManager(0, SBM_BUDGET_BYTES, use_auto_alloc=True)
        sbm.set_name_prefix(name_prefix)

    # attention_tkg's first stack allocations (one_vec, position IDs) run before
    # its own per-tile open_scope, so a scope must be open when we own the sbm.
    if own_sbm:
        sbm.open_scope()
    attention_tkg(
        q=q_sb,
        k_active=k_active_sb,
        v_active=v_active,
        k_prior=k_prior,
        v_prior=v_prior,
        mask=mask,
        out=out_sb,
        cfg=cfg,
        sbm=sbm,
    )
    if own_sbm:
        sbm.close_scope()
    return out_sb
