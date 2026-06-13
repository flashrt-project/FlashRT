# Copyright 2025 XunhaoLai. All rights reserved.
#
# VENDOR NOTE: rewritten to drop the `einops` dependency (the original used
# einops.einsum / einops.rearrange). Logic is equivalent to the SGLang upstream
# `naive/flash_with_topk_idx.py`; only the einsum/rearrange calls were replaced
# with native torch.einsum / torch.reshape so the standalone vendor needs only
# torch (+ triton for the kernels). Original einops expressions are kept in
# comments next to each replacement for auditability.

from typing import Optional

import torch


def naive_flash_decode_with_topk_idx(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    sink: Optional[torch.Tensor],  # [num_heads, head_dim]
    kv_cache: torch.Tensor,  # [max_slots, 2, max_len, num_heads, head_dim]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk: int,
    sm_scale: Optional[float] = None,
    init_blocks: int = 0,
    local_blocks: int = 0,
):
    """Joint reference: lightning-indexer block top-k + sparse attention.

    This is the *contiguous* (non-paged) reference. `kv_cache` here is the dense
    5-D buffer [max_slots, 2(k/v), max_len, num_kv_heads, head_dim], indexed by
    `slot_ids`. It both (a) selects the top-k key blocks via per-block max score
    (with optional forced init/local blocks) and (b) computes the resulting
    sparse attention, returning (o, topk_idx).
    """
    assert (
        kv_cache.shape[2] % block_size == 0
    ), "max cache len must be divisible by block size"
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    original_dtype = q.dtype
    batch_size = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = kv_cache.shape[3]
    gqa_group_size = num_q_heads // num_kv_heads
    head_dim = q.shape[-1]
    # rearrange q: "b (h g) d -> b h g d"
    q = q.float().reshape(batch_size, num_kv_heads, gqa_group_size, head_dim)
    kv_cache_float = kv_cache.float()
    k_sel = kv_cache_float[slot_ids, 0, ...]  # [b, n, h, d]
    v_sel = kv_cache_float[slot_ids, 1, ...]  # [b, n, h, d]
    # einsum(q, k, "b h g d, b n h d -> b h g n")
    qk = torch.einsum("bhgd,bnhd->bhgn", q, k_sel) * sm_scale
    mask = torch.arange(kv_cache.shape[2], device=q.device) < seq_lens[:, None]
    qk = qk.masked_fill(~mask[:, None, None, :], float("-inf"))
    # get score
    score = qk.clone().to(torch.float32)
    # rearrange "b h g (n s) -> b (h g) n s", s=block_size
    n_blocks = score.shape[-1] // block_size
    score = score.reshape(
        batch_size, num_kv_heads, gqa_group_size, n_blocks, block_size
    ).reshape(batch_size, num_q_heads, n_blocks, block_size)
    score = score.max(dim=-1).values  # [batch_size, num_q_heads, num_blocks]
    # post-process score for init_blocks and local_blocks
    INIT_SCORE = 1e30
    LOCAL_SCORE = 1e29
    if init_blocks > 0:
        score[:, :, :init_blocks] = INIT_SCORE
    if local_blocks > 0:
        num_blocks_per_batch = (seq_lens + block_size - 1) // block_size
        for b in range(batch_size):
            num_blks = num_blocks_per_batch[b].item()
            local_start = max(0, num_blks - local_blocks)
            score[b, :, local_start:num_blks] = LOCAL_SCORE
    # compute topk indices per (batch, head)
    # score shape: [batch_size, num_q_heads, num_blocks]
    topk_idx = torch.full(
        (num_q_heads, batch_size, topk),
        fill_value=-1,
        device=score.device,
        dtype=torch.int32,
    )
    num_blocks_per_batch = (seq_lens + block_size - 1) // block_size
    for b in range(batch_size):
        num_blks = num_blocks_per_batch[b].item()
        actual_topk = min(topk, num_blks)
        for h in range(num_q_heads):
            # get topk indices for this (batch, head)
            _, indices = torch.topk(score[b, h, :num_blks], k=actual_topk, dim=-1)
            topk_idx[h, b, :actual_topk] = indices.to(torch.int32)
    # compute attention output with sink
    if sink is not None:
        # sink: [num_q_heads, head_dim] -> [h, g, d]
        sink_reshaped = sink.float().reshape(num_kv_heads, gqa_group_size, head_dim)
        # einsum(q, sink, "b h g d, h g d -> b h g")
        qsink = torch.einsum("bhgd,hgd->bhg", q, sink_reshaped) * sm_scale
        qk_with_sink = torch.cat([qsink[..., None], qk], dim=-1)  # [b, h, g, n+1]
        attn = qk_with_sink.softmax(dim=-1, dtype=torch.float32)
        attn = attn[..., 1:]  # remove sink score
    else:
        attn = qk.softmax(dim=-1, dtype=torch.float32)
    # einsum(attn, v, "b h g n, b n h d -> b h g d")
    o = torch.einsum("bhgn,bnhd->bhgd", attn, v_sel)
    # rearrange "b h g d -> b (h g) d"
    o = o.reshape(batch_size, num_q_heads, head_dim)
    return o.to(original_dtype), topk_idx
