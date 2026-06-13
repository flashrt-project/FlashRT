# SPDX-License-Identifier: Apache-2.0
"""Standalone MiniMax-M3 block-sparse attention Triton kernels (vendored).

Provenance: SGLang PR #27944 (JustinTong0323/sglang @ minimax-m3-upstream,
commit cbe1ffd...), files under
python/sglang/srt/layers/attention/minimax_sparse_ops/. Author XunhaoLai
(NSA-triton). Framework coupling stripped via `._compat`; see VENDOR_NOTES.md.

Public API
----------
Lightning indexer (q.k blockmax score -> top-k block select), paged KV:
    flash_prefill_with_topk_index   (prefill / chunked-prefill)
    flash_decode_with_topk_idx      (single-token decode, split-K)

Block-sparse GQA attention (consumes block_indices from the indexer):
    flash_prefill_with_gqa_share_sparse   (prefill)
    flash_decode_with_gqa_share_sparse    (decode, split-K over top-k blocks)

PyTorch naive references (for correctness checks):
    naive_flash_decode_with_gqa_share_sparse   (attention given topk_idx)
    naive_flash_decode_with_topk_idx           (joint indexer + attention)

All Triton entrypoints expect a paged KV cache:
    k_cache / v_cache : [max_slots, num_kv_heads, head_dim]
    req_to_token      : [max_reqs, max_kv_len]  (logical pos -> physical slot)
    slot_ids          : [batch]                 (per-request row in req_to_token)
    topk_idx          : [num_kv_heads, n, topk] int32, valid ids left-packed,
                        -1 right-padding (the M3 indexer contract).
See VENDOR_NOTES.md for full shape/dtype/layout contracts.
"""

from .common.utils import get_cu_seqblocks, robust_allocator
from .decode.flash_with_topk_idx import flash_decode_with_topk_idx
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse
from .naive.flash_with_topk_idx import naive_flash_decode_with_topk_idx
from .naive.topk_sparse import naive_flash_decode_with_gqa_share_sparse
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse

__all__ = [
    "flash_prefill_with_topk_index",
    "flash_decode_with_topk_idx",
    "flash_prefill_with_gqa_share_sparse",
    "flash_decode_with_gqa_share_sparse",
    "naive_flash_decode_with_gqa_share_sparse",
    "naive_flash_decode_with_topk_idx",
    "get_cu_seqblocks",
    "robust_allocator",
]
