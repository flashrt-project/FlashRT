# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding kernels — the megakernel prologue, the mirror of lm_head/.

A sibling of deltanet/, gqa/ and moe/ rather than a member of one: the embedding is shared verbatim
by the target model's verify pass and the MTP draft head, and belongs to neither attention family.
"""

from .components.embed import (
    all_gather_embed_h,
    embed_compose,
    embed_fwd,
    gather_embed_rows,
    load_token_ids_to_sbuf,
    natural_to_tp2013,
    tp2013_to_natural,
)

__all__ = [
    "all_gather_embed_h",
    "embed_compose",
    "embed_fwd",
    "gather_embed_rows",
    "load_token_ids_to_sbuf",
    "natural_to_tp2013",
    "tp2013_to_natural",
]
