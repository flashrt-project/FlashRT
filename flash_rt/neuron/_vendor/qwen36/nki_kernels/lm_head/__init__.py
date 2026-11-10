# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LM head kernels — final RMSNorm + vocab-parallel projection + per-rank greedy argmax.

A sibling of deltanet/, gqa/ and moe/ rather than a member of one: the lm_head is shared verbatim by
the target model's verify pass and the MTP draft head, and belongs to neither attention family.
"""

from .components.lm_head import lm_head_compose, lm_head_fwd

__all__ = ["lm_head_compose", "lm_head_fwd"]
