# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA head_dim=256 TKG stage kernels — the megakernel building blocks.

qkv_proj · qk_norm · rope · attention · out_proj
"""
