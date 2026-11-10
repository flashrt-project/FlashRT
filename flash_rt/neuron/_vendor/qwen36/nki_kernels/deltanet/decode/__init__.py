# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-generation (TKG) DeltaNet kernels.

fused_layer  — assembled DeltaNet attention megakernel (current decode path).
recurrent    — per-token recurrent kernel (older, flag-gated fallback).
"""
