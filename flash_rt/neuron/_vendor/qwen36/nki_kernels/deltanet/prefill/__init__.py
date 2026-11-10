# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-encoding (CTE) DeltaNet kernels.

chunked_step   — per-chunk-step kernel; numerically stable, the default path.
chunked_fused  — single-kernel fused forward; faster but overflows fp32 for this
                 checkpoint's gating magnitude, so it is gated off (see README).
"""
