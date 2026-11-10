# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MTP draft front end — the embed/norm/eh_proj stage that seeds the draft residual.

Sits between embed/ and the draft decoder layer: it consumes the same token ids embed/ does, adds
the trunk hidden, and emits the tp2013 residual tile the layer composables expect.
"""

from .components.eh_proj import eh_proj_compose, eh_proj_fwd, rms_norm_natural

__all__ = [
    "eh_proj_compose",
    "eh_proj_fwd",
    "rms_norm_natural",
]
