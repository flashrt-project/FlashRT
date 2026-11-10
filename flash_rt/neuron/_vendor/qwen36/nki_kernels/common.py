# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Glue shared by the composable stage kernels across deltanet/, gqa/ and moe/.

Composables that only differ in which weight they consume (e.g. the pre-attention and
post-attention RMSNorms) share one implementation here and keep a role-named wrapper in
their own package, where the layer-specific contract is documented.
"""

import nki.language as nl

from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
from nkilib.core.utils.allocator import create_auto_alloc_manager

H0 = 128  # partition dim (nl.tile_size.pmax)


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def rmsnorm_to_sbuf(
    hidden,
    gamma,
    eps=1e-6,
    hidden_actual=None,
    normed_sb=None,
    single_core_forced=False,
    name_prefix="",
    sbm=None,
):
    """RMSNorm a raw hidden tile into an SBUF-resident [H0, T, H1] tile (zero HBM round-trip).

    100% of the math is ``nkilib.core.subkernels.rmsnorm_tkg`` (square -> reduce over H1 ->
    matmul reduce over H0 -> rsqrt -> gamma, fp32 internal, bf16/fp32 IO); this is caller glue.

    Args:
        hidden:        raw pre-norm hidden, either [B, T, H] HBM or [H0=128, T, H1] SBUF (the
                       megakernel residual); left untouched. rmsnorm_tkg consumes either directly.
        gamma:         [1, H] HBM norm weight in STANDARD form (no +1 applied here).
        eps:           RMSNorm epsilon (config.rms_norm_eps).
        hidden_actual: actual H used for the mean (defaults to H; set when input is padded).
        normed_sb:     optional [H0, T, H1] SBUF output; allocated if None.
        single_core_forced: True forces num_H_shards=1 (tp102) on every core; False keys
                       num_H_shards to the LNC count -- the tp2013 megakernel residual layout.
        name_prefix:   SBUF allocation-name prefix (an instantiating caller must make this unique).
        sbm:           optional BufferManager. Default None allocates a fresh auto-alloc manager (the
                       behaviour every existing caller relies on). A caller running its own MANUAL
                       manager must pass it, so rmsnorm_tkg's tiles are placed inside the region that
                       manager owns instead of being placed independently by the compiler.

    Returns:
        normed_sb: [H0=128, T, H1=H//128] SBUF (same dtype as hidden).
    """
    if hidden.buffer == nl.sbuf:
        h0_in, T, h1 = hidden.shape  # [H0, T, H1] megakernel residual
        kernel_assert(h0_in == H0, "SBUF hidden partition dim must be 128")
        hdim = h0_in * h1
    else:
        B, S, hdim = hidden.shape
        T = B * S
        kernel_assert(hdim % H0 == 0, "hidden H must be divisible by 128")
        h1 = hdim // H0
    kernel_assert(tuple(gamma.shape) == (1, hdim), "gamma must be [1, H]")

    if hidden_actual is None:
        hidden_actual = hdim
    if normed_sb is None:
        normed_sb = nl.ndarray((H0, T, h1), dtype=hidden.dtype, buffer=nl.sbuf)

    if sbm is None:
        sbm = create_auto_alloc_manager()
    sbm.set_name_prefix(name_prefix)
    rmsnorm_tkg(
        input=hidden,
        gamma=gamma,
        output=normed_sb,
        eps=eps,
        hidden_actual=hidden_actual,
        single_core_forced=single_core_forced,
        sbm=sbm,
    )
    return normed_sb
