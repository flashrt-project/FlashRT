"""FlashRT -- GR00T N1.5 on RTX SM89 (Ada, RTX 4090).

The stage sequence is ``models/groot_n15/forward.py``; this target fixes the
device generation, binds the attention sites to FA2 and the projections to
the precision tier's lowering. ``bf16`` runs the reference's algorithms.
``fp8`` packs the vision tower, the VL self-attention, the DiT's
cross-attention K/V and the Qwen3 attention projections to E4M3: on the
fixture observations each of these moves the actions far less than the
Qwen3 MLP projections, whose post-attention inputs carry per-channel
outliers a per-token scale does not absorb, so those stay BF16 -- as do the
patch embedding (588-wide rows) and the 49-row action head, where quantizing
the activation costs more than the narrower weights save. SM89 has no
programmatic dependent launch, so nothing here uses it.
"""
from __future__ import annotations

from flash_rt.models.groot_n15._attention import make_fa2_site
from flash_rt.models.groot_n15._linear import linear_symbols, make_linear
from flash_rt.models.groot_n15.forward import GrootN15Target

GROOT_N15_RTX_SM89 = GrootN15Target(
    name="rtx_sm89",
    compute_capability=(8, 9),
    attention="fa2",
    precisions=("bf16", "fp8"),
    pdl=False,
    make_attention=make_fa2_site,
    make_linear=make_linear,
    linear_symbols=linear_symbols,
    fp8_projections=frozenset({"vision", "vl", "cross_kv", "llm_attention"}),
    build_hint=(
        "Build with: cmake -B build -S . -DGPU_ARCH=89 "
        "-DFLASHRT_ENABLE_GROOT_N15=ON && cmake --build build --target "
        "flash_rt_kernels flash_rt_fa2 flash_rt_groot_n15"),
)
