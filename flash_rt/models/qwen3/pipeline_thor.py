"""FlashRT -- Jetson AGX Thor (SM110) target for the dense Qwen3 BF16 prefill.

The layer sequence is shared with RTX (``models/qwen3/prefill_bf16.py``);
Thor binds the vendored FlashAttention-4 forward
(``flash_rt.hardware.thor.fa4_backend``), the attention path FlashRT's other
Thor models use.
"""
from __future__ import annotations

from flash_rt.models.qwen3._bf16_attention import Fa4CausalAttention
from flash_rt.models.qwen3.prefill_bf16 import Qwen3Bf16Target

QWEN3_4B_THOR = Qwen3Bf16Target(
    name="thor",
    compute_capability=(11, 0),
    attention="fa4",
    pdl=True,
    make_attention=Fa4CausalAttention,
    build_hint=(
        "Build with: cmake -B build -S . -DGPU_ARCH=110 && "
        "cmake --build build --target flash_rt_kernels; FA4 needs the "
        "thor-fa4 extra (nvidia-cutlass-dsl, quack-kernels)"),
)
