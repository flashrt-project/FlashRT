"""FlashRT — Motus weight spec (RTX, torch) — G1 scaffold.

Source checkpoint: ``motus-robotics/Motus`` (Stage 2 pretrained), saved as
DeepSpeed ZeRO-1 layout (``mp_rank_00_model_states.pt``). 5417 named
parameters under top-level key ``module``.

Top-level groups (from ``baseline_artifacts/`` + Motus source):
    wan_model.*           Wan2.2-TI2V-5B DiT (30 layers, hidden 3072)
    vlm.*                 Qwen3-VL-2B (frozen, BF16)
    action_expert.*       30-layer Action Expert (hidden 1024)
    und_expert.*          30-layer Understanding Expert (hidden 512)
    action_module.*       Mirror of action_expert with module-level wrapper
    und_module.*          Mirror of und_expert
    video_module.*        Wraps wan_model views

Motus also has time / patch / text embeddings inside wan_model.

────────────────────────────────────────────────────────────────────
G1 status
────────────────────────────────────────────────────────────────────
Returns an EMPTY ``ModelWeightSpec`` so that frontend ``__init__`` can
import this module without erroring. The full declarative spec
(transforms, FP8 quant decisions, FlatCat sinks) lands in G2 alongside
the eager BF16 forward.

Reading order for G2:
    1. Dump ckpt key list (we already did in baseline run; re-run a
       small probe script for fresh listing).
    2. Group keys by prefix → one LayerBlock per prefix.
    3. Map each Item to a CudaBuffer sink. Start with no Quant() — pure
       BF16 round-trip — then in G4 add Quant() for the GEMM weights.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import ModelWeightSpec


def build_spec() -> ModelWeightSpec:
    """G1: return empty spec. G2 fills in 30+ LayerBlocks for WAN, action,
    und experts plus singletons for time/patch/text embeddings.
    """
    return ModelWeightSpec(
        framework="torch",
        blocks=[],
        singletons=[],
    )
