"""Pi0.5 FP8 runtime site definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pi05QuantSiteSpec:
    name: str
    domain: str
    layer: int | None = None
    fused: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    runtime_active: bool = True


def iter_pi05_fp8_sites(
    *,
    vision_layers: int = 27,
    encoder_layers: int = 18,
    decoder_layers: int = 18,
) -> list[Pi05QuantSiteSpec]:
    """Return Pi0.5 FP8 GEMM sites in FlashRT runtime naming.

    ``runtime_active`` reflects the current Pi0.5 RTX pipeline. A few weights
    are pre-quantized for shape symmetry but are not read by inference, e.g.
    the final encoder layer's post-attention and FFN projections.
    """
    specs: list[Pi05QuantSiteSpec] = []

    for i in range(vision_layers):
        base = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}"
        specs.extend([
            Pi05QuantSiteSpec(
                f"vision_attn_qkv_w_{i}", "vision", i, "qkv",
                (
                    f"{base}.self_attn.q_proj",
                    f"{base}.self_attn.k_proj",
                    f"{base}.self_attn.v_proj",
                ),
            ),
            Pi05QuantSiteSpec(
                f"vision_attn_o_w_{i}", "vision", i, None,
                (f"{base}.self_attn.out_proj",),
            ),
            Pi05QuantSiteSpec(
                f"vision_ffn_up_w_{i}", "vision", i, None,
                (f"{base}.mlp.fc1",),
            ),
            Pi05QuantSiteSpec(
                f"vision_ffn_down_w_{i}", "vision", i, None,
                (f"{base}.mlp.fc2",),
            ),
        ])

    specs.append(Pi05QuantSiteSpec(
        "vision_projector_w", "projector", None, None,
        ("paligemma_with_expert.paligemma.model.multi_modal_projector.linear",),
    ))

    for i in range(encoder_layers):
        base = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"
        is_last = i == encoder_layers - 1
        specs.extend([
            Pi05QuantSiteSpec(
                f"encoder_attn_qkv_w_{i}", "encoder", i, "qkv",
                (
                    f"{base}.self_attn.q_proj",
                    f"{base}.self_attn.k_proj",
                    f"{base}.self_attn.v_proj",
                ),
            ),
            Pi05QuantSiteSpec(
                f"encoder_attn_o_w_{i}", "encoder", i, None,
                (f"{base}.self_attn.o_proj",),
                runtime_active=not is_last,
            ),
            Pi05QuantSiteSpec(
                f"encoder_ffn_gate_up_w_{i}", "encoder", i, "gate_up",
                (
                    f"{base}.mlp.gate_proj",
                    f"{base}.mlp.up_proj",
                ),
                runtime_active=not is_last,
            ),
            Pi05QuantSiteSpec(
                f"encoder_ffn_down_w_{i}", "encoder", i, None,
                (f"{base}.mlp.down_proj",),
                runtime_active=not is_last,
            ),
        ])

    for i in range(decoder_layers):
        base = f"paligemma_with_expert.gemma_expert.model.layers.{i}"
        specs.extend([
            Pi05QuantSiteSpec(
                f"decoder_attn_qkv_w_{i}", "decoder", i, "qkv",
                (
                    f"{base}.self_attn.q_proj",
                    f"{base}.self_attn.k_proj",
                    f"{base}.self_attn.v_proj",
                ),
            ),
            Pi05QuantSiteSpec(
                f"decoder_attn_o_w_{i}", "decoder", i, None,
                (f"{base}.self_attn.o_proj",),
            ),
            Pi05QuantSiteSpec(
                f"decoder_ffn_gate_up_w_{i}", "decoder", i, "gate_up",
                (
                    f"{base}.mlp.gate_proj",
                    f"{base}.mlp.up_proj",
                ),
            ),
            Pi05QuantSiteSpec(
                f"decoder_ffn_down_w_{i}", "decoder", i, None,
                (f"{base}.mlp.down_proj",),
            ),
        ])

    return specs


def pi05_fp8_site_names(runtime_active_only: bool = False) -> set[str]:
    return {
        spec.name
        for spec in iter_pi05_fp8_sites()
        if spec.runtime_active or not runtime_active_only
    }
