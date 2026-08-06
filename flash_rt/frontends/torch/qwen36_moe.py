"""Qwen3.6-35B-A3B text inference.

The language backbone is the same ``qwen3_5_moe`` architecture used by
Nex-N2-mini, so this frontend reuses that implementation.  Nothing in it is
specific to one GPU: it is registered for RTX SM120 and for Jetson AGX Thor
(SM110), and each target's build tiers decide which kernels the shared forward
and decode paths resolve to.  (The base class keeps its ``Rtx`` name, which
predates the Thor path; see :mod:`flash_rt.frontends.torch.nexn2_rtx`.)

The official Qwen3.6 checkpoint also contains a vision tower and an MTP draft
head.  The vision tower is validated but not executed here.  The draft head is
loaded only when ``load_mtp=True`` is passed, which is what
:meth:`Qwen36MoeTextFrontend.generate_spec` needs; ``generate`` never uses it.

See ``docs/qwen36_moe_usage.md`` for the per-architecture build commands.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from typing import Any

from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx


_EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(40)
)

_EXPECTED_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attention_bias": False,
    "attn_output_gate": True,
    "hidden_act": "silu",
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "full_attention_interval": 4,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
}

_EXPECTED_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
}


def _expected_mlp_shapes(prefix: str) -> dict[str, tuple[int, ...]]:
    return {
        prefix + "mlp.gate.weight": (256, 2048),
        prefix + "mlp.experts.gate_up_proj": (256, 1024, 2048),
        prefix + "mlp.experts.down_proj": (256, 2048, 512),
        prefix + "mlp.shared_expert.gate_proj.weight": (512, 2048),
        prefix + "mlp.shared_expert.up_proj.weight": (512, 2048),
        prefix + "mlp.shared_expert.down_proj.weight": (2048, 512),
        prefix + "mlp.shared_expert_gate.weight": (1, 2048),
    }


def _expected_attention_shapes(
        prefix: str, layer_type: str) -> dict[str, tuple[int, ...]]:
    if layer_type == "full_attention":
        return {
            prefix + "self_attn.q_proj.weight": (8192, 2048),
            prefix + "self_attn.k_proj.weight": (512, 2048),
            prefix + "self_attn.v_proj.weight": (512, 2048),
            prefix + "self_attn.o_proj.weight": (2048, 4096),
            prefix + "self_attn.q_norm.weight": (256,),
            prefix + "self_attn.k_norm.weight": (256,),
        }
    return {
        prefix + "linear_attn.in_proj_qkv.weight": (8192, 2048),
        prefix + "linear_attn.in_proj_z.weight": (4096, 2048),
        prefix + "linear_attn.in_proj_a.weight": (32, 2048),
        prefix + "linear_attn.in_proj_b.weight": (32, 2048),
        prefix + "linear_attn.conv1d.weight": (8192, 1, 4),
        prefix + "linear_attn.A_log": (32,),
        prefix + "linear_attn.dt_bias": (32,),
        prefix + "linear_attn.norm.weight": (128,),
        prefix + "linear_attn.out_proj.weight": (2048, 4096),
    }


def _expected_text_shapes(
        layer_types: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    shapes = {
        "lm_head.weight": (248320, 2048),
        "model.language_model.embed_tokens.weight": (248320, 2048),
        "model.language_model.norm.weight": (2048,),
    }
    for i, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{i}."
        shapes[prefix + "input_layernorm.weight"] = (2048,)
        shapes[prefix + "post_attention_layernorm.weight"] = (2048,)
        shapes.update(_expected_mlp_shapes(prefix))
        shapes.update(_expected_attention_shapes(prefix, layer_type))
    return shapes


def _expected_mtp_shapes() -> dict[str, tuple[int, ...]]:
    prefix = "mtp.layers.0."
    shapes = {
        "mtp.fc.weight": (2048, 4096),
        "mtp.norm.weight": (2048,),
        "mtp.pre_fc_norm_embedding.weight": (2048,),
        "mtp.pre_fc_norm_hidden.weight": (2048,),
        prefix + "input_layernorm.weight": (2048,),
        prefix + "post_attention_layernorm.weight": (2048,),
    }
    shapes.update(_expected_attention_shapes(prefix, "full_attention"))
    shapes.update(_expected_mlp_shapes(prefix))
    return shapes


_MTP_KEYS = set(_expected_mtp_shapes())


def _required_text_keys(layer_types: tuple[str, ...]) -> set[str]:
    return set(_expected_text_shapes(layer_types))


def _read_tensor_shapes(
        checkpoint_path: str,
        weight_map: dict[str, str],
        tensor_names: set[str],
) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    shapes = {}
    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(
                safe_open(
                    os.path.join(checkpoint_path, shard),
                    framework="pt",
                    device="cpu",
                )
            )
            for shard in set(weight_map[name] for name in tensor_names)
        }
        for name in tensor_names:
            shapes[name] = tuple(
                readers[weight_map[name]].get_slice(name).get_shape())
    return shapes


def validate_qwen36_moe_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Validate the official BF16 checkpoint before allocating GPU weights."""
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    config_path = os.path.join(checkpoint_path, "config.json")
    index_path = os.path.join(
        checkpoint_path, "model.safetensors.index.json")

    for path in (config_path, index_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Qwen3.6-35B-A3B checkpoint is missing {path!r}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError(
            "Qwen3.6-35B-A3B text frontend requires "
            "model_type='qwen3_5_moe'; "
            f"got {config.get('model_type')!r} in {config_path}")

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"missing text_config object in {config_path}")

    mismatches = []
    for name, expected in _EXPECTED_TEXT_CONFIG.items():
        actual = text_config.get(name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    rope_parameters = text_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        mismatches.append(
            "rope_parameters is missing or is not an object")
    else:
        for name, expected in _EXPECTED_ROPE_PARAMETERS.items():
            actual = rope_parameters.get(name)
            if actual != expected:
                mismatches.append(
                    f"rope_parameters.{name}={actual!r} "
                    f"(expected {expected!r})")
    layer_types = tuple(text_config.get("layer_types") or ())
    if layer_types != _EXPECTED_LAYER_TYPES:
        mismatches.append(
            "layer_types does not match the 30-linear/10-full attention "
            "qwen3_5_moe schedule")
    if mismatches:
        raise ValueError(
            "checkpoint is not compatible with the Qwen3.6-35B-A3B "
            "SM120 text pipeline: " + "; ".join(mismatches))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map object in {index_path}")

    required_text = _required_text_keys(layer_types)
    missing_text = sorted(required_text.difference(weight_map))
    if missing_text:
        preview = ", ".join(missing_text[:8])
        if len(missing_text) > 8:
            preview += f", ... ({len(missing_text)} missing)"
        raise ValueError(
            "Qwen3.6-35B-A3B checkpoint is missing text backbone tensors: "
            + preview)

    missing_mtp = sorted(_MTP_KEYS.difference(weight_map))
    if missing_mtp:
        preview = ", ".join(missing_mtp[:8])
        if len(missing_mtp) > 8:
            preview += f", ... ({len(missing_mtp)} missing)"
        raise ValueError(
            "Qwen3.6-35B-A3B checkpoint is missing its MTP tensor group: "
            + preview)

    shards = sorted(set(weight_map.values()))
    bad_shards = []
    for shard in shards:
        path = os.path.join(checkpoint_path, shard)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            bad_shards.append(shard)
    if bad_shards:
        preview = ", ".join(bad_shards[:8])
        if len(bad_shards) > 8:
            preview += f", ... ({len(bad_shards)} missing or empty)"
        raise FileNotFoundError(
            "Qwen3.6-35B-A3B checkpoint has missing or empty shards: "
            + preview)

    expected_shapes = _expected_text_shapes(layer_types)
    expected_shapes.update(_expected_mtp_shapes())
    actual_shapes = _read_tensor_shapes(
        checkpoint_path, weight_map, set(expected_shapes))
    shape_mismatches = [
        f"{name}={actual_shapes[name]!r} (expected {expected!r})"
        for name, expected in sorted(expected_shapes.items())
        if actual_shapes[name] != expected
    ]
    if shape_mismatches:
        preview = "; ".join(shape_mismatches[:8])
        if len(shape_mismatches) > 8:
            preview += f"; ... ({len(shape_mismatches)} mismatched)"
        raise ValueError(
            "Qwen3.6-35B-A3B checkpoint tensor shape mismatches: "
            + preview)

    return {
        "checkpoint_path": checkpoint_path,
        "text_tensor_count": len(required_text),
        "mtp_tensor_count": len(_MTP_KEYS),
        "vision_tensor_count": sum(
            ".visual." in name for name in weight_map),
        "tensor_count": len(weight_map),
        "shard_count": len(shards),
    }


class Qwen36MoeTextFrontend(Nexn2TorchFrontendRtx):
    """Qwen3.6-35B-A3B text-only frontend (RTX SM120 and Jetson AGX Thor)."""

    _MODEL_LABEL = "Qwen3.6-35B-A3B text"
    _USAGE_DOC = "docs/qwen36_moe_usage.md"

    # The block-scaled 4-bit MMA tier is a build tier, and the prefill now picks
    # its MoE tile from what the module actually has: without the tier it uses
    # the weight-only grouped GEMV, which is slower on a long prompt and
    # otherwise identical. So the tier is a performance requirement, not a
    # correctness one, and demanding it here would refuse a build -- a Jetson
    # one, whose toolchain has no block-scaled mma -- that runs this correctly.
    _REQUIRED_KERNELS = tuple(
        name for name in Nexn2TorchFrontendRtx._REQUIRED_KERNELS
        if not name.startswith(('moe_blocktile_mma', 'moe_m16_mma'))
    ) + ('moe_grouped_w4a16_sm120_bf16',)

    # Same for FA2: prefill and decode both probe the vendored kernel and fall
    # back to a reference, so a target that builds FA4 instead still runs.
    _REQUIRE_FA2 = False

    def __init__(self, checkpoint_path: str, *,
                 device: str = "cuda:0",
                 max_seq: int = 2048,
                 quant: str = "nvfp4",
                 kernelized: bool = True,
                 quant_scope: str = "experts",
                 load_mtp: bool = False,
                 spec_graph_cache_max: int | None = None) -> None:
        """Construct the frontend.

        Args beyond the shared ones (see
        :class:`~flash_rt.frontends.torch.nexn2_rtx.Nexn2TorchFrontendRtx`):

          load_mtp: read the checkpoint's MTP draft head, which
            :meth:`generate_spec` needs. Off by default: it is a full extra
            transformer layer plus its KV, and ``generate`` never reads it.
          spec_graph_cache_max: how many captured speculative windows to keep.
            Each owns a CUDA graph memory pool, and a speculative window is
            larger than a decode step, so this is bounded separately from the
            decode graph cache and much lower. ``None`` takes the runtime
            default, which is sized for the smallest board this path runs on.
        """
        if quant != "nvfp4":
            raise NotImplementedError(
                f"quant={quant!r} is not implemented; only 'nvfp4' is "
                "supported")
        if not kernelized:
            raise NotImplementedError(
                "Qwen3.6-35B-A3B text only supports kernelized=True with "
                "runtime NVFP4 conversion")
        if spec_graph_cache_max is not None:
            spec_graph_cache_max = int(spec_graph_cache_max)
            if spec_graph_cache_max < 1:
                raise ValueError(
                    "spec_graph_cache_max must be at least 1 (got "
                    f"{spec_graph_cache_max}); a speculative step captures a "
                    "graph per position, so a cache that holds none would "
                    "recapture every step")
        # Read by the loader, before any weight is touched, through the base
        # class -- hence set before super().__init__.
        self._load_mtp = bool(load_mtp)
        self._spec_graph_cache_max = spec_graph_cache_max
        contract = validate_qwen36_moe_checkpoint(checkpoint_path)
        super().__init__(
            checkpoint_path,
            device=device,
            max_seq=max_seq,
            quant=quant,
            kernelized=kernelized,
            quant_scope=quant_scope,
        )
        self._checkpoint_contract = contract

    @property
    def load_mtp(self) -> bool:
        """Whether the MTP draft head was loaded (see ``generate_spec``)."""
        return self._load_mtp

    def _decode_state_or_new(self):
        from flash_rt.frontends.torch._nexn2_rtx_decode import Nexn2DecodeState

        if self._decode_state is None:
            self._decode_state = Nexn2DecodeState(
                self._weights, self._user_max_seq, self.device,
                spec_graph_cache_max=self._spec_graph_cache_max)
        return self._decode_state

    def generate_spec(self, max_new_tokens: int, *, k: int = 2):
        """Greedy decode through draft-and-verify with the MTP head.

        Emits exactly what ``generate`` emits: a draft is kept only where the
        model's own argmax agrees with it, so this is a speed change and
        nothing else. Requires ``load_mtp=True`` at construction.
        """
        if self._prompt_ids is None:
            raise ValueError("call set_prompt(...) before generate_spec()")
        if not self._load_mtp:
            raise RuntimeError(
                "speculative decoding needs the MTP draft head, which this "
                "frontend did not load. Construct it with load_mtp=True.")
        if int(k) < 1:
            raise ValueError(
                f"k must be at least 1 (got {k}); k is how many tokens the "
                "draft head proposes per window")

        from flash_rt.frontends.torch._nexn2_rtx_decode import (
            generate_greedy_spec,
        )

        return generate_greedy_spec(
            self._decode_state_or_new(), self._prompt_ids, max_new_tokens,
            int(k), self._fvk, self.device)

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        """Generate tokens with the shared qwen3_5_moe CUDA Graph path."""
        if self._prompt_ids is None:
            raise ValueError("call set_prompt(...) before generate()")
        if do_sample:
            raise NotImplementedError(
                "the Qwen3.6-35B-A3B kernelized path supports greedy "
                "decoding only")

        from flash_rt.frontends.torch._nexn2_rtx_decode import (
            generate_greedy_graph,
        )

        return generate_greedy_graph(
            self._decode_state_or_new(),
            self._prompt_ids,
            max_new_tokens,
            self._fvk,
            self.device,
        )


# The name this frontend shipped under while it was registered for RTX only.
# Kept so an existing import keeps working; new code should use the class
# above, which is what both architectures resolve to.
Qwen36MoeTextFrontendRtx = Qwen36MoeTextFrontend
