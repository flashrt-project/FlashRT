"""FlashRT — LingBot-VLA torch frontend for Thor (SM110) — scaffold.

Class: ``LingbotTorchFrontendThor``. Intended owner of:
    * the loaded LingBot-VLA model (~4B params, ~16.7 GB BF16)
    * the AttentionBackend (declared here; not yet wired into kernels)
    * the LingBotPipelineThor (orchestrates ViT → VLM prefix → 50 × Expert
      denoise steps → action_out_proj)
    * lifecycle: __init__ → set_prompt(prompt, ...) → infer(obs) [...]

Hardware target: Jetson AGX Thor SM110. A Thor-enabled PyTorch build is
required — standard cu128 wheels do NOT include sm_110 kernels. Validated in
the Thor development container.

Reference baseline (upstream LingBot, PyTorch BF16 eager): P50 ~2480 ms for
50 Euler steps (prefix encode + 50 denoise steps); cos_determinism = 1.0.

────────────────────────────────────────────────────────────────────
Status
────────────────────────────────────────────────────────────────────
**Scaffold only — not used by the current runner.** ``__init__`` validates
the checkpoint path layout and builds an empty weight spec; ``set_prompt`` /
``infer`` / ``predict`` raise ``NotImplementedError``. LingBot is not yet
registered in ``load_model`` / ``_PIPELINE_MAP``. The working inference path
is the low-level ``flash_rt.models.lingbot.graph_runner.sample_actions_graph``
(see ``examples/lingbot_quickstart.py`` and ``docs/lingbot_usage.md``); this
class is a placeholder for the eventual stable-frontend integration.
"""

from __future__ import annotations

import logging
import pathlib

from flash_rt.frontends.torch._lingbot_thor_spec import build_spec
from flash_rt.models.lingbot.pipeline_thor import LingBotPipelineDims

logger = logging.getLogger(__name__)


class LingbotTorchFrontendThor:
    """LingBot-VLA torch frontend for Thor SM110."""

    # Static shape constants — source of truth (yaml is metadata only).
    # Verified against the upstream LingBot weight inventory.
    NUM_VIT_BLOCKS = 32
    NUM_VLM_LAYERS = 36
    NUM_EXPERT_LAYERS = 36
    NUM_INFERENCE_STEPS = 50
    ACTION_DIM = 75
    STATE_DIM = 75
    PROJ_WIDTH = 768
    NUM_CAMS = 3
    IMG_SIZE = 224
    TOKENIZER_MAX_LENGTH = 72

    def __init__(self, checkpoint_dir, num_views=3, autotune=3, **kwargs):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = num_views
        self.autotune = autotune

        # Required path layout (mirrors the upstream LingBot ckpt):
        # ``checkpoint_dir`` points at the lingbot-vla-4b/ directory
        # containing ``model.safetensors`` and ``config.json``. Tokenizer
        # / processor files come from QWEN25_PATH (env var or kwarg).
        safetensors_file = self.checkpoint_dir / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(
                f"LingBot-VLA checkpoint not found: {safetensors_file}. "
                f"Expected HuggingFace layout (model.safetensors). "
                f"Download via `modelscope download --model "
                f"Robbyant/lingbot-vla-4b --local_dir lingbot-vla-4b`."
            )
        config_file = self.checkpoint_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"LingBot-VLA config.json not found: {config_file}."
            )
        logger.info(f"[lingbot_vla] checkpoint resolved: {safetensors_file}")

        self.dims = LingBotPipelineDims()

        # Build the (empty) spec to confirm the import path resolves.
        self._spec = build_spec()
        logger.info(
            f"[lingbot_vla] scaffold loaded; spec has "
            f"{len(self._spec.blocks)} block(s), "
            f"{len(self._spec.singletons)} singleton(s); "
            f"dims: VLM={self.dims.vlm_num_layers}L h={self.dims.vlm_hidden_dim}, "
            f"Expert={self.dims.expert_num_layers}L h={self.dims.expert_hidden_dim}, "
            f"ViT={self.dims.vit_num_blocks}L h={self.dims.vit_hidden_dim}"
        )

        # Make the NotImplementedError explicit upfront so callers don't
        # reach infer() before realizing this frontend is a scaffold.
        raise NotImplementedError(
            "LingbotTorchFrontendThor is a scaffold and is not used by the "
            "current runner. Use the low-level "
            "flash_rt.models.lingbot.graph_runner.sample_actions_graph path "
            "(see examples/lingbot_quickstart.py and docs/lingbot_usage.md)."
        )

    # ──────────────────────────────────────────────────────────────
    # Public API stubs (not implemented — use the graph_runner path)
    # ──────────────────────────────────────────────────────────────

    def set_prompt(self, prompt, state=None, images=None, **kwargs):
        """Would tokenize + ViT encode + VLM prefix → fill KV cache + capture."""
        raise NotImplementedError("set_prompt — scaffold; use graph_runner")

    def infer(self, observation):
        """Would replay the denoise CUDA Graph → return the action chunk."""
        raise NotImplementedError("infer — scaffold; use graph_runner")

    def predict(self, images, prompt=None, state=None):
        """``api.VLAModel.predict`` ABI — delegates to set_prompt + infer."""
        raise NotImplementedError("predict — scaffold; use graph_runner")
