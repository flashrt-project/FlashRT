"""FlashRT — LingBot-VLA torch frontend for Thor (SM110) — G1.

Class: ``LingbotTorchFrontendThor``. Owner of:
    * the loaded LingBot-VLA model (~4B params, ~16.7 GB BF16)
    * the AttentionBackend (declared at G1, wired into kernels in G5)
    * the LingBotPipelineThor (orchestrates ViT → VLM prefix → 50 × Expert
      denoise steps → action_out_proj)
    * lifecycle: __init__ → set_prompt(prompt, ...) → infer(obs) [...]

Hardware target: Jetson AGX Thor SM110. NGC PyTorch 2.10 (Thor build) is
required — standard cu128 wheels do NOT include sm_110 kernels. Baseline
runs in container ``lingbot-dev`` (turbo_pi_libero:1.0.0-verify image).

Reference baseline: PyTorch BF16 eager P50 = 2480 ms (50 Euler steps),
116.7 ms prefix encode + 46.2 ms × 50 denoise; cos_determinism = 1.0
(see /workspace/lingbot-vla/baseline_artifacts/).

────────────────────────────────────────────────────────────────────
G1 status
────────────────────────────────────────────────────────────────────
Stub. ``__init__`` validates the checkpoint path layout and builds an
empty weight spec; ``set_prompt`` / ``infer`` / ``predict`` raise
NotImplementedError. The smoke test in
``tests/test_lingbot_g1_smoke.py`` only exercises the construction path
and is expected to catch the NotImplementedError.

Subsequent gates:
    G2  eager BF16 forward via flash_rt abstraction (no fvk yet)
    G3  ViT replaced by fvk patch_embed + block forward (32 blocks)
    G4  VLM 36L replaced by fvk encoder_forward variant (M-RoPE θ=1e6)
    G5  Mixed-Head joint attention (VLM + Expert concat per layer)
    G6  Action Expert 36L + AdaRMSNorm FiLM (cond_dim=768)
    G7  50-step Euler ODE captured in CUDA Graph
    G8  FP8 calibration on baseline_artifacts/inputs + quant rollout
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
    # Verified against /workspace/lingbot-vla/baseline_artifacts/weight_inventory.json.
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

        # G1: build (empty) spec to confirm import path. G2 fills it in.
        self._spec = build_spec()
        logger.info(
            f"[lingbot_vla] G1 scaffold loaded; spec has "
            f"{len(self._spec.blocks)} block(s), "
            f"{len(self._spec.singletons)} singleton(s); "
            f"dims: VLM={self.dims.vlm_num_layers}L h={self.dims.vlm_hidden_dim}, "
            f"Expert={self.dims.expert_num_layers}L h={self.dims.expert_hidden_dim}, "
            f"ViT={self.dims.vit_num_blocks}L h={self.dims.vit_hidden_dim}"
        )

        # Make the NotImplementedError explicit upfront so callers don't
        # waste compute reaching infer() before realizing G2 isn't done.
        raise NotImplementedError(
            "LingbotTorchFrontendThor is at G1 (scaffolding only). "
            "G2 (eager BF16 forward via FlashRT abstraction) is the next "
            "milestone. Baseline reference: "
            "/workspace/lingbot-vla/baseline_artifacts/."
        )

    # ──────────────────────────────────────────────────────────────
    # Public API stubs (filled in G2+)
    # ──────────────────────────────────────────────────────────────

    def set_prompt(self, prompt, state=None, images=None, **kwargs):
        """G2+: tokenize + ViT encode + VLM prefix → fill KV cache + capture."""
        raise NotImplementedError("set_prompt — implemented in G2+")

    def infer(self, observation):
        """G2+: replay denoise CUDA Graph N=50 times → return action chunk."""
        raise NotImplementedError("infer — implemented in G2+")

    def predict(self, images, prompt=None, state=None):
        """``api.VLAModel.predict`` ABI — delegates to set_prompt + infer."""
        raise NotImplementedError("predict — implemented in G2+")
