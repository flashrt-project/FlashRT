"""XR-1 BF16 frontend for gfx1151, with reference and native HIP modes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class XR1TorchFrontendAmdRdna35:
    """Load XR-1 MiBoT deploy or 5B training-format weights with ROCm attention.

    Checkpoint code is loaded with Hugging Face trust_remote_code. Use a local
    snapshot downloaded at a reviewed revision. ``fuse=True`` adds FlashRT
    HIP DiT GEMM/RMSNorm; False keeps upstream ATen ops. ``use_graph=True``
    captures the full GPU forward on first prediction for each input profile.
    """

    def __init__(self, checkpoint_dir, num_views=3, num_steps=5,
                 hardware=None, use_fp8=False, embodiment_tag=None,
                 action_dim=None, attention="sdpa", fuse=False, use_graph=False):
        if hardware not in (None, "amd_rdna35"):
            raise ValueError("XR-1 frontend requires amd_rdna35")
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("XR-1 requires ROCm PyTorch and an available GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx1151":
            raise RuntimeError(f"XR-1 frontend requires gfx1151; found {arch}")
        if use_fp8:
            raise ValueError("XR-1 currently supports BF16 only")
        if attention not in ("sdpa", "eager"):
            raise ValueError("attention must be sdpa or eager")
        for name, value in (("num_steps", num_steps), ("num_views", num_views)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(fuse, bool):
            raise ValueError("fuse must be a boolean")
        if not isinstance(use_graph, bool):
            raise ValueError("use_graph must be a boolean")
        if (action_dim is not None and (isinstance(action_dim, bool)
                or not isinstance(action_dim, int) or action_dim < 1)):
            raise ValueError("action_dim must be a positive integer")
        self.foundation = (Path(checkpoint_dir) / 'model_states.pt').is_file()
        if self.foundation:
            if num_views != 3:
                raise ValueError("XR-1 5B requires three views: ego, left wrist, right wrist")
            from flash_rt.amd.models.xr1.checkpoint_rdna35 import (
                XR1FoundationProcessor, load_foundation_model)
            self.processor = XR1FoundationProcessor(checkpoint_dir)
            self.model = load_foundation_model(checkpoint_dir, attention)
            self.state_dim = self.model.state_shape[-1]
            max_action_dim = self.model.action_shape[-1]
        else:
            self._load_deploy_model(checkpoint_dir, attention)
            self.state_dim = self.model.config.state_dim
            max_action_dim = self.model.config.action_dim
        robot_types = self.processor.list_robot_types()
        if embodiment_tag is None:
            if len(robot_types) != 1:
                raise ValueError(f"Set embodiment_tag to one of {robot_types}")
            embodiment_tag = robot_types[0]
        if embodiment_tag not in robot_types:
            raise ValueError(f"Unknown embodiment {embodiment_tag!r}; available: {robot_types}")
        if action_dim is None:
            if self.foundation:
                action_dim = max_action_dim
            elif embodiment_tag == 'robocasa_mg':
                action_dim = 7
            else:
                raise ValueError("Set action_dim for this deployment's output schema")
        if action_dim > max_action_dim:
            raise ValueError("action_dim exceeds checkpoint action dimension")
        if fuse:
            self.model.requires_grad_(False)
        from flash_rt.amd.models.xr1.pipeline_rdna35 import XR1PipelineRdna35
        self.pipeline = XR1PipelineRdna35(self.model, native=fuse)
        self.use_graph = use_graph
        self.graph_runner = None
        self.num_views, self.num_steps = num_views, num_steps
        self.embodiment_tag, self.action_dim = embodiment_tag, action_dim
        self.prompt = None

    def _load_deploy_model(self, checkpoint_dir, attention):
        from transformers import AutoConfig, AutoModel, AutoProcessor
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        if config.model_type != "mibot":
            raise ValueError("XR-1 requires a deploy-format MiBoT checkpoint")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_dir, trust_remote_code=True, use_fast=False)
        # The outer MiBoT class lacks _supports_sdpa in the released code.
        # Its nested Qwen3-VL supports SDPA; the DiT already calls torch SDPA.
        # Keep the outer wrapper eager and set only the VLM configs to SDPA.
        config.vlm_config._attn_implementation = attention
        config.vlm_config.text_config._attn_implementation = attention
        config.vlm_config.vision_config._attn_implementation = attention
        self.model = AutoModel.from_pretrained(
            checkpoint_dir, config=config, trust_remote_code=True,
            attn_implementation="eager", dtype=torch.bfloat16,
        ).to("cuda").eval()
        for module in self.model.vlm.modules():
            if hasattr(module, "config"):
                module.config._attn_implementation = attention

    def set_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('XR-1 prompt must be a nonempty string')
        self.prompt = prompt

    @torch.inference_mode()
    def infer(self, obs):
        if self.prompt is None:
            raise ValueError("set_prompt must be called before inference")
        images = obs.get("images")
        if images is None:
            images = [obs[k] for k in ("image", "wrist_image", "wrist_image_right")
                      if k in obs]
        if len(images) != self.num_views:
            raise ValueError(f"Expected {self.num_views} images, got {len(images)}")
        if self.foundation:
            images = self.processor.prepare_images(images)
        elif 'action_prefix' in obs:
            raise ValueError('action_prefix is available only for the 5B checkpoint')
        if obs.get("state") is None:
            raise ValueError("XR-1 requires robot proprioceptive state")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if not state.size or state.size > self.state_dim or not np.isfinite(state).all():
            raise ValueError("State must be finite and fit checkpoint state_dim")
        padded = np.zeros((1, 1, self.state_dim), dtype=np.float32)
        padded[0, 0, :state.size] = state
        first_view = 'Ego' if self.foundation else 'Base'
        content = [{"type": "text", "text": f"The following observations are captured from multiple views.\n# {first_view} View\n"}]
        for i, image in enumerate(images):
            if self.foundation and i > 0:
                wrist = 'Left' if i == 1 else 'Right'
                content.append({"type": "text", "text": f"\n# {wrist}-Wrist View\n"})
            elif i == len(images) - 1 and len(images) > 1:
                content.append({"type": "text", "text": "\n# Left-Wrist View\n"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f"\nGenerate robot actions for the task:\n{self.prompt} /no_cot"})
        messages = [{"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            state=padded, robot_type=self.embodiment_tag,
            **({'padding': True, 'images_kwargs': {'do_resize': False}}
               if self.foundation else {}),
            **({'action_prefix': obs['action_prefix']} if 'action_prefix' in obs else {}))
        if self.use_graph:
            if self.graph_runner is None:
                from .xr1_graph import XR1FullGraph
                self.graph_runner = XR1FullGraph(self.pipeline)
            raw_actions = self.graph_runner.run(
                inputs, seed=int(obs.get("seed", 42)), num_steps=self.num_steps,
                embodiment=self.embodiment_tag)
        else:
            data = {k: v.to(device="cuda", dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v.to("cuda") if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()}
            raw_actions = self.pipeline.forward(data, seed=int(obs.get("seed", 42)),
                                                num_steps=self.num_steps,
                                                embodiment=self.embodiment_tag).actions
        actions = self.processor.decode_action(raw_actions,
                                               robot_type=self.embodiment_tag)
        return {"actions": actions[0, :, :self.action_dim].float().cpu().numpy()}
