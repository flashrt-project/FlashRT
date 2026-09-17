"""Drop-in FlashRT TensorRT engine for openpi pi0.5 policies.

Mirrors deployment_scripts.trt_model_forward.setup_pi0_tensorrt_engine from the
Jetson AI Lab openpi tutorial: the policy keeps openpi's transforms, and the
model's sample_actions runs a FlashRT engine built from tools/export_onnx.py
--prompts (inputs images, lang_tokens, noise; output actions).

    import tensorrt as trt
    from openpi_flashrt import setup_pi0_flashrt_engine
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    policy = setup_pi0_flashrt_engine(policy, "pi05.engine", "libflashrt_trt_pi05.so")
    actions = policy.infer(example)["actions"]

The engine is built for a fixed set of cameras (its images input's first
dimension); the observation's image masks must select exactly that many, in
openpi's camera order.
"""
from functools import partial

import tensorrt as trt
import torch

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TOKENS_PER_VIEW = 256


def flashrt_sample_actions(self, device, observation, noise=None, num_steps=None):
    engine = self.trt_engine
    shapes = {name: shape for name, shape, _ in engine.in_meta}
    views = [k for k in IMAGE_KEYS if k in observation.images and bool(observation.image_masks[k].reshape(-1)[0])]
    if len(views) != shapes["images"][0]:
        raise ValueError(f"engine takes {shapes['images'][0]} cameras, observation has {views} unmasked")
    # openpi images are [1, 3, 224, 224] in [-1, 1]; the engine takes HWC fp16
    images = torch.cat([observation.images[k] for k in views], dim=0)
    images = images.permute(0, 2, 3, 1).to(device=device, dtype=torch.float16).contiguous()

    tokens = observation.tokenized_prompt[0][observation.tokenized_prompt_mask[0]]
    tokens = tokens.to(device=device, dtype=torch.int32)
    if (len(views) * TOKENS_PER_VIEW + tokens.numel()) % 2:  # FlashRT keeps the prefix length even
        tokens = torch.cat([tokens, tokens[-1:]])
    tokens = tokens.contiguous()

    horizon, dim = shapes["noise"]
    if noise is None:
        noise = torch.randn(horizon, dim, dtype=torch.float16, device=device)
    else:
        noise = torch.as_tensor(noise).to(device=device, dtype=torch.float16).reshape(horizon, dim).contiguous()

    engine.set_runtime_tensor_shape("images", tuple(images.shape))
    engine.set_runtime_tensor_shape("lang_tokens", tuple(tokens.shape))
    engine.set_runtime_tensor_shape("noise", tuple(noise.shape))
    actions = engine(images=images, lang_tokens=tokens, noise=noise)["actions"]
    return actions[None].float()


def setup_pi0_flashrt_engine(policy, engine_path, plugin_path):
    """Hook a FlashRT pi0.5 engine into an openpi policy (see module docstring)."""
    # The plugin creators must be registered before the engine is deserialized.
    trt.get_plugin_registry().load_library(plugin_path)
    from deployment_scripts.trt_model_forward import setup_pi0_tensorrt_engine

    policy = setup_pi0_tensorrt_engine(policy, engine_path)
    model = policy._model if hasattr(policy, "_model") else policy.model
    sample = partial(flashrt_sample_actions, model)
    model.sample_actions = sample
    if hasattr(policy, "_sample_actions"):
        policy._sample_actions = sample
    print("FlashRT engine hooked to policy sample_actions")
    return policy
