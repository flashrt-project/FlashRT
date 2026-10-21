"""Load XR-1 5B training-format weights without training dependencies."""
from pathlib import Path

import torch


class XR1FoundationProcessor:
    """Qwen3-VL inputs and embodiment statistics for the 30x60 action head.

    Normalization must accompany a training-format checkpoint. Statistics are
    embodiment-specific and are deliberately never inferred from model size.
    """

    def __init__(self, checkpoint):
        import json
        from transformers import AutoProcessor
        root = Path(checkpoint)
        stats_path = root / 'normalization.json'
        if not stats_path.is_file():
            raise FileNotFoundError('XR-1 model_states.pt requires normalization.json '
                                    'with mean/std (30,60), q01/q99 (1,60)')
        stats = json.loads(stats_path.read_text())
        self.stats = {k: torch.as_tensor(stats[k], dtype=torch.float32)
                      for k in ('mean', 'std', 'q01', 'q99')}
        for key, shape in (('mean', (30, 60)), ('std', (30, 60)),
                           ('q01', (1, 60)), ('q99', (1, 60))):
            value = self.stats[key]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f'XR-1 {key} must be finite with shape {shape}')
        if (self.stats['std'] < 0).any():
            raise ValueError('XR-1 std must be nonnegative')
        q01, q99 = self.stats['q01'], self.stats['q99']
        if ((q99 <= q01) & ~((q01 == 0) & (q99 == 0))).any():
            raise ValueError('XR-1 q99 must exceed q01 outside padding')
        self.processor = AutoProcessor.from_pretrained(root / 'tokenizer', use_fast=False)
        self.action_mask = torch.zeros((1, 30, 60), dtype=torch.bfloat16)
        for start, end in ((0, 7), (8, 15), (16, 20)):
            self.action_mask[:, :, start:end] = 1
        self.embodiment = stats.get('embodiment', 'xr1_5b')

    def list_robot_types(self):
        return [self.embodiment]

    def apply_chat_template(self, messages, *, state, robot_type,
                            action_prefix=None, **kwargs):
        if robot_type != self.embodiment:
            raise ValueError(f'Expected embodiment {self.embodiment!r}')
        data = self.processor.apply_chat_template(messages, **kwargs)
        state = torch.as_tensor(state, dtype=torch.float32)
        q01, q99 = self.stats['q01'], self.stats['q99']
        valid = q99 > q01
        normalized = torch.zeros_like(state)
        normalized[..., valid[0]] = (
            2 * (state[..., valid[0]] - q01[..., valid[0]])
            / (q99[..., valid[0]] - q01[..., valid[0]] + 1e-6) - 1)
        data['state'] = normalized.clamp(-1, 1).to(torch.bfloat16)
        data['action_mask'] = self.action_mask.clone()
        data['action'] = torch.zeros_like(self.action_mask)
        if action_prefix is not None:
            prefix = torch.as_tensor(action_prefix, dtype=torch.float32)
            if (prefix.ndim != 2 or prefix.shape[1] != 60
                    or not 0 < prefix.shape[0] <= 30 or not torch.isfinite(prefix).all()):
                raise ValueError('action_prefix must be finite with shape (N,60), 1 <= N <= 30')
            length = prefix.shape[0]
            data['action'][:, :length] = (
                (prefix - self.stats['mean'][:length])
                / (self.stats['std'][:length] + 1e-6)
                * self.action_mask[:, :length])
            data['prefix_length'] = length
        return data

    @staticmethod
    def prepare_images(images):
        """Match the upstream 5B runtime's bounded 32-pixel image resize."""
        import math
        import numpy as np
        from PIL import Image
        result = []
        for image in images:
            if not isinstance(image, Image.Image):
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                    raise ValueError('XR-1 5B images must be PIL RGB or HWC uint8 RGB')
                image = Image.fromarray(array)
            width, height = image.size
            if min(width, height) == 0 or max(width, height) / min(width, height) > 200:
                raise ValueError('Invalid XR-1 image dimensions or aspect ratio')
            new_height = max(32, round(height / 32) * 32)
            new_width = max(32, round(width / 32) * 32)
            if new_height * new_width > 160000:
                scale = math.sqrt(height * width / 160000)
                new_height = max(32, math.floor(height / scale / 32) * 32)
                new_width = max(32, math.floor(width / scale / 32) * 32)
            elif new_height * new_width < 4 * 28 * 28:
                scale = math.sqrt(4 * 28 * 28 / (height * width))
                new_height = math.ceil(height * scale / 32) * 32
                new_width = math.ceil(width / scale / 32) * 32
            result.append(image.convert('RGB').resize((new_width, new_height)))
        return result

    def decode_action(self, actions, robot_type):
        if robot_type != self.embodiment:
            raise ValueError(f'Expected embodiment {self.embodiment!r}')
        mask = self.action_mask.to(actions.device)
        mean, std = (self.stats[k].to(actions.device) for k in ('mean', 'std'))
        return (actions.float() * mask * (std + 1e-6) + mean) * mask


def load_foundation_model(checkpoint, attention):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig
    from flash_rt.amd.models.xr1.upstream.policy import xr1
    root = Path(checkpoint)
    config = Qwen3VLConfig.from_pretrained(root / 'tokenizer')
    with init_empty_weights():
        model = xr1(vlm_config=config, attention=attention,
                    ffn_gradient_checkpointing=False, async_train=False)
    saved = torch.load(root / 'model_states.pt', map_location='cpu', mmap=True,
                       weights_only=True)['module']
    weights = {key.removeprefix('model.'): value for key, value in saved.items()}
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval().to('cuda')
