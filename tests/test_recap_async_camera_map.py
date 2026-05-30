import sys
import types

import numpy as np
import torch


checkpoint_stub = types.ModuleType("training.rl.checkpoint")
checkpoint_stub.save_lora_state = lambda trainer, path: path
sys.modules.setdefault("training.rl.checkpoint", checkpoint_stub)

trainer_stub = types.ModuleType("training.trainers.pi05_torch_trainer")
trainer_stub.Pi05Trainer = object
sys.modules.setdefault("training.trainers.pi05_torch_trainer", trainer_stub)

from training.rl import train_recap as tr


class _FakeConfig:
    action_dim = 7


class _FakeModel:
    config = _FakeConfig()


class _FakeTrainer:
    is_compiled = True
    device = "cpu"
    model = _FakeModel()

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(()))

    def trainable_parameters(self):
        return [self._param]

    def reset_peak_memory(self):
        pass

    def peak_memory_bytes(self):
        return 0


class _FakeDataset:
    def build_chunk_starts(self, *, action_horizon):
        assert action_horizon == 10
        return np.array([0, 1, 2], dtype=np.int64)

    def has_acp_column(self):
        return False


def test_train_recap_policy_passes_camera_map_to_async_loader(monkeypatch):
    seen = {}

    def fake_loader(*args, **kwargs):
        seen["camera_map"] = kwargs.get("camera_map")
        seen["action_dim_target"] = kwargs.get("action_dim_target")
        return []

    monkeypatch.setattr(tr, "make_step_dataloader", fake_loader)

    camera_map = {
        "base_0_rgb": "front",
        "left_wrist_0_rgb": "wrist",
        "right_wrist_0_rgb": None,
    }
    cfg = tr.RecapTrainConfig(
        num_steps=1,
        batch_size=2,
        use_acp=False,
        camera_map=camera_map,
    )

    tr.train_recap_policy(
        _FakeTrainer(),
        _FakeDataset(),
        tokenizer=lambda tasks: (_ for _ in ()).throw(
            AssertionError("tokenizer should not be called")),
        observation_builder=lambda *args, **kwargs: None,
        config=cfg,
        derive_acp_if_missing=False,
    )

    assert seen["camera_map"] == camera_map
    assert seen["action_dim_target"] == 7
