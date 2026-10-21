XR-1 inference sources vendored from
https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 at
`0dd7aef8dc87296246aae812a1f59ccb708e5546`, under Apache-2.0.

Original files:

- `xr1/mibot/models/VLA/XR1.py` → `policy.py`
- `xr1/mibot/models/VLM/qwen3vl.py` → `qwen3vl.py`
- `xr1/mibot/utils/model_utils.py` → `model_utils.py`

ROCm changes: local relative imports, removal of the mmengine registration,
injectable Qwen3-VL config and attention implementation, and replacement of
Liger RMSNorm/RoPE with the corresponding Transformers PyTorch functions.
Training code is retained to preserve the upstream model's structure and
checkpoint keys. FlashRT exposes inference only.
