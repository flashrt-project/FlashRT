# Vendored from XiaomiRobotics/Xiaomi-Robotics-1 at
# 0dd7aef8dc87296246aae812a1f59ccb708e5546 (Apache-2.0).
# ROCm adaptation: native Transformers RMSNorm/RoPE, injectable VLM config
# and attention implementation, and no training framework registration.
# Copyright (C) 2026 Xiaomi Corporation.
from functools import wraps
from typing import Any, Callable

import torch


def auto_cast(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(model_self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.bfloat16)
        return func(model_self, batch, *args, **kwargs)

    return wrapper
