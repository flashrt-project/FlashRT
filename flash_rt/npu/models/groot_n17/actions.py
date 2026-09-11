"""Batched inverse of the GR00T N1.7 action encoding.

The official decode costs 23 ms a frame on this box — a quarter of the whole
frame — and none of it is arithmetic. It builds one pose object per horizon
step: 81 of them, each running a Gram-Schmidt, a couple of ``isclose`` checks
and a cross product on 3x3 matrices in Python. The same work over the whole
horizon at once is a handful of batched operations.

Everything here is driven by the processor's own parameters, read once at setup.
Nothing is hardcoded, and any combination of representation, type and format
this has not been validated against is refused by name rather than approximated
— the relative end-effector path in particular is a pose composition, and
getting it subtly wrong would show up as a plausible trajectory rather than as
an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` to ``(..., 3, 3)``, the reference's Gram-Schmidt.

    The six values are the first two *rows* of the rotation matrix; the second
    is orthogonalised against the first and the third is their cross product.
    """
    first, second = rot6d[..., :3], rot6d[..., 3:]
    row1 = first / first.norm(dim=-1, keepdim=True)
    row2 = second - (row1 * second).sum(-1, keepdim=True) * row1
    row2 = row2 / row2.norm(dim=-1, keepdim=True)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)


@dataclass(frozen=True)
class _Modality:
    key: str
    start: int
    width: int
    low: torch.Tensor
    span: torch.Tensor
    relative: bool
    end_effector: bool


class ActionDecoder:
    """Normalised model output to robot-space actions, over the whole horizon.

    ``__call__`` takes the ``(1, horizon, action_dim)`` tensor the action head
    produces and the observation's raw state, and returns one tensor per
    modality in the layout the caller's policy expects.
    """

    #: Representations this implementation has been validated against. A
    #: checkpoint asking for anything else is refused: the alternative is a
    #: silently different trajectory.
    SUPPORTED = {("absolute", False), ("relative", False), ("relative", True)}

    def __init__(self, processor, embodiment_tag: str, *, device="npu:0",
                 dtype=torch.float32):
        state_action = processor.state_action_processor
        config = state_action.modality_configs[embodiment_tag]["action"]
        params = state_action.norm_params[embodiment_tag]["action"]
        if not state_action.use_relative_action:
            raise NotImplementedError(
                "this decoder implements the relative-action contract; the "
                "checkpoint's processor has it disabled")
        self.horizon = len(config.delta_indices)
        self.device = device
        self.dtype = dtype
        self.modalities: list[_Modality] = []
        start = 0
        configs = config.action_configs or []
        if len(configs) != len(config.modality_keys):
            raise NotImplementedError(
                "every action modality needs a representation; this checkpoint "
                f"declares {len(configs)} for {len(config.modality_keys)} modalities")
        for key, action_config in zip(config.modality_keys, configs):
            width = int(torch.as_tensor(params[key]["dim"]).reshape(-1)[0].item())
            relative = str(action_config.rep).rsplit(".", 1)[-1].lower() == "relative"
            end_effector = str(action_config.type).rsplit(".", 1)[-1].lower() == "eef"
            if (("relative" if relative else "absolute"), end_effector) not in self.SUPPORTED:
                raise NotImplementedError(
                    f"action modality {key!r} is {action_config.rep} / "
                    f"{action_config.type}, which this decoder has not been "
                    "validated for")
            if end_effector and str(action_config.format).rsplit(".", 1)[-1].lower() \
                    != "xyz_rot6d":
                raise NotImplementedError(
                    f"end-effector modality {key!r} is in {action_config.format}; "
                    "this decoder implements the translation-and-6D-rotation form")
            if (action_config.state_key or key) != key:
                raise NotImplementedError(
                    f"modality {key!r} references state {action_config.state_key!r}; "
                    "this decoder reads the state of the same name")
            low = torch.as_tensor(params[key]["min"]).to(device, dtype)
            high = torch.as_tensor(params[key]["max"]).to(device, dtype)
            self.modalities.append(_Modality(
                key=key, start=start, width=width,
                low=low.reshape(-1, width) if low.numel() > width else low.reshape(width),
                span=(high - low).reshape(low.shape), relative=relative,
                end_effector=end_effector))
            start += width
        self.action_dim = start

    # ------------------------------------------------------------------
    def __call__(self, normalized: torch.Tensor, state: dict) -> dict:
        """``(1, horizon, action_dim)`` normalised in, robot-space out."""
        flat = normalized.reshape(-1, normalized.shape[-1])[:self.horizon]
        if flat.shape[0] != self.horizon:
            raise ValueError(
                f"the decoder was built for a {self.horizon}-step horizon, got "
                f"{flat.shape[0]}")
        out = {}
        for modality in self.modalities:
            block = flat[:, modality.start:modality.start + modality.width]
            values = ((block.to(self.dtype).clamp(-1.0, 1.0) + 1.0) / 2.0
                      * modality.span + modality.low)
            if modality.relative:
                # The reference takes the last state timestep as the frame.
                reference = torch.as_tensor(state[modality.key]).to(
                    self.device, self.dtype).reshape(-1, modality.width)[-1]
                values = (self._compose(values, reference) if modality.end_effector
                          else values + reference)
            out[modality.key] = values.unsqueeze(0)
        return out

    @staticmethod
    def _compose(relative: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compose a relative pose trajectory onto a reference frame.

        The reference does this as ``T_ref @ T_rel`` per step over homogeneous
        matrices; over a horizon it is one batched product.
        """
        rotation_reference = rot6d_to_matrix(reference[3:])
        rotation = rotation_reference @ rot6d_to_matrix(relative[:, 3:])
        translation = (relative[:, :3] @ rotation_reference.transpose(-1, -2)
                       + reference[:3])
        return torch.cat((translation, rotation[:, :2, :].reshape(-1, 6)), dim=-1)
