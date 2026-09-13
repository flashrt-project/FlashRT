"""GR00T N1.7 on Ascend: the frontend `load_model` routes to.

The compute is in ``flash_rt/npu/models/groot_n17/`` — the ported backbone and
action head, the three native kernels, the image path, and the action decode.
This is the adapter that makes it reachable, and it keeps the same four-call
contract the model's other backends have:

    model = flash_rt.load_model(checkpoint, config="groot_n17",
                                framework="torch", hardware="npu")
    fe = model.pipeline
    fe.set_hf_processor(processor)
    fe.set_prompt(aux=aux, prompt=instruction)
    state = fe.normalize_state(state_dict)
    out = fe.infer(state, aux=aux)                 # a fresh observation
    actions = fe.denormalize_action(out, state_dict=state_dict)

``VLAModel.predict`` is not this model's path on any backend: it calls
``set_prompt`` with a prompt string alone, and every GR00T N1.7 frontend needs
the observation bundle as well. Reach ``model.pipeline`` as above.

What ``aux`` has to carry is the part that depends on the prompt and the camera
geometry rather than on the frame, which is why the captured graph has exactly
three inputs. The keys are listed in ``PROMPT_KEYS`` and checked on the way in,
because a missing one would otherwise surface as a shape error several calls
later. Producing them is the caller's job here exactly as it is on Thor and on
CDNA4 — the official processor and the reference backbone's rotary embeddings
are the source, and no part of that is Ascend-specific.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from flash_rt.models.groot_n17.embodiments import (
    EMBODIMENT_NUM_VIEWS,
    EMBODIMENT_TAG_TO_INDEX,
)
from flash_rt.npu.models.groot_n17 import backbone as bb
from flash_rt.npu.models.groot_n17 import pipeline as pl
from flash_rt.npu.models.groot_n17 import preprocess as pre
from flash_rt.npu.models.groot_n17.actions import ActionDecoder, StateEncoder
from flash_rt.npu.models.groot_n17.captured import CapturedFrame
from flash_rt.npu.models.groot_n17.weights import load_frame

#: Everything ``set_prompt`` needs that is a function of the prompt and the
#: camera geometry. ``CapturedFrame`` documents what each one is.
PROMPT_KEYS = ("views", "image_mask", "attention_mask", "vit_cos", "vit_sin",
               "llm_cos", "llm_sin", "text_embeds", "patch_shape",
               "patch_positions")

#: One of these has to reach ``infer`` for a fresh observation: the camera's
#: uint8 frames, which this frontend transforms on the die, or the patch rows if
#: the caller already has them.
FRAME_KEYS = ("frames", "patches")


class GrootN17TorchFrontendNpu:
    """The Ascend BF16 tier for GR00T N1.7.

    One captured graph holds the whole model — patch projection, ViT, DeepStack,
    the truncated language model, the VL adapter, and the four denoise steps over
    the 32-layer DiT — so a frame refills three buffers and replays. Building it
    is what ``set_prompt`` does; it cannot happen in ``__init__`` because the
    graph's shapes come from the prompt.
    """

    def __init__(self, checkpoint_dir, *, num_views: int = 2,
                 embodiment_tag: str | None = None,
                 action_horizon: int = pl.ACTION_HORIZON,
                 steps: int = pl.NUM_STEPS, device: str = "npu:0",
                 use_int8: bool = False):
        if use_int8:
            # Measured and not shipped: the quantised GEMM wins 3 us a call while
            # the kernel types it adds to the DiT's per-layer loop make the
            # launches already in that loop slower. Refused rather than silently
            # served as BF16, which would be measured as a regression.
            raise NotImplementedError(
                "the Ascend GR00T N1.7 tier is BF16; there is no INT8 tier to "
                "select. Pass precision='auto' or precision='bf16'.")
        tag = embodiment_tag or "oxe_droid_relative_eef_relative_joint"
        if tag not in EMBODIMENT_TAG_TO_INDEX:
            raise ValueError(
                f"embodiment_tag {tag!r} is not in the checkpoint's table; known "
                f"tags: {sorted(EMBODIMENT_TAG_TO_INDEX)}")
        expected = EMBODIMENT_NUM_VIEWS.get(tag)
        if expected is not None and int(num_views) != expected:
            raise ValueError(
                f"embodiment {tag!r} was trained with {expected} camera "
                f"view(s), got num_views={num_views}")
        self.device = device
        self.embodiment_tag = tag
        self.embodiment_id = EMBODIMENT_TAG_TO_INDEX[tag]
        self.num_views = int(num_views)
        self.horizon = int(action_horizon)
        self.steps = int(steps)
        self._processor = None
        self._encoder = None
        self._decoder = None
        self._transform = None
        self._frame = None
        self._prompt = None
        self._latency_ms: list[float] = []

        weights = load_frame(checkpoint_dir)
        self._backbone = bb.BoundBackbone(weights, device=device)
        self._chain = pl.BoundChain(weights, self.embodiment_id, device=device,
                                    steps=self.steps, horizon=self.horizon)
        del weights                      # 6.5 GB of host tensors

    # ── the host half of the contract ─────────────────────────────────
    def set_hf_processor(self, processor) -> None:
        """The official processor owns the normalisation parameters and the
        modality layout, so the state encode and the action decode are built
        from it rather than from a second copy of those numbers."""
        self._processor = processor
        self._encoder = StateEncoder(processor, self.embodiment_tag, pl.STATE_DIM,
                                     device=self.device)
        self._decoder = ActionDecoder(processor, self.embodiment_tag,
                                      device=self.device)

    def normalize_state(self, state_dict: dict) -> torch.Tensor:
        """``{modality: (..., dim)}`` to the ``(1, 1, 132)`` the head reads."""
        if self._encoder is None:
            raise RuntimeError("call set_hf_processor before normalize_state")
        return self._encoder(state_dict)

    def denormalize_action(self, action_normed: torch.Tensor,
                           state_dict: dict | None = None) -> dict:
        """The head's output to robot-space actions, one tensor per modality."""
        if self._decoder is None:
            raise RuntimeError("call set_hf_processor before denormalize_action")
        if state_dict is None:
            raise ValueError(
                "this embodiment's actions are relative, so denormalising them "
                "needs the observation's state as the reference frame")
        return self._decoder(action_normed, state_dict)

    # ── the graph ─────────────────────────────────────────────────────
    def set_prompt(self, *, aux: dict, prompt: str | None = None) -> None:
        """Build and capture the frame graph for one prompt.

        Everything that depends only on the prompt is resolved here and then
        only read: the rope tables, the causal extent, where the image tokens
        sit, and the instruction's embeddings. Changing the prompt means calling
        this again.
        """
        missing = [key for key in PROMPT_KEYS if key not in aux]
        if missing:
            raise ValueError(
                f"aux is missing {missing}; a GR00T N1.7 prompt bundle carries "
                f"{list(PROMPT_KEYS)} (see the class docstring for where they "
                "come from)")
        if int(aux["views"]) != self.num_views:
            raise ValueError(
                f"aux describes {int(aux['views'])} camera view(s) and this "
                f"frontend was built for {self.num_views}")
        self._frame = CapturedFrame(self._backbone, self._chain, aux)
        self._frame.capture()
        self._prompt = prompt
        self._transform = None            # its geometry comes from the frames

    # ── per frame ─────────────────────────────────────────────────────
    def infer(self, state_normalized: torch.Tensor, *, aux: dict | None = None,
              initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        """One observation in, ``(1, horizon, 132)`` normalised actions out.

        ``aux`` carries the frame: ``frames``, the camera's ``(views, H, W, 3)``
        uint8 on the device, which this transforms and projects; or ``patches``
        if the caller already has the projection's rows. Omitting ``aux``
        entirely would mean reusing the backbone features from the previous
        frame, which this tier does not implement -- the captured graph is the
        whole model, and a features-only replay is a second graph that has not
        been measured.
        """
        if self._frame is None:
            raise RuntimeError("call set_prompt before infer")
        if aux is None:
            raise NotImplementedError(
                "this tier captures the whole frame, so every call needs the "
                "observation; reusing the previous frame's backbone features is "
                "a separate graph (CapturedChain) and is not wired here")
        present = [key for key in FRAME_KEYS if aux.get(key) is not None]
        if len(present) != 1:
            raise ValueError(
                f"a frame needs exactly one of {list(FRAME_KEYS)}, got {present}")

        started = time.perf_counter()
        patches = (aux["patches"] if present[0] == "patches"
                   else self._project(aux["frames"]))
        noise = self._noise(initial_noise)
        self._frame.fill(patches, state_normalized, noise)
        actions = self._frame.replay()
        self._latency_ms.append((time.perf_counter() - started) * 1000.0)
        return actions.float()

    def _project(self, frames) -> torch.Tensor:
        """The reference's evaluation transform and the patch projection, on the
        die. The transform is built on the first frame because its plan is fixed
        by the camera's geometry, which the prompt bundle does not carry."""
        if not isinstance(frames, torch.Tensor):
            frames = torch.as_tensor(np.asarray(frames))
        if frames.device.type != "npu":
            raise ValueError(
                f"frames must be on an Ascend device, got {frames.device}; the "
                "transform is handed their address, and the upload is part of "
                "the frame's cost so it belongs where the caller can see it")
        if self._transform is None:
            if frames.dim() != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    f"frames must be (views, H, W, 3) uint8, got "
                    f"{tuple(frames.shape)}")
            self._transform = pre.EvalImageTransform(
                int(frames.shape[1]), int(frames.shape[2]), device=self.device,
                images=int(frames.shape[0]))
        return pre.patch_rows(self._transform(frames))

    def _noise(self, initial_noise) -> torch.Tensor:
        if initial_noise is None:
            return torch.randn(1, self.horizon, pl.ACTION_DIM,
                               dtype=torch.bfloat16, device=self.device)
        noise = torch.as_tensor(initial_noise)
        if tuple(noise.shape[-2:]) != (self.horizon, pl.ACTION_DIM):
            raise ValueError(
                f"initial_noise must end in ({self.horizon}, {pl.ACTION_DIM}), "
                f"got {tuple(noise.shape)}")
        return noise.reshape(1, self.horizon, pl.ACTION_DIM).to(
            self.device, torch.bfloat16)

    # ── reporting ─────────────────────────────────────────────────────
    def get_latency_stats(self) -> dict:
        """Per-call wall clock of `infer`, which is the graph replay plus the
        image path and the noise draw -- not the whole served frame, which also
        has the caller's transforms and the action decode in it."""
        if not self._latency_ms:
            return {"calls": 0}
        samples = sorted(self._latency_ms)
        return {
            "calls": len(samples),
            "median_ms": samples[len(samples) // 2],
            "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
            "min_ms": samples[0],
        }

    def precision_spec(self) -> dict:
        """BF16 throughout. 910/A2 parts have no FP8 tensor hardware, and the
        INT8 tier was measured and not shipped."""
        return {"tier": "bf16", "weights": "bf16", "activations": "bf16",
                "native_kernels": ("groot_n17 dit attention",
                                   "groot_n17 fused add-and-normalise",
                                   "groot_n17 evaluation image resize")}
