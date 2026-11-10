"""Pi0.5 served on the arithmetic of its trainer (SM80, BF16).

The native Pi0.5 pipelines own their arithmetic: fused projections, a BF16
denoising state, time conditioning precomputed per step, an output
projection pre-scaled by the step size. That is what makes them fast, and
it is why a trainer that later recomputes a rollout lands near it rather
than on it. For imitation learning the distance does not matter. For an
on-policy update it does: the update weighs every recorded denoising
transition by the ratio of its probability under the trainer's forward to
its probability under the rollout's, and before any update that ratio has
to be exactly one.

This frontend serves the same checkpoint through FlashRT-Train's policy
schedule. The prefix (vision tower and language model) runs once per
observation and keeps every layer's keys and values; the action expert is
evaluated against them at every denoising step. Each of those calls is the
trainer's own forward restricted to its rows, on GEMM algorithms that do
not change with the row count, so:

  * a sample's actions do not depend on how many samples share its batch;
  * the trainer's forward at a recorded ``(x, t)`` returns the rollout's
    velocity bit for bit, in any minibatch and order;
  * an update takes effect in the next rollout with nothing copied: the
    rollout and the update read the same parameter buffers.

The trainer is ``self.trainer`` (``flashrt_train.native.Trainer``, created
with ``objective="policy"``); ``docs/pi05_isomorphic.md`` shows the loop.

Needs the ``flashrt_train`` package and its native library
(``FRT_TRAIN_LIB``). It does not use FlashRT's own kernel module.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Optional, Sequence, Union

import numpy as np
import torch

from flash_rt.core.utils.actions import LIBERO_ACTION_DIM, unnormalize_actions
from flash_rt.core.utils.pi05_prompt import PI05_STATE_PROMPT_MAX_LEN, format_pi05_prompt

logger = logging.getLogger(__name__)

ACTION_DIM = 32
IMG_HW = 224
NUM_VIEWS = 3
CHUNK_SIZE = 50
NUM_STEPS_DEFAULT = 10

_TRAINABLE = ("none", "expert", "lora")


_TOKENIZERS: dict = {}


def _token_ids(prompt_text: str, max_len: int, state=None) -> list:
    """The prompt's token ids, by the tokenizer the native frontends use (openpi's when importable, else the
    SentencePiece model through the FlashRT locator)."""
    key = ("openpi", int(max_len))
    if key not in _TOKENIZERS:
        try:
            from openpi.models.tokenizer import PaligemmaTokenizer
            _TOKENIZERS[key] = PaligemmaTokenizer(max_len=max_len)
        except (ImportError, FileNotFoundError, OSError, RuntimeError):
            _TOKENIZERS[key] = None
    if _TOKENIZERS[key] is not None:
        tokens, mask = _TOKENIZERS[key].tokenize(prompt_text, state=state)
        return [int(t) for t in tokens[: int(mask.sum())]]
    if "sp" not in _TOKENIZERS:
        from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece
        _TOKENIZERS["sp"] = load_paligemma_sentencepiece()
    from flash_rt.utils.paligemma_tokenizer import encode_pi05_prompt
    sp = _TOKENIZERS["sp"]
    if state is None:
        return encode_pi05_prompt(sp, prompt_text)
    return list(sp.Encode(format_pi05_prompt(prompt_text, state), add_bos=True))


class Pi05TorchFrontendIsomorphic:
    """Pi0.5 rollouts that the trainer reproduces bit for bit.

    Args:
        checkpoint_dir: directory with ``model.safetensors`` (openpi PyTorch
            layout) and the norm stats the native frontends read.
        num_views: three; the trainer's forward takes three cameras.
        chunk_size: action horizon (the checkpoint's; 50 for the base model).
        max_prompt_len: the text columns of the forward. Every prompt is
            padded to it under a mask, so a sample's layout does not depend
            on the other prompts of its batch.
        num_steps: denoising steps.
        denoise_trace: return every step's state and velocity.
        sde: accept ``sde_sigma`` / ``step_noise`` (the stochastic sampler).
        trainable: what ``self.trainer`` updates. ``"none"`` serves only;
            ``"expert"`` trains the action expert over a frozen prefix;
            ``"lora"`` trains the PEFT adapter in ``adapter``. Other forms
            go through ``trainer_kwargs``.
        adapter: PEFT LoRA adapter directory (``trainable="lora"``).
        arena_gib: device memory the trainer's activation arena takes.
        device: CUDA ordinal.
        trainer_kwargs: further ``flashrt_train.native.Trainer`` arguments.
    """

    def __init__(self,
                 checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = NUM_VIEWS,
                 chunk_size: int = CHUNK_SIZE,
                 max_prompt_len: int = PI05_STATE_PROMPT_MAX_LEN,
                 num_steps: int = NUM_STEPS_DEFAULT,
                 *,
                 denoise_trace: bool = False,
                 sde: bool = False,
                 trainable: str = "none",
                 adapter: Optional[str] = None,
                 arena_gib: float = 16.0,
                 device: int = 0,
                 hardware: Optional[str] = None,
                 trainer_kwargs: Optional[dict] = None):
        if int(num_views) != NUM_VIEWS:
            raise ValueError(
                f"the isomorphic Pi0.5 frontend takes {NUM_VIEWS} camera views (the trainer's forward), got {num_views}")
        if trainable not in _TRAINABLE:
            raise ValueError(f"trainable must be one of {_TRAINABLE}, got {trainable!r}")
        if (trainable == "lora") != (adapter is not None):
            raise ValueError("adapter is given with trainable='lora', and only then")
        try:
            from flashrt_train.native import Trainer
        except ImportError as e:
            raise ImportError(
                "the isomorphic Pi0.5 frontend runs on FlashRT-Train: put the flashrt_train package on PYTHONPATH and "
                "point FRT_TRAIN_LIB at its native library") from e

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"safetensors not found at {safetensors_path}")
        from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
        self.norm_stats = load_norm_stats(pi05_candidates(checkpoint_dir), checkpoint_dir=checkpoint_dir)

        self.num_views = NUM_VIEWS
        self.chunk_size = int(chunk_size)
        self.max_prompt_len = int(max_prompt_len)
        self._num_steps = int(num_steps)
        self._denoise_trace = bool(denoise_trace)
        self._sde = bool(sde)
        self.trainable = trainable
        self._device = torch.device("cuda", int(device))

        form: dict = {}
        if trainable == "expert":
            form["frozen"] = ["paligemma."]
        elif trainable == "lora":
            form["adapter"] = str(adapter)
        kwargs = dict(objective="policy", gemm_algo="invariant", text_tokens=self.max_prompt_len,
                      suffix_len=self.chunk_size, optimizer=trainable != "none", device=int(device),
                      arena_bytes=int(float(arena_gib) * (1 << 30)), **form)
        kwargs.update(trainer_kwargs or {})
        if kwargs.get("gemm_algo") != "invariant":
            raise ValueError("the isomorphic frontend needs gemm_algo='invariant': another choice makes a sample depend on its batch")
        self.trainer = Trainer(str(safetensors_path), **kwargs)

        self._prompts: list = []        # (text, state) per slot
        self._tokens: Optional[torch.Tensor] = None   # int64 [B, max_prompt_len] on the device
        self._mask: Optional[np.ndarray] = None       # uint8 [B, max_prompt_len] on the host
        self._images: Optional[list] = None           # three bf16 [B, 3, 224, 224]
        self._last = None
        self.latency_records: list = []
        logger.info("Pi05TorchFrontendIsomorphic: %s, chunk %d, text %d, %d steps, trainable=%s",
                    safetensors_path, self.chunk_size, self.max_prompt_len, self._num_steps, trainable)

    # ---- prompts -------------------------------------------------------------------------------------------------
    def _tokenize(self, prompts: Sequence[str], states) -> None:
        if states is None:
            states = [None] * len(prompts)
        if len(states) != len(prompts):
            raise ValueError("one state per prompt")
        tokens = np.zeros((len(prompts), self.max_prompt_len), dtype=np.int64)
        mask = np.zeros((len(prompts), self.max_prompt_len), dtype=np.uint8)
        for b, (text, state) in enumerate(zip(prompts, states)):
            ids = _token_ids(text, self.max_prompt_len, state)
            if len(ids) > self.max_prompt_len:
                raise ValueError(f"prompt {b} has {len(ids)} tokens; max_prompt_len is {self.max_prompt_len}")
            tokens[b, : len(ids)] = ids
            mask[b, : len(ids)] = 1
        self._prompts = list(zip(prompts, states))
        self._tokens = torch.from_numpy(tokens).to(self._device)
        self._mask = mask

    def set_prompt(self, prompt_text: str, state=None) -> None:
        """One prompt, for :meth:`infer`."""
        self._tokenize([prompt_text], None if state is None else [state])

    def set_prompt_batch(self, prompts: Sequence[str], states=None) -> None:
        """One prompt per sample, for :meth:`infer_batch`. Any width; the width may change between calls."""
        self._tokenize(list(prompts), states)

    def set_prompt_tokens(self, tokens, mask) -> None:
        """Token ids and their mask directly (``[B, max_prompt_len]``), for callers that tokenize themselves."""
        tokens = np.asarray(torch.as_tensor(tokens).cpu(), dtype=np.int64)
        mask = np.asarray(torch.as_tensor(mask).cpu()).astype(np.uint8)
        if tokens.ndim != 2 or tokens.shape[1] != self.max_prompt_len or mask.shape != tokens.shape:
            raise ValueError(f"tokens and mask must be [B, {self.max_prompt_len}]")
        self._prompts = [(None, None)] * tokens.shape[0]
        self._tokens = torch.from_numpy(np.ascontiguousarray(tokens)).to(self._device)
        self._mask = np.ascontiguousarray(mask)

    # ---- inputs --------------------------------------------------------------------------------------------------
    def _camera_tensors(self, observations: Sequence[dict]) -> list:
        """Three bf16 ``[B, 3, 224, 224]`` tensors in [-1, 1] from uint8 HWC frames (the native frontends' scaling)."""
        cams = [[] for _ in range(NUM_VIEWS)]
        for obs in observations:
            if "images" in obs:
                frames = list(obs["images"])
            else:
                frames = [obs["image"], obs["wrist_image"], obs["wrist_image_right"]]
            if len(frames) < NUM_VIEWS:
                raise ValueError(f"an observation needs {NUM_VIEWS} frames, got {len(frames)}")
            for v in range(NUM_VIEWS):
                im = np.asarray(frames[v])
                if im.shape != (IMG_HW, IMG_HW, 3):
                    raise ValueError(f"frames must be ({IMG_HW}, {IMG_HW}, 3), got {im.shape}")
                cams[v].append(torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0))
        return [torch.stack(c).permute(0, 3, 1, 2).to(self._device, torch.bfloat16).contiguous() for c in cams]

    def _draw(self, shape, value, generator) -> torch.Tensor:
        if value is not None:
            if generator is not None:
                raise ValueError("pass either noise or generator, not both")
            t = torch.as_tensor(value).to(self._device, torch.float32)
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"noise must have shape {tuple(shape)}, got {tuple(t.shape)}")
            return t.contiguous()
        t = torch.empty(shape, dtype=torch.float32, device=self._device)
        return t.normal_(generator=generator) if generator is not None else t.normal_()

    def denoise_timesteps(self) -> list:
        """Flow time of each step, as the trainer's schedule accumulates it in fp32."""
        dt = np.float32(-1.0) / np.float32(self._num_steps)
        out, t = [], np.float32(1.0)
        for _ in range(self._num_steps):
            out.append(float(t))
            t = np.float32(t + dt)
        return out

    # ---- inference -----------------------------------------------------------------------------------------------
    def _rollout(self, observations: Sequence[dict], noise, generator, step_noise, sde_sigma, return_noise: bool) -> list:
        if self._tokens is None:
            raise RuntimeError("set_prompt must be called before infer")
        B = len(observations)
        if int(self._tokens.shape[0]) != B:
            raise ValueError(f"{B} observations for {int(self._tokens.shape[0])} prompts")
        if (sde_sigma is not None or step_noise is not None) and not self._sde:
            raise ValueError("step_noise / sde_sigma need a frontend built with sde=True")
        if step_noise is not None and sde_sigma is None:
            raise ValueError("step_noise needs sde_sigma")
        S, steps = self.chunk_size, self._num_steps
        t0 = time.perf_counter()

        images = self._camera_tensors(observations)
        init_gen = None if (noise is not None and sde_sigma is not None and step_noise is None) else generator
        x0 = self._draw((B, S, ACTION_DIM), noise, init_gen)
        sigma = eps = None
        if sde_sigma is not None:
            sigma = np.asarray(sde_sigma, dtype=np.float32).reshape(-1)
            if sigma.shape[0] != steps or not np.isfinite(sigma).all() or (sigma < 0).any():
                raise ValueError(f"sde_sigma must hold {steps} finite non-negative values")
            eps = self._draw((steps, B, S, ACTION_DIM), step_noise, generator)
        actions = torch.empty_like(x0)
        trace_x = trace_v = None
        if self._denoise_trace:
            trace_x = torch.empty(steps, B, S, ACTION_DIM, dtype=torch.float32, device=self._device)
            trace_v = torch.empty_like(trace_x)

        self.trainer.policy_prefix(images, self._tokens, self._mask)
        self.trainer.policy_sample(x0, actions, steps=steps, sigma=sigma, step_noise=eps, trace_x=trace_x, trace_v=trace_v)
        self.trainer.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # what a replay needs, on the device: the trainer's forward takes exactly these
        self._last = {"images": images, "tokens": self._tokens, "mask": self._mask, "noise": x0, "actions": actions,
                      "trace_x": trace_x, "trace_v": trace_v, "step_noise": eps, "sde_sigma": sigma}

        raw = actions.cpu().numpy()
        tx = trace_x.cpu().numpy() if trace_x is not None else None
        tv = trace_v.cpu().numpy() if trace_v is not None else None
        dt = np.float32(-1.0) / np.float32(steps)
        results = []
        for b in range(B):
            entry = {"actions": unnormalize_actions(raw[b], self.norm_stats)[:, :LIBERO_ACTION_DIM], "raw_actions": raw[b]}
            if return_noise:
                entry["noise"] = x0[b].cpu().numpy()
            if tx is not None:
                entry["denoise_trace"] = {"x": tx[:, b], "v": tv[:, b], "delta": dt * tv[:, b], "timesteps": self.denoise_timesteps()}
            if eps is not None:
                entry["step_noise"] = eps[:, b].cpu().numpy()
                entry["sde_sigma"] = [float(v) for v in sigma]
            results.append(entry)
        return results

    def infer(self, observation: dict, debug: bool = False, *, noise=None, generator=None, step_noise=None, sde_sigma=None,
              return_noise: bool = False) -> dict:
        """One observation. The arguments and the result follow ``Pi05TorchFrontendRtx.infer``.

        ``noise`` is ``(chunk_size, 32)``, ``step_noise`` ``(num_steps, chunk_size, 32)``. The result always carries
        ``"raw_actions"``; with ``denoise_trace=True`` the trace holds ``x[s]`` (the state entering step ``s``), ``v[s]``
        (the velocity the trainer's forward returns at ``(x[s], t[s])``) and ``delta[s] = dt * v[s]``, all fp32.
        """
        noise = None if noise is None else torch.as_tensor(noise)[None]
        step_noise = None if step_noise is None else torch.as_tensor(step_noise)[:, None]
        out = self._rollout([observation], noise, generator, step_noise, sde_sigma, return_noise)[0]
        if debug:
            logger.info("Raw actions[0,:5]: %s; latency %.1f ms", out["raw_actions"][0, :5], self.latency_records[-1])
        return out

    def infer_batch(self, observations: list, *, noise=None, generator=None, step_noise=None, sde_sigma=None,
                    return_noise: bool = False) -> list:
        """One rollout per observation, as one batch. ``noise`` is ``(B, chunk_size, 32)``, ``step_noise``
        ``(num_steps, B, chunk_size, 32)``. Every sample's result is what :meth:`infer` gives it alone."""
        return self._rollout(list(observations), noise, generator, step_noise, sde_sigma, return_noise)

    # ---- what the trainer side needs --------------------------------------------------------------------------------
    @property
    def last_rollout(self) -> Optional[dict]:
        """Device tensors of the last call: ``images`` (three bf16 ``[B, 3, 224, 224]``), ``tokens``, ``mask``,
        ``noise``, ``actions``, ``trace_x`` / ``trace_v`` (``[steps, B, chunk, 32]`` with ``denoise_trace``),
        ``step_noise``, ``sde_sigma``. ``trainer.policy_forward(images, tokens, mask, x, t, out)`` on any subset of
        the samples and any recorded ``(x, t)`` returns the trace's velocity."""
        return self._last

    def export_prefix(self) -> torch.Tensor:
        """The last observations' prefix as flat records (``[B, trainer.policy_record_bytes()]`` bytes on the device):
        what a replay buffer keeps so that an update over a frozen prefix does not run it again
        (``trainer.policy_import`` + ``policy_forward_kept``)."""
        if self._last is None:
            raise RuntimeError("no rollout yet")
        B = int(self._last["noise"].shape[0])
        records = torch.empty(B, self.trainer.policy_record_bytes(), dtype=torch.uint8, device=self._device)
        self.trainer.policy_export(records)
        return records

    def get_latency_stats(self) -> dict:
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {"count": len(lat), "mean_ms": float(np.mean(lat)), "std_ms": float(np.std(lat)), "min_ms": float(np.min(lat)),
                "max_ms": float(np.max(lat)), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                "hz": float(1000 / np.mean(lat))}
