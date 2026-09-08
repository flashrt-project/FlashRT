"""Pi0.5 Ascend NPU frontend (torch_npu / CANN).

Mirrors the AMD frontend's public contract
(``flash_rt.amd.frontends.torch.pi05.Pi05TorchFrontendAmd``) but drives
the captured NPU pipeline in ``flash_rt/npu/models/pi05/captured.py``
and replaces CUDA/HIP-graph capture with ``torch.npu.graph`` capture.

Execution model (BF16 or real-data calibrated static encoder INT8):

- the checkpoint's BF16 weights live on the NPU;
- ``set_prompt`` tokenizes the prompt (openpi layout, SentencePiece
  fallback), computes its language embeddings into a static NPU buffer,
  and builds a per-length ``_CapturedRunner`` that captures the whole
  fixed frame (vision + encoder + num_steps denoise) into one NPU graph;
- ``infer`` only refills the static image/noise buffers, replays the
  graph and reads the actions back (unnormalised to robot space).

The fp32 CPU weights are kept as ``self.wref`` so an eager fp32 CPU pass
is available for cosine gating (``reference_actions``).

``state_prompt_mode="fixed"`` and the RL/batched surfaces are not ported
yet; they raise NotImplementedError like the AMD frontend's RTX-only
surfaces do.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
from flash_rt.npu.models.pi05 import pipeline as npu_pl
from flash_rt.npu.models.pi05 import fast as npu_fast

logger = logging.getLogger(__name__)

_NPU_ONLY = object()


from flash_rt.npu.models.pi05.captured import _CapturedRunner


class Pi05TorchFrontendNpu:
    """Pi0.5 on Huawei Ascend via torch_npu + NPU graph capture."""

    def __init__(self, checkpoint_dir, num_views=2, chunk_size=10,
                 max_prompt_len=48, num_steps=10, vision_pool_factor=1,
                 vision_num_layers=None, cache_frames=1, use_fp8=False,
                 hardware=None, fp8_layout=None, state_prompt_mode="exact",
                 state_prompt_fixed_max_len=None, use_int8=False, **kwargs):
        if state_prompt_mode != "exact" or state_prompt_fixed_max_len is not None:
            raise NotImplementedError("NPU supports exact cached prompt buckets; fixed padded prompts are not implemented")
        if vision_num_layers not in (None, npu_pl.VIS_L):
            raise NotImplementedError("NPU inference requires the complete vision tower")
        if int(chunk_size) <= 0 or int(num_steps) <= 0:
            raise ValueError("chunk_size and num_steps must be positive")
        from flash_rt.npu.core import device
        device.ensure_npu()
        from flash_rt.npu.core.native_kernels import DecoderRope, GatedAdaRms
        self._decoder_rope = DecoderRope()
        self._ada_kernel = GatedAdaRms()
        import ctypes
        import torch
        try:
            query_soc = self._decoder_rope.library.flashrt_npu_soc_version
        except AttributeError as exc:
            raise ImportError("Rebuild the NPU library with scripts/npu/build.sh") from exc
        query_soc.restype = ctypes.c_char_p
        compiled_soc = query_soc().decode()
        running_soc = device.device_name(torch.npu.current_device())
        if compiled_soc != running_soc:
            raise RuntimeError(f"NPU library targets {compiled_soc}, but the current device is {running_soc}")
        ckpt = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir = ckpt
        if num_views not in (2, 3):
            raise ValueError("num_views must be 2 or 3")
        self.num_views = num_views
        self.chunk_size = int(chunk_size)
        self.num_steps = int(num_steps)
        self.use_int8 = bool(use_int8)
        self._int8_calibrated = False
        if use_fp8:
            logger.warning("NPU backend has no FP8 tier; ignoring use_fp8=True")
        if vision_pool_factor != 1:
            raise NotImplementedError("vision_pool_factor != 1 not ported")
        if cache_frames != 1:
            raise NotImplementedError("temporal K/V caching not ported yet")

        from flash_rt.npu.core.native_kernels import EncoderRope, EulerUpdate
        self._encoder_rope = EncoderRope()
        self._euler_kernel = EulerUpdate()

        # weights: fp32 CPU reference kept for gating; BF16/NPU for serving
        self.wref = npu_pl.load_weights_fp32(ckpt / "model.safetensors")
        self.wb = npu_fast.make_vision_padded_weights(
            {k: self._to_serving(k, v) for k, v in self.wref.items()})

        # per-step time conditioning (computed once, fp32/npu)
        self.conds = []
        for s in range(self.num_steps):
            te = npu_pl.time_embedding(1.0 - s / self.num_steps,
                                       npu_pl.DEC_D).to("npu")
            c = F.silu(F.linear(te, self.wb["time_mlp_in.weight"],
                                self.wb["time_mlp_in.bias"]))
            c = F.silu(F.linear(c, self.wb["time_mlp_out.weight"],
                                self.wb["time_mlp_out.bias"]))
            self.conds.append(c)

        # fast serving path: merged decoder GEMMs + precomputed AdaRMS styles,
        # and fused encoder norms (npu_rms_norm / npu_add_rms_norm; encoder
        # QKV merge is NOT used — measured slower).
        self.wfast = npu_fast.make_fast_weights(self.wb)
        self.styles = npu_fast.make_styles_opt(self.wb, self.conds)
        self.wfe = npu_fast.make_encoder_opt_weights(self.wb)
        self._encoder_bf16 = dict(self.wfe)

        # observation normalisation stats (openpi assets or lerobot meta)
        self.norm_stats = load_norm_stats(
            pi05_candidates(ckpt), checkpoint_dir=ckpt, strict=True)

        # prompt-length → captured runner cache
        self._runners = {}
        self.current_prompt_len = 0
        self._lat = []
        self._current_prompt_text = None
        self._current_state = None
        self._native_io = True

    # ── weight conversion ──────────────────────────────────────────────
    @staticmethod
    def _to_serving(key: str, t: torch.Tensor) -> torch.Tensor:
        """Keep serving GEMMs BF16; retain explicit FP32 setup operations."""
        if key == "action_in_proj.weight":
            return t.to(torch.bfloat16).to(torch.float32).to("npu")
        if ".vision_model." in key and ("layer_norm" in key or "post_layernorm" in key):
            # Native vision normalization always consumes BF16 parameters.
            return t.to(torch.bfloat16).to("npu")
        if ".vision_model.encoder.layers." in key and key.endswith(".bias"):
            # Preserve BF16 bias values; native biased GEMM consumes FP32 bias.
            return t.to(torch.bfloat16).to(torch.float32).to("npu")
        if (".dense." in key or "layer_norm" in key or key.endswith(".bias")
                or "patch_embedding" in key):
            return t.to(torch.float32).to("npu")
        return t.to(torch.bfloat16).to("npu")

    # ── prompt handling ────────────────────────────────────────────────
    def set_prompt(self, prompt_text: str, state: Optional[np.ndarray] = None):
        """Tokenise the (state-conditional) prompt and build the captured graph."""
        tokens = self._tokenize(prompt_text, state)
        self._current_tokens = tokens
        self._set_lang(len(tokens), tokens)
        self.current_prompt_len = len(tokens)
        self._current_prompt_text = prompt_text
        self._current_state = None if state is None else np.asarray(state).copy()

    def _tokenize(self, prompt_text: str, state) -> list:
        try:
            from openpi.models.tokenizer import PaligemmaTokenizer
            max_len = 200 if state is not None else 48
            t, m = PaligemmaTokenizer(max_len=max_len).tokenize(
                prompt_text, state=state)
            return [int(x) for x in t[: int(m.sum())]]
        except Exception:
            from flash_rt.utils.paligemma_tokenizer import (
                load_paligemma_sentencepiece)
            sp = load_paligemma_sentencepiece()
            if state is None:
                # 108 is PaliGemma's '\n' prompt-end separator (openpi layout)
                return [sp.bos_id()] + sp.Encode(prompt_text) + [108]
            from flash_rt.core.utils.pi05_prompt import format_pi05_prompt
            return sp.Encode(format_pi05_prompt(prompt_text, state),
                             add_bos=True)

    def _set_lang(self, lang_len: int, tokens: list):
        ids = torch.tensor(tokens, dtype=torch.long, device="npu")
        emb = F.embedding(ids, self.wb[npu_pl._LM]) * float(npu_pl.ENC_D ** 0.5)
        runner = self._runners.get(lang_len)
        if runner is None:
            runner = _CapturedRunner(self.wb, self.num_views, lang_len,
                                     self.chunk_size, self.num_steps,
                                     self.conds, wfast=self.wfast,
                                     styles=self.styles, wfe=self.wfe,
                                     norm_stats=self.norm_stats if self._native_io else None,
                                     decoder_rope=self._decoder_rope, ada_kernel=self._ada_kernel,
                                     encoder_rope=self._encoder_rope, euler_kernel=self._euler_kernel,
                                     cache_only=True, defer_residual=True, paged_attention=True)
            self._runners[lang_len] = runner
        from contextlib import nullcontext
        with runner.native.lock if runner.native is not None else nullcontext():
            runner.lang.copy_(emb)
            # Prompt updates are setup work. Finish the current-stream upload
            # before a cached graph consumes it on its dedicated replay stream.
            if runner._graph is not None:
                torch.npu.current_stream().synchronize()
        if runner._graph is None:
            runner.capture()

    # ── observation contract ───────────────────────────────────────────
    def _stack_observation(self, observation: dict) -> torch.Tensor:
        keys = ("image", "wrist_image", "wrist_image_right")
        if "images" in observation:
            imgs = list(observation["images"])
        else:
            imgs = [observation[k] for k in keys[: self.num_views]]
        if len(imgs) != self.num_views:
            raise ValueError(
                f"observation must carry {self.num_views} views; got {len(imgs)}")
        out = []
        for img in imgs:
            a = np.asarray(img)
            if a.dtype != np.uint8 or a.shape != (224, 224, 3):
                raise ValueError(
                    "each view must be a uint8 (224,224,3) array")
            f = a.astype(np.float32) / 127.5 - 1.0          # [-1,1]
            out.append(torch.from_numpy(f).permute(2, 0, 1))  # CHW
        return torch.stack(out).to("npu")                    # (nv,3,224,224)

    # ── serving API (mirrors the AMD frontend) ─────────────────────────
    def calibrate(self, observations=None, *, percentile=99.9):
        """Freeze encoder INT8 scales from real camera observations.

        Samples contain camera arrays and may supply ``prompt``, normalized
        ``state`` and ``noise``. Missing prompts use the current prompt;
        missing diffusion noise uses a reproducible model noise draw.
        Call only while this frontend is idle, before serving requests.
        """
        if not self.use_int8:
            return None
        if observations is None:
            raise ValueError("INT8 calibration requires real observations")
        import hashlib
        from flash_rt.npu.models.pi05.quantization import calibrate_encoder
        from flash_rt.npu.core.native_kernels import RowQuantizer, GeluMulQuant, RmsRowQuant
        quantizer = RowQuantizer()
        fingerprints = []
        rng = np.random.default_rng(0)

        def make_runner(sample, weights):
            prompt = sample.get("prompt", self._current_prompt_text)
            if prompt is None:
                raise ValueError("set a prompt or include prompt in every sample")
            tokens = self._tokenize(str(prompt), sample.get("state", self._current_state))
            images = self._stack_observation(sample)
            noise = np.asarray(sample.get("noise", rng.standard_normal(
                (self.chunk_size, npu_pl.ACTION_DIM))), dtype=np.float32)
            if noise.shape != (self.chunk_size, npu_pl.ACTION_DIM) or not np.isfinite(noise).all():
                raise ValueError("invalid calibration diffusion noise")
            digest = hashlib.sha256()
            digest.update(images.cpu().numpy().tobytes())
            digest.update(np.asarray(tokens, dtype=np.int64).tobytes())
            digest.update(noise.tobytes())
            fingerprints.append(digest.hexdigest())
            runner = _CapturedRunner(self.wb, self.num_views, len(tokens),
                self.chunk_size, self.num_steps, self.conds, wfast=self.wfast,
                styles=self.styles, wfe=weights, decoder_rope=self._decoder_rope,
                ada_kernel=self._ada_kernel, encoder_rope=self._encoder_rope)
            ids = torch.tensor(tokens, device="npu", dtype=torch.long)
            runner.lang.copy_(F.embedding(ids, self.wb[npu_pl._LM]) * npu_pl.ENC_D ** 0.5)
            runner.fill(images, torch.tensor(noise, device="npu"))
            return runner

        bound, report = calibrate_encoder(self._encoder_bf16, observations,
            make_runner, self.num_views * npu_pl.VIS_TOKENS_PER_VIEW,
            percentile, quantizer, GeluMulQuant(), RmsRowQuant(), attention_output_quant=True)
        report["sample_sha256"] = fingerprints
        self.wfe = bound
        self._calibration_report = report
        self._int8_calibrated = True
        self._runners.clear()
        if self._current_prompt_text is not None:
            self._set_lang(len(self._current_tokens), self._current_tokens)
        return report

    def calibrate_with_real_data(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)

    def warm_state_prompt_buckets(self, prompt_text, states, sample_observation):
        lens = []
        for s in states:
            self.set_prompt(prompt_text, state=s)
            lens.append(self.current_prompt_len)
        return lens

    def infer(self, observation: dict, noise: Optional[np.ndarray] = None,
              debug: bool = False):
        if self.use_int8 and not self._int8_calibrated:
            raise RuntimeError("call calibrate_with_real_data before INT8 inference")
        runner = self._runners.get(self.current_prompt_len)
        if runner is None or runner._graph is None:
            raise RuntimeError("call set_prompt(...) before infer()")
        if runner.native is not None:
            return self._infer_native(runner, observation, noise)
        imgs = self._stack_observation(observation)
        noise_t = None
        if noise is not None:
            noise_t = torch.tensor(np.asarray(noise, dtype=np.float32),
                                   device="npu")
        t0 = torch.npu.Event(enable_timing=True)
        t1 = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        runner.fill(imgs, noise_t)
        t0.record()
        runner.replay()
        t1.record()
        torch.npu.synchronize()
        self._lat.append(t0.elapsed_time(t1))
        raw = runner.out.cpu().numpy()                     # (chunk,32) fp32
        unnorm = unnormalize_actions(raw, self.norm_stats)
        robot = unnorm[:, :LIBERO_ACTION_DIM]
        return {"actions": robot, "raw_actions": raw}

    def _infer_native(self, runner, observation, noise):
        images = observation.get("images")
        if images is None:
            keys = ("image", "wrist_image", "wrist_image_right")
            images = [observation[key] for key in keys[:self.num_views]]
        if len(images) != self.num_views:
            raise ValueError(f"expected {self.num_views} camera views")
        with runner.native.lock:
            for index, image in enumerate(images):
                image = np.asarray(image)
                if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                    raise ValueError("each view must be a uint8 (224,224,3) array")
                np.copyto(runner.host_images.array[index], image)
            if noise is None:
                runner.host_noise.array[:] = np.random.standard_normal(runner.host_noise.shape)
            else:
                noise = np.asarray(noise, dtype=np.float32)
                if noise.shape != runner.host_noise.shape:
                    raise ValueError(f"noise must have shape {runner.host_noise.shape}")
                np.copyto(runner.host_noise.array, noise)
            runner.native.execute()
            self._lat.append(runner.native.last_replay_ms)
            # Returned arrays own their storage and survive the next replay.
            return {"actions": runner.host_robot.array.copy(),
                    "raw_actions": runner.host_raw.array.copy()}

    # ── reports / precision ────────────────────────────────────────────
    def reference_actions(self, observation: dict, noise: Optional[np.ndarray]):
        """fp32 CPU reference over the same observation/noise (gating arm)."""
        dev = "cpu"
        imgs = self._stack_observation(observation).to(dev)
        tokens = self._runner_tokens()
        ids = torch.tensor(tokens, dtype=torch.long)
        noise_t = None
        if noise is not None:
            noise_t = torch.tensor(np.asarray(noise, dtype=np.float32))
        a = npu_pl.sample(imgs, ids, noise_t, self.wref,
                          num_steps=self.num_steps)
        return a.numpy()

    def _runner_tokens(self):
        return self._current_tokens

    def get_latency_stats(self) -> dict:
        lat = np.array(self._lat[-500:], dtype=np.float64)
        if lat.size == 0:
            return {}
        lat = np.sort(lat)
        n = lat.size
        return {
            "count": int(n), "mean_ms": float(lat.mean()),
            "std_ms": float(lat.std()), "min_ms": float(lat.min()),
            "max_ms": float(lat.max()), "p50_ms": float(lat[n // 2]),
            "p95_ms": float(lat[min(n - 1, int(n * 0.95))]),
            "hz": float(1000.0 / lat.mean()),
        }

    @property
    def precision_spec(self):
        from flash_rt.core.precision_spec import ModelPrecisionSpec, PrecisionSpec
        from flash_rt.npu.core.linear import StaticRowInt8Weight
        if not self._int8_calibrated:
            return ModelPrecisionSpec(source="manual")
        spec = ModelPrecisionSpec(source="calibration")
        report = self._calibration_report
        for key, weight in self.wfe.items():
            if not isinstance(weight, StaticRowInt8Weight):
                continue
            if (key.startswith(f"{npu_pl._EP}.{npu_pl.ENC_L - 1}.")
                    and (".mlp." in key or ".self_attn.o_proj." in key)):
                continue  # These calibrated bindings are dead in cache-only serving.
            spec.weight_specs[key] = PrecisionSpec(dtype="int8", granularity="per_channel",
                axis=0, scale_source="manual", scale=weight.weight_scales.cpu().numpy().copy())
            scales = weight.activation_scales.cpu().numpy()
            metadata = dict(dtype="int8", scale_source="calibration",
                calibration_method=report["method"], calibration_samples=report["samples"],
                calibration_percentile=report["percentile"])
            spec.activation_specs[key + ".image_tokens"] = PrecisionSpec(
                granularity="per_channel", axis=0, scale=scales[:-1].copy(), **metadata)
            spec.activation_specs[key + ".language_tokens"] = PrecisionSpec(
                scale=scales[-1:].copy(), **metadata)
        spec.validate()
        return spec

    # RTX-only / not-yet-ported surfaces fail loudly
    def set_rl_mode(self, *a, **k):
        raise NotImplementedError("set_rl_mode is not ported to the NPU backend")

    def set_batched_mode(self, *a, **k):
        raise NotImplementedError("batched serving is not ported to the NPU backend")
