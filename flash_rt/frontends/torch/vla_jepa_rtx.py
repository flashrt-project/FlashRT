"""FlashRT — LeRobot VLA-JEPA frontend for the RTX (SM120) backend.

Wraps :class:`flash_rt.models.vla_jepa.pipeline_rtx.VlaJepaRtxPipeline` with the
model-specific weight/geometry/preprocessing layer:

  * prompt construction exactly as ``lerobot.policies.vla_jepa.Qwen3VLInterface``
    (special ``<|action_i|>`` / ``<|embodied_action|>`` token expansion + chat
    template), tokenized once per task and staged into static buffers;
  * the reference image path (uint8 frame -> ``resize_images_to`` area resize ->
    Qwen3-VL processor resize/normalize -> patchify);
  * state normalization from the checkpoint's own normalizer statistics;
  * action post-processing (clip -> gripper pre-snap -> MIN_MAX unnormalize ->
    gripper binarize) exactly as ``lerobot``'s postprocessor pipeline;
  * CUDA-Graph capture of the whole steady-state pipeline.

Usage::

    fe = VlaJepaTorchFrontendRtx(ckpt, qwen_dir=..., device='cuda:0')
    fe.set_prompt("put the white mug on the plate", max_pixels=None)
    actions = fe.infer(frames_u8, state_raw)     # [7, 7] float32 numpy
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch

from flash_rt.frontends.torch._vla_jepa_rtx_spec import (
    DEFAULT_LLM_ATTENTION,
    LLM_ATTENTION_ROUTES,
)

PROMPT_TEMPLATE = ("Your task is {instruction}. Infer the temporal dynamics from "
                   "frames {actions} and produce the corresponding policy actions {e_actions}.")
SPECIAL_ACTION_TOKEN = "<|action_{}|>"
EMBODIED_ACTION_TOKEN = "<|embodied_action|>"

# Kernel symbols the frontend calls directly. Checked once before capture so a
# build without -DFLASHRT_ENABLE_VLA_JEPA=ON fails with an actionable message
# instead of an AttributeError at the first launch.
_REQUIRED_KERNELS = (
    "VlaJepaKernelPolicy",
    "vla_frames_area_resize",
    "vla_frames_bicubic_resize",
    "vla_frames_bicubic_patchify",
    "vla_frames_patchify",
    "vla_state_normalize",
    "vla_dit_step_rows",
    "vla_action_postprocess",
    "embedding_lookup_bf16",
    "rms_norm",
    # Model-scoped <128,128,128> W4A16 twins; the shared default tile is
    # unchanged for every other model.
    "fp4_w4a16_gemm_sm120_bf16out_k128",
    "fp4_w4a16_gemm_residual_sm120_bf16out_k128",
    "fp4_w4a16_gemm_sm120_bf16out_bias_k128",
)


def _require_kernels(fvk) -> None:
    missing = [name for name in _REQUIRED_KERNELS if not hasattr(fvk, name)]
    if missing:
        raise RuntimeError(
            "VLA-JEPA RTX kernels are missing from flash_rt_kernels: "
            f"{missing[:4]}{'...' if len(missing) > 4 else ''}. "
            "Rebuild with -DFLASHRT_ENABLE_VLA_JEPA=ON on GPU_ARCH=120/121 "
            "(and -DFLASHRT_BUILD_QWEN3_VL=ON for the Qwen3-VL module).")


class VlaJepaTorchFrontendRtx:
    """VLA-JEPA (Qwen3-VL-2B + DiT-B) inference frontend, RTX SM120."""

    def __init__(self, checkpoint: str, *, qwen_dir: str | None = None,
                 action_ckpt: str | None = None, device: str = "cuda:0",
                 num_views: int = 2, seq_len: int = 256, num_image_tokens: int = 128,
                 chunk_size: int = 7, action_dim: int = 7, state_dim: int = 8,
                 num_cond_tokens: int = 32, num_action_tokens_per_timestep: int = 8,
                 num_video_frames: int = 8, jepa_tubelet_size: int = 2,
                 num_inference_timesteps: int = 4, timestep_buckets: int = 1000,
                 resize_images_to: tuple[int, int] | None = (224, 224),
                 use_cuda_graph: bool = True, gemm: str = "cublaslt",
                 llm_precision: str = "nvfp4", nvfp4_sites: str = "all",
                 nvfp4_act_mode: str = "dynamic",
                 llm_attention: str = DEFAULT_LLM_ATTENTION,
                 dit_precision: str = "nvfp4",
                 verbose: bool = False) -> None:
        from flash_rt.models.vla_jepa import VlaJepaRtxPipeline

        if llm_precision not in ("bf16", "fp8", "fp8_residual_safe", "nvfp4"):
            raise ValueError(
                "llm_precision must be 'bf16', 'fp8', 'fp8_residual_safe' or "
                f"'nvfp4', got {llm_precision!r}")
        if dit_precision not in ("bf16", "nvfp4"):
            raise ValueError(
                f"dit_precision must be 'bf16' or 'nvfp4', got {dit_precision!r}")
        if llm_attention not in LLM_ATTENTION_ROUTES:
            raise ValueError(
                f"llm_attention must be one of {LLM_ATTENTION_ROUTES}, "
                f"got {llm_attention!r}")
        self.checkpoint = str(checkpoint)
        self.qwen_dir = str(qwen_dir or os.path.join(self.checkpoint, "flashrt_qwen3vl"))
        self.action_ckpt = str(action_ckpt or self.checkpoint)
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VlaJepaTorchFrontendRtx requires a CUDA device; the RTX "
                "SM120/SM121 native path has no CPU/MPS fallback")
        cap = torch.cuda.get_device_capability(torch.device(device))
        if cap < (12, 0):
            raise RuntimeError(
                f"VlaJepaTorchFrontendRtx requires SM120/SM121 (Blackwell), "
                f"got compute capability {cap[0]}.{cap[1]}; this frontend has "
                "no SM89/Thor/AMD fallback")
        self.num_views = int(num_views)
        self.seq_len = int(seq_len)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.num_cond = int(num_cond_tokens)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.timestep_buckets = int(timestep_buckets)
        self.resize_images_to = resize_images_to
        self.use_cuda_graph = bool(use_cuda_graph)

        cfg = json.load(open(os.path.join(self.checkpoint, "config.json")))
        self.cfg = cfg
        self.num_action_tokens_per_timestep = int(
            cfg.get("num_action_tokens_per_timestep", num_action_tokens_per_timestep))
        self.num_video_frames = int(cfg.get("num_video_frames", num_video_frames))
        self.jepa_tubelet_size = int(cfg.get("jepa_tubelet_size", jepa_tubelet_size))
        self.image_keys = [k for k, v in cfg["input_features"].items() if v["type"] == "VISUAL"]
        self.gripper_dim = int(cfg.get("gripper_dim", 6))
        self.gripper_threshold = float(cfg.get("gripper_threshold", 0.5))
        self.clip_normalized_actions = bool(cfg.get("clip_normalized_actions", True))
        self.pre_snap_gripper_action = bool(cfg.get("pre_snap_gripper_action", True))
        self.binarize_gripper_action = bool(cfg.get("binarize_gripper_action", True))

        from flash_rt import flash_rt_kernels as _fvk
        _require_kernels(_fvk)
        self._kernel_policy = _fvk.VlaJepaKernelPolicy()

        # The frontend owns checkpoint IO; the pipeline receives the loaded
        # tensors in its bf16 kernel layout (see _vla_jepa_rtx_spec).
        from flash_rt.frontends.torch._vla_jepa_rtx_spec import load_weights
        qwen_cfg = json.load(open(os.path.join(self.qwen_dir, "config.json")))
        weights = self._kernel_policy.run(load_weights,
            self.qwen_dir, self.action_ckpt, device=device,
            num_llm_layers=int(qwen_cfg["text_config"]["num_hidden_layers"]),
            num_dit_layers=int(cfg.get("action_num_layers", 16)),
            num_inference_steps=self.num_inference_timesteps)

        self.pipe = self._kernel_policy.run(VlaJepaRtxPipeline,
            self.qwen_dir, action_ckpt=self.action_ckpt, weights=weights,
            device=device,
            num_views=self.num_views, seq_len=self.seq_len,
            num_image_tokens=num_image_tokens, chunk_size=chunk_size,
            action_dim=action_dim, state_dim=state_dim, num_cond_tokens=num_cond_tokens,
            num_inference_steps=self.num_inference_timesteps,
            timestep_buckets=self.timestep_buckets, gemm=gemm,
            llm_precision=llm_precision, nvfp4_sites=nvfp4_sites,
            nvfp4_act_mode=nvfp4_act_mode,
            llm_attention=llm_attention,
            dit_precision=dit_precision,
            verbose=verbose)

        self._build_tokenizer()
        self._load_norm_stats()
        # hand the checkpoint's own processor/statistics to the pipeline so the
        # whole raw-frame -> chunk boundary can live inside one CUDA Graph
        self.pipe.processor = self.processor
        self.pipe.resize_images_to = self.resize_images_to or (224, 224)
        self.pipe.state_mean = self.state_mean
        self.pipe.state_std = self.state_std
        self.pipe.state_eps = self.state_eps
        self.pipe.action_min = self.action_min
        self.pipe.action_max = self.action_max
        self.pipe.gripper_dim = self.gripper_dim
        self.pipe.gripper_threshold = self.gripper_threshold
        self.pipe.nvfp4_sites = nvfp4_sites
        self.pipe.clip_normalized_actions = self.clip_normalized_actions
        self.pipe.pre_snap_gripper_action = self.pre_snap_gripper_action
        self.pipe.binarize_gripper_action = self.binarize_gripper_action
        self._prompt_ids: np.ndarray | None = None
        self._graph = None
        # The captured critical path runs at high priority so the forked
        # off-path branches (DeepStack merger MLPs) only fill idle SMs; the
        # graph's kernel nodes keep the capture streams' priorities.
        self._graph_stream = torch.cuda.Stream(device=device, priority=-1)
        # Pinned host staging for the per-decision boundary. The captured graph
        # owns the H2D/D2H copies from/to these buffers, so replay has no
        # per-call allocation, pageable copy, or CPU gather on the critical
        # path; staging a decision is a flat memcpy into pinned memory.
        # The device frame buffer is CHW, so the caller's HWC view is staged
        # with a permute whose source strides match the cache's memory order.
        self._pin_frames = torch.empty(self.num_views, 3, 256, 256,
                                       dtype=torch.uint8, pin_memory=True)
        self._pin_state = torch.empty(self.state_dim, dtype=torch.float32,
                                      pin_memory=True)
        self._initialize_noise()
        self._pin_out = torch.empty(self.chunk_size, self.action_dim,
                                    dtype=torch.float32, pin_memory=True)
        self.latency_records: list[float] = []

    # ───────────────────────── prompt / tokenizer ─────────────────────────

    def _build_tokenizer(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.qwen_dir)
        tok = self.processor.tokenizer
        max_action_tokens = self.chunk_size * 4
        self.action_tokens = []
        self.action_token_ids = []
        for i in range(max_action_tokens):
            piece = SPECIAL_ACTION_TOKEN.format(i)
            self.action_tokens.append(piece)
            if piece not in tok.get_vocab():
                tok.add_tokens([piece], special_tokens=True)
            self.action_token_ids.append(tok.convert_tokens_to_ids(piece))
        if EMBODIED_ACTION_TOKEN not in tok.get_vocab():
            tok.add_tokens([EMBODIED_ACTION_TOKEN], special_tokens=True)
        self.embodied_token_id = tok.convert_tokens_to_ids(EMBODIED_ACTION_TOKEN)
        num_action_prompt_steps = self.num_video_frames // self.jepa_tubelet_size - 1
        self.replace_prompt = "".join(
            t * self.num_action_tokens_per_timestep
            for t in self.action_tokens[:num_action_prompt_steps])
        self.embodied_replace_prompt = EMBODIED_ACTION_TOKEN * self.num_cond

    def build_prompt_ids(self, instruction: str, images: list[torch.Tensor]) -> dict[str, Any]:
        """Tokenize the VLA-JEPA prompt for one instruction (once per task)."""
        text = PROMPT_TEMPLATE.format(
            instruction=instruction, actions=self.replace_prompt,
            e_actions=self.embodied_replace_prompt)
        messages = [[{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": text}],
        }]]
        out = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            processor_kwargs={"padding": False, "return_tensors": "pt",
                              "do_rescale": False})
        return out

    def set_prompt(self, instruction: str, observation_images_u8: np.ndarray) -> None:
        """Stage the prompt for one task. ``observation_images_u8`` is a
        representative ``[V, H, W, 3]`` uint8 observation (used only for the
        image geometry, which is resolution-fixed)."""
        imgs = [torch.from_numpy(np.ascontiguousarray(observation_images_u8[v]))
                .permute(2, 0, 1).float().div(255.0) for v in range(self.num_views)]
        if self.resize_images_to is not None:
            h, w = self.resize_images_to
            imgs = [torch.nn.functional.interpolate(im.float()[None], size=(h, w),
                                                    mode="area")[0] for im in imgs]
        out = self.build_prompt_ids(instruction, imgs)
        ids = out["input_ids"][0].cpu().numpy()
        grid = out["image_grid_thw"].cpu().numpy()
        if ids.shape[0] > self.seq_len:
            raise ValueError(f"prompt length {ids.shape[0]} exceeds seq_len {self.seq_len}")
        self.pipe.stage_prompt(ids, grid)
        self.pipe.set_embodied_positions(self.embodied_token_id)
        # The captured graph bakes in the (static) buffer addresses and the
        # per-prompt geometry. Every task prompt has the same structure (same
        # image spans, same padded sequence length), so the graph stays valid;
        # only a geometry change forces a recapture.
        geom = (tuple(self.pipe.vision_spans), self.pipe.seq_len, self.pipe._tail_row0)
        if getattr(self, "_graph_geom", None) != geom:
            self._graph = None
            self._graph_geom = geom
        self._prompt_ids = ids
        self._image_grid = grid

    @torch.no_grad()
    def calibrate(self, samples, *, warm: int = 2, verbose: bool = False) -> dict:
        """Accumulate real-observation activation statistics before capture.

        ``samples`` is an iterable of ``(frames_u8, state_raw)`` pairs using the
        same shapes as :meth:`infer`. Call :meth:`set_prompt` first so the
        vision spans and tokenized prompt are staged. The default dynamic
        per-row activation scales do not need this; it is required for the
        ``static``/``clipped`` activation modes and for calibrated per-layer
        selection. Invalidates any captured graph.
        """
        observations = [{"frames": np.asarray(frames), "state": np.asarray(state)}
                        for frames, state in samples]
        if not observations:
            raise ValueError("calibrate() needs at least one (frames, state) sample")
        stats = self.pipe.calibrate_nvfp4(observations, warm=warm, verbose=verbose)
        self._graph = None
        return stats

    # ───────────────────────── normalization stats ─────────────────────────

    def _load_norm_stats(self) -> None:
        from safetensors.torch import load_file

        pre = load_file(os.path.join(self.checkpoint,
                                     "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        post = load_file(os.path.join(self.checkpoint,
                                      "policy_postprocessor_step_2_unnormalizer_processor.safetensors"))
        st = pre["observation.state.mean"], pre["observation.state.std"]
        self.state_mean = st[0].float().cuda()
        self.state_std = st[1].float().cuda()
        self.state_eps = 1e-8
        a_min, a_max = post["action.min"], post["action.max"]
        self.action_min = a_min.float().cuda()
        self.action_max = a_max.float().cuda()

    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(state, dtype=np.float32)).cuda()
        return (t - self.state_mean) / (self.state_std + self.state_eps)

    # ───────────────────────── image preprocessing ─────────────────────────

    def preprocess_frames(self, frames_u8: np.ndarray) -> torch.Tensor:
        """uint8 [V, H, W, 3] -> patchified pixel_values [P, 1536] bf16 (device).

        Mirrors the reference: float/[0,1] -> area resize to ``resize_images_to``
        -> the Qwen3-VL image processor (smart resize + /0.5 mean-std) -> 16x16
        patchify with temporal_patch_size frames.
        """
        imgs = [torch.from_numpy(np.ascontiguousarray(frames_u8[v]))
                .permute(2, 0, 1).float().div(255.0) for v in range(self.num_views)]
        if self.resize_images_to is not None:
            h, w = self.resize_images_to
            imgs = [torch.nn.functional.interpolate(im.float()[None], size=(h, w),
                                                    mode="area")[0] for im in imgs]
        out = self.processor.image_processor(
            images=[im for im in imgs], return_tensors="pt", do_rescale=False,
            input_data_format="channels_first")
        return out["pixel_values"].to(torch.bfloat16).to(self.device)

    # ───────────────────────── post-processing ─────────────────────────

    def postprocess(self, actions_norm: torch.Tensor) -> torch.Tensor:
        a = actions_norm.float().clone()
        if self.clip_normalized_actions:
            a = a.clamp(-1.0, 1.0)
        if self.pre_snap_gripper_action and a.shape[-1] > self.gripper_dim:
            a[..., self.gripper_dim] = (a[..., self.gripper_dim] >= self.gripper_threshold).float()
        # lerobot MIN_MAX inverse (UnnormalizerProcessorStep): the forward map
        # is 2*(x-min)/(max-min) - 1, so the inverse is (x+1)/2*(max-min)+min.
        a = (a + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min
        if self.binarize_gripper_action and a.shape[-1] > self.gripper_dim:
            a[..., self.gripper_dim] = 1.0 - 2.0 * (a[..., self.gripper_dim] > self.gripper_threshold).float()
        return a

    # ───────────────────────── inference ─────────────────────────

    def _upload_pinned(self) -> None:
        """Pinned host -> device staging (eager path only; the captured graph
        records these same copies as its leading/final memcpy nodes)."""
        self.pipe.raw_frames.copy_(self._pin_frames, non_blocking=True)
        self.pipe.raw_state.copy_(self._pin_state, non_blocking=True)
        self.pipe.actions.copy_(self._pin_noise, non_blocking=True)

    def set_observation(self, frames_u8: np.ndarray, state_raw: np.ndarray) -> None:
        # ``torch.from_numpy`` keeps the caller's strides, so a CHW-ordered
        # HWC view permutes to a contiguous CHW buffer at memory speed;
        # numpy's ascontiguousarray would instead pay a ~0.44 ms CPU gather
        # per decision on this representation.
        self._pin_frames.copy_(torch.from_numpy(np.asarray(frames_u8)).permute(0, 3, 1, 2))
        self._pin_state.copy_(torch.from_numpy(np.asarray(state_raw, dtype=np.float32)))
        if not self.use_cuda_graph:
            self._upload_pinned()

    def _initialize_noise(self) -> None:
        self._pin_noise = torch.empty(self.chunk_size, self.action_dim,
                                      dtype=torch.bfloat16, pin_memory=True)
        self.set_noise()

    def set_noise(self, noise: np.ndarray | None = None) -> None:
        """Set reusable noise; None restores the fixed seed-0 Gaussian default."""
        if noise is None:
            noise = np.random.default_rng(0).standard_normal(
                (self.chunk_size, self.action_dim)).astype(np.float32)
        self._pin_noise.copy_(torch.from_numpy(np.asarray(noise, dtype=np.float32)))
        if not self.use_cuda_graph:
            self.pipe.actions.copy_(self._pin_noise, non_blocking=True)

    @torch.no_grad()
    def infer(self, frames_u8: np.ndarray | None = None, state_raw: np.ndarray | None = None,
              noise: np.ndarray | None = None, actions_norm: torch.Tensor | None = None) -> np.ndarray:
        _require_kernels(self.pipe._kernels())
        if frames_u8 is not None or state_raw is not None:
            self.set_observation(frames_u8, state_raw)
        # Capture (once) before staging the flow-matching start state: the
        # capture warmups run the Euler loop and would otherwise leave the
        # action buffer holding their accumulated result.
        if self.use_cuda_graph:
            self.record_graph()
        if noise is not None:
            self.set_noise(noise)
        if actions_norm is not None:
            # The graph starts by copying the pinned start state into the
            # action buffer, so a normalized start state takes the same path.
            self._pin_noise.copy_(actions_norm.to(torch.bfloat16))
            if not self.use_cuda_graph:
                self.pipe.actions.copy_(self._pin_noise, non_blocking=True)
        if self.use_cuda_graph:
            self._graph.replay()
        else:
            # Every decision starts from staged noise, including infer() calls
            # that reuse the previous observation. The Euler loop mutates actions.
            self.pipe.actions.copy_(self._pin_noise, non_blocking=True)
            self.pipe.forward()
        torch.cuda.synchronize()
        if self.use_cuda_graph:
            return self._pin_out.numpy().copy()
        return self.pipe.out_actions.cpu().numpy().copy()

    def record_graph(self, external_stream: int | None = None) -> None:
        if self._graph is not None:
            return
        s = self._graph_stream.cuda_stream if external_stream is None else external_stream
        self._upload_pinned()
        for _ in range(3):
            self.pipe.forward(stream=s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self._graph_stream):
            # Per-decision boundary transfers live in the graph: the replay
            # reads the pinned observation/noise and leaves the delivered
            # chunk in pinned memory. Addresses are fixed at capture time.
            self.pipe.raw_frames.copy_(self._pin_frames, non_blocking=True)
            self.pipe.raw_state.copy_(self._pin_state, non_blocking=True)
            self.pipe.actions.copy_(self._pin_noise, non_blocking=True)
            self.pipe.forward()
            self._pin_out.copy_(self.pipe.out_actions, non_blocking=True)
        self._graph = g

    def clear_graph(self) -> None:
        self._graph = None
