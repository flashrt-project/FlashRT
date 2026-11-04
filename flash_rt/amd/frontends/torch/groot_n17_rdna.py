"""GR00T N1.7 BF16 frontend for gfx1100, gfx1151 and gfx1201."""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from flash_rt.amd.hardware.capabilities_rdna import load_capabilities
from flash_rt.amd.models.groot_n17.pipeline_rdna import GrootN17PipelineRdna
from flash_rt.executors.torch_weights import MultiSafetensorsSource
from flash_rt.executors.weight_loader import WeightLoader
from flash_rt.models.groot_n17.embodiments import EMBODIMENT_TAG_TO_INDEX
from flash_rt.models.groot_n17.weight_spec_orin import ORIN_WEIGHT_SPEC
from flash_rt.models.groot_n17.mrope_table import (
    RopeConfig, apply_interleaved_mrope, build_position_ids_for_segments,
    compute_inv_freq,
)


def _make_transformer_ops(kernels, capabilities):
    """Select model-specific RDNA operator policies once during setup."""
    from flash_rt.amd.hardware.rdna.attention import attention_bf16
    from flash_rt.amd.models.groot_n17.pipeline_rdna import _TransformerOps

    # Same-device official BF16 action gates select the attention and weight
    # layout together. gfx1100/gfx1201 retain nn.Linear's physical N-by-K
    # layout and Torch dispatch; gfx1151 uses tiled long attention and the
    # library's first K-by-N heuristic. Timed algorithm selection and small-M
    # substitutions exceed the action gates. Other models keep their policy.
    policies = {
        "gfx1100": (None, True),
        "gfx1151": (attention_bf16, False),
        "gfx1201": (None, True),
    }
    attention, weight_transposed = policies[capabilities.gpu_arch]
    return _TransformerOps(kernels, long_attention=attention, gemm_autotune=False,
                           weight_transposed=weight_transposed)


def _process_observation(processor, observation, embodiment_tag, *, num_views,
                        image_history):
    """Run the official image, instruction and state transforms once.

    Input is the official batched observation dictionary with ``video``,
    ``state`` and ``language`` groups. No model forward is needed to prepare it.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not isinstance(observation, dict) or any(
            not isinstance(observation.get(key), dict) for key in ("video", "state", "language")):
        raise ValueError("observation requires video, state and language dictionaries")
    if len(observation["video"]) != num_views:
        raise ValueError(f"expected {num_views} camera streams")
    flat = {}
    for group in ("video", "state", "language"):
        for key, value in observation[group].items():
            if group == "language":
                # Policy observations carry B x language-history; inference
                # consumes the latest instruction for each batch item.
                value = np.asarray(value)
                if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 1:
                    raise ValueError("language must have shape (1, history)")
                flat[key] = value[:, -1].tolist()
            else:
                value = np.asarray(value)
                if group == "video":
                    if (value.ndim != 5 or value.shape[:2] != (1, image_history)
                            or value.shape[-1] != 3 or value.dtype != np.uint8):
                        raise ValueError("video must be uint8 (1, image_history, height, width, 3)")
                elif value.ndim != 3 or value.shape[:2] != (1, 1):
                    raise ValueError("state must have shape (1, 1, channels)")
                flat[f"{group}.{key}"] = value
    tag = EmbodimentTag.resolve(embodiment_tag)
    batch = processor.process_observation(flat, tag)
    return batch


def _prompt_aux(batch, image_token_id, *, device):
    """Build N1.7 M-RoPE from token segments, independent of the backbone."""
    ids = batch["input_ids"].detach().cpu()
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("GR00T N1.7 supports batch size one")
    mask = ids == image_token_id
    grids = batch["image_grid_thw"].detach().cpu()
    lengths, kinds, segments = [], [], []
    tokens = mask[0].tolist()
    offset, image = 0, 0
    while offset < len(tokens):
        visual = tokens[offset]
        end = offset + 1
        while end < len(tokens) and tokens[end] == visual:
            end += 1
        lengths.append(end - offset)
        kinds.append("image" if visual else "text")
        if visual:
            if image >= len(grids):
                raise ValueError("more image-token segments than image grids")
            segments.append(tuple(int(v) for v in grids[image]))
            image += 1
        else:
            segments.append(None)
        offset = end
    if image != len(grids):
        raise ValueError("image grids do not match the tokenized observation")
    config = RopeConfig()
    positions = build_position_ids_for_segments(
        segment_lengths=lengths, segment_kinds=kinds, segment_grids=segments,
        cfg=config, device=device)
    # The official BF16 policy casts inv_freq along with the model. Text
    # rotary then promotes that rounded buffer back to FP32 for the angles.
    inv = compute_inv_freq(config, device=device).bfloat16().float()
    freqs = positions[..., None].float() * inv
    freqs = apply_interleaved_mrope(freqs, config.mrope_section)
    angles = torch.cat((freqs, freqs), dim=-1).squeeze(0)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    return {"input_ids": ids, "visual_pos_masks": mask, "grid_thw": grids,
            "rope_cos": cos.unsqueeze(0), "rope_sin": sin.unsqueeze(0),
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"]}


def prepare_observation(processor, observation, embodiment_tag, *, num_views,
                        metadata=None):
    """Return normalized state and raw aux using the official eval processor.

    ``metadata`` is an optional previous result; it only avoids rebuilding
    M-RoPE when token IDs and image grids match. Pixels and state stay fresh.
    This function does not bind a prompt or execute model weights.
    """
    processor.eval()
    video = processor.modality_configs[embodiment_tag]["video"]
    batch = _process_observation(
        processor, observation, embodiment_tag, num_views=num_views,
        image_history=len(video.delta_indices))
    if (metadata is None
            or not torch.equal(batch["input_ids"], metadata["input_ids"])
            or not torch.equal(batch["image_grid_thw"], metadata["grid_thw"])):
        image_id = processor.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        metadata = _prompt_aux(batch, image_id, device="cpu")
    aux = {k: v for k, v in metadata.items() if k not in ("pixel_values", "attention_mask")}
    aux.update(pixel_values=batch["pixel_values"], attention_mask=batch["attention_mask"])
    return batch["state"], aux


def frames_from_observation(observation, camera_order):
    """Flatten batch-one video into (history * cameras, H, W, 3) uint8.

    Official order is time first, then cameras in checkpoint modality order.
    The temporal copies inside each vision patch are unrelated to this history.
    """
    videos = observation["video"]
    if set(videos) != set(camera_order) or not camera_order:
        raise ValueError("video keys must match the checkpoint camera order")
    values = [np.asarray(videos[key]) for key in camera_order]
    if any(v.ndim != 5 or v.shape[0] != 1 or v.shape[-1] != 3
           or v.dtype != np.uint8 or v.shape != values[0].shape for v in values):
        raise ValueError("video must be matching uint8 (1, history, H, W, 3) arrays")
    frames = np.stack([v[0] for v in values], axis=1)
    return frames.reshape(-1, *frames.shape[-3:])


def _host_frames(frames):
    """Accept the same tensor/ndarray/sequence forms as Orin's frames= API."""
    def array(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.uint8:
                raise ValueError("frames must be uint8 camera bytes")
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        raise ValueError("frames must contain tensors or numpy arrays")

    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValueError("frames must not be empty")
        frames = np.stack([array(frame) for frame in frames])
    elif isinstance(frames, (torch.Tensor, np.ndarray)):
        frames = array(frames)
    else:
        raise ValueError("frames must be a tensor, ndarray or ordered sequence, not a mapping")
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8 camera bytes")
    if frames.ndim != 4 or frames.shape[-1] != 3 or any(n == 0 for n in frames.shape):
        raise ValueError("frames must have shape (images, H, W, 3) with nonempty dimensions")
    return np.ascontiguousarray(frames)


def process_frames(processor, frames, expected_grid):
    """Prepare fresh pixels without re-tokenizing or executing model weights.

    Uses the checkpoint's official eval image transforms and HF image processor,
    including non-square resize, crop and smart-resize rules. It omits the
    lossless PIL round trips in process_observation. Processing is on the host;
    device frames are copied to host explicitly. No approximate GPU resize is
    substituted, and no CPU/transfer cost is hidden in the reported boundary.
    """
    frames = _host_frames(frames)
    if frames.shape[0] != expected_grid.shape[0]:
        raise ValueError("frames count must match every camera/history image from set_prompt")
    processor.eval()
    if processor.use_albumentations:
        from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
        images, _ = apply_with_replay(processor.eval_image_transform, frames)
    else:
        images = processor.eval_image_transform(
            torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0))[0]
        images = list(images)
    batch = processor.processor.image_processor(
        images=images, input_data_format="channels_first", return_tensors="pt")
    if not torch.equal(batch["image_grid_thw"].cpu(), expected_grid.cpu()):
        raise ValueError("frames changed image grid; call set_prompt with the new observation")
    return batch["pixel_values"]


class GrootN17TorchFrontendAmdRdna:
    """Full vision-to-action BF16 execution with a per-prompt HIP graph.

    Use explicit ``set_prompt(aux=...)``, normalized-state ``infer`` and
    ``denormalize_action``, as on the other GR00T N1.7 frontends. Each observation
    supplies normalized state and either unprojected ``pixel_values`` or the
    projected observation bundle used by CUDA/CDNA. Omitting ``aux`` explicitly
    reuses the last observation, following the existing N1.7 contract.
    """

    _PROMPT_KEYS = ("input_ids", "grid_thw", "visual_pos_masks", "rope_cos", "rope_sin")

    def __init__(self, checkpoint_path, *, num_views=2,
                 embodiment_tag="oxe_droid_relative_eef_relative_joint",
                 device="cuda:0", hardware=None, action_horizon=None,
                 use_fp8=False, use_backbone_graph=True, fuse_image_embeds=False):
        if use_fp8:
            raise ValueError("GR00T N1.7 on RDNA supports BF16 only")
        if not isinstance(fuse_image_embeds, bool):
            raise TypeError("fuse_image_embeds must be a bool")
        self._fuse_image_embeds = fuse_image_embeds
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("GR00T N1.7 RDNA requires PyTorch with ROCm")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("RDNA inference requires a ROCm CUDA device")
        self.checkpoint_path = Path(checkpoint_path)
        if not isinstance(num_views, int) or isinstance(num_views, bool) or num_views < 1:
            raise ValueError("num_views must be a positive integer")
        self.num_views, self.embodiment_tag = num_views, embodiment_tag
        mapping_path = self.checkpoint_path / "embodiment_id.json"
        mapping = (json.loads(mapping_path.read_text()) if mapping_path.exists()
                   else EMBODIMENT_TAG_TO_INDEX)
        if embodiment_tag not in mapping:
            raise ValueError(f"unknown embodiment_tag {embodiment_tag!r}")
        self._slot = int(mapping[embodiment_tag])
        processor_config = json.loads((self.checkpoint_path / "processor_config.json").read_text())
        video = processor_config["processor_kwargs"]["modality_configs"][embodiment_tag]["video"]
        self.image_history = len(video["delta_indices"])
        self.config = json.loads((self.checkpoint_path / "config.json").read_text())
        self._validate_config()
        self.action_horizon = (self.config["action_horizon"] if action_horizon is None
                               else action_horizon)
        if (not isinstance(self.action_horizon, int) or isinstance(self.action_horizon, bool)
                or not 1 <= self.action_horizon <= self.config["action_horizon"]):
            raise ValueError("action_horizon must be a positive integer within the checkpoint horizon")
        self.steps = int(self.config["num_inference_timesteps"])
        if not isinstance(use_backbone_graph, bool):
            raise TypeError("use_backbone_graph must be a bool")
        self.use_backbone_graph = use_backbone_graph
        if self.action_horizon < 1 or self.steps < 1:
            raise ValueError("action_horizon and denoising steps must be positive")
        self._graph = None
        self._graph_output = None
        self._runtime = None
        self._processor = None
        self._processed_prompt = None
        self.latency_records = []
        with torch.cuda.device(self.device):
            from flash_rt.amd import flash_rt_amd_kernels as kernels
            props = torch.cuda.get_device_properties(self.device)
            self._caps = load_capabilities(kernels, props.gcnArchName, hardware)
            self.hardware = self._caps.hardware
            self.ops = _make_transformer_ops(kernels, self._caps)
            from flash_rt.amd.hardware.rdna._native import kernels as require_native_ops
            require_native_ops()
            weights = self._load_weights()
            self.pipeline = GrootN17PipelineRdna(
                weights, self.config, self.ops, device=self.device,
                horizon=self.action_horizon, steps=self.steps)

    def _validate_config(self):
        expected = {"max_action_dim": 132, "max_state_dim": 132,
                    "backbone_embedding_dim": 2048, "select_layer": 16,
                    "use_alternate_vl_dit": True, "use_vlln": True,
                    "use_vl_self_attention": True, "add_pos_embed": True,
                    "apply_sincos_state_encoding": False, "exclude_state": False,
                    "reproject_vision": False, "use_future_tokens": False,
                    "hidden_size": 1024}
        for name, value in expected.items():
            if self.config.get(name) != value:
                raise ValueError(f"unsupported GR00T N1.7 config {name}={self.config.get(name)!r}; expected {value!r}")
        dit = self.config["diffusion_model_cfg"]
        for name, value in {"num_layers": 32, "num_attention_heads": 32,
                            "attention_head_dim": 48, "output_dim": 1024,
                            "interleave_self_attention": True, "norm_type": "ada_norm"}.items():
            if dit.get(name) != value:
                raise ValueError(f"unsupported GR00T N1.7 DiT {name}={dit.get(name)!r}")
        adapter = self.config["vl_self_attention_cfg"]
        for name, value in {"num_layers": 4, "num_attention_heads": 32,
                            "attention_head_dim": 64, "positional_embeddings": None}.items():
            if adapter.get(name) != value:
                raise ValueError(f"unsupported GR00T N1.7 VL adapter {name}={adapter.get(name)!r}")

    def _load_weights(self):
        shards = sorted(str(p) for p in self.checkpoint_path.glob("model*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model safetensors in {self.checkpoint_path}")
        weights = SimpleNamespace()
        # Slice the embodiment tables before uploading; avoid storing all slots.
        source = MultiSafetensorsSource(shards, device="cpu")
        WeightLoader(source=source, target=weights, spec=ORIN_WEIGHT_SPEC).run()
        for name, value in vars(weights).items():
            if name.startswith(("_st_enc_", "_ac_enc_", "_ac_dec_")):
                value = value[self._slot]
            if isinstance(value, torch.Tensor):
                value = value.to(self.device).contiguous()
                if value.ndim == 2 and name not in (
                        "_embed_tokens_w", "_vit_pos_embed", "_ah_pos_embed_w"):
                    value = self.ops.prepare_weight(value)
            elif isinstance(value, list):
                value = [self.ops.prepare_weight(v.to(self.device).contiguous())
                         if v.ndim == 2 else v.to(self.device).contiguous() for v in value]
            setattr(weights, name, value)
        if self.action_horizon > weights._ah_pos_embed_w.shape[0]:
            raise ValueError("action_horizon exceeds checkpoint position embedding capacity")
        return weights

    def set_hf_processor(self, processor):
        processor.eval()
        self._processor = processor
        self._processed_prompt = None

    def _hf_processor(self):
        if self._processor is None:
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
            self.set_hf_processor(Gr00tN1d7Processor.from_pretrained(self.checkpoint_path))
        return self._processor

    def prepare_observation(self, observation):
        """Prepare raw official observations without binding a prompt or running the model.

        Returns normalized state and an aux bundle for ``set_prompt`` / ``infer``.
        The instruction comes from ``observation["language"]``. Image processing,
        state normalization and prompt M-RoPE preparation stay outside the graph.
        """
        state, aux = prepare_observation(
            self._hf_processor(), observation, self.embodiment_tag,
            num_views=self.num_views, metadata=self._processed_prompt)
        self._processed_prompt = {k: v.clone() for k, v in aux.items()
                                  if k not in ("pixel_values", "attention_mask")}
        return state, aux

    def normalize_state(self, state_dict):
        self._hf_processor()
        from gr00t.data.embodiment_tags import EmbodimentTag
        tag = EmbodimentTag.resolve(self.embodiment_tag)
        values = {k.removeprefix("state."): np.asarray(v, dtype=np.float32).reshape(1, -1)
                  for k, v in state_dict.items()}
        normalized = self._processor.state_action_processor.apply_state(state=values, embodiment_tag=tag.value)
        # The checkpoint's modality ordering, not the caller's dictionary order.
        keys = self._processor.modality_configs[tag.value]["state"].modality_keys
        flat = np.concatenate([normalized[k] for k in keys], axis=-1)
        if flat.shape[-1] > 132:
            raise ValueError("state exceeds the checkpoint's 132 padded dimensions")
        flat = np.pad(flat, ((0, 0), (0, 132 - flat.shape[-1])))
        return torch.from_numpy(flat).reshape(1, 1, 132)

    def denormalize_action(self, action_normed, state_dict=None):
        self._hf_processor()
        from gr00t.data.embodiment_tags import EmbodimentTag
        states = {k.removeprefix("state."): np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
                  for k, v in (state_dict or {}).items()}
        if action_normed.ndim == 2:
            action_normed = action_normed.unsqueeze(0)
        decoded = self._processor.decode_action(
            action_normed.detach().float().cpu().numpy(),
            EmbodimentTag.resolve(self.embodiment_tag), states)
        return {k: torch.as_tensor(v).float() for k, v in decoded.items()}

    def _validate_prompt(self, aux):
        if not isinstance(aux, dict):
            raise TypeError("aux must be an observation dictionary")
        raw = self._fuse_image_embeds
        required = self._PROMPT_KEYS if raw else self._PROMPT_KEYS[1:]
        required += ("pixel_values",) if raw else ("pixel_features", "llm_input_embeds")
        missing = [k for k in required if k not in aux]
        if missing:
            raise ValueError(f"GR00T N1.7 aux is missing {missing}")
        if any(not isinstance(aux[k], torch.Tensor) for k in required):
            raise TypeError("aux inputs must be tensors")
        grid = aux["grid_thw"]
        if grid.shape != (self.num_views * self.image_history, 3):
            raise ValueError("grid_thw must describe every camera and image-history frame")
        if grid.dtype not in (torch.int32, torch.int64):
            raise ValueError("grid_thw must contain integers")
        if torch.any(grid <= 0) or torch.any(grid[:, 1:] % 2):
            raise ValueError("vision grids must be positive with even spatial dimensions")
        rows = int(grid.prod(-1).sum())
        mask = aux["visual_pos_masks"]
        if mask.ndim != 2 or mask.shape[0] != 1 or mask.dtype != torch.bool:
            raise ValueError("visual_pos_masks must be a boolean (1, tokens) tensor")
        tokens = mask.shape[1]
        if int(mask.sum()) != rows // 4 or int(mask.sum()) == tokens:
            raise ValueError("image token mask must match the grid and include language tokens")
        if raw:
            ids = aux["input_ids"]
            if ids.shape != mask.shape or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer (1, tokens) tensor")
            if bool(((ids < 0) | (ids >= self.pipeline.w._embed_tokens_w.shape[0])).any()):
                raise ValueError("input_ids contains an out-of-vocabulary token")
        for key in ("rope_cos", "rope_sin"):
            if aux[key].shape not in ((1, tokens, 128), (tokens, 128)):
                raise ValueError(f"{key} must contain one 128-wide row per LLM token")
        self._validate_attention_mask(aux, tokens)
        shapes = ({"pixel_values": (rows, 1536)} if raw else
                  {"pixel_features": (rows, 1024), "llm_input_embeds": (1, tokens, 2048)})
        for key, shape in shapes.items():
            if tuple(aux[key].shape) != shape or not aux[key].is_floating_point():
                raise ValueError(f"{key} must be a floating tensor with shape {shape}")
        return shapes, tokens

    @staticmethod
    def _validate_attention_mask(aux, tokens):
        if "attention_mask" in aux:
            mask = aux["attention_mask"]
            if not isinstance(mask, torch.Tensor) or mask.shape != (1, tokens) or not bool(mask.all()):
                raise ValueError("attention_mask must describe an unpadded batch of one")

    def _invalidate_graphs(self):
        self._graph = self._graph_output = None
        self._backbone_graph = self._dit_graph = self._dit_output = None

    @torch.no_grad()
    def set_prompt(self, *, aux: dict, prompt: str | None = None):
        """Bind one observation and prompt; raw or CUDA/CDNA projected aux.

        Raw aux uses token IDs as the instruction authority. Projected aux uses
        the caller's fused LLM embeddings. Explicit setup replacement invalidates
        every captured stage. Caller tensors are copied into owned buffers.
        """
        shapes, tokens = self._validate_prompt(aux)
        with torch.cuda.device(self.device):
            self._invalidate_graphs()
            self._metadata = {k: aux[k].detach().cpu().clone() for k in self._PROMPT_KEYS
                              if k in aux and (k != "input_ids" or self._fuse_image_embeds)}
            self.pipeline.bind_prompt(aux, fuse_image_embeds=self._fuse_image_embeds)
            self._inputs = {k: aux[k].to(self.device, torch.bfloat16).contiguous().clone()
                            for k in shapes}
            self._features = torch.empty((tokens, 2048), device=self.device, dtype=torch.bfloat16)
            self._have_features = False
            self._state = torch.zeros((1, 132), device=self.device, dtype=torch.bfloat16)
            self._noise = torch.zeros((self.pipeline.horizon, 132), device=self.device, dtype=torch.bfloat16)
            self._runtime = True
            self._prompt = prompt
            self.latency_records.clear()

    def _backbone(self):
        pixels = self._inputs.get("pixel_values", self._inputs.get("pixel_features"))
        self._features.copy_(self.pipeline.backbone(pixels, self._inputs.get("llm_input_embeds")))
        return self._features

    def _denoise(self):
        return self.pipeline.denoise(self._features, self._state, self._noise)

    def _forward(self):
        self._backbone()
        return self._denoise()

    def _capture(self, forward):
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        current.wait_stream(stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = forward()
        current.wait_stream(stream)
        return graph, result

    @staticmethod
    def _positive_integer(value, name, maximum=None):
        if (not isinstance(value, int) or isinstance(value, bool) or value < 1
                or (maximum is not None and value > maximum)):
            raise ValueError(f"{name} must be a positive integer" +
                             (f" no greater than {maximum}" if maximum else ""))
        return value

    def _configure_denoising(self, horizon, steps, buckets):
        pipeline = self.pipeline
        if (horizon, steps, buckets) == (pipeline.horizon, pipeline.steps, pipeline.num_timestep_buckets):
            return
        self._invalidate_graphs()
        pipeline.horizon, pipeline.steps, pipeline.num_timestep_buckets = horizon, steps, buckets
        pipeline._prepare_schedule()
        self._noise = torch.empty((horizon, 132), device=self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def infer(self, state_normalized, *, aux=None, frames=None, initial_noise=None,
              num_inference_timesteps=None, action_horizon=None,
              num_timestep_buckets=None, use_dit_graph=True, use_graph=None):
        """Return normalized FP32 ``(1, horizon, 132)`` actions.

        Fresh aux refreshes vision/backbone; omitted aux reuses the last
        observation. ``use_dit_graph`` controls denoising capture. Backbone
        capture is selected at construction with ``use_backbone_graph``.
        ``use_graph=False`` is the RDNA convenience for fully eager execution.
        When both stages use graphs, fresh observations use one combined graph.
        ``frames`` follows Orin's uint8 image API and needs fuse_image_embeds=True.
        Image preparation uses official host transforms outside the HIP graph.
        """
        if not isinstance(state_normalized, torch.Tensor) or not state_normalized.is_floating_point():
            raise TypeError("infer expects a normalized state tensor; use normalize_state")
        if self._runtime is None:
            raise RuntimeError("call set_prompt before infer")
        started = time.perf_counter()
        if frames is not None:
            if not self._fuse_image_embeds:
                raise ValueError("frames= needs fuse_image_embeds=True")
            if aux is not None and not isinstance(aux, dict):
                raise TypeError("aux must be an observation dictionary")
            if aux is not None and "pixel_values" in aux:
                raise ValueError("frames= and aux['pixel_values'] cannot both be supplied")
            # Stored metadata is owned by this instance. Optional supplied
            # metadata still passes the same validation as an ordinary aux call.
            pixels = process_frames(self._hf_processor(), frames, self._metadata["grid_thw"])
            aux = {**(aux or {}), "pixel_values": pixels}
        horizon = self._positive_integer(
            self.action_horizon if action_horizon is None else action_horizon,
            "action_horizon", self.config["action_horizon"])
        steps = self._positive_integer(self.steps if num_inference_timesteps is None
                                       else num_inference_timesteps, "num_inference_timesteps")
        buckets = self._positive_integer(self.config["num_timestep_buckets"] if num_timestep_buckets is None
                                         else num_timestep_buckets, "num_timestep_buckets")
        if tuple(state_normalized.shape) not in ((1, 1, 132), (1, 132)):
            raise ValueError("normalized state must have shape (1, 1, 132)")
        if initial_noise is not None and (not isinstance(initial_noise, torch.Tensor)
                or not initial_noise.is_floating_point() or tuple(initial_noise.shape) not in (
                    (1, horizon, 132), (horizon, 132))):
            raise ValueError("initial_noise must have shape (1, action_horizon, 132)")
        if aux is not None:
            if not isinstance(aux, dict):
                raise TypeError("aux must be an observation dictionary")
            for key, expected in self._metadata.items():
                if key in aux:
                    if not isinstance(aux[key], torch.Tensor):
                        raise TypeError(f"{key} must be a tensor")
                    if not torch.equal(aux[key].detach().cpu(), expected):
                        raise ValueError(f"observation changed {key}; call set_prompt to rebuild the graph")
            self._validate_attention_mask(aux, self._features.shape[0])
            for key, buffer in self._inputs.items():
                value = aux.get(key)
                if (not isinstance(value, torch.Tensor) or value.shape != buffer.shape
                        or not value.is_floating_point()):
                    raise ValueError(f"every observation must supply matching {key}")
        for name, value in (("use_dit_graph", use_dit_graph), ("use_graph", use_graph)):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        with torch.cuda.device(self.device):
            self._configure_denoising(horizon, steps, buckets)
            if aux is not None:
                for key, buffer in self._inputs.items():
                    buffer.copy_(aux[key])
            self._state.copy_(state_normalized.reshape(1, 132))
            if initial_noise is None:
                self._noise.normal_()
            else:
                self._noise.copy_(initial_noise.reshape(horizon, 132))
            backbone_graph = self.use_backbone_graph and use_graph is not False
            dit_graph = use_dit_graph and use_graph is not False
            if aux is not None and backbone_graph and dit_graph:
                if self._graph is None:
                    self._graph, self._graph_output = self._capture(self._forward)
                self._graph.replay()
                result = self._graph_output
            else:
                if aux is not None or not self._have_features:
                    if backbone_graph:
                        if self._backbone_graph is None:
                            self._backbone_graph, _ = self._capture(self._backbone)
                        self._backbone_graph.replay()
                    else:
                        self._backbone()
                if dit_graph:
                    if self._dit_graph is None:
                        self._dit_graph, self._dit_output = self._capture(self._denoise)
                    self._dit_graph.replay()
                    result = self._dit_output
                else:
                    result = self._denoise()
            self._have_features = True
            result = result.float()
            torch.cuda.current_stream(self.device).synchronize()
            self.latency_records.append((time.perf_counter() - started) * 1000)
            return result

    def predict(self, *args, **kwargs):
        """Alias of ``infer``, returning normalized actions."""
        return self.infer(*args, **kwargs)

    @property
    def precision_spec(self):
        return None

    def get_latency_stats(self):
        if not self.latency_records:
            return {}
        return {"count": len(self.latency_records),
                "p50_ms": float(np.median(self.latency_records)),
                "mean_ms": float(np.mean(self.latency_records))}
