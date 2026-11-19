"""LingBot-VLA-v2 BF16 torch frontend for gfx1100, gfx1151 and gfx1201."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import torch

from flash_rt.amd.hardware.capabilities_rdna import load_capabilities
from flash_rt.amd.models.lingbot_vla_v2.pipeline_rdna import (
    LingbotVlaV2PipelineRdna,
    _TransformerOps,
)
from flash_rt.executors.torch_weights import MultiSafetensorsSource
from flash_rt.executors.weight_loader import WeightLoader
from flash_rt.models.lingbot_vla_v2.weight_spec import build_spec

# ════════════════════════════════════════════════════════════════════
#  Policy configuration
# ════════════════════════════════════════════════════════════════════


def _load_policy_config(checkpoint, policy_config, num_steps):
    """Load RoboTwin defaults and apply checkpoint or explicit policy settings."""
    import yaml

    defaults = yaml.safe_load(
        files("flash_rt.configs").joinpath("lingbot_vla_v2.yaml").read_text()
    )["policy"]
    if policy_config is None:
        path = checkpoint / "lingbotvla_cli.yaml"
        if not path.exists():
            path = checkpoint / "config.json"
        policy_config = path if path.exists() else {}
    if isinstance(policy_config, dict):
        values = policy_config
    else:
        values = yaml.safe_load(Path(policy_config).read_text())
    if "model" in values:
        values = {**values["model"], **values["train"]}
    config = {**defaults, **values}
    config["n_action_steps"] = config["chunk_size"]
    if num_steps is not None:
        config["num_steps"] = num_steps
    steps = config["num_steps"]
    if not isinstance(steps, int) or not 1 <= steps <= 256:
        raise ValueError("num_steps must be an integer from 1 to 256 for the BF16 schedule")
    supported = {
        "adanorm_time": True,
        "final_norm_adanorm": False,
        "action_fp32": False,
        "router_activation": "sigmoid",
        "use_shared_expert_gate": False,
        "use_moe": True,
        "enable_expert_vision": False,
    }
    for key, expected in supported.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"unsupported LingBot-VLA-v2 setting {key}={config[key]!r}")
    align = config["align_params"] = config["align_params"] or {}
    if align and (
        align.get("mode", "query") != "query" or align.get("video", {}).get("use_cls_loss", False)
    ):
        raise ValueError("LingBot-VLA-v2 RDNA supports query alignment without video CLS tokens")
    return config


class LingbotVlaV2TorchFrontendAmdRdna:
    """Batch-1 BF16 frontend with observation preparation and HIP Graph replay.

    State normalization and action decoding use the policy's robot transforms.
    """

    _PROMPT_KEYS = ("image_grid_thw", "img_masks", "lang_tokens", "lang_masks")

    def __init__(
        self,
        checkpoint_path,
        *,
        policy_config=None,
        processor_path=None,
        num_views=3,
        num_steps=None,
        action_horizon=None,
        device="cuda:0",
        hardware=None,
        use_fp8=False,
    ):
        if use_fp8:
            raise ValueError("LingBot-VLA-v2 on RDNA supports BF16 only")
        self.checkpoint_path = Path(checkpoint_path)
        self.config = _load_policy_config(self.checkpoint_path, policy_config, num_steps)
        self.processor_path = processor_path or self.config.get("tokenizer_path")
        if self.processor_path is None:
            raise ValueError(
                "processor_path is required for the Qwen3-VL configuration and processor"
            )
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("LingBot-VLA-v2 RDNA requires PyTorch with ROCm")
        self.device = torch.device(device)
        self.num_views = num_views
        self.action_horizon = (
            self.config["n_action_steps"] if action_horizon is None else action_horizon
        )
        if num_views < 1 or not 1 <= self.action_horizon <= self.config["n_action_steps"]:
            raise ValueError(
                "num_views must be positive and action_horizon must fit the policy chunk"
            )
        from transformers import AutoConfig

        self.qwen_config = AutoConfig.from_pretrained(self.processor_path).to_dict()
        if self.qwen_config["model_type"] != "qwen3_vl":
            raise ValueError("processor_path must contain a Qwen3-VL configuration")
        self._processor = None
        self._metadata = None
        self._graph = None
        self._graph_output = None
        self._capture_stream = None
        self._prefix_graph = self._prefix_graph_cache = None
        self._prefix_cache = self._decode_graph = self._decode_graph_output = None
        self._split_pool = None
        self._prefix_ready = False
        with torch.cuda.device(self.device), torch.inference_mode():
            from flash_rt.amd import flash_rt_amd_kernels as kernels
            from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend

            props = torch.cuda.get_device_properties(self.device)
            self.hardware = load_capabilities(kernels, props.gcnArchName, hardware).hardware
            # Describe the official nn.Linear storage directly through hipBLASLt.
            ops = _TransformerOps(
                kernels, gemm=Rdna35GemmBackend(kernels, autotune=False, native_nt=True)
            )
            weights = self._load_weights()
            self.pipeline = LingbotVlaV2PipelineRdna(
                weights, self.qwen_config["vision_config"], ops
            )
            self.state_dim = weights._state_proj_w.shape[0]
            self.action_dim = weights._action_out_proj_w.shape[1]
            self._state = torch.zeros((1, self.state_dim), dtype=torch.bfloat16, device=self.device)
            self._noise = torch.zeros(
                (1, self.config["n_action_steps"], self.action_dim), dtype=torch.bfloat16,
                device=self.device,
            )

    def _load_weights(self):
        shards = sorted(self.checkpoint_path.glob("model*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model safetensors in {self.checkpoint_path}")
        weights = SimpleNamespace()
        WeightLoader(
            source=MultiSafetensorsSource(shards, device="cpu"), target=weights, spec=build_spec()
        ).run()
        for name, value in vars(weights).items():
            if isinstance(value, list):
                value = [tensor.to(self.device).contiguous() for tensor in value]
            else:
                value = value.to(self.device).contiguous()
            setattr(weights, name, value)
        return weights

    def set_hf_processor(self, processor):
        """Use an already-loaded Qwen3-VL processor, including offline callers."""
        self._processor = processor

    def _hf_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self.set_hf_processor(
                AutoProcessor.from_pretrained(self.processor_path, padding_side="right")
            )
        return self._processor

    # ════════════════════════════════════════════════════════════════════
    #  Observation preparation and prompt binding
    # ════════════════════════════════════════════════════════════════════

    def prepare_observation(self, images, prompt, *, img_masks=None):
        """Pack resized uint8 HWC views and tokenize the policy's chat prompt."""
        if len(images) != self.num_views:
            raise ValueError(f"expected {self.num_views} camera images")
        processor = self._hf_processor()
        pixels, grids = [], []
        for image in images:
            image = torch.as_tensor(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
                raise ValueError("camera images must be uint8 HWC with three channels")
            processed = processor.image_processor(image.permute(2, 0, 1).contiguous())
            pixels.append(torch.as_tensor(processed["pixel_values"]).reshape(-1, self._patch_width))
            grids.append(torch.as_tensor(processed["image_grid_thw"]).reshape(3))
        if self.config.get("use_qwen3_chat_template", True):
            prompt = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
            )
        else:
            if not prompt.startswith("<bos>"):
                prompt = f"<bos>{prompt}"
            if not prompt.endswith("\n"):
                prompt = f"{prompt}\n"
        tokenized = processor.tokenizer(
            [prompt], padding="max_length", padding_side="right", truncation=True,
            max_length=self.config["tokenizer_max_length"], return_tensors="pt",
        )
        return {
            "images": torch.cat(pixels).bfloat16(),
            "image_grid_thw": torch.stack(grids),
            "img_masks": (
                torch.ones(self.num_views, dtype=torch.bool)
                if img_masks is None
                else torch.as_tensor(img_masks).bool()
            ),
            "lang_tokens": tokenized["input_ids"],
            "lang_masks": tokenized["attention_mask"].bool(),
        }

    @property
    def _patch_width(self):
        vision = self.qwen_config["vision_config"]
        return vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2

    def _observation_aux(self, aux):
        grid = torch.as_tensor(aux["image_grid_thw"]).detach().cpu().reshape(-1, 3)
        images = torch.as_tensor(aux["images"])
        tokens = torch.as_tensor(aux["lang_tokens"]).detach().cpu()
        masks = torch.as_tensor(aux["lang_masks"]).detach().cpu()
        image_masks = torch.as_tensor(aux["img_masks"]).detach().cpu().reshape(-1)
        merge = self.qwen_config["vision_config"]["spatial_merge_size"]
        if (
            grid.shape != (self.num_views, 3)
            or (grid <= 0).any()
            or (grid[:, 1:] % merge != 0).any()
        ):
            raise ValueError("image_grid_thw must match the camera count and spatial merge size")
        if (
            images.shape[-1] != self._patch_width
            or images.numel() != int(grid.prod(-1).sum()) * self._patch_width
        ):
            raise ValueError("images must contain processor-packed patches matching image_grid_thw")
        if (
            tokens.ndim not in (1, 2)
            or (tokens.ndim == 2 and tokens.shape[0] != 1)
            or tokens.shape != masks.shape
            or image_masks.numel() != self.num_views
        ):
            raise ValueError("lang_tokens, lang_masks and img_masks require batch size one")
        metadata = {
            "image_grid_thw": grid.long(),
            "img_masks": image_masks[None].bool(),
            "lang_tokens": tokens.reshape(1, -1).long(),
            "lang_masks": masks.reshape(1, -1).bool(),
        }
        return images.reshape(-1, self._patch_width), metadata

    @torch.inference_mode()
    def set_prompt(self, *, aux, prompt=None):
        """Bind packed observation inputs and invalidate the previous inference graph."""
        pixels, metadata = self._observation_aux(aux)
        with torch.cuda.device(self.device):
            self._graph, self._graph_output = None, None
            self._prefix_graph = self._prefix_graph_cache = None
            self._prefix_cache = self._decode_graph = self._decode_graph_output = None
            self._split_pool = None
            self._prefix_ready = False
            self._metadata = {key: value.clone() for key, value in metadata.items()}
            self._pixels = pixels.to(device=self.device, dtype=torch.bfloat16).contiguous().clone()
            self.pipeline.bind_vision(metadata["image_grid_thw"])
            self.pipeline.bind_prefix(
                metadata["lang_tokens"], metadata["lang_masks"], metadata["img_masks"],
                qwen_config=self.qwen_config, policy_config=self.config,
            )
            if not hasattr(self.pipeline, "action_cfg"):
                self.pipeline.bind_action(self.config)

    # ════════════════════════════════════════════════════════════════════
    #  Graph capture and normalized action inference
    # ════════════════════════════════════════════════════════════════════

    def _forward(self):
        return self.pipeline.infer(self._pixels, self._state, self._noise)

    def _capture(self, forward=None, *, pool=None):
        forward = self._forward if forward is None else forward
        current = torch.cuda.current_stream(self.device)
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=self.device)
        stream = self._capture_stream
        stream.wait_stream(current)
        multistream = self.pipeline.shared_stream is not None
        with torch.cuda.stream(stream):
            self.pipeline.ops.prefix_attention.begin_exact_calibration()
            if multistream:
                weight = self.pipeline.action_qkv[0][0]
                self.pipeline.ops.gemm.begin_exact_calibration(
                    self.pipeline.horizon + 1, weight.shape[1], weight.shape[0], biased=True
                )
            forward()
            self.pipeline.ops.prefix_attention.end_exact_calibration()
            if multistream:
                self.pipeline.ops.gemm.end_exact_calibration()
            for _ in range(1 if multistream else 2):
                forward()
        current.wait_stream(stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        if multistream:
            # Warm the multi-stream HIP Graph scheduler before the retained capture.
            with torch.cuda.graph(graph, stream=stream, pool=pool):
                warm_output = forward()
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            graph.reset()
            del warm_output
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            output = forward()
        current.wait_stream(stream)
        return graph, output

    @torch.inference_mode()
    def prefill_observation(self, *, aux, use_graph=True):
        """Prepare reusable image/text context for subsequent action samples."""
        pixels, metadata = self._observation_aux(aux)
        if self._metadata is None or any(
            not torch.equal(metadata[key], self._metadata[key]) for key in self._PROMPT_KEYS
        ):
            self.set_prompt(aux=aux)
        with torch.cuda.device(self.device):
            self._prefix_ready = False
            self._pixels.copy_(pixels)
            if use_graph:
                if self._prefix_graph is None:
                    self._split_pool = torch.cuda.graph_pool_handle()
                    current = torch.cuda.current_stream(self.device)
                    if self._capture_stream is None:
                        self._capture_stream = torch.cuda.Stream(device=self.device)
                    stream = self._capture_stream
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        self.pipeline.prefill(self._pixels, cache_only=True)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream, pool=self._split_pool):
                        _, cache = self.pipeline.prefill(self._pixels, cache_only=True)
                    current.wait_stream(stream)
                    self._prefix_graph, self._prefix_graph_cache = graph, cache
                self._prefix_graph.replay()
                cache = self._prefix_graph_cache
            else:
                _, cache = self.pipeline.prefill(self._pixels, cache_only=True)
            if cache is not self._prefix_cache:
                self._decode_graph = self._decode_graph_output = None
            self._prefix_cache = cache
            self._prefix_ready = True

    def _forward_cached(self):
        return self.pipeline.denoise(self._state, self._prefix_cache, self._noise)

    def _set_action_inputs(self, state_normalized, initial_noise):
        if not isinstance(state_normalized, torch.Tensor):
            raise TypeError("inference expects a normalized state tensor")
        if state_normalized.shape not in ((self.state_dim,), (1, self.state_dim)):
            raise ValueError(
                f"normalized state must have shape (1, {self.state_dim}) or ({self.state_dim},)"
            )
        if initial_noise is not None and initial_noise.shape != self._noise.shape:
            raise ValueError(f"initial_noise must have shape {tuple(self._noise.shape)}")
        self._state.copy_(state_normalized.reshape(1, -1))
        if initial_noise is None:
            self._noise.normal_()
        else:
            self._noise.copy_(initial_noise)

    @torch.inference_mode()
    def infer_cached(self, state_normalized, *, initial_noise=None, use_graph=True):
        """Sample actions from the most recently prepared image/text context."""
        if not self._prefix_ready:
            raise RuntimeError("call prefill_observation before infer_cached")
        with torch.cuda.device(self.device):
            self._set_action_inputs(state_normalized, initial_noise)
            if use_graph:
                if self._decode_graph is None:
                    self._decode_graph, self._decode_graph_output = self._capture(
                        self._forward_cached, pool=self._split_pool
                    )
                self._decode_graph.replay()
                result = self._decode_graph_output
            else:
                result = self._forward_cached()
            return result[:, : self.action_horizon].float()

    @torch.inference_mode()
    def infer(self, state_normalized, *, aux, initial_noise=None, use_graph=True):
        """Fresh packed observation + normalized state → normalized FP32 action chunk."""
        if self._metadata is None:
            raise RuntimeError("call set_prompt with observation aux before infer")
        pixels, metadata = self._observation_aux(aux)
        for key in self._PROMPT_KEYS:
            if not torch.equal(metadata[key], self._metadata[key]):
                raise ValueError(f"observation changed {key}; call set_prompt to rebuild the graph")
        with torch.cuda.device(self.device):
            self._set_action_inputs(state_normalized, initial_noise)
            self._prefix_ready = False
            self._pixels.copy_(pixels)
            if use_graph:
                if self._graph is None:
                    self._graph, self._graph_output = self._capture()
                self._graph.replay()
                result = self._graph_output
            else:
                result = self._forward()
            return result[:, : self.action_horizon].float()

    def predict(self, *args, **kwargs):
        """Alias of infer, returning normalized actions."""
        return self.infer(*args, **kwargs)
