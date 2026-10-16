"""DM05 Torch IO and graph frontend for RTX SM120."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_runtime(checkpoint, num_views, num_steps):
    try:
        from playground.dm05_libero import DM05Exp
    except ImportError as exc:
        raise ImportError("DM05 requires the model distribution providing playground.dm05_libero") from exc
    exp = DM05Exp(task="inference", **{
        name: DM05Exp.__dataclass_fields__[name].default_factory()
        for name in ("model_config", "data_config", "inference_config")})
    exp.model_config.model_name_or_path = str(checkpoint)
    exp.model_config.chunk_size = 10
    exp.model_config.vision_attn_implementation = "sdpa"
    exp.inference_config.output_action_dim = 7
    exp.inference_config.image_prompts = ["Head", "Left wrist"][:num_views]
    exp._initialize_inference_runtime()
    exp.inference_config.diffusion_steps = num_steps
    model = exp.inference_config.model
    try:
        from opendm.model.dm05.dm05_lora import unwrap_dm05_model
    except ImportError:
        pass
    else:
        model = unwrap_dm05_model(model)
    return exp, exp.inference_config, model


class Dm05TorchFrontendRtx:
    """Own the DM05 model, IO transforms, calibration and CUDA graphs.

    The supported tier is mixed NVFP4/FP8/INT8 and requires use_fp4=True.
    Calibration must be explicitly loaded or fitted before inference.
    No method on the model distribution's runtime is replaced.
    """

    def __init__(self, checkpoint, num_views=2, num_steps=10, use_fp4=False):
        if not use_fp4:
            raise ValueError("DM05 currently supports the mixed NVFP4/FP8/INT8 tier; pass use_fp4=True")
        if num_views not in (1, 2):
            raise ValueError("DM05 supports one or two configured camera views")
        if num_steps is None:
            num_steps = 10
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("DM05 kernels require an SM120 CUDA device")
        from flash_rt import _extensions
        _extensions.require("flash_rt_dm05", config="dm05")
        from flash_rt.models.dm05.pipeline_rtx import Dm05Config, Dm05PipelineRtx
        from flash_rt.models.dm05.preprocess import build_transforms
        self._checkpoint_path = str(checkpoint)
        self._graphs = {}
        self._graph_stream = torch.cuda.Stream()
        self._calibrating = False
        self.calibrated = False
        self._prompt = None
        self._exp, self._runtime, model = _load_runtime(checkpoint, num_views, num_steps)
        self._native = Dm05PipelineRtx(model, num_views=num_views, steps=num_steps,
                                     invalidate_graphs=self._graphs.clear)
        self._native.configure(Dm05Config())
        self._host_out = torch.empty(self._native.action.T, self._native.A,
                                     dtype=torch.float32, pin_memory=True)
        self._native.host_out = self._host_out
        self._native._host_out_np = self._host_out.numpy()
        transforms, self._tokenizer = build_transforms(self._runtime.input_transform.transforms)
        self._host_transforms = [t for t in transforms if type(t).__name__ != "ToDevice"]
        self._output_dim = int(self._runtime.output_action_dim)
        if not 0 < self._output_dim <= self._native.A:
            raise ValueError("DM05 output_action_dim must fit the checkpoint action dimension")
        mask = torch.ones(self._native.A) if self._runtime.full_action_mask else torch.zeros(self._native.A)
        if not self._runtime.full_action_mask:
            mask[:self._output_dim] = 1
        self._action_mask = mask.to(torch.bfloat16).float().cuda()
        self._prompt_info = {}
        default_state = Path(checkpoint) / "calibration_state.pt"
        if default_state.is_file():
            self.load_calibration_state(default_state)

    def set_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("DM05 needs a non-empty text prompt")
        self._prompt = prompt

    def load_calibration_state(self, state):
        self.calibrated = False
        import torch
        if not isinstance(state, dict):
            state = torch.load(state, map_location="cuda", weights_only=True)
        self._native.load_calibration_state(state)
        self.calibrated = True

    def calibration_state(self):
        return self._native.calibration_state()

    def calibrate(self, observations, *, percentile=99.9, max_samples=None, verbose=False):
        # This model's static scale fitting is max-based; do not silently promise
        # a percentile implementation it does not have.
        if percentile != 100.0:
            raise ValueError("DM05 calibration is max-based; pass percentile=100.0")
        if isinstance(observations, dict):
            observations = [observations]
        if max_samples is not None:
            if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0:
                raise ValueError("max_samples must be a positive integer")
            from itertools import islice
            observations = list(islice(observations, max_samples))
        else:
            observations = list(observations)
        import torch
        self.calibrated = False
        self._calibrating = True
        try:
            if not observations:
                raise ValueError("DM05 calibration needs representative observations")
            with torch.no_grad():
                self._native.begin_calibration()
                for observation in observations:
                    self.infer(observation)
                result = self._native.finish_calibration()
                result["observations"] = len(observations)
        finally:
            self._calibrating = False
            self._graphs.clear()
        self.calibrated = True
        return result

    def _execute(self):
        import torch
        n = self._native
        if self._calibrating:
            n._run(torch.cuda.current_stream().cuda_stream)
            return
        key = (n.prefix.seq, tuple(n.prefix.segments), n.action.use_mask, n.raw_mode, n.host_staged)
        graph = self._graphs.get(key)
        if graph is None:
            from flash_rt.models.dm05 import fp8
            stream = self._graph_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fp8.set_autotune(True)
                try:
                    n._run(stream.cuda_stream)
                finally:
                    fp8.set_autotune(False)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    n._run(stream.cuda_stream)
            torch.cuda.current_stream().wait_stream(stream)
            self._graphs[key] = graph
        graph.replay()

    def infer(self, observation, *, noise=None):
        import torch
        if not self.calibrated and not self._calibrating:
            raise RuntimeError("DM05 needs calibration: load_calibration_state(...) or calibrate(..., percentile=100.0)")
        prompt = observation.get("prompt", self._prompt)
        if not prompt:
            raise ValueError("DM05 needs a prompt")
        images = observation.get("images")
        if images is None:
            images = [observation[k] for k in ("image", "wrist_image") if k in observation]
        from PIL import Image
        images = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im)) for im in images]
        if len(images) != self._native.vision.nv:
            raise ValueError("DM05 image count must match num_views")
        if any(im.width != im.height for im in images) or len({im.size for im in images}) != 1:
            raise ValueError("DM05 currently supports equally sized square views only")
        if "state" not in observation:
            raise ValueError("DM05 requires robot state")
        if observation.get("history_images") is not None:
            raise NotImplementedError("DM05 history images are not supported")
        data = self._runtime._prepare_model_input(text=prompt, images=images,
                    states=np.asarray(observation["state"]).copy(), robot_type=observation.get("robot_type"))
        with torch.no_grad():
            return {"actions": self._predict_data(data, noise=noise)}

    def _predict_data(self, data, *, noise=None):
        import os
        import torch
        from flash_rt.models.dm05.pipeline_rtx import segments_from_token_types
        n, runtime = self._native, self._runtime
        original_state = np.asarray(data["state"]).copy()
        key = self._tokenizer._key(data)
        data = dict(data)
        x = data
        for transform in self._host_transforms:
            x = transform(x)
        pixels = x["pixel_values"]
        if x.get("history_mask") is not None or (not isinstance(pixels, np.ndarray) and pixels.dtype != torch.uint8):
            raise NotImplementedError("DM05 frontend requires raw uint8 views without image history")
        state = data["state"] if runtime.use_transformed_state else original_state
        seed = os.environ.get("DM05_INFERENCE_SEED")
        if noise is not None:
            if tuple(noise.shape) != (n.action.T, n.A) or not bool(torch.isfinite(noise).all()):
                raise ValueError("DM05 noise must be finite with shape (chunk, checkpoint action_dim)")
            n.action.set_noise(noise.to(device="cuda", dtype=torch.bfloat16))
            n._noise_seed = None
        elif seed is not None:
            seed = int(seed)
            # Isolate reproducibility from the caller's global RNG state.
            if n._noise_seed != seed:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                n.action.set_noise(torch.randn(1, n.action.T, n.A, generator=generator,
                                              device="cuda", dtype=torch.bfloat16)[0])
                n._noise_seed = seed
        else:
            n.action.set_noise(torch.randn(1, n.action.T, n.A, device="cuda", dtype=torch.bfloat16)[0])
            n._noise_seed = None
        n.prefix.staged = True
        if self._tokenizer.inner.add_state:
            key = (key, x["input_ids"].numpy().tobytes())
        if key != n._prompt_key:
            info = self._prompt_info.get(key)
            if info is None:
                info = (x["input_ids"].reshape(-1).numpy().copy(), segments_from_token_types(x["token_type_ids"]))
                self._prompt_info[key] = info
            n.prefix.set_prompt_host(info[0], n.image_token_id, info[1])
            n._prompt_key = key
        n.action.set_action_mask(self._action_mask)
        pixels = (pixels if isinstance(pixels, np.ndarray) else pixels.numpy()).reshape(n.vision.nv, *pixels.shape[-3:])
        if n.host_in is None or tuple(n.host_in.shape) != pixels.shape:
            n.host_in = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=True)
            n.raw_u8 = torch.empty(pixels.shape, dtype=torch.uint8, device="cuda")
            n._host_in_np = n.host_in.numpy()
            self._graphs.clear()
        np.copyto(n._host_in_np, pixels)
        if not (n.raw_mode and n.host_staged):
            n.raw_mode = n.host_staged = True
            self._graphs.clear()
        self._execute()
        torch.cuda.current_stream().synchronize()
        actions = n._host_out_np[:, :self._output_dim].copy()
        out = runtime.output_transform({"action": actions, "state": state, "meta_data": data["meta_data"]})
        result = np.asarray(out["action"])
        if result.ndim != 2 or result.shape[-1] != self._output_dim or not np.isfinite(result).all():
            raise RuntimeError("DM05 returned invalid final actions")
        return result
