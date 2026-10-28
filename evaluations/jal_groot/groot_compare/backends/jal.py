"""Official GR00T policy adapters with caller-supplied diffusion noise.

Imports of PyTorch, NumPy, GR00T and TensorRT are delayed until construction.
The eager sampler retains the upstream function's bytecode: only its private
``torch.randn`` lookup is redirected. No global PyTorch function is patched.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types


ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ISAAC_REVISION = "9c7e746b2cd37a810070a98ef41d290a07e806c2"


class _NoiseTorchProxy:
    def __init__(self, torch_module):
        self.module = torch_module
        self.noise = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.module, name)

    def randn(self, *args, **kwargs):
        if self.noise is None:
            raise RuntimeError("The official sampler requested noise outside get_action")
        if args or set(kwargs) != {"size", "dtype", "device"}:
            raise RuntimeError("Unsupported upstream diffusion-noise call signature")
        if tuple(kwargs["size"]) != tuple(self.noise.shape):
            raise RuntimeError("Official diffusion noise shape differs from paired noise")
        if kwargs["dtype"] != self.noise.dtype or kwargs["device"] != self.noise.device:
            raise RuntimeError("Official diffusion noise dtype/device differs from paired noise")
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Official sampler requested more than one noise tensor")
        return self.noise.clone()


def _install_noise_sampler(action_head, torch_module):
    original = action_head.get_action_with_features
    raw = inspect.unwrap(original)
    if isinstance(raw, types.MethodType):
        raw = raw.__func__
    if not isinstance(raw, types.FunctionType):
        raise RuntimeError("Unsupported official action sampler implementation")
    source = textwrap.dedent(inspect.getsource(raw))
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch" and node.func.attr == "randn"
    ]
    if len(calls) != 1 or raw.__globals__.get("torch") is not torch_module:
        raise RuntimeError("Expected exactly one official torch.randn diffusion initializer")
    proxy = _NoiseTorchProxy(torch_module)
    private_globals = dict(raw.__globals__, torch=proxy)
    replacement = types.FunctionType(
        raw.__code__, private_globals, raw.__name__, raw.__defaults__, raw.__closure__
    )
    replacement.__kwdefaults__ = raw.__kwdefaults__
    functools.update_wrapper(replacement, raw)
    action_head.get_action_with_features = types.MethodType(
        torch_module.no_grad()(replacement), action_head
    )
    return proxy, original, hashlib.sha256(source.encode()).hexdigest()


def _refresh_trt_noise(action_head, runner, noise):
    """Update both setup noise and the buffer already referenced by a CUDA graph."""
    action_head.init_actions.copy_(noise)
    state = runner._graph_state
    if state is not None:
        if tuple(state.initial_actions.shape) != tuple(noise.shape):
            raise RuntimeError("TensorRT graph noise shape differs from paired noise")
        state.initial_actions.copy_(noise)


class _JalBackend:
    def _init_policy(self, checkpoint, cpu_threads, kwargs):
        import numpy as np
        import torch

        source_dir = Path(kwargs.pop(
            "source_dir", Path(__file__).resolve().parents[2] / "vendor" / "Isaac-GR00T"
        )).resolve()
        if not (source_dir / "gr00t" / "policy" / "gr00t_policy.py").is_file():
            raise FileNotFoundError(f"Pinned Isaac-GR00T source missing: {source_dir}")
        sys.path.insert(0, str(source_dir))
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        imported = Path(inspect.getfile(Gr00tPolicy)).resolve()
        if source_dir not in imported.parents:
            raise RuntimeError(f"Another Isaac-GR00T checkout is already imported: {imported}")
        self.device = kwargs.pop("device", "cuda:0")
        embodiment = kwargs.pop("embodiment_tag", "LIBERO_PANDA")
        steps = kwargs.pop("denoising_steps", 4)
        if kwargs:
            raise TypeError(f"Unknown JAL backend options: {sorted(kwargs)}")
        if cpu_threads < 1 or steps != 4:
            raise ValueError("This paired LIBERO comparison requires CPU threads >= 1 and 4 steps")
        if not torch.cuda.is_available():
            raise RuntimeError("JAL GR00T comparison requires CUDA")
        torch.set_num_threads(cpu_threads)
        self._np, self._torch = np, torch
        self.source_dir = source_dir
        self._closed, self._active = False, False
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(embodiment),
            model_path=checkpoint, device=self.device, strict=True,
        )
        self.action_head = self.policy.model.action_head
        self.action_head.num_inference_timesteps = steps
        self.noise_shape = (
            1, int(self.action_head.config.action_horizon), int(self.action_head.action_dim)
        )
        video = self.policy.modality_configs["video"]
        if list(video.modality_keys) != ["image"] or list(video.delta_indices) != [0]:
            raise ValueError("Use the shared single-camera checkpoint overlay: image, frame [0]")
        if set(self.policy.modality_configs["action"].modality_keys) != set(ACTION_KEYS):
            raise ValueError("This adapter requires the seven LIBERO physical-action keys")
        self.output_horizon = len(self.policy.modality_configs["action"].delta_indices)
        self.metadata = {
            "isaac_gr00t_revision": ISAAC_REVISION,
            "source_dir": str(source_dir),
            "checkpoint": str(Path(checkpoint).resolve()),
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "denoising_steps": steps,
            "noise_shape": list(self.noise_shape),
            "decoded_action_shape": [1, self.output_horizon, len(ACTION_KEYS)],
            "camera_keys": ["image"],
            "noise_contract": "caller float32 rounded to BF16 before each backend",
            "io_boundary": "official policy.get_action: raw observations to decoded CPU actions",
        }

    def _noise(self, noise):
        np, torch = self._np, self._torch
        array = np.asarray(noise)
        if array.shape != self.noise_shape or not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"Expected floating noise with shape {self.noise_shape}")
        if not np.isfinite(array).all():
            raise ValueError("Noise must contain only finite values")
        return torch.as_tensor(np.array(array, dtype=np.float32, copy=True)).to(
            device=self.device, dtype=torch.bfloat16
        ).contiguous()

    def _actions(self, result):
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Official policy did not return (actions, info)")
        actions, _ = result
        if set(actions) != set(ACTION_KEYS):
            raise RuntimeError("Decoded action keys differ from the LIBERO contract")
        for key in ACTION_KEYS:
            value = actions[key]
            if (not isinstance(value, self._np.ndarray)
                    or value.shape != (1, self.output_horizon, 1)
                    or value.dtype != self._np.float32
                    or not self._np.isfinite(value).all()):
                raise RuntimeError(f"Invalid physical action output: {key}")
        return {key: actions[key] for key in ACTION_KEYS}

    def input_fingerprints(self, observation: dict) -> dict:
        """Hash official preprocessing outputs on CPU, outside inference timing."""
        if self._closed or self._active:
            raise RuntimeError("Backend is closed or already processing an observation")
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        self.policy.check_observation(observation)
        processed = []
        for obs in self.policy._unbatch_observation(observation):
            step = self.policy._to_vla_step_data(obs)
            processed.append(self.policy.processor([
                {"type": MessageType.EPISODE_STEP.value, "content": step}
            ]))
        inputs = _rec_to_dtype(
            self.policy.collate_fn(processed), dtype=self._torch.bfloat16
        )["inputs"]
        fingerprints = {}
        for key in ("pixel_values", "input_ids", "image_grid_thw", "state"):
            tensor = inputs[key]
            if not isinstance(tensor, self._torch.Tensor) or tensor.device.type != "cpu":
                raise RuntimeError(f"Expected CPU processor tensor for fingerprint: {key}")
            tensor = tensor.detach().cpu()
            if tensor.dtype == self._torch.bfloat16:
                tensor = tensor.float()
            array = self._np.ascontiguousarray(tensor.numpy())
            fingerprints[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        return fingerprints

    def synchronize(self):
        self._torch.cuda.synchronize(self.device)

    def close(self):
        if not self._closed:
            self.synchronize()
            self.policy = None
            self.action_head = None
            self._closed = True


class TorchBackend(_JalBackend):
    """Official BF16 eager policy, or explicitly requested JAL DiT compilation."""

    def __init__(self, checkpoint: str, cpu_threads: int = 2, **kwargs):
        compile_mode = kwargs.pop("compile_mode", None)
        if compile_mode not in (None, "max-autotune"):
            raise ValueError("compile_mode must be None (eager) or 'max-autotune' (JAL recipe)")
        self._init_policy(checkpoint, cpu_threads, kwargs)
        self._sampler, self._original_sampler, source_sha = _install_noise_sampler(
            self.action_head, self._torch
        )
        if compile_mode:
            self.action_head.model.forward = self._torch.compile(
                self.action_head.model.forward, mode=compile_mode
            )
        self.metadata.update(
            backend="pytorch", label="PyTorch eager BF16" if not compile_mode else
            "PyTorch BF16, DiT torch.compile(max-autotune)",
            compile_mode=compile_mode, sampler_source_sha256=source_sha,
            noise_injection="private sampler globals; exactly one randn replaced",
        )

    def get_action(self, observation: dict, noise) -> dict:
        if self._closed or self._active:
            raise RuntimeError("Backend is closed or already processing an observation")
        self._active = True
        try:
            self._sampler.noise = self._noise(noise)
            self._sampler.calls = 0
            with self._torch.inference_mode():
                result = self.policy.get_action(observation)
            if self._sampler.calls != 1:
                raise RuntimeError("Official sampler did not consume the paired noise exactly once")
            return self._actions(result)
        finally:
            self._sampler.noise = None
            self._active = False

    def close(self):
        if not self._closed:
            self.action_head.get_action_with_features = self._original_sampler
            self._original_sampler = None
            self._sampler.noise = None
        super().close()


class TensorRTBackend(_JalBackend):
    """JAL optimized mixed-NVFP4 policy; refuses silent engine/graph fallback."""

    def __init__(self, checkpoint: str, engine_dir: str, cpu_threads: int = 2, **kwargs):
        engine_path = Path(engine_dir).resolve()
        export = json.loads((engine_path / "export_metadata.json").read_text())
        if (export.get("quantization") != "mixed_nvfp4"
                or not export.get("optimized_action_head")):
            raise ValueError("Expected JAL optimized mixed_nvfp4 engine metadata")
        self._init_policy(checkpoint, cpu_threads, kwargs)
        deployment = self.source_dir / "scripts" / "deployment"
        sys.path.insert(0, str(deployment))
        from _trt_contract import resolve_batch_size
        from trt_model_forward import close_tensorrt_engines, setup_tensorrt_engines

        self._close_engines = close_tensorrt_engines
        resolve_batch_size(str(engine_path), 1, source="groot_compare")
        self.action_head.init_actions = self._torch.empty(
            self.noise_shape, device=self.device, dtype=self._torch.bfloat16
        )
        try:
            setup_tensorrt_engines(self.policy, str(engine_path), mode="n17_full_pipeline")
            self.runner = getattr(self.action_head, "_trt_optimization_runner", None)
            backbone = self.policy.model.backbone
            if (self.runner is None or getattr(backbone, "vit_engine", None) is None
                    or getattr(backbone, "llm_engine", None) is None
                    or self.runner.vl_sa_engine is None):
                raise RuntimeError("Required optimized TRT component fell back to PyTorch")
            original_signature = self.runner._signature

            def checked_signature(backbone_output, action_input):
                signature = original_signature(backbone_output, action_input)
                if (self.runner._graph_state is not None
                        and signature != self.runner._graph_signature):
                    raise RuntimeError("Observation would trigger eager TensorRT graph fallback")
                return signature

            self.runner._signature = checked_signature
            self.metadata.update(
                backend="tensorrt", label="TensorRT optimized mixed NVFP4",
                compile_mode=None, engine_dir=str(engine_path), export_metadata=export,
                noise_injection="init_actions plus explicit persistent graph-buffer copy",
                graph_replays=0,
            )
        except Exception:
            self._close_engines(self.policy)
            super().close()
            raise

    def get_action(self, observation: dict, noise) -> dict:
        if self._closed or self._active:
            raise RuntimeError("Backend is closed or already processing an observation")
        self._active = True
        try:
            paired_noise = self._noise(noise)
            with self._torch.inference_mode():
                # Graph buffers are inference tensors; update them under the
                # same mode used when the upstream runtime allocated them.
                _refresh_trt_noise(self.action_head, self.runner, paired_noise)
                result = self.policy.get_action(observation)
            if self.runner._capture_failed or self.runner._graph_state is None:
                raise RuntimeError("Optimized TRT CUDA graph was not used; refusing fallback result")
            self.metadata["graph_replays"] += 1
            return self._actions(result)
        finally:
            self._active = False

    def close(self):
        if not self._closed:
            self.synchronize()
            self._close_engines(self.policy)
            self.runner = None
        super().close()
