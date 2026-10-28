"""FlashRT's fixed-prompt Thor GR00T backend for a shared comparison harness.

Only ``capture_calibration`` loads the official model. It runs during separate
setup, using a calibration observation excluded from the evaluation split.
The serving backend never recalibrates or calls an official model forward.
GPU dependencies are imported lazily so the harness can be inspected on CPU.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
NOISE_SHAPE = (1, 40, 132)
CAPTURE_SCHEMA = "groot-comparison-calibration-v1"


def _observation_prompt(observation: dict) -> str:
    """Enforce the single-camera, single-observation LIBERO I/O contract."""
    if set(observation) != {"video", "state", "language"}:
        raise ValueError("Expected the official video/state/language observation")
    if set(observation["video"]) != {"image"}:
        raise ValueError("This comparison requires exactly the image camera")
    image = np.asarray(observation["video"]["image"])
    if image.ndim != 5 or image.shape[:2] != (1, 1) or image.shape[-1] != 3:
        raise ValueError("Expected RGB image shape (1, 1, height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("Raw camera images must be uint8")
    if set(observation["state"]) != set(ACTION_KEYS):
        raise ValueError("Expected LIBERO x/y/z/roll/pitch/yaw/gripper state")
    for key, value in observation["state"].items():
        value = np.asarray(value)
        expected = (1, 1, 2 if key == "gripper" else 1)
        if value.shape != expected or value.dtype != np.float32:
            raise ValueError(f"State {key} must be float32 with shape {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"State {key} contains nonfinite values")
    language = observation["language"]
    if len(language) != 1:
        raise ValueError("Expected one language modality")
    value = next(iter(language.values()))
    if (not isinstance(value, list) or len(value) != 1
            or not isinstance(value[0], list) or len(value[0]) != 1
            or not isinstance(value[0][0], str)):
        raise ValueError("Language must contain one batch of one instruction")
    return value[0][0]


def observation_sha256(observation: dict) -> str:
    """Identify calibration data without retaining a mutable observation alias."""
    _observation_prompt(observation)
    digest = hashlib.sha256()
    for group in sorted(observation):
        for key, value in sorted(observation[group].items()):
            digest.update(json.dumps([group, key]).encode())
            if group == "language":
                digest.update(json.dumps(value, ensure_ascii=False).encode())
            else:
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode())
                digest.update(json.dumps(array.shape).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_identity(checkpoint: str) -> dict:
    """Check local configuration identity; the run manifest pins full weights."""
    root = Path(checkpoint)
    files = {}
    for name in ("config.json", "processor_config.json", "statistics.json",
                 "embodiment_id.json", "model.safetensors.index.json"):
        path = root / name
        if path.exists():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif name in ("config.json", "processor_config.json"):
            raise FileNotFoundError(path)
    weights = {p.name: p.stat().st_size for p in sorted(root.glob("*.safetensors"))}
    if not weights:
        raise FileNotFoundError(f"No checkpoint safetensors under {root}")
    return {"metadata_sha256": files, "weight_file_sizes": weights}


def _hook_module():
    """Load the pinned vendor hook helper without altering global sys.path."""
    candidates = []
    if os.environ.get("FLASHRT_ROOT"):
        candidates.append(Path(os.environ["FLASHRT_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[2] / "vendor" / "FlashRT")
    spec = importlib.util.find_spec("flash_rt")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent.parent)
    for root in candidates:
        path = root / "tests" / "_helpers" / "groot_n17" / "capture_aux_multi.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_comparison_capture_aux", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("FlashRT capture_aux_multi.py not found; set FLASHRT_ROOT")


def _capture_aux(policy, observation, torch):
    helper = _hook_module()
    aux = {}
    hooks = helper._install_hooks(policy, aux)
    backbone = policy.model.backbone.model
    visual = backbone.model.visual
    original_visual = visual.forward
    handle = None
    numpy_state = np.random.get_state()
    try:
        def capture_tokens(module, args, kwargs):
            aux["input_ids"] = kwargs["input_ids"].detach().cpu().clone()

        handle = backbone.register_forward_pre_hook(capture_tokens, with_kwargs=True)

        def capture_pixels(hidden_states, grid_thw, **kwargs):
            aux["pixel_values"] = hidden_states.detach().cpu().clone()
            return original_visual(hidden_states, grid_thw, **kwargs)

        visual.forward = capture_pixels
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            np.random.seed(0)
            with torch.inference_mode():
                result = policy.get_action(observation)
        return aux, result[0] if isinstance(result, tuple) else result
    finally:
        np.random.set_state(numpy_state)
        if handle is not None:
            handle.remove()
        visual.forward = original_visual
        helper._restore_hooks(hooks)


def capture_calibration(checkpoint: str, observation: dict, output_path: str,
                        cpu_threads: int = 2) -> dict:
    """Capture one separate calibration observation; never called by evaluation.

    Returns metadata and writes a trusted local torch capture. The caller owns
    the calibration/evaluation split and must exclude this observation from its
    scored samples. Existing captures are not overwritten.
    """
    prompt = _observation_prompt(observation)
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(target)
    identity = _checkpoint_identity(checkpoint)
    stored_inputs = copy.deepcopy(observation)

    import torch
    import gr00t.model  # noqa: F401; registers the official processor/model.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    torch.set_num_threads(cpu_threads)
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve("LIBERO_PANDA"),
                        model_path=checkpoint, device="cuda:0")
    head = None
    try:
        head = policy.model.action_head
        shape = (1, int(head.action_horizon), int(head.action_dim))
        if shape != NOISE_SHAPE:
            raise ValueError(f"Unsupported GR00T noise shape: {shape}")
        head.num_inference_timesteps = 4
        aux, actions = _capture_aux(policy, stored_inputs, torch)
        if tuple(aux["initial_noise"].shape) != NOISE_SHAPE:
            raise ValueError("Official captured noise has an unexpected shape")
        metadata = {
            "schema": CAPTURE_SCHEMA,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_identity": identity,
            "checkpoint_identity_scope": "metadata hashes and weight sizes; use run manifest for full weight identity",
            "prompt": prompt,
            "observation_sha256": observation_sha256(stored_inputs),
            "noise_shape": list(NOISE_SHAPE),
            "num_inference_timesteps": 4,
            "cpu_threads": cpu_threads,
            "calibration_samples": 1,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
        os.close(fd)
        try:
            torch.save({"metadata": metadata, "aux": aux, "inputs": stored_inputs,
                        "actions": actions}, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return metadata
    finally:
        head = None
        del policy
        gc.collect()
        torch.cuda.empty_cache()


class FlashRTBackend:
    """One-camera, batch-one, four-step GR00T with frozen prompt/calibration."""

    noise_shape = NOISE_SHAPE

    def __init__(self, checkpoint: str, calibration_capture: str,
                 cpu_threads: int = 2, **kwargs: Any):
        use_fp4 = kwargs.pop("use_fp4", True)
        device = kwargs.pop("device", "cuda:0")
        steps = kwargs.pop("denoising_steps", 4)
        horizon = kwargs.pop("action_horizon", 40)
        if kwargs:
            raise TypeError(f"Unknown FlashRT options: {', '.join(sorted(kwargs))}")
        if cpu_threads < 1 or steps != 4 or horizon != 40:
            raise ValueError("This adapter requires positive CPU threads, four steps and horizon 40")
        if device not in ("cuda", "cuda:0"):
            raise ValueError("The pinned FlashRT loader supports this comparison on CUDA device 0 only")
        device = "cuda:0"

        import torch
        import gr00t.model  # noqa: F401
        import flash_rt
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

        self._torch = torch
        self._model = self._frontend = self._prep = self._processor = None
        self._closed = False
        torch.set_num_threads(cpu_threads)
        # Only load captures created locally by capture_calibration.
        capture = torch.load(calibration_capture, map_location="cpu", weights_only=False)
        meta = capture["metadata"]
        if meta.get("schema") != CAPTURE_SCHEMA:
            raise ValueError("Use a dedicated capture_calibration artifact for this adapter")
        if meta["checkpoint_identity"] != _checkpoint_identity(checkpoint):
            raise ValueError("Calibration capture belongs to a different checkpoint/configuration")
        self._prompt = _observation_prompt(capture["inputs"])
        if self._prompt != meta["prompt"]:
            raise ValueError("Calibration prompt metadata mismatch")
        if observation_sha256(capture["inputs"]) != meta["observation_sha256"]:
            raise ValueError("Calibration observation metadata mismatch")
        if tuple(meta["noise_shape"]) != NOISE_SHAPE:
            raise ValueError("Calibration capture uses an unsupported action shape")
        aux = capture["aux"]
        self._expected_tokens = aux["input_ids"].cpu().clone()
        self._expected_grid = aux["grid_thw"].cpu().clone()
        self._pixel_shape = tuple(aux["pixel_values"].shape)
        self._rec_to_dtype = _rec_to_dtype
        self._message_type = MessageType.EPISODE_STEP.value
        self._device = device
        try:
            self._model = flash_rt.load_model(
                checkpoint, config="groot_n17", framework="torch", hardware="thor",
                num_views=1, embodiment_tag="libero_sim", use_fp4=use_fp4,
                device=device,
            )
            self._frontend = self._model.pipeline
            requested_tier = "fp4" if use_fp4 else "fp8"
            if getattr(self._frontend, "_DIT_QUANT", None) != requested_tier:
                raise RuntimeError(f"FlashRT did not select requested {requested_tier} tier")
            # Existing caches may have been rewritten by another calibration
            # experiment. Bake this capture before graph construction, avoiding
            # both stale caches and the unsafe post-capture calibrate() path.
            self._frontend._load_calibration_cache = lambda: None
            try:
                self._frontend.set_prompt(aux=aux, prompt=self._prompt)
            finally:
                del self._frontend._load_calibration_cache
            self._processor = self._frontend._hf_processor()
            self._processor.eval()
            tag = EmbodimentTag.resolve("LIBERO_PANDA")
            self._prep = Gr00tPolicy.__new__(Gr00tPolicy)
            self._prep.processor, self._prep.embodiment_tag = self._processor, tag
            self._prep.modality_configs = self._processor.get_modality_configs()[tag.value]
            self._prep.language_key = self._prep.modality_configs["language"].modality_keys[0]
            if list(self._prep.modality_configs["video"].modality_keys) != ["image"]:
                raise ValueError("Checkpoint processor must use the single image camera")
            if len(self._prep.modality_configs["action"].delta_indices) != 16:
                raise ValueError("Checkpoint processor must decode 16 action steps")
            # Verify processor parity on setup data only; held-out pixels may change.
            processed = self._prepare(capture["inputs"])
            self._validate_prompt_grid(processed)
            if not torch.equal(processed["pixel_values"].cpu(),
                               aux["pixel_values"].to(processed["pixel_values"].dtype).cpu()):
                raise ValueError("Calibration pixels differ from official preprocessing")
            self.metadata = {
                "backend": "flashrt",
                "precision": "FP8 backbone / NVFP4 action head" if use_fp4 else "FP8",
                "checkpoint": str(Path(checkpoint).resolve()),
                "calibration_capture": str(Path(calibration_capture).resolve()),
                "calibration_observation_sha256": meta["observation_sha256"],
                "calibration_samples": 1,
                "calibration_frozen": True,
                "calibration_cache_policy": "recomputed from the dedicated capture before graph construction",
                "prompt": self._prompt,
                "image_grid_thw": self._expected_grid.tolist(),
                "noise_shape": list(NOISE_SHAPE),
                "noise_input": "caller-supplied CPU float32; copied to GPU and cast to bfloat16 per call",
                "action_shape": [1, 16, 7],
                "num_inference_timesteps": 4,
                "cpu_threads": torch.get_num_threads(),
                "device": str(self._frontend.device),
                "timing_boundary": "raw CPU observation/noise to physical CPU action arrays",
            }
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"FlashRT cleanup also failed: {cleanup_error}")
            raise

    def _prepare(self, observation: dict):
        steps = [self._processor([{
            "type": self._message_type,
            "content": self._prep._to_vla_step_data(obs),
        }]) for obs in self._prep._unbatch_observation(observation)]
        return self._rec_to_dtype(self._processor.collator(steps),
                                  self._torch.bfloat16)["inputs"]

    def _validate_prompt_grid(self, processed):
        for key, expected in (("input_ids", self._expected_tokens),
                              ("image_grid_thw", self._expected_grid)):
            actual = processed[key].cpu()
            if not self._torch.equal(actual, expected.to(actual.dtype)):
                raise ValueError(f"Changed {key}; construct a new backend for this prompt/grid")
        if tuple(processed["pixel_values"].shape) != self._pixel_shape:
            raise ValueError("Image patch shape differs from the captured graph")
        if tuple(processed["state"].shape) != (1, 1, 132):
            raise ValueError("Expected normalized state shape (1, 1, 132)")

    def input_fingerprints(self, observation: dict) -> dict:
        """Hash official processed inputs outside inference and timing.

        BF16 values are represented as float32 because NumPy cannot directly
        expose BF16 storage. This preserves every BF16 value exactly and gives
        the JAL and FlashRT stacks an identical serialization contract.
        """
        if self._closed:
            raise RuntimeError("FlashRT backend is closed")
        if _observation_prompt(observation) != self._prompt:
            raise ValueError("Changed instruction; construct a new FlashRT backend")
        with self._torch.inference_mode():
            processed = self._prepare(observation)
            self._validate_prompt_grid(processed)
            result = {}
            for key in ("pixel_values", "input_ids", "image_grid_thw", "state"):
                value = processed[key].detach().cpu()
                if value.dtype == self._torch.bfloat16:
                    value = value.float()
                array = np.ascontiguousarray(value.numpy())
                result[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                               "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            return result

    def get_action(self, observation: dict, noise: np.ndarray) -> dict:
        if self._closed:
            raise RuntimeError("FlashRT backend is closed")
        if _observation_prompt(observation) != self._prompt:
            raise ValueError("Changed instruction; construct a new FlashRT backend")
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != self.noise_shape or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite diffusion noise of shape {self.noise_shape}")
        torch = self._torch
        with torch.inference_mode():
            processed = self._prepare(observation)
            self._validate_prompt_grid(processed)
            self._frontend._backbone_features = self._frontend.run_backbone_graph({
                "pixel_values": processed["pixel_values"].to(self._device),
            })
            initial_noise = torch.from_numpy(np.ascontiguousarray(noise)).to(
                device=self._device, dtype=torch.bfloat16).contiguous()
            normalized = self._frontend.infer(
                processed["state"].to(self._device), initial_noise=initial_noise,
                num_inference_timesteps=4, action_horizon=40,
            )
            state = {"state." + key: value for key, value in observation["state"].items()}
            decoded = self._frontend.denormalize_action(normalized, state_dict=state)
            if set(decoded) != set(ACTION_KEYS):
                raise ValueError("Decoded action keys do not match LIBERO")
            actions = {key: decoded[key].detach().float().cpu().numpy().copy()
                       for key in ACTION_KEYS}
        for key, value in actions.items():
            if value.shape != (1, 16, 1) or not np.isfinite(value).all():
                raise ValueError(f"Invalid physical action output for {key}")
        return actions

    def synchronize(self):
        if not self._closed:
            self._torch.cuda.synchronize(self._device)

    def close(self):
        if self._closed:
            return
        error = None
        try:
            self._torch.cuda.synchronize(self._device)
        except BaseException as exc:
            error = exc
        # A failed CUDA synchronization must not leave graph/model references
        # alive or prevent this backend from becoming closed.
        try:
            if self._frontend is not None:
                for name in ("_k_dit_graph", "_kbb_graph"):
                    if hasattr(self._frontend, name):
                        delattr(self._frontend, name)
        finally:
            self._prep = self._processor = self._frontend = self._model = None
            self._closed = True
            gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note(f"FlashRT empty_cache also failed: {exc}")
        if error is not None:
            raise error
