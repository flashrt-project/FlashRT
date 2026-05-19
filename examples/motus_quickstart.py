#!/usr/bin/env python3
"""Motus FlashRT quickstart.

Runs the Motus E2E inference path with the same input bundle shape used by
the upstream Motus reference: prompt embeddings, VLM inputs, first frame, and
robot state are prepared outside FlashRT and passed to set_prompt()/infer().

Examples:
    # Set MOTUS_ROOT, MOTUS_CHECKPOINT, MOTUS_WAN_PATH, MOTUS_VLM_PATH, and
    # MOTUS_INPUT_BUNDLE first, or pass the same paths as CLI arguments.
    #
    # Default validated Stage3 fast profile
    python examples/motus_quickstart.py --benchmark 5

    # Explicit FP8 trajectory baseline profile
    python examples/motus_quickstart.py --fp4-profile off --benchmark 5

    # Explicit FP4/NVFP4 experiment profile
    python examples/motus_quickstart.py --fp4-profile on --benchmark 5

    # Alias for the default fast profile.
    python examples/motus_quickstart.py --fp4-profile fast --benchmark 5
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys
import time
import types
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


def _load_tensor(path: pathlib.Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu")


def _cosine(a, b) -> float:
    import torch

    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    return float(torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)).item())


def _load_inputs(bundle: pathlib.Path):
    inputs = bundle / "inputs"
    first_frame = _load_tensor(inputs / "first_frame.pt")
    state = _load_tensor(inputs / "state.pt")
    instruction = (inputs / "instruction.txt").read_text().strip()
    t5_embeds = _load_tensor(inputs / "t5_embed.pt")
    vlm_inputs = _load_tensor(inputs / "vlm_inputs.pt")
    if not isinstance(t5_embeds, list):
        t5_embeds = [t5_embeds]
    if not isinstance(vlm_inputs, list):
        vlm_inputs = [vlm_inputs]
    return first_frame, state, instruction, t5_embeds, vlm_inputs


def _load_optional_refs(bundle: pathlib.Path):
    outputs = bundle / "outputs"
    actions_path = outputs / "predicted_actions.pt"
    frames_path = outputs / "predicted_frames.pt"
    if not actions_path.exists() or not frames_path.exists():
        return None, None
    return _load_tensor(frames_path), _load_tensor(actions_path)


def _set_motus_runtime_defaults() -> None:
    os.environ.setdefault("FLASH_RT_MOTUS_AUTO_G7_23_VAE_FP8", "1")
    os.environ.setdefault("FLASH_RT_MOTUS_STATIC_MOD_SKIP_LAST_N", "0")
    os.environ.setdefault("FLASH_RT_MOTUS_STATIC_MOD_FP8", "1")
    os.environ.setdefault("FLASH_RT_MOTUS_STATIC_MOD_FP8_LAST_N", "6")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _install_deepspeed_stub() -> None:
    if "deepspeed.comm.comm" in sys.modules:
        return
    import importlib.machinery
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.__spec__ = importlib.machinery.ModuleSpec(
        "deepspeed", loader=None)
    comm_pkg = types.ModuleType("deepspeed.comm")
    comm_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "deepspeed.comm", loader=None)
    comm = types.ModuleType("deepspeed.comm.comm")
    comm.__spec__ = importlib.machinery.ModuleSpec(
        "deepspeed.comm.comm", loader=None)
    comm.get_rank = lambda: 0
    comm.get_world_size = lambda: 1
    comm.is_available = lambda: False
    comm.is_initialized = lambda: False
    comm.barrier = lambda *args, **kwargs: None
    comm.init_process_group = lambda *args, **kwargs: None
    comm.all_to_all = lambda *args, **kwargs: None
    deepspeed.comm = comm_pkg
    comm_pkg.comm = comm
    sys.modules.setdefault("deepspeed", deepspeed)
    sys.modules.setdefault("deepspeed.comm", comm_pkg)
    sys.modules.setdefault("deepspeed.comm.comm", comm)


def _install_optional_import_stubs(motus_root: pathlib.Path) -> None:
    import torch

    if "wan" not in sys.modules:
        wan = types.ModuleType("wan")
        wan.__path__ = [str(motus_root / "bak" / "wan")]
        sys.modules["wan"] = wan
    if "wan.modules" not in sys.modules:
        wan_modules = types.ModuleType("wan.modules")
        wan_modules.__path__ = [str(motus_root / "bak" / "wan" / "modules")]
        sys.modules["wan.modules"] = wan_modules
    if "wan.utils" not in sys.modules:
        wan_utils = types.ModuleType("wan.utils")
        wan_utils.__path__ = [str(motus_root / "bak" / "wan" / "utils")]
        sys.modules["wan.utils"] = wan_utils
    if "wan.utils.fm_solvers_unipc" not in sys.modules:
        fm_solvers_unipc = types.ModuleType("wan.utils.fm_solvers_unipc")

        class FlowUniPCMultistepScheduler:
            def __init__(self, num_train_timesteps=1000, shift=1.0, **kwargs):
                self.num_train_timesteps = num_train_timesteps
                self.shift = shift
                self.timesteps = torch.empty(0, dtype=torch.int64)

            def set_timesteps(self, num_inference_steps, device=None, shift=None,
                              **kwargs):
                if shift is not None:
                    self.shift = shift
                self.timesteps = torch.linspace(
                    self.num_train_timesteps - 1,
                    0,
                    num_inference_steps,
                    device=device,
                    dtype=torch.float32,
                ).to(torch.int64)

            def step(self, model_output, timestep, sample, return_dict=False,
                     **kwargs):
                prev = sample + model_output * 0
                if return_dict:
                    return {"prev_sample": prev}
                return (prev,)

        fm_solvers_unipc.FlowUniPCMultistepScheduler = FlowUniPCMultistepScheduler
        sys.modules["wan.utils.fm_solvers_unipc"] = fm_solvers_unipc
    if "imageio" not in sys.modules:
        imageio = types.ModuleType("imageio")
        imageio.config = types.SimpleNamespace(video_extensions=())
        sys.modules["imageio"] = imageio
    if "easydict" not in sys.modules:
        easydict = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

            def __setattr__(self, key, value):
                self[key] = value

            def __delattr__(self, key):
                del self[key]

        easydict.EasyDict = EasyDict
        sys.modules["easydict"] = easydict
    if "diffusers" not in sys.modules:
        diffusers = types.ModuleType("diffusers")
        configuration_utils = types.ModuleType("diffusers.configuration_utils")
        models_pkg = types.ModuleType("diffusers.models")
        modeling_utils = types.ModuleType("diffusers.models.modeling_utils")

        class ConfigMixin:
            def register_to_config(self, **kwargs):
                cfg = getattr(self, "config", types.SimpleNamespace())
                for key, value in kwargs.items():
                    setattr(cfg, key, value)
                self.config = cfg

        class ModelMixin(torch.nn.Module):
            pass

        def register_to_config(fn):
            sig = inspect.signature(fn)

            def wrapped(self, *args, **kwargs):
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                cfg = types.SimpleNamespace()
                for key, value in bound.arguments.items():
                    if key != "self":
                        setattr(cfg, key, value)
                self.config = cfg
                return fn(self, *args, **kwargs)

            return wrapped

        configuration_utils.ConfigMixin = ConfigMixin
        configuration_utils.register_to_config = register_to_config
        modeling_utils.ModelMixin = ModelMixin
        models_pkg.modeling_utils = modeling_utils
        diffusers.configuration_utils = configuration_utils
        diffusers.models = models_pkg
        sys.modules["diffusers"] = diffusers
        sys.modules["diffusers.configuration_utils"] = configuration_utils
        sys.modules["diffusers.models"] = models_pkg
        sys.modules["diffusers.models.modeling_utils"] = modeling_utils
    if "ftfy" not in sys.modules:
        import importlib.machinery
        ftfy = types.ModuleType("ftfy")
        ftfy.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
        ftfy.fix_text = lambda text: text
        sys.modules["ftfy"] = ftfy


def _install_wan_config_filter() -> None:
    orig_load = json.load

    def load_strip_diffusers_metadata(fp, *args, **kwargs):
        cfg = orig_load(fp, *args, **kwargs)
        if isinstance(cfg, dict):
            cfg.pop("_class_name", None)
            cfg.pop("_diffusers_version", None)
        return cfg

    json.load = load_strip_diffusers_metadata


def _patch_qwen3vl_image_features(frontend) -> None:
    import torch

    vlm = getattr(getattr(frontend.model, "und_module", None), "vlm_model", None)
    if vlm is None or getattr(vlm, "_flashrt_quickstart_patched", False):
        return
    orig = vlm.get_image_features

    def get_image_features_compat(pixel_values, image_grid_thw=None, **kwargs):
        # Older transformers Qwen3VL.get_image_features may not accept
        # return_dict — try with it, fall back without.
        try:
            out = orig(pixel_values, image_grid_thw,
                        return_dict=True, **kwargs)
        except TypeError:
            out = orig(pixel_values, image_grid_thw, **kwargs)
        if hasattr(out, "pooler_output"):
            return out.pooler_output, getattr(out, "deepstack_features", None)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[-1]
        return out, None

    vlm.get_image_features = get_image_features_compat
    qwen_model = getattr(vlm, "model", None)
    if qwen_model is not None and not getattr(qwen_model, "_flashrt_quickstart_rope", False):
        orig_rope = qwen_model.get_rope_index

        def get_rope_index_compat(input_ids, *args, **kwargs):
            old_args = (
                args and isinstance(args[0], torch.Tensor)
                and args[0].ndim == 2 and args[0].shape[-1] == 3)
            old_kwargs = "image_grid_thw" in kwargs and "mm_token_type_ids" not in kwargs
            if old_args or old_kwargs:
                image_grid_thw = args[0] if old_args else kwargs.get("image_grid_thw")
                video_grid_thw = (
                    args[1] if old_args and len(args) > 1
                    else kwargs.get("video_grid_thw"))
                attention_mask = (
                    args[2] if old_args and len(args) > 2
                    else kwargs.get("attention_mask"))
                image_token_id = getattr(
                    getattr(qwen_model, "config", None), "image_token_id", None)
                if image_token_id is None:
                    image_token_id = getattr(
                        getattr(vlm, "config", None), "image_token_id", None)
                if image_token_id is None:
                    image_token_id = 151655
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                mm_token_type_ids[input_ids == image_token_id] = 1
                try:
                    return orig_rope(
                        input_ids,
                        mm_token_type_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                    )
                except TypeError:
                    # Older transformers Qwen3VL: get_rope_index has
                    # signature (input_ids, image_grid_thw, video_grid_thw,
                    # attention_mask) with no mm_token_type_ids.
                    return orig_rope(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask,
                    )
            return orig_rope(input_ids, *args, **kwargs)

        qwen_model.get_rope_index = get_rope_index_compat
        qwen_model._flashrt_quickstart_rope = True
    vlm._flashrt_quickstart_patched = True


def _infer_once(pipe, first_frame, state, seed: int):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        t0 = _now()
        frames, actions = pipe.infer(first_frame, state=state)
        dt_ms = (_now() - t0) * 1000.0
    return frames, actions, dt_ms


def main() -> None:
    env_motus_root = os.environ.get("FLASH_RT_MOTUS_ROOT") or os.environ.get("MOTUS_ROOT")
    env_motus_root_path = pathlib.Path(env_motus_root) if env_motus_root else None

    def _default_from_root(*parts: str) -> str | None:
        if env_motus_root_path is None:
            return None
        return str(env_motus_root_path.joinpath(*parts))

    parser = argparse.ArgumentParser(description="FlashRT Motus quickstart")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MOTUS_CHECKPOINT")
        or _default_from_root("pretrained_models", "Motus"),
        help="Motus checkpoint directory")
    parser.add_argument(
        "--motus-root",
        default=env_motus_root,
        help="Upstream Motus repo path used for imports")
    parser.add_argument(
        "--wan-path",
        default=os.environ.get("MOTUS_WAN_PATH")
        or _default_from_root("pretrained_models", "Wan2.2-TI2V-5B"),
        help="Wan2.2 checkpoint/config directory")
    parser.add_argument(
        "--vlm-path",
        default=os.environ.get("MOTUS_VLM_PATH")
        or _default_from_root("pretrained_models", "Qwen3-VL-2B-Instruct"),
        help="Qwen3-VL checkpoint directory")
    parser.add_argument(
        "--input-bundle",
        default=os.environ.get("MOTUS_INPUT_BUNDLE")
        or _default_from_root("baseline_artifacts"),
        help="Directory containing inputs/ and optional outputs/")
    parser.add_argument(
        "--fp4-profile",
        default="fast",
        choices=["fast", "fast-cache", "fast-tiny", "on", "off"],
        help="Motus precision profile: fast is the default validated "
             "Stage3 profile. off is the explicit FP8 trajectory baseline. "
             "on enables the Motus FP4/NVFP4 experiment profile.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of Motus denoising steps. The committed latency/cosine "
             "baseline uses 10; other values are algorithm experiments and "
             "will recapture the CUDA graph.")
    parser.add_argument("--autotune", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=5,
                        help="Number of timed graph replays after warmup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-output", default=None,
                        help="Optional .pt path for {'frames','actions'}")
    parser.add_argument("--no-compare", action="store_true",
                        help="Do not compare against optional bundle outputs")
    args = parser.parse_args()

    required = {
        "--motus-root or FLASH_RT_MOTUS_ROOT/MOTUS_ROOT": args.motus_root,
        "--checkpoint or MOTUS_CHECKPOINT": args.checkpoint,
        "--wan-path or MOTUS_WAN_PATH": args.wan_path,
        "--vlm-path or MOTUS_VLM_PATH": args.vlm_path,
        "--input-bundle or MOTUS_INPUT_BUNDLE": args.input_bundle,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required Motus paths: " + ", ".join(missing))

    os.environ["FLASH_RT_MOTUS_ROOT"] = str(pathlib.Path(args.motus_root))
    os.environ["FLASH_RT_MOTUS_FP4_PROFILE"] = args.fp4_profile
    _set_motus_runtime_defaults()

    import torch

    assert torch.cuda.is_available(), "Motus FlashRT requires CUDA"
    torch.cuda.reset_peak_memory_stats()
    _install_deepspeed_stub()
    _install_optional_import_stubs(pathlib.Path(args.motus_root))
    _install_wan_config_filter()

    bundle = pathlib.Path(args.input_bundle)
    first_frame, state, instruction, t5_embeds, vlm_inputs = _load_inputs(bundle)
    ref_frames, ref_actions = (None, None) if args.no_compare else (
        _load_optional_refs(bundle))

    print(f"[motus.quickstart] checkpoint={args.checkpoint}")
    print(f"[motus.quickstart] input_bundle={bundle}")
    print(f"[motus.quickstart] fp4_profile={args.fp4_profile}")
    print(f"[motus.quickstart] num_inference_steps={args.num_inference_steps}")

    t0 = time.perf_counter()
    from flash_rt.frontends.torch.motus_rtx import MotusTorchFrontendRtx

    pipe = MotusTorchFrontendRtx(
        checkpoint_dir=args.checkpoint,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        num_inference_steps=args.num_inference_steps,
        autotune=args.autotune,
    )
    _patch_qwen3vl_image_features(pipe)
    print(f"[motus.quickstart] load wall={time.perf_counter() - t0:.1f}s")

    pipe.set_prompt(instruction, t5_embeds=t5_embeds, vlm_inputs=vlm_inputs)

    print("[motus.quickstart] warmup #1: FP8 calibration + CUDA graph capture")
    _, _, t_calib = _infer_once(pipe, first_frame, state, args.seed)
    print(f"[motus.quickstart] warmup #1 done: {t_calib:.1f} ms")

    print("[motus.quickstart] warmup #2: graph replay")
    _, _, t_capture = _infer_once(pipe, first_frame, state, args.seed)
    print(f"[motus.quickstart] warmup #2 done: {t_capture:.1f} ms")

    print("[motus.quickstart] warm replay")
    frames, actions, t_replay = _infer_once(pipe, first_frame, state, args.seed)
    print(f"[motus.quickstart] warm replay done: {t_replay:.3f} ms")

    times = []
    for i in range(max(0, args.benchmark)):
        frames, actions, dt_ms = _infer_once(pipe, first_frame, state, args.seed)
        times.append(dt_ms)
        print(f"[motus.quickstart] replay {i + 1}/{args.benchmark}: {dt_ms:.3f} ms")

    if times:
        ordered = sorted(times)
        p50 = ordered[len(ordered) // 2]
        print(f"[motus.quickstart] graph P50={p50:.3f} ms "
              f"min={min(times):.3f} max={max(times):.3f}")

    print(f"[motus.quickstart] frames shape={tuple(frames.shape)} "
          f"dtype={frames.dtype}")
    print(f"[motus.quickstart] actions shape={tuple(actions.shape)} "
          f"dtype={actions.dtype}")
    print("[motus.quickstart] cuda memory: "
          f"peak_allocated={torch.cuda.max_memory_allocated()/1e9:.3f} GB "
          f"peak_reserved={torch.cuda.max_memory_reserved()/1e9:.3f} GB "
          f"current_allocated={torch.cuda.memory_allocated()/1e9:.3f} GB")

    if ref_frames is not None and ref_actions is not None:
        print("[motus.quickstart] cosine vs bundle outputs: "
              f"action={_cosine(actions, ref_actions):.6f} "
              f"frames={_cosine(frames, ref_frames):.6f}")

    if args.save_output:
        out_path = pathlib.Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"frames": frames.detach().cpu(),
                    "actions": actions.detach().cpu()}, out_path)
        print(f"[motus.quickstart] saved {out_path}")


if __name__ == "__main__":
    main()
