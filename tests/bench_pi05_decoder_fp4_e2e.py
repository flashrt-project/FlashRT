#!/usr/bin/env python3
"""Locked-clock end-to-end A/B for the Pi0.5 Thor production NVFP4 path.

The suite launches FP8 and FP4 in separate processes. Both use the same
checkpoint, explicit prompt tokens, eight LIBERO observations, calibration,
and matched NumPy noise seeds. The FP4 mode combines all 17 live encoder FFN
layers with the decoder NVFP4 path. Latency covers the complete
``infer()`` call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


EXPECTED_DEVICE = "NVIDIA Thor"
EXPECTED_CC = (11, 0)
DEFAULT_WARMUP = 300
LATENCY_GROUP_COUNT = 10
RESULT_SCHEMA_VERSION = 2
README_REFERENCE_P50_MS = {1: 23.01, 2: 27.17, 3: 31.74}
REGRESSION_BASELINE_P50_MS = {1: 30.5, 2: 36.3, 3: 42.8}
REQUIRED_REGRESSION_MARGIN_MS = 2.0
TAIL_LATENCY_MS = 40.0
PROMPT_TOKENS = [
    2, 18075, 908, 573, 3118, 3963, 578,
    2040, 665, 575, 573, 24655, 108,
]

# The published Thor configuration, written as the sweep-knob values that
#   flash_rt.load_model(config="pi05", hardware="thor", framework="torch",
#                       use_fp4=True, use_fp4_decoder=True, use_fa4=True)
# resolves to (either through the load_model preset or through a frontend
# default load_model leaves untouched). --construct load_model refuses to run
# with any other value, so a published number can only come from a
# configuration the public API reproduces.
PUBLIC_API_PRESET = {
    "encoder_gu_mode": "p1",
    "encoder_p1_combiner": "epilogue_hw",
    "encoder_down_variant": 7,
    "encoder_down_x_variant": 6,
    "decoder_gate_up_variant": 10,
    "decoder_weight_format": "nvfp4",
    "decoder_act_format": "nvfp4",
    "decoder_rht": 0,
    "decoder_fused_attn": 0,
    "decoder_fused_geglu": 1,
    "awq_alpha": 0.8,
    "encoder_attn_o_fp4": 1,
    "encoder_attn_qkv_fp4": 0,
    "siglip_ffn_fp4": 1,
    "encoder_fp4_layer_count": 17,
}


def latency_group_medians(samples_ms: list[float]) -> list[float]:
    """Keep timing groups in run order so late warmup remains visible."""
    if len(samples_ms) < LATENCY_GROUP_COUNT:
        raise ValueError(
            f"latency diagnostics require at least {LATENCY_GROUP_COUNT} samples")
    return [
        float(np.median(group))
        for group in np.array_split(samples_ms, LATENCY_GROUP_COUNT)
    ]


def public_api_kwargs(mode: str, checkpoint: str, num_views: int) -> dict:
    """The exact load_model() call behind each published latency column."""
    kwargs = dict(
        checkpoint=checkpoint,
        framework="torch",
        config="pi05",
        hardware="thor",
        num_views=num_views,
        autotune=3,
        use_fa4=True,
    )
    if mode == "fp4":
        kwargs.update(use_fp4=True, use_fp4_decoder=True)
    return kwargs


def machine_state() -> dict[str, object]:
    power = subprocess.run(
        ["nvpmodel", "-q"], check=False, capture_output=True, text=True)
    if power.returncode != 0 or "NV Power Mode: MAXN" not in power.stdout:
        raise RuntimeError("E2E benchmark requires `sudo nvpmodel -m 0`")

    gpu_nodes: dict[str, dict[str, int]] = {}
    for name in ("gpu-gpc-0", "gpu-nvd-0"):
        matches = list(Path("/sys/devices").glob(f"**/{name}/devfreq/{name}"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {name} devfreq node, found {len(matches)}")
        values = {
            key: int((matches[0] / key).read_text().strip())
            for key in ("min_freq", "max_freq", "cur_freq")
        }
        if len(set(values.values())) != 1:
            raise RuntimeError(
                f"E2E benchmark requires locked {name} clocks, got {values}")
        gpu_nodes[name] = values

    emc_path = Path("/sys/kernel/nvpmodel_clk_cap/emc")
    if not emc_path.is_file():
        raise RuntimeError(f"missing EMC cap node: {emc_path}")
    emc_cap = int(emc_path.read_text().strip())
    if emc_cap <= 0:
        raise RuntimeError(f"invalid EMC cap: {emc_cap}")
    return {
        "power_mode": "MAXN",
        "gpu_devfreq_hz": gpu_nodes,
        "emc_cap_hz": emc_cap,
        "locked": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-mode", choices=("fp8", "fp4"))
    parser.add_argument(
        "--checkpoint", default=os.environ.get("PI05_CHECKPOINT"),
        help="Pi0.5 checkpoint directory (defaults to $PI05_CHECKPOINT)")
    parser.add_argument(
        "--fixture", help="exact N=8 fixture; defaults to the selected view count")
    parser.add_argument("--num-views", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help="warmup calls per mode; approximately 300 are recommended for "
             "stable Thor timing")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--encoder-gu-mode", choices=("p1", "merged"), default="p1")
    parser.add_argument(
        "--encoder-p1-combiner",
        choices=("direct", "lut", "lut_native", "epilogue", "epilogue_hw"),
        default="epilogue_hw")
    parser.add_argument("--encoder-down-variant", type=int, default=7)
    parser.add_argument("--encoder-down-x-variant", type=int, default=6)
    parser.add_argument("--decoder-gate-up-variant", type=int, default=10)
    parser.add_argument(
        "--decoder-weight-format", choices=("nvfp4", "e0m3"),
        default="nvfp4",
        help="Decoder projection weight element format: nvfp4 (E2M1, "
             "default) or e0m3 (uniform INT4 via the runtime instruction "
             "descriptor)")
    parser.add_argument(
        "--decoder-act-format", choices=("nvfp4", "e0m3"),
        default="nvfp4",
        help="Decoder activation element format (e0m3 requires "
             "--decoder-weight-format e0m3)")
    parser.add_argument(
        "--decoder-rht", type=int, choices=(0, 1), default=0,
        help="Per-16 Hadamard rotation on decoder weights + activations "
             "(requires --decoder-act-format e0m3)")
    parser.add_argument(
        "--decoder-fused-attn", type=int, choices=(0, 1), default=0,
        help="Fold the decoder attention seqused mask into the softmax "
             "kernel. Only reaches the fixed-shape state-prompt path, "
             "which this suite does not exercise")
    parser.add_argument(
        "--decoder-fused-geglu", type=int, choices=(0, 1), default=1,
        help="Fused-epilogue decoder FFN: one interleaved GeGLU GEMM "
             "replaces the gate_up GEMM + GeGLU kernel (nvfp4 weights only)")
    parser.add_argument("--awq-alpha", type=float, default=0.8)
    parser.add_argument(
        "--encoder-attn-o-fp4", type=int, choices=(0, 1), default=1,
        help="NVFP4 encoder attention O projections")
    parser.add_argument(
        "--encoder-attn-qkv-fp4", type=int, choices=(0, 1), default=0,
        help="NVFP4 encoder attention QKV projections (AWQ-calibrated). "
             "Off by default: passes every gate but the encoder-M QKV GEMM "
             "is not weight-bandwidth-bound, so FP4 only matches FP8 while "
             "the extra quantize step costs ~0.3 ms and precision margin")
    parser.add_argument(
        "--siglip-ffn-fp4", type=int, choices=(0, 1), default=1,
        help="NVFP4 SigLIP FFN on all 27 vision layers")
    parser.add_argument(
        "--encoder-fp4-layer-count", type=int, choices=range(18),
        default=17, help="FP4-quantize this many leading live encoder FFNs")
    parser.add_argument(
        "--construct", choices=("load_model", "frontend"),
        default="load_model",
        help="How to build the pipelines. 'load_model' (default, and the "
             "mode the published latency table is measured in) goes through "
             "the public flash_rt.load_model() API and rejects any sweep "
             "knob that deviates from the published preset. 'frontend' "
             "instantiates the frontend classes directly and is what the "
             "non-default sweep knobs require.")
    parser.add_argument(
        "--cuda-profile", action="store_true",
        help="capture one stable-state infer with cudaProfilerStart/Stop")
    args = parser.parse_args()

    if args.warmup < 5 or args.iters < 20:
        raise ValueError("strict E2E requires --warmup >= 5 and --iters >= 20")
    if args.construct == "load_model":
        deviations = {
            name: getattr(args, name)
            for name, expected in PUBLIC_API_PRESET.items()
            if getattr(args, name) != expected
        }
        if deviations:
            raise ValueError(
                "--construct load_model runs exactly the configuration "
                "flash_rt.load_model() produces, so these knobs cannot be "
                f"overridden: {deviations} (published preset: "
                f"{ {k: PUBLIC_API_PRESET[k] for k in deviations} }). "
                "Re-run with --construct frontend to sweep them; results "
                "from that mode are not public-API numbers.")
    if args.awq_alpha <= 0:
        raise ValueError("--awq-alpha must be positive")
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (or set $PI05_CHECKPOINT)")
    if args.fixture is None:
        fixture_dir = os.environ.get("PI05_FIXTURE_DIR")
        if fixture_dir is None:
            parser.error(
                "--fixture is required (or set $PI05_FIXTURE_DIR to the "
                "directory holding libero_obs_<views>v_n8.npz)")
        args.fixture = os.path.join(
            fixture_dir, f"libero_obs_{args.num_views}v_n8.npz")
    state = machine_state()

    if args.child_mode is not None:
        import torch

        import flash_rt.flash_rt_fp4 as fvk_fp4
        import flash_rt.flash_rt_kernels as fvk
        from flash_rt.hardware.thor import fa4_backend

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device_name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        if device_name != EXPECTED_DEVICE or capability != EXPECTED_CC:
            raise RuntimeError(
                f"expected {EXPECTED_DEVICE} {EXPECTED_CC}, got "
                f"{device_name} {capability}")
        if not fvk_fp4.has_nvfp4():
            raise RuntimeError("flash_rt_fp4 lacks NVFP4 support")
        if not fa4_backend.is_available():
            raise RuntimeError(
                "strict FP4+FA4 E2E requires active FA4: "
                f"{fa4_backend.status()}")
        if args.output_dir is None:
            raise ValueError("child mode requires --output-dir")

        if args.construct == "load_model":
            from flash_rt import load_model
            pipe = load_model(
                **public_api_kwargs(
                    args.child_mode, args.checkpoint, args.num_views)
            ).pipeline
        elif args.child_mode == "fp8":
            from flash_rt.frontends.torch.pi05_thor import (
                Pi05TorchFrontendThor)
            pipe = Pi05TorchFrontendThor(
                args.checkpoint, num_views=args.num_views, autotune=3,
                use_fa4=True)
        else:
            from flash_rt.frontends.torch.pi05_thor_fp4 import (
                Pi05TorchFrontendThorFP4)
            pipe = Pi05TorchFrontendThorFP4(
                args.checkpoint, num_views=args.num_views, autotune=3,
                use_fp4_encoder_ffn=True,
                fp4_layers=tuple(range(args.encoder_fp4_layer_count)),
                use_awq=True,
                awq_alpha=args.awq_alpha,
                use_p1_split_gu=args.encoder_gu_mode == "p1",
                encoder_p1_combiner=args.encoder_p1_combiner,
                encoder_down_variant=args.encoder_down_variant,
                encoder_down_x_variant=args.encoder_down_x_variant,
                decoder_gate_up_variant=args.decoder_gate_up_variant,
                decoder_weight_format=args.decoder_weight_format,
                decoder_act_format=args.decoder_act_format,
                decoder_rht=bool(args.decoder_rht),
                decoder_fused_attn=bool(args.decoder_fused_attn),
                decoder_fused_geglu=bool(args.decoder_fused_geglu),
                use_fp4_decoder=True,
                use_fa4=True,
                use_fp4_encoder_attn=bool(args.encoder_attn_o_fp4),
                use_fp4_encoder_attn_qkv=bool(args.encoder_attn_qkv_fp4),
                use_fp4_siglip_ffn=bool(args.siglip_ffn_fp4))

        data = np.load(args.fixture)
        count = int(data["n"])
        if count != 8:
            raise ValueError(f"strict E2E fixture requires 8 samples, got {count}")
        observations = []
        for index in range(count):
            observation = {
                "image": data[f"img_{index}"],
                "state": data[f"state_{index}"],
            }
            if args.num_views >= 2:
                observation["wrist_image"] = data[f"wrist_{index}"]
            if args.num_views == 3:
                observation["wrist_image_right"] = data[
                    f"wrist_right_{index}"]
            observations.append(observation)

        pipe.set_prompt(PROMPT_TOKENS)
        pipe.calibrate(observations, percentile=99.9, verbose=False)
        for index in range(args.warmup):
            pipe.infer(observations[index % count])

        raw_outputs = []
        action_outputs = []
        for index, observation in enumerate(observations):
            np.random.seed(args.seed + index)
            output = pipe.infer(observation)
            raw_outputs.append(pipe._g_noise.float().cpu().numpy())
            action_outputs.append(output["actions"])

        if args.cuda_profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            pipe.infer(observations[0])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

        latencies = []
        for index in range(args.iters):
            start = time.perf_counter()
            pipe.infer(observations[index % count])
            latencies.append((time.perf_counter() - start) * 1000.0)
        torch.cuda.synchronize()

        raw_array = np.stack(raw_outputs)
        action_array = np.stack(action_outputs)
        if not np.isfinite(raw_array).all() or not np.isfinite(action_array).all():
            raise RuntimeError(f"{args.child_mode} produced non-finite actions")
        np.savez(
            Path(args.output_dir) / f"{args.child_mode}_actions.npz",
            raw=raw_array, actions=action_array)
        result = {
            "mode": args.child_mode,
            "construction": args.construct,
            "public_api_call": (
                public_api_kwargs(
                    args.child_mode, "<checkpoint>", args.num_views)
                if args.construct == "load_model" else None),
            "device": device_name,
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash_rt_fp4_so": str(Path(fvk_fp4.__file__).resolve()),
            "flash_rt_fp4_sha256": hashlib.sha256(
                Path(fvk_fp4.__file__).read_bytes()).hexdigest(),
            "flash_rt_kernels_so": str(Path(fvk.__file__).resolve()),
            "flash_rt_kernels_sha256": hashlib.sha256(
                Path(fvk.__file__).read_bytes()).hexdigest(),
            "clock_state": state,
            "precision_config": (
                {
                    "encoder": "fp8",
                    "decoder": "fp8",
                    "attention": "fa4_siglip_encoder",
                }
                if args.child_mode == "fp8"
                else {
                    "encoder": "nvfp4_ffn",
                    "encoder_fp4_layers": list(range(
                        args.encoder_fp4_layer_count)),
                    "encoder_awq": True,
                    "encoder_awq_alpha": args.awq_alpha,
                    "encoder_attn_o": (
                        "nvfp4" if args.encoder_attn_o_fp4 else "fp8"),
                    "encoder_attn_qkv": (
                        "nvfp4_awq" if args.encoder_attn_qkv_fp4
                        else "fp8"),
                    "siglip_ffn": (
                        "nvfp4_all_27_layers" if args.siglip_ffn_fp4
                        else "fp8"),
                    "encoder_gu_mode": args.encoder_gu_mode,
                    "encoder_p1_combiner": args.encoder_p1_combiner,
                    "encoder_down_variant": args.encoder_down_variant,
                    "encoder_down_x_variant": args.encoder_down_x_variant,
                    "decoder": "nvfp4_all_projections",
                    "decoder_weight_format": args.decoder_weight_format,
                    "decoder_act_format": args.decoder_act_format,
                    "decoder_rht": bool(args.decoder_rht),
                    "decoder_fused_attn": bool(args.decoder_fused_attn),
                    "decoder_fused_geglu": bool(args.decoder_fused_geglu),
                    "decoder_gate_up_variant": args.decoder_gate_up_variant,
                    "attention": "fa4_siglip_encoder",
                }
            ),
            "fa4_status": fa4_backend.status(),
            "fa4_runtime": {
                "nvidia_cutlass_dsl": importlib.metadata.version(
                    "nvidia-cutlass-dsl"),
                "quack_kernels": importlib.metadata.version("quack-kernels"),
                "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
                "flash_attention_arch": os.environ.get(
                    "FLASH_ATTENTION_ARCH"),
            },
            "num_views": args.num_views,
            "denoise_steps": 10,
            "warmup": args.warmup,
            "iters": args.iters,
            "p50_ms": statistics.median(latencies),
            "p95_ms": float(np.percentile(latencies, 95)),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "latency_group_medians_ms": latency_group_medians(latencies),
            "samples_ms": latencies,
        }
        print("__E2E_RESULT__ " + json.dumps(result, sort_keys=True), flush=True)
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo_root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise RuntimeError(
            "strict E2E suite requires a clean tracked worktree; commit first")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True).stdout.strip()
    output_dir = Path(
        args.output_dir or f"/tmp/flashrt-pi05-decoder-fp4-e2e-{stamp}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    child_results = {}
    for mode in ("fp8", "fp4"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--child-mode", mode,
            "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--fixture", str(Path(args.fixture).resolve()),
            "--num-views", str(args.num_views),
            "--output-dir", str(output_dir.resolve()),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--seed", str(args.seed),
            "--encoder-gu-mode", args.encoder_gu_mode,
            "--encoder-p1-combiner", args.encoder_p1_combiner,
            "--encoder-down-variant", str(args.encoder_down_variant),
            "--encoder-down-x-variant", str(args.encoder_down_x_variant),
            "--decoder-gate-up-variant", str(args.decoder_gate_up_variant),
            "--decoder-weight-format", args.decoder_weight_format,
            "--decoder-act-format", args.decoder_act_format,
            "--decoder-rht", str(args.decoder_rht),
            "--decoder-fused-attn", str(args.decoder_fused_attn),
            "--decoder-fused-geglu", str(args.decoder_fused_geglu),
            "--awq-alpha", str(args.awq_alpha),
            "--encoder-fp4-layer-count", str(args.encoder_fp4_layer_count),
            "--encoder-attn-o-fp4", str(args.encoder_attn_o_fp4),
            "--encoder-attn-qkv-fp4", str(args.encoder_attn_qkv_fp4),
            "--siglip-ffn-fp4", str(args.siglip_ffn_fp4),
            "--construct", args.construct,
        ]
        child = subprocess.run(
            command, check=False, capture_output=True, text=True,
            env=os.environ.copy())
        if child.returncode != 0:
            raise RuntimeError(
                f"{mode} child failed rc={child.returncode}\n"
                f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}")
        lines = [
            line.removeprefix("__E2E_RESULT__ ")
            for line in child.stdout.splitlines()
            if line.startswith("__E2E_RESULT__ ")
        ]
        if len(lines) != 1:
            raise RuntimeError(
                f"expected one result from {mode} child, got {len(lines)}")
        child_results[mode] = json.loads(lines[0])

    fp8 = np.load(output_dir / "fp8_actions.npz")
    fp4 = np.load(output_dir / "fp4_actions.npz")
    raw_fp8 = fp8["raw"].astype(np.float64)
    raw_fp4 = fp4["raw"].astype(np.float64)
    action_fp8 = fp8["actions"].astype(np.float64)
    action_fp4 = fp4["actions"].astype(np.float64)
    raw_cosines = []
    action_cosines = []
    for index in range(raw_fp8.shape[0]):
        lhs = raw_fp4[index].reshape(-1)
        rhs = raw_fp8[index].reshape(-1)
        raw_cosines.append(float(
            lhs @ rhs / (np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1e-12)))
        lhs = action_fp4[index].reshape(-1)
        rhs = action_fp8[index].reshape(-1)
        action_cosines.append(float(
            lhs @ rhs / (np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1e-12)))

    raw_lhs = raw_fp4.reshape(-1)
    raw_rhs = raw_fp8.reshape(-1)
    action_lhs = action_fp4.reshape(-1)
    action_rhs = action_fp8.reshape(-1)
    fp8_p50 = float(child_results["fp8"]["p50_ms"])
    fp4_p50 = float(child_results["fp4"]["p50_ms"])
    fp4_p95 = float(child_results["fp4"]["p95_ms"])
    readme_reference = README_REFERENCE_P50_MS[args.num_views]
    regression_baseline = REGRESSION_BASELINE_P50_MS[args.num_views]
    regression_limit = regression_baseline - REQUIRED_REGRESSION_MARGIN_MS
    comparison = {
        "raw_cosine": float(
            raw_lhs @ raw_rhs /
            (np.linalg.norm(raw_lhs) * np.linalg.norm(raw_rhs) + 1e-12)),
        "raw_min_sample_cosine": min(raw_cosines),
        "raw_max_abs": float(np.max(np.abs(raw_fp4 - raw_fp8))),
        "action_cosine": float(
            action_lhs @ action_rhs /
            (np.linalg.norm(action_lhs) * np.linalg.norm(action_rhs) + 1e-12)),
        "action_min_sample_cosine": min(action_cosines),
        "action_max_abs": float(np.max(np.abs(action_fp4 - action_fp8))),
        "fp8_p50_ms": fp8_p50,
        "fp4_p50_ms": fp4_p50,
        "fp4_p95_ms": fp4_p95,
        "p50_delta_ms": fp4_p50 - fp8_p50,
        "p50_speedup": fp8_p50 / fp4_p50,
        "num_views": args.num_views,
        "readme_reference_p50_ms": readme_reference,
        "regression_baseline_p50_ms": regression_baseline,
        "required_regression_margin_ms": REQUIRED_REGRESSION_MARGIN_MS,
        "regression_limit_p50_ms": regression_limit,
    }
    gates = {
        "raw_cosine_at_least_0_995": comparison["raw_cosine"] >= 0.995,
        "raw_min_sample_cosine_at_least_0_995": (
            comparison["raw_min_sample_cosine"] >= 0.995),
        "action_cosine_at_least_0_999": comparison["action_cosine"] >= 0.999,
        "action_min_sample_cosine_at_least_0_995": (
            comparison["action_min_sample_cosine"] >= 0.995),
        "fp4_p50_faster_than_fp8": fp4_p50 < fp8_p50,
        "fp4_fa4_p50_within_regression_limit": fp4_p50 <= regression_limit,
    }
    if args.num_views == 2:
        gates["fp4_fa4_2v_p95_at_most_40_ms"] = fp4_p95 <= TAIL_LATENCY_MS
    if args.num_views == 3:
        gates["fp4_fa4_3v_p50_at_most_40_ms"] = fp4_p50 <= TAIL_LATENCY_MS
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "construction": args.construct,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "fixture": str(Path(args.fixture).resolve()),
        "prompt_tokens": PROMPT_TOKENS,
        "clock_state": state,
        "children": child_results,
        "comparison": comparison,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("__E2E_SUITE__ " + json.dumps(result, sort_keys=True))
    print(f"Artifacts: {output_dir}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
