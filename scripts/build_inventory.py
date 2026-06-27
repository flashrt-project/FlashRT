#!/usr/bin/env python3
"""Report the CUDA/C++ translation units compiled into each FlashRT pybind
module, read from a configured CMake build directory.

This is a build-structure measurement tool for the VLA-deployment kernel-build
split. It does NOT change anything at runtime; it only reports the current
compile surface so source-gating work (slim builds) can be measured and
reviewed unit by unit.

Source of truth: ``<build>/CMakeFiles/<target>.dir/DependInfo.cmake``. CMake
writes one of these per target after ``cmake`` configure, listing every source
the target compiles directly. This is generator-agnostic (Makefiles or Ninja)
and reflects the *configured* options (e.g. GPU_ARCH, FLASHRT_* gates), which
is exactly the surface we want to shrink.

Usage:
    python scripts/build_inventory.py                 # default build dir: ./build
    python scripts/build_inventory.py --build out      # custom build dir
    python scripts/build_inventory.py --json           # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# pybind modules whose compile surface this cleanup cares about.
TARGETS = (
    "flash_rt_kernels",
    "flash_rt_qwen3_vl_kernels",
    "flash_rt_fa2",
)

# Categorization of the flash_rt_kernels flat source list, mirroring the
# "Current Build Layout" groups in AGENTS.md. Used only to make the report
# readable; gating decisions still live in CMakeLists.txt.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "generic_shared": (
        "csrc/gemm/gemm_runner.cu",
        "csrc/kernels/norm.cu",
        "csrc/kernels/activation.cu",
        "csrc/kernels/rope.cu",
        "csrc/kernels/elementwise.cu",
        "csrc/kernels/fusion.cu",
        "csrc/kernels/patch_embed.cu",
        "csrc/kernels/softmax.cu",
        "csrc/kernels/attention_cublas.cu",
        "csrc/kernels/attention_mha.cu",
        "csrc/kernels/attention_mha_causal.cu",
        "csrc/kernels/decoder_fused.cu",
        "csrc/kernels/embedding_lookup_bf16.cu",
        "csrc/kernels/bf16_matmul_bf16.cu",
        "csrc/attention/fmha_dispatch.cu",
    ),
    "qwen36_linear_attention": (
        "csrc/kernels/causal_conv1d_qwen36.cu",
        "csrc/kernels/linear_attention/gated_delta_wy_bf16.cu",
        "csrc/kernels/linear_attention/gated_delta_wy_bf16_mma_fla.cu",
        "csrc/kernels/linear_attention/gated_delta_wy_output_o_mma_fla.cu",
        "csrc/kernels/linear_attention/gated_delta_wy_recompute_wu_mma_fla.cu",
        "csrc/kernels/gated_deltanet_qwen36.cu",
        "csrc/kernels/rms_norm_gated_silu_qwen36.cu",
        "csrc/kernels/silu_mul_qwen36.cu",
        "csrc/kernels/qwen36_misc.cu",
        "csrc/kernels/bf16_matvec_qwen36.cu",
        "csrc/kernels/bf16_matmul_qwen36.cu",
        "csrc/kernels/bf16_matmul_qwen36_thor.cu",
    ),
    "sm120_nvfp4_named": (
        "csrc/quantize/fp8_block128_to_nvfp4_swizzled.cu",
        "csrc/quantize/bf16_weight_to_nvfp4_swizzled.cu",
        "csrc/kernels/silu_mul_to_nvfp4_swizzled.cu",
        "csrc/kernels/fp4_w4a4_matvec_sm120.cu",
        "csrc/kernels/fp4_w4a4_mma_sm120.cu",
        "csrc/kernels/fp4_w4a4_mma_warpsplit_sm120.cu",
        "csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cu",
    ),
    "motus_video_fp8_history": (
        "csrc/quantize/ada_layer_norm_fp8.cu",
        "csrc/quantize/awq_quant_fp8_static_bf16.cu",
        "csrc/quantize/bf16_ndhwc_to_ncdhw_transpose.cu",
        "csrc/quantize/bf16_quant_fp8_ncdhw_to_ndhwc.cu",
        "csrc/quantize/bf16_rms_silu_ncdhw.cu",
        "csrc/quantize/bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4.cu",
        "csrc/quantize/bias_gelu_quantize_fp8.cu",
    ),
    "dit_video": (
        "csrc/kernels/dit_bf16.cu",
        "csrc/kernels/attention_dit_bf16.cu",
    ),
    "qwen3_family": (
        "csrc/kernels/rope_qwen3.cu",
        "csrc/kernels/qwen3_qkv_post_proc.cu",
    ),
}

_SRC_RE = re.compile(r"csrc/[A-Za-z0-9_/]+\.(?:cu|cpp)")


def target_sources(build_dir: Path, target: str) -> list[str] | None:
    """Return the sorted, de-duplicated csrc TUs a target compiles directly.

    Returns None if the target's DependInfo.cmake is absent (target not
    configured in this build dir).
    """
    dep = build_dir / "CMakeFiles" / f"{target}.dir" / "DependInfo.cmake"
    if not dep.is_file():
        return None
    text = dep.read_text(encoding="utf-8", errors="replace")
    return sorted(set(_SRC_RE.findall(text)))


def categorize(sources: list[str]) -> dict[str, list[str]]:
    """Bucket a target's sources by the AGENTS.md layout groups."""
    known = {src: cat for cat, srcs in CATEGORIES.items() for src in srcs}
    buckets: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    buckets["other"] = []
    for src in sources:
        buckets.setdefault(known.get(src, "other"), []).append(src)
    return {cat: srcs for cat, srcs in buckets.items() if srcs}


def collect(build_dir: Path) -> dict[str, object]:
    report: dict[str, object] = {"build_dir": str(build_dir), "targets": {}}
    for target in TARGETS:
        srcs = target_sources(build_dir, target)
        if srcs is None:
            report["targets"][target] = {"configured": False}
            continue
        entry: dict[str, object] = {"configured": True, "count": len(srcs)}
        if target == "flash_rt_kernels":
            cats = categorize(srcs)
            entry["categories"] = {c: len(v) for c, v in cats.items()}
            entry["category_sources"] = cats
        entry["sources"] = srcs
        report["targets"][target] = entry
    return report


def print_report(report: dict[str, object]) -> None:
    print(f"FlashRT build inventory  (build dir: {report['build_dir']})")
    print("=" * 64)
    targets: dict[str, dict] = report["targets"]  # type: ignore[assignment]
    for target in TARGETS:
        entry = targets[target]
        if not entry.get("configured"):
            print(f"\n{target}: NOT configured in this build dir")
            continue
        print(f"\n{target}: {entry['count']} directly-compiled TUs")
        cats = entry.get("categories")
        if cats:
            for cat, n in cats.items():
                print(f"    {cat:<28} {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default="build", type=Path,
                    help="configured CMake build directory (default: ./build)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table")
    args = ap.parse_args(argv)

    if not args.build.is_dir():
        print(f"error: build dir '{args.build}' does not exist; run cmake "
              f"configure first", file=sys.stderr)
        return 2

    report = collect(args.build)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
