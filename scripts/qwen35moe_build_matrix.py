#!/usr/bin/env python3
"""What each qwen3_5_moe build tier adds, and what a build without them has.

The claim these tiers make is that a build with every qwen3_5_moe option off
compiles the same sources and exports the same symbols it did before they
existed. That is a property of the gates, so it is checked by reading them:
which translation units CMake adds under each tier, and which ``m.def`` names
sit inside the matching preprocessor guard in the bindings.

Reading the gates rather than building has a specific limit and a specific
advantage. It cannot catch a kernel that fails to compile -- only a build does
that, and the configure matrix printed at the end is how to run one. It can
catch the thing a single build cannot: a source or a symbol that leaks into a
configuration nobody built.

    python scripts/qwen35moe_build_matrix.py            # print the matrix
    python scripts/qwen35moe_build_matrix.py --check    # exit 1 on a leak

``tests/test_qwen35moe_build_matrix.py`` runs the same checks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# CMake option -> the compile definition it sets -> what the bindings guard on.
TIERS = {
    "FLASHRT_ENABLE_QWEN35MOE_CORE": "FLASHRT_HAVE_QWEN35MOE_CORE",
    "FLASHRT_ENABLE_QWEN35MOE_W4A16": "FLASHRT_HAVE_QWEN35MOE_W4A16",
    "FLASHRT_ENABLE_QWEN35MOE_W4A4": "FLASHRT_HAVE_QWEN35MOE_W4A4",
}

# Gates that are not tiers but are still model-specific: the grouped MoE GEMM
# object, which only the weight-only tier on sm_110 builds.
EXTRA_GATES = ("FLASHRT_HAVE_QWEN35MOE_GROUPED_SM100",)

# Sources gated somewhere other than a tier, with the gate that owns each.
# Checked by name because that is the whole point: the grouped MoE GEMM used to
# be a second source in an object library every Thor build compiles.
ELSEWHERE = {
    "csrc/gemm/fp4/cutlass_nvfp4_moe_grouped_sm100.cu":
        "qwen35moe_nvfp4_grouped_sm100_obj",
}


def _cmake_text() -> str:
    return (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def _bindings_text() -> str:
    return (ROOT / "csrc" / "bindings.cpp").read_text(encoding="utf-8")


def tier_sources() -> dict[str, list[str]]:
    """Sources CMake adds inside each ``if(<tier>)`` block."""
    text = _cmake_text()
    out: dict[str, list[str]] = {}
    for option in TIERS:
        # The block runs from `if(<option>)` to its matching endif at column 0.
        start = text.index(f"if({option})\n")
        depth, i, end = 0, start, None
        for line in text[start:].split("\n"):
            stripped = line.strip()
            if stripped.startswith("if(") or stripped.startswith("if ("):
                depth += 1
            elif stripped == "endif()":
                depth -= 1
                if depth == 0:
                    end = i + len(line)
                    break
            i += len(line) + 1
        if end is None:                                     # pragma: no cover
            raise AssertionError(f"unterminated if({option}) in CMakeLists.txt")
        block = text[start:end]
        out[option] = sorted(re.findall(r"(csrc/[\w/]+\.cu)\b", block))
    return out


def guarded_symbols() -> dict[str, list[str]]:
    """``m.def`` names inside each gate's ``#ifdef`` region in bindings.cpp."""
    text = _bindings_text()
    gates = list(TIERS.values()) + list(EXTRA_GATES)
    out: dict[str, list[str]] = {gate: [] for gate in gates}
    active: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#if"):
            gate = next((g for g in gates if g in stripped), None)
            active.append(gate)
        elif stripped.startswith("#endif"):
            if active:
                active.pop()
        else:
            name = re.match(r'm\.def\("([\w]+)"', stripped)
            if name:
                for gate in active:
                    if gate is not None:
                        out[gate].append(name.group(1))
    return {gate: sorted(set(names)) for gate, names in out.items()}


def ungated_model_sources() -> list[str]:
    """Tier sources CMake also adds outside their tier block.

    A source listed under a tier and again anywhere else is compiled by a build
    that turned every tier off, which is the leak these gates exist to prevent.
    Checked by counting occurrences, not by guessing from names: a name says
    nothing about which gate compiles a file.
    """
    text = _cmake_text()
    leaked = []
    for srcs in tier_sources().values():
        for src in srcs:
            if text.count(src) != 1:
                leaked.append(src)
    for src, owner in ELSEWHERE.items():
        block = text[text.index(f"add_library({owner}"):]
        block = block[:block.index("endif()")]
        if text.count(src) != 1 or src not in block:
            leaked.append(src)
    return sorted(set(leaked))


def ungated_model_symbols() -> list[str]:
    """Tier symbols that bindings.cpp also defines outside their gate."""
    text = _bindings_text()
    gated = guarded_symbols()
    known = {name for names in gated.values() for name in names}
    seen_outside = []
    gates = list(TIERS.values()) + list(EXTRA_GATES)
    active: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#if"):
            active.append(next((g for g in gates if g in stripped), None))
        elif stripped.startswith("#endif"):
            if active:
                active.pop()
        else:
            name = re.match(r'm\.def\("([\w]+)"', stripped)
            if name and name.group(1) in known:
                if not any(g is not None for g in active):
                    seen_outside.append(name.group(1))
    return sorted(set(seen_outside))


# The configurations a reviewer should be able to reproduce, and what each is
# for. These are configure lines, not builds: the first four should configure,
# and the last is expected to fail.
CONFIGURE_MATRIX = [
    ("baseline sm_120",
     "-DGPU_ARCH=120",
     "no qwen3_5_moe source or symbol"),
    ("baseline sm_110",
     "-DGPU_ARCH=110",
     "no qwen3_5_moe source or symbol; no FA2"),
    ("sm_110 supported",
     "-DGPU_ARCH=110 -DFLASHRT_ENABLE_QWEN35MOE_CORE=ON "
     "-DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON -DFLASHRT_ENABLE_THOR_FA2=ON",
     "core + weight-only tiers, grouped MoE GEMM, FA2"),
    ("sm_120 supported",
     "-DGPU_ARCH=120 -DFLASHRT_ENABLE_QWEN35MOE=ON",
     "all three tiers"),
    ("sm_110 block-scaled (must fail)",
     "-DGPU_ARCH=110 -DFLASHRT_ENABLE_QWEN35MOE_W4A4=ON",
     "FATAL_ERROR: the block-scaled MMA tier needs sm_120a/sm_121a"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a tier source or symbol is ungated")
    args = parser.parse_args()

    sources, symbols = tier_sources(), guarded_symbols()
    for option, gate in TIERS.items():
        print(f"\n{option}  ->  {gate}")
        print(f"  sources ({len(sources[option])}):")
        for src in sources[option]:
            print(f"    {src}")
        print(f"  symbols ({len(symbols[gate])}):")
        for name in symbols[gate]:
            print(f"    {name}")
    for gate in EXTRA_GATES:
        print(f"\n{gate}")
        print(f"  symbols ({len(symbols[gate])}):")
        for name in symbols[gate]:
            print(f"    {name}")

    leaked_sources = ungated_model_sources()
    leaked_symbols = ungated_model_symbols()
    print("\nungated model-named sources:",
          ", ".join(leaked_sources) or "none")
    print("ungated tier symbols:", ", ".join(leaked_symbols) or "none")

    print("\nconfigure matrix:")
    for name, flags, expect in CONFIGURE_MATRIX:
        print(f"  {name}\n    cmake -S . -B build {flags}\n    -> {expect}")

    if args.check and (leaked_sources or leaked_symbols):
        print("\nFAIL: the above enter a build with every tier off",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
