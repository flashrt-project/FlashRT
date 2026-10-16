"""What to say when the compiled half is not there.

This distribution is pure Python. It carries the frontends, the structure
catalog, the host adapters and the kernel sources, but no ``.so``: kernels
either arrive through the kernel hub, which the structures layer talks to,
or are built locally for the models actually being run. So the first thing
a fresh install meets is an absent extension, and that moment has to carry
its own instructions — a bare ``ModuleNotFoundError: No module named
'flash_rt.flash_rt_kernels'`` reads as a broken package rather than as a
step the user has not taken yet.

The structures layer already answers this way: a refusal names its reason
and the rung below it. This module gives the native path the same manners.
"""

from __future__ import annotations

import importlib
import importlib.util

#: extension module → what it serves. ``flash_rt_kernels`` is the core
#: library every native frontend needs; the rest are per-family.
EXTENSIONS = {
    "flash_rt_kernels": "core kernels — every native frontend needs these",
    "flash_rt_fa2": "FlashAttention-2, the RTX attention path",
    "flash_rt_qwen3_vl_kernels": "Qwen3-VL SM89 FP8 kernels",
}

#: model family → the CMake switch that builds its kernels and nothing
#: else. Pair any of these with ``-DFLASHRT_SLIM_BUILD=ON``.
BUILD_SWITCHES = {
    "dm05": "FLASHRT_ENABLE_DM05",
    "lingbot": "FLASHRT_ENABLE_LINGBOT",
    "motus": "FLASHRT_ENABLE_MOTUS",
    "qwen3_5_moe": "FLASHRT_ENABLE_QWEN35MOE",
    "nexn2": "FLASHRT_ENABLE_QWEN35MOE",
    "qwen3_vl": "FLASHRT_BUILD_QWEN3_VL",
    "melband_roformer": "FLASHRT_ENABLE_MELBAND_ROFORMER",
    "omnivoice": "FLASHRT_ENABLE_OMNIVOICE",
    "audio_codebook": "FLASHRT_ENABLE_AUDIO_CODEBOOK",
    "minimax_remover": "FLASHRT_ENABLE_MINIMAX_REMOVER",
}

_REPO = "https://github.com/flashrt-project/FlashRT"


def present(name: str = "flash_rt_kernels") -> bool:
    """True if the named extension is importable beside this package."""
    try:
        return importlib.util.find_spec("flash_rt." + name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def missing() -> list[str]:
    """The extensions this install does not have."""
    return [n for n in EXTENSIONS if not present(n)]


def build_command(config: str | None = None) -> str:
    """The shortest build that produces what ``config`` needs."""
    switch = BUILD_SWITCHES.get(config or "")
    if switch:
        return ("cmake -B build -S . -DFLASHRT_SLIM_BUILD=ON "
                "-D%s=ON && cmake --build build -j$(nproc)" % switch)
    return "cmake -B build -S . && cmake --build build -j$(nproc)"


def require(name: str = "flash_rt_kernels", *, config: str | None = None):
    """Return the extension module, or refuse with reason and next step.

    Raises :class:`ImportError` so call sites that already guard imports
    keep behaving as they did when the extension shipped in the wheel.
    """
    if present(name):
        try:
            return importlib.import_module("flash_rt." + name)
        except ImportError as exc:
            # Built, but not for this environment. Worth saying plainly:
            # the fix is a rebuild, not a build, and the two look alike
            # from the traceback alone.
            raise ImportError(
                "flash_rt.{name} is present but will not load here:\n"
                "    {exc}\n"
                "The extension is compiled against a specific Python, torch\n"
                "and CUDA ABI. Rebuild it in the environment you are running:\n"
                "    {build}".format(name=name, exc=exc,
                                     build=build_command(config))) from exc
    raise ImportError(
        "flash_rt.{name} is not built ({what}).\n"
        "\n"
        "This distribution ships pure Python; the CUDA extensions are built\n"
        "from the source tree, for the hardware and the models you run:\n"
        "    git clone {repo} && cd FlashRT\n"
        "    pip install -e .\n"
        "    {build}\n"
        "The editable install matters: the build writes the extensions into\n"
        "the clone's own flash_rt/ directory, so that clone has to be what\n"
        "your interpreter imports.\n"
        "Add -DGPU_ARCH=120 for RTX 5090, 110 for Jetson Thor, 89 for RTX\n"
        "4090; see the Build section of the README for the full table.\n"
        "\n"
        "To run without a local build, use the structures layer instead: it\n"
        "obtains kernels from the kernel hub and refuses legibly when one is\n"
        "unavailable.\n"
        "    from flash_rt import structures".format(
            name=name, what=EXTENSIONS.get(name, "compiled extension"),
            repo=_REPO, build=build_command(config)))


def report() -> str:
    """One line per extension: present, or the switch that builds it."""
    lines = []
    for name, what in EXTENSIONS.items():
        mark = "present" if present(name) else "absent"
        lines.append("  %-28s %-8s  %s" % (name, mark, what))
    if missing():
        lines.append("")
        lines.append("  build:  " + build_command())
    return "\n".join(lines)
