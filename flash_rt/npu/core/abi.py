"""Identity checks for the Ascend shared objects.

The backend ships one separately loaded shared object per translation unit —
four for Pi0.5 (the vector dispatch unit, the gate/up cube unit, the decoder
GEMM and the decode attention) and three for the GR00T N1.7 action head and
image path, each group built only when that model is selected. They are
separate because the kernel headers define a per-translation-unit tiling symbol
and a cube-only unit cannot share a file scope with a mixed one, and each
honours its own environment override so a developer can swap one out.

Nothing else stops a process from mixing a freshly built library with a stale
one, or with one compiled for a different part. Both failures are silent: the
symbol names are identical either way and only the contract behind them has
moved. So every library exports the same pair of identity symbols (see
``csrc/npu/abi.h``) and every loader checks them here before binding anything.
"""
import ctypes as C

# Bumped whenever an exported entry point's signature or contract changes.
# ``scripts/npu/build.sh`` compiles the same number into every shared object.
#
# Adding a new unit with new symbols is not such a change: the contract behind
# every already-published entry point is untouched, and bumping would refuse
# every shared object a user has already built for no reason. Nor does
# iterating on a *new* entry point before it is released — the number tracks
# what has shipped, not how many times an unreleased signature was edited.
ABI_VERSION = 1

# The only part these kernels are validated for. The host tiling names it and
# several kernels divide work by its twenty cube cores.
SOC_VERSION = "Ascend910B4"


def _read(library, name, restype):
    try:
        symbol = getattr(library, name)
    except AttributeError as exc:
        raise ImportError(
            f"the Ascend library at {getattr(library, '_name', '?')} does not export "
            f"{name}; rebuild every library with scripts/npu/build.sh") from exc
    symbol.restype = restype
    symbol.argtypes = []
    return symbol()


def verify(library, role: str) -> str:
    """Check one loaded shared object and return the SoC it was built for.

    ``role`` names the library in the error, because the whole point of the
    check is to say *which* of the four is out of step.
    """
    abi = _read(library, "flashrt_npu_abi_version", C.c_int)
    if abi != ABI_VERSION:
        raise ImportError(
            f"the Ascend {role} library is ABI {abi}, this build of FlashRT "
            f"expects {ABI_VERSION}; rebuild every library with "
            f"scripts/npu/build.sh (a partial rebuild is what this catches)")
    compiled = _read(library, "flashrt_npu_soc_version", C.c_char_p).decode()
    if compiled != SOC_VERSION:
        raise ImportError(
            f"the Ascend {role} library targets {compiled}, but these kernels "
            f"are validated for {SOC_VERSION} only")
    running = _running_soc()
    if running is not None and compiled != running:
        raise RuntimeError(
            f"the Ascend {role} library targets {compiled}, but the current "
            f"device is {running}")
    return compiled


def _running_soc():
    """The device's own name, or None when there is no device to ask.

    None means exactly one thing: no Ascend runtime, or no usable device. That
    keeps both halves of the check usable on a machine with no NPU, which is
    where most of the tests for this file run.

    It does not mean "the query failed". Once a device is present, failing to
    name it is a fault and propagates: swallowing it would skip the hardware
    half of the check on precisely the machine the check exists for.
    """
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:
        return None
    if not torch.npu.is_available():
        return None
    return str(torch.npu.get_device_name(torch.npu.current_device()))
