"""Exact RDNA BF16 targets and their setup-time provider bindings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RdnaCapabilities:
    hardware: str
    gpu_arch: str
    streamed_smallm_max_rows: int = 0

    def make_backends(self, kernels, dtype):
        from . import rdna3, rdna35, rdna4

        providers = {
            "amd_rdna3": (rdna3.Rdna3GemmBackend, rdna3.Rdna3AttentionBackend),
            "amd_rdna35": (rdna35.Rdna35GemmBackend, rdna35.Rdna35AttentionBackend),
            "amd_rdna4": (rdna4.Rdna4GemmBackend, rdna4.Rdna4AttentionBackend),
        }
        gemm, attention = providers[self.hardware]
        return gemm(kernels, dtype), attention(kernels)


RDNA3 = RdnaCapabilities("amd_rdna3", "gfx1100", streamed_smallm_max_rows=16)
RDNA35 = RdnaCapabilities("amd_rdna35", "gfx1151")
RDNA4 = RdnaCapabilities("amd_rdna4", "gfx1201", streamed_smallm_max_rows=16)
_TARGETS = {target.gpu_arch: target for target in (RDNA3, RDNA35, RDNA4)}
SUPPORTED_HARDWARE = tuple(target.hardware for target in _TARGETS.values())


def load_capabilities(kernels, device_arch: str, expected_hardware=None):
    """Validate request, PyTorch device, extension device and compiled ISA."""
    target = _TARGETS.get(device_arch.split(":", 1)[0])
    if target is None:
        raise RuntimeError(
            f"unsupported RDNA device {device_arch!r}; "
            "expected gfx1100, gfx1151 or gfx1201")
    if expected_hardware not in (None, target.hardware):
        raise RuntimeError(
            f"RDNA device {device_arch!r} requires hardware={target.hardware!r}, "
            f"got {expected_hardware!r}")
    info = dict(kernels.build_info())
    build = str(info.get("gpu_arch", "")).split(":", 1)[0]
    extension_device = str(kernels.device_arch()).split(":", 1)[0]
    if (info.get("backend") != "rdna" or info.get("wave_size") != 32
            or info.get("hardware") != target.hardware
            or (build, extension_device) != (target.gpu_arch,) * 2):
        raise RuntimeError(
            f"{target.hardware} requires matching {target.gpu_arch} RDNA kernels "
            f"and device; build={info!r}, device={device_arch!r}, "
            f"extension_device={extension_device!r}")
    required = (
        "RdnaGemmRunner", "attention_encoder_gqa_rdna",
        "attention_decoder_gqa_rdna", "attention_decoder_gqa_splitkey_rdna",
        "smallm_wmma_bf16_rdna", "smallm_wmma_bf16_residual_rdna",
    )
    missing = [name for name in required if not callable(getattr(kernels, name, None))]
    if missing:
        raise RuntimeError("RDNA extension is missing required kernels: " + ", ".join(missing))
    return target
