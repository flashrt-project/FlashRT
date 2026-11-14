"""Higgs Audio v3 TTS with explicit CUDA and RDNA frontend routing.

The acoustic backbone is a standard dense Qwen3-4B; audio is produced by a
fused multi-codebook embedding/head over an 8-codebook neural codec at 25 Hz.
The hand-written forward path lives in
``flash_rt.frontends.torch.higgs_audio_v3_rtx``.
"""

from flash_rt.models.higgs_audio_v3.pipeline_rtx import (
    HiggsAudioV3Dims,
    HiggsAudioV3Pipeline,
)


def create_frontend(checkpoint_path, *, hardware="auto", **kwargs):
    """Construct the TTS frontend; its API is independent of the VLA wrapper.

    Pass the same ``device``, ``fp8`` and capacity options as to the direct
    CUDA/RDNA constructor. Explicit RDNA hardware IDs are checked against the
    selected device before loading the checkpoint.
    """
    import torch
    if hardware == "auto" and not torch.version.hip:
        # Preserve the original CUDA constructor, including its device handling
        # and compiled-kernel probes. The registry is not a CUDA whitelist.
        from flash_rt.frontends.torch.higgs_audio_v3_rtx import HiggsAudioV3TorchFrontendRtx
        return HiggsAudioV3TorchFrontendRtx(checkpoint_path, **kwargs)

    from flash_rt.hardware import resolve_pipeline_class
    if hardware == "auto":
        from flash_rt.amd.hardware.capabilities_rdna import device_capabilities
        with torch.cuda.device(kwargs.get("device", "cuda:0")):
            arch = device_capabilities(torch.cuda.get_device_properties().gcnArchName).hardware
    else:
        arch = hardware
    frontend = resolve_pipeline_class("higgs_audio_v3", "torch", arch)
    if arch in ("amd_rdna3", "amd_rdna35", "amd_rdna4"):
        kwargs["hardware"] = arch
    return frontend(checkpoint_path, **kwargs)


__all__ = ["HiggsAudioV3Dims", "HiggsAudioV3Pipeline", "create_frontend"]
