"""XR-1 RDNA compute adapters composed from existing FlashRT HIP kernels.

The upstream model owns attention, flow matching and checkpoint parameters.
Optional native adapters replace DiT linears/RMSNorm. Vision patch Conv3d
retains MIOpen: a mathematically equivalent GEMM changed BF16 rounding enough
to fail action parity on real camera observations.
"""
from __future__ import annotations

from types import MethodType
import torch


def _hip_linear(module, x):
    x = x.contiguous()
    out = torch.empty((*x.shape[:-1], module.weight.shape[0]),
                      dtype=x.dtype, device=x.device)
    module._xr1_gemm.linear(out, x, module._xr1_weight, module.bias)
    return out


def _hip_rms_norm(module, x):
    x = x.contiguous()
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("XR-1 native RMSNorm requires ROCm BF16 input")
    out = torch.empty_like(x)
    module._xr1_kernels.rms_norm_rdna(
        out.data_ptr(), x.data_ptr(), x.numel() // x.shape[-1], x.shape[-1],
        module.variance_epsilon, torch.cuda.current_stream(x.device).cuda_stream)
    # Preserve upstream's BF16 rounding before applying the learned scale.
    return module.weight * out


class XR1PipelineRdna35:
    """Instance-local native adapters; reference mode requires no extension."""

    def __init__(self, model, *, native=False):
        self.model = model
        self.native = native
        self.gemm = None
        if not native:
            return
        try:
            from flash_rt.amd import flash_rt_amd_kernels as kernels
        except ImportError as exc:
            raise ImportError(
                'XR-1 native mode requires gfx1151 FlashRT HIP kernels. Build with '
                '`bash scripts/amd/build_amd.sh gfx1151`, or pass fuse=False.'
            ) from exc
        from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend
        info = dict(kernels.build_info())
        if info.get("gpu_arch") != "gfx1151" or info.get("backend") != "rdna35":
            raise RuntimeError("XR-1 native path requires an extension built for gfx1151")
        self.gemm = Rdna35GemmBackend(kernels)
        for module in model.dit.modules():
            if isinstance(module, torch.nn.Linear):
                module._xr1_weight = module.weight.t().contiguous()
                module._xr1_gemm = self.gemm
                module.forward = MethodType(_hip_linear, module)
            elif module.__class__.__name__ in ("Qwen3VLTextRMSNorm", "Qwen2RMSNorm"):
                module._xr1_kernels = kernels
                module.forward = MethodType(_hip_rms_norm, module)

    @torch.inference_mode()
    def forward(self, inputs, *, seed, num_steps, embodiment):
        # Both upstream formats seed PyTorch globally. Confine that effect to
        # this prediction, preserving the application's CPU and GPU RNG state.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            if hasattr(self.model, 'action_shape'):
                from types import SimpleNamespace
                self.model.num_steps = num_steps
                torch.manual_seed(seed)
                actions = self.model.generate(dict(inputs))
                return SimpleNamespace(actions=actions)
            return self.model(**inputs, seed=seed, num_steps=num_steps, task_id=embodiment)
