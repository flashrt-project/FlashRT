"""ROCm backend capability helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RocmBackendInfo:
    name: str = "rocm"
    extension_name: str = "flash_rt_rocm_kernels"
    available: bool = False
    device_name: str | None = None
    hip_version: str | None = None
    supports_graph: bool = False
    supports_fp8_dtype: bool = False
    supports_fp8_gemm: bool = False
    supports_fp4: bool = False

    @property
    def supports_fp8(self) -> bool:
        return self.supports_fp8_gemm


def _torch_rocm_info() -> tuple[bool, str | None, str | None, bool, bool]:
    try:
        import torch
    except Exception:
        return False, None, None, False, False

    hip_version = getattr(torch.version, "hip", None)
    available = bool(torch.cuda.is_available() and hip_version)
    device_name = torch.cuda.get_device_name(0) if available else None
    supports_graph = bool(available and hasattr(torch.cuda, "CUDAGraph"))
    supports_fp8_dtype = bool(available and hasattr(torch, "float8_e4m3fnuz"))
    return available, device_name, hip_version, supports_graph, supports_fp8_dtype


def _probe_fp8_gemm() -> bool:
    try:
        import torch
        from flash_rt import flash_rt_rocm_kernels as rocm
    except Exception:
        return False

    if not bool(torch.cuda.is_available() and getattr(torch.version, "hip", None)):
        return False
    if not hasattr(torch, "float8_e4m3fnuz"):
        return False
    try:
        a = torch.randn(16, 32, device="cuda", dtype=torch.float32) * 0.5
        b = torch.randn(32, 16, device="cuda", dtype=torch.float32) * 0.5
        a_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        b_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        a8 = (a / a_scale).to(torch.float8_e4m3fnuz)
        b8 = (b / b_scale).to(torch.float8_e4m3fnuz)
        rocm.hipblaslt_matmul_fp8_e4m3fnuz_bf16(a8, b8, a_scale, b_scale)
        rocm.hip_sync()
    except RuntimeError as exc:
        if "hipBLASLt did not return a usable" in str(exc):
            return False
        raise
    except Exception:
        return False
    return True


def get_backend_info(*, probe_fp8_gemm: bool = False) -> RocmBackendInfo:
    available, device_name, hip_version, supports_graph, supports_fp8_dtype = (
        _torch_rocm_info()
    )
    supports_fp8_gemm = _probe_fp8_gemm() if probe_fp8_gemm else False
    return RocmBackendInfo(
        available=available,
        device_name=device_name,
        hip_version=hip_version,
        supports_graph=supports_graph,
        supports_fp8_dtype=supports_fp8_dtype,
        supports_fp8_gemm=supports_fp8_gemm,
    )


__all__ = ["RocmBackendInfo", "get_backend_info"]
