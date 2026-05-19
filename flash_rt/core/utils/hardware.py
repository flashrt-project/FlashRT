"""FlashRT — Hardware detection utilities."""

import subprocess


def get_gpu_sm_version() -> int:
    """Detect GPU SM version (e.g., 89 for 4090, 110 for Thor, 120 for 5090)."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return major * 10 + minor
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cc = result.stdout.strip().split("\n")[0]
            return int(cc.replace(".", ""))
    except Exception:
        pass
    return 0


def get_gpu_name() -> str:
    """Get GPU product name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "Unknown"


def supports_fp8() -> bool:
    """Check if GPU supports FP8 (SM >= 89)."""
    return get_gpu_sm_version() >= 89


def supports_nvfp4() -> bool:
    """Check if GPU supports NVFP4 block-scaled (SM >= 120)."""
    return get_gpu_sm_version() >= 120
