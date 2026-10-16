"""NVFP4 (W4A4) linear sites on FlashRT's sm120 kernels, raw-pointer interface.

    weight (offline)   bf16_weight_to_nvfp4_swizzled   -> packed (N, K/2) + swizzled SFB + alpha
    activation         quantize_bf16_to_nvfp4_swizzled_clipped(group_amax)   (static, calibrated)
                       quantize_bf16_to_nvfp4_swizzled                       (dynamic; calibration pass)
    GEMM               fp4_w4a16_gemm_sm120_bf16out_pingpong

The calibrated parameter is the per-16-column group amax, fitted from real
observations and multiplied by a clip factor; the kernel caps each row's own block
amax at it, so per-token adaptivity survives while outliers are bounded. Admission:
K >= 64, K % 64 == 0, N % 16 == 0.
"""
from __future__ import annotations

import torch
from flash_rt import flash_rt_kernels as fvk


def _ru(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def eligible(n: int, k: int) -> bool:
    return k >= 64 and k % 64 == 0 and n % 16 == 0


class Nvfp4Linear:
    def __init__(self, weight: torch.Tensor, max_m: int, name: str = "?"):
        n, k = int(weight.shape[0]), int(weight.shape[1])
        if not eligible(n, k):
            raise ValueError(f"{name}: N={n} K={k} not NVFP4-eligible")
        self.name, self.N, self.K = name, n, k
        d = weight.device
        wb = weight.detach().to(torch.bfloat16).contiguous()
        self.w_packed = torch.empty(n, k // 2, dtype=torch.uint8, device=d)
        self.w_sfb = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(n, k), dtype=torch.uint8, device=d)
        scratch = torch.empty(1, dtype=torch.float32, device=d)
        alpha = torch.empty(1, dtype=torch.float32, device=d)
        fvk.bf16_weight_to_nvfp4_swizzled(wb.data_ptr(), self.w_packed.data_ptr(),
                                          self.w_sfb.data_ptr(), scratch.data_ptr(),
                                          alpha.data_ptr(), n, k, 0)
        torch.cuda.synchronize(d)
        self.alpha = float(alpha.item())
        m = _ru(int(max_m), 128)
        self.a_packed = torch.empty(m, k // 2, dtype=torch.uint8, device=d)
        self.a_sfa = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(m, k), dtype=torch.uint8, device=d)
        self.group_amax = torch.zeros(k // 16, dtype=torch.float32, device=d)
        self.calibrating = False
        self.calibrated = False
        self.n_obs = 0

    def quantize(self, x: torch.Tensor, m: int, stream: int) -> None:
        """Quantize activation rows x[:m] (bf16, (m, K) contiguous) into the scratch."""
        if self.calibrating:
            a = x[:m].abs().float().reshape(m, self.K // 16, 16).amax(2).amax(0)
            torch.maximum(self.group_amax, a, out=self.group_amax)
            self.n_obs += 1
            fvk.quantize_bf16_to_nvfp4_swizzled(x.data_ptr(), self.a_packed.data_ptr(),
                                                self.a_sfa.data_ptr(), m, self.K, stream)
            return
        if not self.calibrated:
            raise RuntimeError(f"{self.name}: static NVFP4 site used before calibration")
        fvk.quantize_bf16_to_nvfp4_swizzled_clipped(x.data_ptr(), self.group_amax.data_ptr(),
                                                    self.a_packed.data_ptr(),
                                                    self.a_sfa.data_ptr(), m, self.K, stream)

    def reset_calibration(self) -> None:
        self.calibrating, self.calibrated, self.n_obs = True, False, 0
        self.group_amax.zero_()

    def gemm(self, out: torch.Tensor, m: int, stream: int, act: "Nvfp4Linear | None" = None) -> None:
        """out[:m] = act(x) @ W^T in bf16. `act` reuses another site's quantized
        activation (gate/up read the same input), which is exact."""
        a = self if act is None else act
        fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong(
            a.a_packed.data_ptr(), self.w_packed.data_ptr(), out.data_ptr(), m, self.N, self.K,
            a.a_sfa.data_ptr(), self.w_sfb.data_ptr(), self.alpha, stream)

    def finish_calibration(self, clip: float, consumer: bool = False) -> None:
        """Freeze group_amax * clip. A consumer site (reads another site's quantized
        activation) never quantizes itself, so it needs no statistic."""
        if consumer:
            self.calibrating, self.calibrated = False, True
            return
        if self.n_obs == 0:
            raise RuntimeError(f"{self.name}: no real observations recorded")
        self.group_amax.mul_(clip).clamp_(min=1e-6)
        self.calibrating, self.calibrated = False, True
