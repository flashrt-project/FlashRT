"""Static FP8 (E4M3, per-tensor) linear sites on FlashRT kernels, raw-pointer interface.

    weight (offline)   w8 = W / s_w, s_w = amax|W| / 448                 (N, K) row-major
    activation         quantize_fp8_static(x, s_a[slot])                 static, calibrated
                       quantize_fp8_device + fp8_accumulate_scale_max     calibration pass
    GEMM               GemmRunner.fp8_nt_dev(A (M,K), W (N,K)) -> bf16   device scales

A site owns several activation-scale SLOTS so one weight can carry a separate
calibrated scale per diffusion step, the way FlashRT's Pi0.5 decoder does
(pipeline_rtx._fp8_decoder_step_scale_name): the action expert's activations
drift with the flow time, and one shared scale would be sized for the widest step.
"""
from __future__ import annotations

import torch
from flash_rt import flash_rt_kernels as fvk

E4M3 = 448.0
_GEMM = None


_TUNE = False
_TUNED: set = set()


def set_autotune(flag: bool) -> None:
    """While set, the first eager GEMM of each (M, N, K) runs FlashRT's cuBLASLt autotune,
    whose choice GemmRunner caches for that shape. Enabled only for the pre-capture warm-up."""
    global _TUNE
    _TUNE = flag


def _gemm():
    global _GEMM
    if _GEMM is None:
        _GEMM = fvk.GemmRunner()
    return _GEMM


class Fp8Linear:
    def __init__(self, weight: torch.Tensor, max_m: int, slots: int = 1, name: str = "?"):
        self.name = name
        self.N, self.K = int(weight.shape[0]), int(weight.shape[1])
        d = weight.device
        w = weight.detach().float()
        sw = w.abs().amax().clamp(min=1e-12) / E4M3
        self.w8 = (w / sw).to(torch.float8_e4m3fn).contiguous()
        self.w_scale = sw.reshape(1).to(torch.float32).contiguous()
        self.a8 = torch.empty(int(max_m), self.K, dtype=torch.float8_e4m3fn, device=d)
        self.a_scale = torch.zeros(int(slots), dtype=torch.float32, device=d)
        self.scratch = torch.zeros(1, dtype=torch.float32, device=d)
        self.calibrating = False
        self.calibrated = False
        self.autotune = True     # False keeps cuBLASLt's heuristic algorithm for this shape

    def quantize(self, x: torch.Tensor, m: int, stream: int, slot: int = 0) -> None:
        n = m * self.K
        sp = self.a_scale[slot:slot + 1]
        if self.calibrating:
            fvk.quantize_fp8_device(x.data_ptr(), self.a8.data_ptr(), self.scratch.data_ptr(), n, stream)
            fvk.fp8_accumulate_scale_max(self.scratch.data_ptr(), sp.data_ptr(), stream)
            self._last_scale = self.scratch
            return
        if not self.calibrated:
            raise RuntimeError(f"{self.name}: static FP8 site used before calibration")
        fvk.quantize_fp8_static(x.data_ptr(), self.a8.data_ptr(), sp.data_ptr(), n, stream)
        self._last_scale = sp

    def gemm(self, out: torch.Tensor, m: int, stream: int, slot: int = 0,
             act: "Fp8Linear | None" = None) -> None:
        if act is not None and not isinstance(act, Fp8Linear):
            raise TypeError(f"{self.name}: cannot reuse a non-FP8 quantized activation")
        a = self if act is None else act
        # during calibration the GEMM runs with the dynamic scale just computed
        sa = a.scratch if a.calibrating else a.a_scale[slot:slot + 1]
        if _TUNE and self.autotune and not a.calibrating and (m, self.N, self.K) not in _TUNED:
            _TUNED.add((m, self.N, self.K))
            _gemm().autotune_fp8_nt_dev(a.a8.data_ptr(), self.w8.data_ptr(), out.data_ptr(), m, self.N,
                                        self.K, sa.data_ptr(), self.w_scale.data_ptr(), 16)
        _gemm().fp8_nt_dev(a.a8.data_ptr(), self.w8.data_ptr(), out.data_ptr(), m, self.N, self.K,
                           sa.data_ptr(), self.w_scale.data_ptr(), stream)

    def reset_calibration(self) -> None:
        self.calibrating, self.calibrated = True, False
        self.a_scale.zero_()

    def finish_calibration(self, clip: float = 1.0, consumer: bool = False) -> None:
        """Freeze the per-slot activation scales (max over the calibration pass; `clip`
        applies to NVFP4 sites only and is accepted here for a uniform interface)."""
        if not consumer and bool((self.a_scale <= 0).any()):
            raise RuntimeError(f"{self.name}: a scale slot saw no calibration data")
        self.calibrating, self.calibrated = False, True
