"""Hub v3 executable forms for ``gated_delta_core``."""

from __future__ import annotations

import torch

from ...guard import PROCEED, GuardedSeam
from .. import hub_kernel


class HubV3GatedDeltaCore(GuardedSeam, torch.nn.Module):
    """Single-token H=32/48, D=128 recurrence with explicit state output.

    The log-decay ``g`` binds in the dtype the host actually exposes:
    BF16 through the original entry, FP32 through the ``gf32`` twin —
    the 27B-class cached-decode hosts keep ``g`` in FP32, and rounding
    it through BF16 (or casting in the hot path) is what qualification
    used to refuse here.
    """

    def __init__(self, sample: torch.Tensor,
                 g_dtype: torch.dtype = torch.bfloat16,
                 state_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if sample.dtype != torch.bfloat16:
            raise ValueError("gated_delta_core v3 requires BF16 Q/K/V")
        if g_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                "gated_delta_core v3 serves BF16 or FP32 log-decay only")
        if state_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                "gated_delta_core v3 serves BF16 or FP32 state only")
        if state_dtype is torch.float32 and g_dtype is not torch.float32:
            raise ValueError(
                "gated_delta_core v3 has no BF16-g/FP32-state entry; no "
                "host has exposed that combination")
        self._g_dtype = g_dtype
        self._state_dtype = state_dtype
        if sample.ndim != 4 or sample.shape[0] != 1 \
                or sample.shape[1] != 1 \
                or sample.shape[2] not in (32, 48) \
                or sample.shape[3] != 128:
            raise ValueError(
                "gated_delta_core v3 requires Q shape "
                "(1,1,H,128) with H=32 or H=48; the published "
                "sequence API has no explicit state output")
        if not sample.is_contiguous():
            raise ValueError("gated_delta_core v3 requires contiguous Q/K/V")
        self.heads = int(sample.shape[2])
        self._ops = hub_kernel("flashrt/gated-delta-attention", ">=3")
        if g_dtype is torch.float32:
            name = ("gated_delta_recurrent_inout_gf32_sf32_bf16"
                    if state_dtype is torch.float32
                    else "gated_delta_recurrent_inout_gf32_bf16")
            step = getattr(self._ops, name, None)
            if step is None:
                raise ValueError(
                    "refused: the installed gated-delta-attention build "
                    f"predates the {name} entry; a release carrying it "
                    "is required")
        else:
            step = self._ops.gated_delta_recurrent_inout_bf16
        self._step = step
        self.register_buffer(
            "_state_out",
            torch.empty(
                1, self.heads, 128, 128,
                device=sample.device, dtype=self._state_dtype),
            persistent=False,
        )
        self.register_buffer(
            "_out",
            torch.empty(
                1, self.heads, 128,
                device=sample.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self._frt_arm(
            dtypes=(torch.bfloat16,), device=sample.device, k=128,
            rows=self.heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor | None,
        *,
        output_final_state: bool,
        use_qk_l2norm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if query.ndim != 4 or query.shape[0] != 1 \
                or query.shape[1:] != (1, self.heads, 128):
            raise ValueError(
                "gated_delta_core v3 query shape moved after binding")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("gated_delta_core v3 Q/K/V shapes differ")
        if not (query.is_contiguous() and key.is_contiguous()
                and value.is_contiguous()):
            raise ValueError("gated_delta_core v3 requires contiguous Q/K/V")
        if log_decay.shape != query.shape[:3] \
                or beta.shape != log_decay.shape:
            raise ValueError("gated_delta_core v3 gating shapes differ")
        if log_decay.dtype != self._g_dtype \
                or beta.dtype != torch.bfloat16:
            raise ValueError(
                f"gated_delta_core v3 bound {self._g_dtype} log-decay "
                "and BF16 beta; the host's dtypes moved after binding")
        if state is None or state.shape != (1, self.heads, 128, 128):
            raise ValueError("gated_delta_core v3 state shape differs")
        if state.dtype != self._state_dtype or not state.is_contiguous():
            raise ValueError(
                f"gated_delta_core v3 bound contiguous {self._state_dtype} "
                "state; the host's state moved after binding")
        # One custom op. The caller's state is read-only and the final state is
        # written into graph-stable storage for snapshot and rollback.
        out, state_out = self._step(
            query[:, 0], key[:, 0], value[:, 0],
            log_decay[:, 0], beta[:, 0], state,
            use_qk_l2norm=use_qk_l2norm,
            state_out=self._state_out,
            out=self._out,
        )
        return out[:, None], state_out if output_final_state else None


def bind_gated_delta_core(sample: dict[str, torch.Tensor]):
    """Bind v3 decode recurrence and launch the observed real sample once.

    The entry is chosen by the observed sample's log-decay dtype — the
    form the host actually calls with, not a preference."""
    state = sample.get("state")
    core = HubV3GatedDeltaCore(
        sample["query"], g_dtype=sample["g"].dtype,
        state_dtype=(state.dtype if state is not None
                     else torch.bfloat16))
    with torch.no_grad():
        core(
            sample["query"], sample["key"], sample["value"],
            sample["g"], sample["beta"], sample.get("state"),
            output_final_state=bool(sample.get("output_final_state", True)),
            use_qk_l2norm=bool(sample.get("use_qk_l2norm", True)),
        )
    guard = core._frt_guard
    if guard is not None:
        guard.calls = 0
    return core
