"""FlashRT -- RTX DiT attention backend for GROOT N1.7.

The validated N1.7 RTX path is the action-head DiT self/cross attention
route.  Backbone ViT, LLM, and VL self-attention stay on the shared N1.7
frontend/calibration path and are not exposed by this backend contract.
"""

from __future__ import annotations


class RtxFlashAttnBackendGrootN17:
    """FA2-backed DiT attention slots for GROOT N1.7 on RTX hardware."""

    SITES = ("dit_self", "dit_cross")

    def __init__(
        self,
        *,
        num_vit_groups: int,
        llm_seq_max: int,
        vl_self_attn_seq_max: int,
        sa: int,
        s_kv_text: int,
        s_kv_image: int,
        num_dit_cross_blocks: int = 16,
        device: str = "cuda",
        slot_dtype=None,
    ):
        import torch

        self._torch = torch
        self._device = device
        # Q/K/V/O slot dtype. Default bf16 (FP8 production path). The full-FP16
        # baseline frontend passes torch.float16; run() dispatches fwd_fp16
        # automatically on q.dtype, so no other change is needed.
        bf16 = slot_dtype if slot_dtype is not None else torch.bfloat16

        self._num_vit_groups = int(num_vit_groups)
        self._llm_seq_max = int(llm_seq_max)
        self._vl_self_attn_seq_max = int(vl_self_attn_seq_max)
        self._sa = int(sa)
        self._s_kv_text = int(s_kv_text)
        self._s_kv_image = int(s_kv_image)
        self._dit_kv_seq = max(self._s_kv_text, self._s_kv_image)
        self._num_dit_cross_blocks = int(num_dit_cross_blocks)

        if self._sa <= 0:
            raise ValueError("sa must be positive")
        if self._s_kv_text <= 0 or self._s_kv_image <= 0:
            raise ValueError("DiT cross K/V sequence lengths must be positive")
        if self._num_dit_cross_blocks <= 0:
            raise ValueError("num_dit_cross_blocks must be positive")

        self.dit_self_Q = torch.empty(self._sa, 32, 48, dtype=bf16, device=device)
        self.dit_self_K = torch.empty_like(self.dit_self_Q)
        self.dit_self_V = torch.empty_like(self.dit_self_Q)

        self.dit_cross_Q = torch.empty(self._sa, 32, 48, dtype=bf16, device=device)
        self.dit_cross_K = [
            torch.empty(self._dit_kv_seq, 32, 48, dtype=bf16, device=device)
            for _ in range(self._num_dit_cross_blocks)
        ]
        self.dit_cross_V = [
            torch.empty(self._dit_kv_seq, 32, 48, dtype=bf16, device=device)
            for _ in range(self._num_dit_cross_blocks)
        ]

        try:
            from flash_rt import flash_rt_fa2 as fa2
        except ImportError:
            raise RuntimeError(
                "GROOT N1.7 RTX backend requires FlashRT's vendored FA2 "
                "module (`flash_rt.flash_rt_fa2`). Build it with CMake on "
                "RTX targets; do not install/use the upstream `flash_attn` "
                "pip package for this path.")
        self._fa2 = fa2
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        def lse(seq: int, heads: int):
            return torch.empty(
                1, heads, self._round_up_128(seq),
                dtype=torch.float32, device=device)

        # Preallocated FA2 outputs/LSE give stable returned pointers.
        self.dit_self_O = torch.empty_like(self.dit_self_Q)
        self.dit_cross_O = torch.empty_like(self.dit_cross_Q)
        self._dit_self_lse = lse(self._sa, 32)
        self._dit_cross_lse = lse(self._sa, 32)

    def sites(self) -> tuple[str, ...]:
        return self.SITES

    def head_dim(self, site: str) -> int:
        return {
            "dit_self": 48,
            "dit_cross": 48,
        }[site]

    def num_q_heads(self, site: str) -> int:
        return {
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def num_kv_heads(self, site: str) -> int:
        return {
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def get_ptrs(self) -> dict:
        """Bulk pointer view for callers that bind slots by tensor name."""
        return {
            "dit_self_Q": self.dit_self_Q.data_ptr(),
            "dit_self_K": self.dit_self_K.data_ptr(),
            "dit_self_V": self.dit_self_V.data_ptr(),
            "dit_self_O": self.dit_self_O.data_ptr(),
            "dit_cross_Q": self.dit_cross_Q.data_ptr(),
            "dit_cross_O": self.dit_cross_O.data_ptr(),
            "dit_cross_K": [t.data_ptr() for t in self.dit_cross_K],
            "dit_cross_V": [t.data_ptr() for t in self.dit_cross_V],
        }

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        if site == "dit_self":
            self._check_layer(site, layer_idx, 16)
            return {
                "Q": self.dit_self_Q.data_ptr(),
                "K": self.dit_self_K.data_ptr(),
                "V": self.dit_self_V.data_ptr(),
                "O": self.dit_self_O.data_ptr(),
            }
        if site == "dit_cross":
            self._check_layer(site, layer_idx, self._num_dit_cross_blocks)
            return {
                "Q": self.dit_cross_Q.data_ptr(),
                "K": self.dit_cross_K[layer_idx].data_ptr(),
                "V": self.dit_cross_V[layer_idx].data_ptr(),
                "O": self.dit_cross_O.data_ptr(),
            }
        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq=None,
        stream: int = 0,
    ) -> int:
        if site == "dit_self":
            self._check_layer(site, layer_idx, 16)
            self._check_seq("dit_self q_seq", q_seq, self._sa)
            if kv_seq is not None and int(kv_seq) != int(q_seq):
                raise ValueError("dit_self is self-attention; kv_seq must equal q_seq")
            q = self.dit_self_Q[:q_seq].unsqueeze(0)
            k = self.dit_self_K[:q_seq].unsqueeze(0)
            v = self.dit_self_V[:q_seq].unsqueeze(0)
            return self._run_fa2(
                q, k, v, self.dit_self_O[:q_seq].unsqueeze(0),
                self._dit_self_lse, causal=False, stream=stream)

        if site == "dit_cross":
            self._check_layer(site, layer_idx, self._num_dit_cross_blocks)
            self._check_seq("dit_cross q_seq", q_seq, self._sa)
            if kv_seq is None:
                raise ValueError("dit_cross requires kv_seq")
            self._check_seq("dit_cross kv_seq", kv_seq, self._dit_kv_seq)
            q = self.dit_cross_Q[:q_seq].unsqueeze(0)
            k = self.dit_cross_K[layer_idx][:kv_seq].unsqueeze(0)
            v = self.dit_cross_V[layer_idx][:kv_seq].unsqueeze(0)
            return self._run_fa2(
                q, k, v, self.dit_cross_O[:q_seq].unsqueeze(0),
                self._dit_cross_lse, causal=False, stream=stream)

        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    def _run_fa2(self, q, k, v, o, lse, *, causal: bool = False,
                 stream: int = 0) -> int:
        fwd = self._fa2.fwd_bf16 if q.dtype == self._torch.bfloat16 else self._fa2.fwd_fp16
        if causal:
            if q.dtype != self._torch.bfloat16:
                raise RuntimeError(
                    "vendored FA2 causal entry is currently bf16-only")
            fwd = self._fa2.fwd_bf16_causal
        B, Sq, Hq, D = q.shape
        Sk, Hk = k.shape[1], k.shape[2]
        fwd(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0,
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=1.0 / (D ** 0.5),
            num_sms=self._num_sms,
            stream=stream,
        )
        return o.data_ptr()

    @staticmethod
    def _round_up_128(x: int) -> int:
        return ((int(x) + 127) // 128) * 128

    @staticmethod
    def _check_layer(site: str, layer_idx: int, num_layers: int) -> None:
        if not (0 <= int(layer_idx) < int(num_layers)):
            raise IndexError(
                f"{site} layer_idx={layer_idx} out of range [0, {num_layers})")

    @staticmethod
    def _check_seq(name: str, seq: int, limit: int) -> None:
        if not (1 <= int(seq) <= int(limit)):
            raise ValueError(f"{name}={seq} out of range [1, {limit}]")


def make_groot_n17_rtx_attention_spec(
    *,
    num_vit_groups: int,
    llm_seq_max: int,
    vl_self_attn_seq_max: int,
    sa: int,
    s_kv_text: int,
    s_kv_image: int,
) -> dict:
    """Return canonical RTX N1.7 attention dimensions without allocating CUDA."""
    return {
        "dit_self": {
            "num_layers": 16,
            "num_q_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 48,
            "max_q_seq": int(sa),
            "max_kv_seq": int(sa),
            "dtype": "bf16",
        },
        "dit_cross": {
            "num_layers": 16,
            "num_q_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 48,
            "max_q_seq": int(sa),
            "max_kv_seq": max(int(s_kv_text), int(s_kv_image)),
            "dtype": "bf16",
        },
    }
