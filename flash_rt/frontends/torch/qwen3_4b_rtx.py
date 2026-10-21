"""FlashRT — PyTorch frontend for Qwen3-4B (dense BF16) on RTX.

Class name ``Qwen3_4BTorchFrontendRtx`` follows ``docs/adding_new_model.md``
§0 naming. Targets HuggingFace ``Qwen/Qwen3-4B`` geometry:

    hidden=2560, layers=36, GQA 32q/8kv, head_dim=128, intermediate=9728

Reuses ``RtxFlashAttnBackendQwen3`` + flash_rt_kernels BF16 primitives
(same stack as Higgs Audio's Qwen3-4B backbone). Scratch buffers are
sized from config — never the Qwen3-8B NVFP4 constants.

Precision-sensitive defaults (leave-one-out vs HF at S=256/1024 with
bf16-aligned RoPE tables): only ``use_hf_qkn_rope`` and ``use_sep_qkv``
are required for full-seq cosine >= 0.999. Other toggles default to the
FlashRT fast path:

  * ``use_sdpa``         — SDPA vs FA2 (default FA2; no material cosine hit)
  * ``use_hf_norm``      — HF RMSNorm vs fvk (default fvk; no material hit)
  * ``use_hf_qkn_rope``  — HF q/k RMSNorm + rotate_half RoPE vs fused kernels
  * ``use_sep_gateup``   — separate gate/up + F.silu vs fused (default fused)
  * ``use_sep_qkv``      — separate q/k/v vs fused qkv (needed at S=256)
  * ``use_torch_qkv``    — F.linear for sep q/k/v (default True; cuBLASLt
                           QKV drops S=256 full-seq cosine below 0.999)

Public surface:
  * ``forward_prefill(input_ids) -> logits [S, vocab]``
  * ``forward_prefill_llm_only(input_ids) -> hidden [S, H]`` (no lm_head)
  * ``inject`` via ``_qwen3_4b_rtx_bf16_weights.inject_hf_state_dict``
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._qwen3_4b_rtx_bf16_weights import (
    extract_weights_qwen3_4b_bf16,
    inject_hf_state_dict,
    qwen3_4b_config_dict,
)
from flash_rt.models.qwen3.pipeline_rtx import Qwen3_4BDims

BF16 = torch.bfloat16


def _hf_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match HF ``Qwen3RMSNorm``: fp32 variance, weight applied in input dtype."""
    input_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_hf(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``apply_rotary_pos_emb`` with rotate_half; cos/sin are (S, HD/2)."""
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)  # (S, 1, HD)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3_4BTorchFrontendRtx:
    """Qwen3-4B BF16 inference frontend (PyTorch + RTX, FA2 attention)."""

    DIMS = Qwen3_4BDims()

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        device: str = "cuda:0",
        max_seq: int = 2048,
        max_q_seq: int | None = None,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        config: Mapping[str, Any] | None = None,
        alloc_own_forward_buffers: bool = True,
        use_sdpa: bool | None = None,
        use_hf_norm: bool = False,
        use_hf_qkn_rope: bool = True,
        use_sep_gateup: bool = False,
        use_sep_qkv: bool = True,
        use_torch_qkv: bool = True,
    ) -> None:
        """Load BF16 weights and allocate attention / scratch buffers.

        Args:
          checkpoint_path: optional HF-style dir (config.json + weights).
            May be None when ``state_dict`` is provided (random-weight tests).
          device: CUDA device string.
          max_seq: KV cache length.
          max_q_seq: max prefill Q rows (defaults to max_seq).
          state_dict: in-memory HF state_dict (overrides checkpoint weights).
          config: HF config dict / Qwen3Config; defaults to Qwen3-4B dims.
          alloc_own_forward_buffers: allocate attn + RoPE at construction.
          use_sdpa: use torch SDPA instead of FA2. ``None`` auto-selects SDPA
            when ``flash_rt_fa2`` is missing (Thor / sm_110).
          use_hf_norm: use HF-style RMSNorm instead of fvk fused norms.
          use_hf_qkn_rope: use HF q/k RMSNorm + rotate_half RoPE (needed
            for long-seq HF cosine).
          use_sep_gateup: separate gate/up GEMM + F.silu instead of fused.
          use_sep_qkv: separate q/k/v GEMMs instead of fused qkv (needed
            for mid-seq HF cosine, e.g. S=256).
          use_torch_qkv: use ``F.linear`` for sep q/k/v (needed for S=256
            cosine > 0.999 with current cuBLASLt).
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_seq = int(max_seq)
        self.max_q_seq = int(max_q_seq) if max_q_seq is not None else self.max_seq
        if use_sdpa is None:
            try:
                from flash_rt import flash_rt_fa2 as _fa2  # noqa: F401
                use_sdpa = False
            except ImportError:
                use_sdpa = True
        self.use_sdpa = bool(use_sdpa)
        self.use_hf_norm = bool(use_hf_norm)
        self.use_hf_qkn_rope = bool(use_hf_qkn_rope)
        self.use_sep_gateup = bool(use_sep_gateup)
        self.use_sep_qkv = bool(use_sep_qkv)
        self.use_torch_qkv = bool(use_torch_qkv)
        self._attn = None
        self._rope_cos = None
        self._rope_sin = None
        self._rope_cos_fp32 = None
        self._rope_sin_fp32 = None
        self._WL: list[dict[str, torch.Tensor]] = []
        self._embed = None
        self._lm_head = None
        self._final_norm = None
        self._weights = None
        self._cfg: dict[str, Any] | None = None
        self._cur_pos = 0

        handles, cfg = extract_weights_qwen3_4b_bf16(
            device=device,
            checkpoint_path=checkpoint_path,
            state_dict=state_dict,
            config=config if config is not None else (
                qwen3_4b_config_dict() if state_dict is not None else None),
            assert_dims=True,
        )
        self._weights = handles
        self._cfg = cfg
        self._rebuild_packed_weights()
        if alloc_own_forward_buffers:
            self._alloc_buffers()
            self._build_rope_table()

    # ── weight packing ──

    def _rebuild_packed_weights(self) -> None:
        """Concatenate q/k/v and gate/up for single GEMM launches."""
        w = self._weights.ptrs
        self._embed = w["embed"]
        self._lm_head = w["lm_head"]
        self._final_norm = w["final_norm"]
        self._WL = []
        for L in w["layers"]:
            qkv = torch.cat([L["q"], L["k"], L["v"]], 0).contiguous()
            gu = torch.cat([L["gate"], L["up"]], 0).contiguous()
            self._WL.append({
                "qkv": qkv,
                "q": L["q"],
                "k": L["k"],
                "v": L["v"],
                "gu": gu,
                "gate": L["gate"],
                "up": L["up"],
                "o": L["o"],
                "down": L["down"],
                "in_norm": L["in_norm"],
                "post_norm": L["post_norm"],
                "qn": L["qn"],
                "kn": L["kn"],
            })

    def load_hf_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Replace weights from an HF state_dict (used by cosine tests)."""
        inject_hf_state_dict(self, state_dict, config=config)

    # ── buffers / RoPE ──

    def _alloc_buffers(self) -> None:
        c = self._cfg
        kwargs = dict(
            max_seq=self.max_seq,
            max_q_seq=self.max_q_seq,
            dtype=BF16,
            num_layers=c["num_layers"],
            num_q_heads=c["num_q_heads"],
            num_kv_heads=c["num_kv_heads"],
            head_dim=c["head_dim"],
            device=self.device,
        )
        # FA2 path keeps the RTX backend (it requires flash_rt_fa2).
        # SDPA path (Thor / sm_110, or an explicit use_sdpa=True) uses the
        # existing Thor SDPA backend so construction does not import FA2.
        if self.use_sdpa:
            from flash_rt.hardware.thor.attn_backend_qwen3 import (
                ThorAttnBackendQwen3,
            )
            self._attn = ThorAttnBackendQwen3(**kwargs)
        else:
            from flash_rt.hardware.rtx.attn_backend_qwen3 import (
                RtxFlashAttnBackendQwen3,
            )
            self._attn = RtxFlashAttnBackendQwen3(**kwargs)

    def _build_rope_table(self) -> None:
        """Build rotate_half RoPE tables matching HF ``Qwen3RotaryEmbedding``.

        Critical: HuggingFace ``model.to(dtype=bfloat16)`` also casts the
        ``inv_freq`` buffer to bf16, permanently quantizing the frequencies.
        ``rotary_emb.forward`` then does ``inv_freq.float()``, which expands
        those already-rounded values — it does NOT recover fp32 precision.
        We mirror that cast so long-seq cosine vs HF stays bit-identical.
        """
        hd = self._cfg["head_dim"]
        theta = self._cfg["rope_theta"]
        inv = 1.0 / (theta ** (
            torch.arange(0, hd, 2, device=self.device, dtype=torch.float32) / hd))
        # Match HF buffer dtype after ``.to(bfloat16)``.
        inv = inv.to(BF16).float()
        pos = torch.arange(self.max_seq, device=self.device, dtype=torch.float32)
        f = torch.outer(pos, inv)
        # Fused kernels expect (S, HD/2) bf16; HF path needs fp32 half-tables.
        self._rope_cos_fp32 = f.cos().contiguous()
        self._rope_sin_fp32 = f.sin().contiguous()
        self._rope_cos = self._rope_cos_fp32.to(BF16).contiguous()
        self._rope_sin = self._rope_sin_fp32.to(BF16).contiguous()

    def reset_state(self) -> None:
        if self._attn is not None:
            self._attn.reset_cache()
        self._cur_pos = 0

    # ── GEMM helpers ──

    def _gemm(self, fvk, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor,
              M: int, N: int, K: int, stream: int) -> None:
        """BF16 GEMM with kernel preference, torch matmul fallback."""
        if M > 1 and hasattr(fvk, "bf16_matmul_cublaslt_bf16"):
            fvk.bf16_matmul_cublaslt_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, stream)
            return
        if hasattr(fvk, "bf16_matmul_bf16"):
            fvk.bf16_matmul_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, stream)
            return
        # Fallback (kernels not built / unavailable).
        out.copy_(torch.nn.functional.linear(x, w))

    def _embed_lookup(self, fvk, ids: torch.Tensor, h: torch.Tensor,
                      S: int, H: int, stream: int) -> None:
        if hasattr(fvk, "embedding_lookup_bf16"):
            fvk.embedding_lookup_bf16(
                ids.data_ptr(), self._embed.data_ptr(), h.data_ptr(), S, H, stream)
        else:
            h.copy_(self._embed[ids])

    def _rms_norm_into(
        self, fvk, src: torch.Tensor, weight: torch.Tensor, dst: torch.Tensor,
        S: int, H: int, eps: float, stream: int,
    ) -> None:
        if self.use_hf_norm:
            dst.copy_(_hf_rms_norm(src, weight, eps))
        else:
            fvk.rms_norm(src.data_ptr(), weight.data_ptr(),
                         dst.data_ptr(), S, H, eps, stream)

    def _residual_add_rms_norm_into(
        self, fvk, residual: torch.Tensor, delta: torch.Tensor,
        weight: torch.Tensor, dst: torch.Tensor,
        S: int, H: int, eps: float, stream: int,
    ) -> None:
        if self.use_hf_norm:
            residual.add_(delta)
            dst.copy_(_hf_rms_norm(residual, weight, eps))
        else:
            fvk.residual_add_rms_norm(
                residual.data_ptr(), delta.data_ptr(),
                weight.data_ptr(), dst.data_ptr(), S, H, eps, stream)

    def _residual_add(
        self, fvk, residual: torch.Tensor, delta: torch.Tensor,
        S: int, H: int, stream: int,
    ) -> None:
        if self.use_hf_norm:
            residual.add_(delta)
        else:
            fvk.residual_add(residual.data_ptr(), delta.data_ptr(), S * H, stream)

    def _qkn_rope_prefill(
        self, fvk, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        w: dict[str, torch.Tensor], L: int, S: int,
        NQ: int, NKV: int, HD: int, eps: float, stream: int,
    ) -> None:
        be = self._attn
        if self.use_hf_qkn_rope:
            qh = q.view(S, NQ, HD)
            kh = k.view(S, NKV, HD)
            vh = v.view(S, NKV, HD)
            qn = _hf_rms_norm(qh, w["qn"], eps)
            kn = _hf_rms_norm(kh, w["kn"], eps)
            # Cast half-tables to q dtype like HF RotaryEmbedding.forward.
            cos_h = self._rope_cos_fp32[:S].to(dtype=qn.dtype)
            sin_h = self._rope_sin_fp32[:S].to(dtype=qn.dtype)
            qr, kr = _apply_rope_hf(qn, kn, cos_h, sin_h)
            be.Q_buf[0, :S].copy_(qr)
            be.K_cache[L, :S].copy_(kr)
            be.V_cache[L, :S].copy_(vh)
            return

        rc, rs = self._rope_cos, self._rope_sin
        fvk.qwen3_q_norm_rope_qstage_prefill_bf16(
            q.data_ptr(), w["qn"].data_ptr(),
            rc[:S].data_ptr(), rs[:S].data_ptr(),
            be.Q_buf[:, :S].data_ptr(), NQ, S, int(q.stride(0)),
            int(be.Q_buf[:, :S].stride(1)), eps, stream)
        fvk.qwen3_k_norm_rope_kvwrite_prefill_bf16(
            k.data_ptr(), v.data_ptr(), w["kn"].data_ptr(),
            rc[:S].data_ptr(), rs[:S].data_ptr(),
            be.K_cache[L, :S].data_ptr(),
            be.V_cache[L, :S].data_ptr(),
            NKV, S, int(k.stride(0)),
            int(be.K_cache[L, :S].stride(0)),
            eps, stream)

    def _attn_prefill(
        self, L: int, S: int, NQ: int, NKV: int, HD: int, stream: int,
    ) -> torch.Tensor:
        be = self._attn
        be.O_buf[:, :S].zero_()
        scale = HD ** -0.5
        if self.use_sdpa:
            # Match HF ``sdpa_attention_forward``: attn_mask=None + is_causal
            # enables ``enable_gqa=True`` (no repeat_kv) on torch>=2.5 CUDA.
            qh = be.Q_buf[:, :S].transpose(1, 2).contiguous()  # (1, NQ, S, HD)
            kh = be.K_cache[L:L + 1, :S].transpose(1, 2).contiguous()
            vh = be.V_cache[L:L + 1, :S].transpose(1, 2).contiguous()
            out_h = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=None, is_causal=True, scale=scale,
                enable_gqa=True)
            be.O_buf[:, :S].copy_(out_h.transpose(1, 2))
        else:
            be.run("full", L, S, kv_seq=S, causal=True, stream=stream,
                   softmax_scale=scale)
        return be.O_buf[:, :S].reshape(S, NQ * HD).contiguous()

    def _swiglu(
        self, fvk, xn2: torch.Tensor, w: dict[str, torch.Tensor],
        Dg: torch.Tensor, g: torch.Tensor, u: torch.Tensor, act: torch.Tensor,
        S: int, H: int, INTER: int, stream: int,
    ) -> None:
        if self.use_sep_gateup:
            self._gemm(fvk, xn2, w["gate"], g, S, INTER, H, stream)
            self._gemm(fvk, xn2, w["up"], u, S, INTER, H, stream)
            act.copy_(F.silu(g) * u)
            return
        self._gemm(fvk, xn2, w["gu"], Dg, S, 2 * INTER, H, stream)
        if hasattr(fvk, "silu_mul_qwen36_bf16_merged"):
            fvk.silu_mul_qwen36_bf16_merged(
                Dg.data_ptr(), act.data_ptr(), S, INTER, stream)
        else:
            g_view = Dg[:, :INTER].contiguous()
            u_view = Dg[:, INTER:].contiguous()
            fvk.silu_mul_qwen36_bf16(
                g_view.data_ptr(), u_view.data_ptr(), act.data_ptr(),
                S * INTER, stream)

    # ── forward ──

    @torch.no_grad()
    def forward_prefill(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        *,
        return_last_only: bool = False,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Run causal prefill; return logits ``[S, vocab]`` (or ``[1, vocab]``).

        Matches HF ``Qwen3ForCausalLM``: embed → 36× (attn+MLP) → final
        RMSNorm → lm_head on every position (unless ``return_last_only``).
        If ``return_hidden``, skip lm_head and return ``[S, H]`` bf16.
        """
        from flash_rt import flash_rt_kernels as fvk

        if self._attn is None or self._rope_cos is None:
            raise RuntimeError("buffers not allocated")

        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.detach().view(-1).tolist()
        else:
            ids_list = list(input_ids)
        S = len(ids_list)
        if S == 0:
            raise ValueError("empty input_ids")
        if S > self.max_seq:
            raise ValueError(f"seq len {S} > max_seq {self.max_seq}")
        if S > self.max_q_seq:
            raise ValueError(f"seq len {S} > max_q_seq {self.max_q_seq}")

        c = self._cfg
        H = c["hidden"]
        NQ, NKV, HD = c["num_q_heads"], c["num_kv_heads"], c["head_dim"]
        INTER, EPS = c["intermediate"], c["rms_norm_eps"]
        NQK, KV = NQ * HD, NKV * HD
        V = c["vocab_size"]
        NL = c["num_layers"]
        dev = self.device
        s = torch.cuda.current_stream().cuda_stream

        self.reset_state()
        ids_t = torch.tensor(ids_list, device=dev, dtype=torch.int64)
        h = torch.empty(S, H, device=dev, dtype=BF16)
        xn = torch.empty(S, H, device=dev, dtype=BF16)
        xn2 = torch.empty(S, H, device=dev, dtype=BF16)
        Dq = torch.empty(S, NQK + 2 * KV, device=dev, dtype=BF16)
        q_buf = torch.empty(S, NQK, device=dev, dtype=BF16)
        k_buf = torch.empty(S, KV, device=dev, dtype=BF16)
        v_buf = torch.empty(S, KV, device=dev, dtype=BF16)
        Dg = torch.empty(S, 2 * INTER, device=dev, dtype=BF16)
        g = torch.empty(S, INTER, device=dev, dtype=BF16)
        u = torch.empty(S, INTER, device=dev, dtype=BF16)
        act = torch.empty(S, INTER, device=dev, dtype=BF16)
        tmp = torch.empty(S, H, device=dev, dtype=BF16)

        self._embed_lookup(fvk, ids_t, h, S, H, s)
        self._rms_norm_into(fvk, h, self._WL[0]["in_norm"], xn, S, H, EPS, s)

        for L in range(NL):
            w = self._WL[L]
            if self.use_sep_qkv:
                # Separate q/k/v match HF nn.Linear reduction order.
                # cuBLASLt on these projections drops S=256 full cosine <0.999;
                # F.linear restores parity (use_torch_qkv=True by default).
                if self.use_torch_qkv:
                    # out= form avoids an extra alloc (CUDA-Graph friendlier).
                    torch.mm(xn, w["q"].t(), out=q_buf)
                    torch.mm(xn, w["k"].t(), out=k_buf)
                    torch.mm(xn, w["v"].t(), out=v_buf)
                else:
                    self._gemm(fvk, xn, w["q"], q_buf, S, NQK, H, s)
                    self._gemm(fvk, xn, w["k"], k_buf, S, KV, H, s)
                    self._gemm(fvk, xn, w["v"], v_buf, S, KV, H, s)
                q, k, v = q_buf, k_buf, v_buf
            else:
                self._gemm(fvk, xn, w["qkv"], Dq, S, NQK + 2 * KV, H, s)
                q = Dq[:, :NQK]
                k = Dq[:, NQK:NQK + KV]
                v = Dq[:, NQK + KV:]
            self._qkn_rope_prefill(fvk, q, k, v, w, L, S, NQ, NKV, HD, EPS, s)
            ao = self._attn_prefill(L, S, NQ, NKV, HD, s)
            self._gemm(fvk, ao, w["o"], tmp, S, H, NQK, s)
            self._residual_add_rms_norm_into(
                fvk, h, tmp, w["post_norm"], xn2, S, H, EPS, s)
            self._swiglu(fvk, xn2, w, Dg, g, u, act, S, H, INTER, s)
            self._gemm(fvk, act, w["down"], tmp, S, H, INTER, s)
            if L < NL - 1:
                self._residual_add_rms_norm_into(
                    fvk, h, tmp, self._WL[L + 1]["in_norm"], xn, S, H, EPS, s)
            else:
                self._residual_add(fvk, h, tmp, S, H, s)

        # Final RMSNorm + lm_head over all positions (HF parity).
        hn = torch.empty(S, H, device=dev, dtype=BF16)
        self._rms_norm_into(fvk, h, self._final_norm, hn, S, H, EPS, s)
        self._cur_pos = S
        if return_hidden:
            return hn
        if return_last_only:
            logits = torch.empty(1, V, device=dev, dtype=BF16)
            self._gemm(fvk, hn[S - 1:S].contiguous(), self._lm_head,
                       logits, 1, V, H, s)
        else:
            logits = torch.empty(S, V, device=dev, dtype=BF16)
            self._gemm(fvk, hn, self._lm_head, logits, S, V, H, s)
        return logits

    @torch.no_grad()
    def forward_prefill_llm_only(
        self,
        input_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Prefill without lm_head — hidden states ``[S, H]`` (latency llm metric)."""
        return self.forward_prefill(input_ids, return_hidden=True)
