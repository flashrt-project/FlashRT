"""FlashRT -- dense Qwen3 BF16 prefill: the decoder sequence every target shares.

The layer sequence here is the same on every device; hardware enters only
through a :class:`Qwen3Bf16Target`, which names the device the path was
validated on and supplies the attention binding
(``models/qwen3/pipeline_<hw>.py``). Buffers
and weights are owned by the frontend runtime and arrive here as raw device
pointers, so a step issues the same launches with the same arguments every
time and can be captured into a CUDA graph.

Numerical contract -- HuggingFace ``Qwen3ForCausalLM`` in BF16, PyTorch eager:

* q, k, v, gate and up stay separate projections, as in HF. Every GEMM is
  ``bf16_matmul_cublaslt_ref_bf16``: by default cuBLASLt's heuristic first
  choice -- the algorithm torch's own matmul runs at these shapes -- so its
  bits. With ``tuned_gemms`` it may run a faster candidate that matched that
  choice on random probe operands, which is checked rather than guaranteed.
  Fusing q/k/v changes the algorithm for some sequence lengths and with it
  the reduction order.
* RMSNorm, the residual add, the q/k norm + RoPE and the SiLU gate run the
  ``*_hf_bf16`` kernels, which round where HF rounds and reduce in torch's
  order (``torch_reduce`` picks the tree of the running PyTorch release).
* The RoPE tables are built by the runtime the way HF builds them, from the
  ``inv_freq`` precision the declared baseline carries.
* Attention is the target's backend; it is the one stage that is close to,
  rather than equal to, the reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

#: flash_rt_kernels entries the prefill sequence launches.
KERNEL_SYMBOLS = (
    "embedding_lookup_bf16",
    "bf16_matmul_cublaslt_ref_bf16",
    "rms_norm_hf_bf16",
    "residual_add_rms_norm_hf_bf16",
    "qk_rms_norm_rope_hf_bf16",
    "silu_mul_hf_bf16",
)


@dataclass(frozen=True)
class Qwen3_4BDims:
    """Static dim contract for HuggingFace Qwen/Qwen3-4B (dense BF16).

    Matches https://huggingface.co/Qwen/Qwen3-4B -- same GQA geometry as
    Qwen3-8B (32q/8kv/hd=128) but hidden=2560 / intermediate=9728.
    """

    hidden: int = 2560
    num_layers: int = 36
    vocab_size: int = 151_936
    intermediate: int = 9728
    num_q_heads: int = 32
    num_kv_heads: int = 8           # GQA 4:1
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    max_pos: int = 40_960
    rms_norm_eps: float = 1e-6


def torch_reduce_tree(version: str | None = None) -> int:
    """The ``torch_reduce`` tree of the ``*_hf_bf16`` kernels for a release.

    PyTorch 2.10 reversed the intra-warp shuffle of its CUDA reduction and
    vectorized 128-wide rows, so the HF reference rounds differently on either
    side of it. ``version`` defaults to the running PyTorch.
    """
    v = (version or torch.__version__).split("+")[0].split(".")
    major, minor = int(v[0]), int("".join(c for c in v[1] if c.isdigit()) or 0)
    return 1 if (major, minor) >= (2, 10) else 0


@dataclass(frozen=True)
class Qwen3Bf16Target:
    """One device the BF16 prefill is bound to.

    ``name`` is the ``flash_rt.hardware`` arch key. ``compute_capability`` is
    the exact device generation the path was validated on; other devices are
    refused rather than assumed to behave the same. ``make_attention`` builds
    the attention binding over the runtime's q/k/v/o buffers.
    """

    name: str
    compute_capability: tuple[int, int]
    attention: str
    pdl: bool
    make_attention: Callable[..., Any]
    build_hint: str


def require_kernels(fvk: Any, target: Qwen3Bf16Target) -> None:
    """Raise unless ``flash_rt_kernels`` carries every entry the sequence uses."""
    missing = [name for name in KERNEL_SYMBOLS if not hasattr(fvk, name)]
    if missing:
        raise RuntimeError(
            f"Qwen3 BF16 prefill on {target.name} needs flash_rt_kernels "
            f"entries {missing}, which this build does not carry. "
            f"{target.build_hint}")


class Qwen3Bf16Prefill:
    """The decoder layer sequence over runtime-owned buffers (batch 1).

    ``layers`` holds one pointer table per decoder layer (``in_norm``, ``q``,
    ``k``, ``v``, ``q_norm``, ``k_norm``, ``o``, ``post_norm``, ``gate``,
    ``up``, ``down``); ``model`` holds ``embed``, ``final_norm`` and
    ``lm_head``. ``bufs`` points at the persistent activations, each sized for
    the runtime's largest prompt: ``ids`` (int64), ``h`` / ``xn`` / ``t``
    ``[S, hidden]``, ``q`` / ``o`` ``[S, nq * hd]``, ``k`` / ``v``
    ``[S, nkv * hd]``, ``g`` / ``u`` ``[S, intermediate]``, ``cos`` / ``sin``
    ``[S, hd / 2]``. ``attention.run(S, stream)`` attends ``q`` over ``k`` /
    ``v`` causally into ``o``.
    """

    def __init__(self, fvk: Any, dims: Qwen3_4BDims,
                 layers: Sequence[Mapping[str, int]],
                 model: Mapping[str, int], bufs: Mapping[str, int],
                 attention: Any, *, pdl: bool, torch_reduce: int,
                 tuned_gemms: bool = False) -> None:
        if len(layers) != dims.num_layers:
            raise ValueError(
                f"{len(layers)} layer tables for {dims.num_layers} layers")
        self._fvk = fvk
        self._d = dims
        self._layers = [dict(w) for w in layers]
        self._model = dict(model)
        self._b = dict(bufs)
        self._attn = attention
        self._pdl = bool(pdl)
        self._tree = int(torch_reduce)
        self._tuned = bool(tuned_gemms)

    def _gemm(self, x: int, w: int, out: int, m: int, n: int, k: int,
              stream: int) -> None:
        self._fvk.bf16_matmul_cublaslt_ref_bf16(x, w, out, m, n, k, self._tuned,
                                                stream)

    def run(self, S: int, stream: int) -> None:
        """Prefill ``S`` tokens from ``bufs['ids']``.

        Leaves the final-normed hidden states in ``bufs['xn']``.
        """
        f, d, b = self._fvk, self._d, self._b
        H, I = d.hidden, d.intermediate
        NQ, NKV, HD = d.num_q_heads, d.num_kv_heads, d.head_dim
        QD, KD = NQ * HD, NKV * HD
        eps, tree, pdl = d.rms_norm_eps, self._tree, self._pdl
        gemm = self._gemm

        f.embedding_lookup_bf16(b["ids"], self._model["embed"], b["h"], S, H,
                                stream)
        f.rms_norm_hf_bf16(b["h"], self._layers[0]["in_norm"], b["xn"], S, H,
                           eps, tree, pdl, stream)
        last = d.num_layers - 1
        for i, w in enumerate(self._layers):
            gemm(b["xn"], w["q"], b["q"], S, QD, H, stream)
            gemm(b["xn"], w["k"], b["k"], S, KD, H, stream)
            gemm(b["xn"], w["v"], b["v"], S, KD, H, stream)
            f.qk_rms_norm_rope_hf_bf16(
                b["q"], QD, NQ, b["k"], KD, NKV, w["q_norm"], w["k_norm"],
                b["cos"], b["sin"], S, HD, eps, tree, pdl, stream)
            self._attn.run(S, stream)
            gemm(b["o"], w["o"], b["t"], S, H, QD, stream)
            f.residual_add_rms_norm_hf_bf16(
                b["h"], b["t"], w["post_norm"], b["xn"], S, H, eps, tree, pdl,
                stream)
            gemm(b["xn"], w["gate"], b["g"], S, I, H, stream)
            gemm(b["xn"], w["up"], b["u"], S, I, H, stream)
            f.silu_mul_hf_bf16(b["g"], b["u"], b["g"], S * I, pdl, stream)
            gemm(b["g"], w["down"], b["t"], S, H, I, stream)
            nxt = (self._model["final_norm"] if i == last
                   else self._layers[i + 1]["in_norm"])
            f.residual_add_rms_norm_hf_bf16(
                b["h"], b["t"], nxt, b["xn"], S, H, eps, tree, pdl, stream)

    def lm_head(self, S: int, rows: int, out: int, stream: int) -> None:
        """``out[rows, vocab] = lm_head(xn[S - rows:S])`` after :meth:`run`."""
        H = self._d.hidden
        src = self._b["xn"] + (S - rows) * H * 2   # bf16 rows
        self._gemm(src, self._model["lm_head"], out, rows, self._d.vocab_size,
                   H, stream)
