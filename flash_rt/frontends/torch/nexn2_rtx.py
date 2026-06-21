"""FlashRT -- Nex-N2-mini inference frontend (PyTorch + RTX SM120).

LLM frontend for the qwen3_5_moe 35B-A3B model. Surface:
    - ``__init__(checkpoint_path, *, kernelized=, quant_scope=, ...)``
    - ``set_prompt(text)``          -- tokenizes for the next call
    - ``infer()``                   -- single forward, returns logits
    - ``generate(max_new_tokens)``  -- greedy decode
    - ``latency_records``           -- list[float] populated by infer()

Two backends share this surface:
  * ``kernelized=True`` (production): NVFP4 weights loaded directly via the
    fvk loader and run through the SM120 kernel forward / decode -- fits the
    32 GB RTX 5090 (the BF16 model does not). Requires the gated kernel build
    (-DFLASHRT_ENABLE_QWEN35MOE=ON). See docs/nexn2_usage.md.
  * ``kernelized=False`` (reference): the BF16 HF model, used to lock the
    golden cosine fixture. Large (35B total params) -- loads with HF device
    mapping and may offload to host RAM.
"""

from __future__ import annotations

import time

from flash_rt.models.nexn2.pipeline_rtx import Nexn2Pipeline

# Kernels the kernelized path calls; checked up front so a build without
# -DFLASHRT_ENABLE_QWEN35MOE=ON fails clearly instead of after loading the 35B
# checkpoint and crashing mid-forward on a missing symbol.
_REQUIRED_FVK = (
    'w16a16_gemm_sm120_bf16', 'moe_blocktile_mma_sm120_bf16',
    'moe_weighted_sum_sm120_bf16', 'moe_router_topk_sm120_bf16',
    'qwen36_partial_rope_qk_bf16', 'causal_conv1d_qwen36_bf16',
    'gdn_recurrent_seq_sm120_bf16',
)


def _require_kernels(fvk) -> None:
    """Raise a clear RuntimeError if the gated qwen3_5_moe kernels or the FA2
    module are missing (build was not configured with
    -DFLASHRT_ENABLE_QWEN35MOE=ON, or flash_rt_fa2 is absent)."""
    missing = [s for s in _REQUIRED_FVK if not hasattr(fvk, s)]
    if missing:
        raise RuntimeError(
            "Nex-N2 kernelized path needs the qwen3_5_moe SM120 kernels, which "
            "are absent from flash_rt_kernels (missing: "
            f"{', '.join(missing)}). Rebuild on an SM120 toolchain with "
            "-DFLASHRT_ENABLE_QWEN35MOE=ON. See docs/nexn2_usage.md.")
    try:
        from flash_rt import flash_rt_fa2 as _fa2
    except Exception as e:                                  # pragma: no cover
        raise RuntimeError(
            "Nex-N2 full attention needs the vendored FA2 module "
            "(flash_rt_fa2), which failed to import. Build with FA2 enabled "
            "(ENABLE_FA2, auto-on for SM120).") from e
    if not hasattr(_fa2, 'fwd_bf16_causal'):                # pragma: no cover
        raise RuntimeError(
            "flash_rt_fa2 is present but lacks fwd_bf16_causal; rebuild the "
            "FA2 module.")


class Nexn2TorchFrontendRtx:
    """Nex-N2-mini inference frontend (PyTorch + RTX SM120)."""

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0',
                 max_seq: int = 2048,
                 quant: str = 'nvfp4',
                 kernelized: bool = False,
                 quant_scope: str = 'full') -> None:
        """Construct the frontend.

        Args:
          checkpoint_path: HF-style checkpoint directory.
          device: cuda device string for the kernelized path.
          max_seq: maximum sequence length (KV + scratch sized to this).
          quant: weight quantization format for the kernelized path. Only
            ``'nvfp4'`` is implemented (NVFP4 W4A16 for full-attn + MoE GEMM;
            GDN in_proj kept BF16); any other value raises NotImplementedError.
          kernelized: when False (default) load the BF16 HF reference model
            (correctness baseline; the 35B-A3B weights do not fit the 32 GB
            card). When True load the NVFP4-quantized weights directly via the
            fvk loader and run the kernel forward/decode -- the production path.
          quant_scope: kernelized-only. ``'experts'`` (recommended) = only
            the routed experts are NVFP4; the dense projections run on the
            deterministic BF16-weight w16a16 GEMM, so prefill cos vs the BF16
            golden is ~0.99 and bit-reproducible. ``'full'`` additionally
            NVFP4-quantises the non-red-line dense projections (q/k/v/o /
            out_proj / shared) for a smaller footprint at lower cos.
        """
        if quant != 'nvfp4':
            raise NotImplementedError(
                f"quant={quant!r} is not implemented; only 'nvfp4' is "
                "supported (the kernelized path quantizes via "
                "extract_weights_nexn2_nvfp4).")

        self.checkpoint_path = checkpoint_path
        self.device = device
        self._user_max_seq = int(max_seq)
        self._quant_format = quant
        self._kernelized = bool(kernelized)
        self._quant_scope = quant_scope
        self._tokenizer = None
        self._prompt_ids = None
        self._pipeline: Nexn2Pipeline | None = None
        self._weights = None
        self._fvk = None
        self._decode_state = None
        self.latency_records: list[float] = []

        if self._kernelized:
            self._build_kernelized_nvfp4()
        else:
            self._build_phase1_reference()

    def _build_phase1_reference(self) -> None:
        """Load tokenizer + the BF16 HF reference model (kernelized=False).

        This is the correctness baseline only; the production path is the
        kernelized forward/decode (kernelized=True).
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            device_map='auto',
        )
        hf_model.eval()
        self._pipeline = Nexn2Pipeline(hf_model)

    def _build_kernelized_nvfp4(self) -> None:
        """Load NVFP4 weights via the fvk loader and arm the kernel forward.

        No HF model: the 35B-A3B checkpoint does not fit a 32 GB RTX 5090
        in BF16. The loader streams each shard, quantizes the large GEMMs
        to NVFP4 (GDN in_proj / norms / router kept BF16) and frees the
        BF16 source as it goes, fitting in ~22 GB.
        """
        from transformers import AutoTokenizer

        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import (
            extract_weights_nexn2_nvfp4,
        )

        _require_kernels(fvk)            # fail fast before loading the 35B ckpt

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self._fvk = fvk
        self._weights = extract_weights_nexn2_nvfp4(
            self.checkpoint_path, fvk, device=self.device,
            quant_scope=self._quant_scope)

    @property
    def tokenizer(self):
        """The HF tokenizer loaded from the checkpoint."""
        return self._tokenizer

    def set_prompt(self, text: str) -> None:
        """Tokenize ``text`` for the next ``infer()`` / ``generate()`` call."""
        enc = self._tokenizer(text, return_tensors='pt')
        self._prompt_ids = enc['input_ids'].to(self.device)

    def infer(self):
        """Single forward pass over the current prompt; returns logits.

        Returns:
            logits: (B, S, vocab_size) tensor.
        """
        if self._prompt_ids is None:
            raise ValueError('call set_prompt(...) before infer()')
        if self._kernelized:
            import torch

            from flash_rt.frontends.torch._nexn2_rtx_forward import (
                nexn2_forward_nvfp4,
            )
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = nexn2_forward_nvfp4(
                    self._weights, self._prompt_ids, self._fvk, self.device)
            torch.cuda.synchronize()
            self.latency_records.append(time.perf_counter() - t0)
            return logits.unsqueeze(0)        # (1, S, vocab)
        t0 = time.perf_counter()
        logits = self._pipeline.forward(self._prompt_ids)
        self.latency_records.append(time.perf_counter() - t0)
        return logits

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        """Autoregressive generate over the current prompt.

        Kernelized path: greedy M=1 decode over the fvk kernels (KV cache +
        GDN recurrent/conv state). Reference path: HF .generate().
        """
        if self._prompt_ids is None:
            raise ValueError('call set_prompt(...) before generate()')
        if self._kernelized:
            from flash_rt.frontends.torch._nexn2_rtx_decode import (
                Nexn2DecodeState, generate_greedy,
            )
            if self._decode_state is None:
                self._decode_state = Nexn2DecodeState(
                    self._weights, self._user_max_seq, self.device)
            return generate_greedy(
                self._decode_state, self._prompt_ids, max_new_tokens,
                self._fvk, self.device)
        return self._pipeline.generate(
            self._prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
