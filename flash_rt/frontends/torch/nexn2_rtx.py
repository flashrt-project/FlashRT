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
          quant: weight quantization format for the kernelized path.
            * ``'nvfp4'`` (default): NVFP4 W4A16 for full-attn + MoE GEMM;
              GDN in_proj kept BF16.
            * ``'fp8'``: FP8 E4M3 block-128 weights.
          kernelized: when False (default) load the BF16 HF reference and
            delegate forward to it (Phase-1 fixture). When True load the
            NVFP4-quantized weights directly via the fvk loader and run the
            kernelized forward -- the production seam (fits the RTX 5090;
            the BF16 reference does not).
          quant_scope: kernelized-only. ``'experts'`` (recommended) = only
            the routed experts are NVFP4; the dense projections run on the
            deterministic BF16-weight w16a16 GEMM, so prefill cos vs the BF16
            golden is ~0.99 and bit-reproducible. ``'full'`` additionally
            NVFP4-quantises the non-red-line dense projections (q/k/v/o /
            out_proj / shared) for a smaller footprint at lower cos.
        """
        if quant not in ('fp8', 'nvfp4'):
            raise ValueError(f"quant must be 'fp8' or 'nvfp4', got {quant!r}")

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
        """Load tokenizer + HF reference model and wrap it in the Pipeline.

        Replaced kernel-by-kernel in Phase 2+; the seams (Pipeline object,
        tokenizer, prompt ids) stay identical.
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
