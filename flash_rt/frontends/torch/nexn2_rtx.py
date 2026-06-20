"""FlashRT -- Nex-N2-mini inference frontend (PyTorch + RTX SM120).

Phase-1 surface area (frozen to keep tests stable across phases):
    - ``__init__(checkpoint_path)`` -- loads tokenizer + builds Pipeline
    - ``set_prompt(text)``          -- tokenizes for the next infer()
    - ``infer()``                   -- single forward, returns logits
    - ``generate(max_new_tokens)``  -- delegates to HF .generate()

Future surface (added in later phases, signatures kept stable):
    - ``calibrate_with_real_data(prompts)`` -- Phase 3 NVFP4/FP8 calibration
    - ``latency_records``                   -- list[float] populated by infer()

Phase 1 loads the HF reference model (BF16) to lock the cosine fixture.
The reference is large (35B total params), so it loads with HF device
mapping and may offload to host RAM; the production path (Phase 3) loads
NVFP4-quantized weights directly and fits the RTX 5090.
"""

from __future__ import annotations

import time

from flash_rt.models.nexn2.pipeline_rtx import Nexn2Pipeline


class Nexn2TorchFrontendRtx:
    """Nex-N2-mini inference frontend (PyTorch + RTX SM120)."""

    def __init__(self, checkpoint_path: str, *,
                 device: str = 'cuda:0',
                 max_seq: int = 2048,
                 quant: str = 'nvfp4') -> None:
        """Construct the frontend.

        Args:
          checkpoint_path: HF-style checkpoint directory.
          device: cuda device string for the kernelized path.
          max_seq: maximum sequence length (KV + scratch sized to this).
          quant: weight quantization format for the kernelized path.
            * ``'nvfp4'`` (default): NVFP4 W4A16 for full-attn + MoE GEMM;
              GDN in_proj kept BF16.
            * ``'fp8'``: FP8 E4M3 block-128 weights.
            In Phase 1 this only records intent; the shim loads the BF16
            HF reference regardless.
        """
        if quant not in ('fp8', 'nvfp4'):
            raise ValueError(f"quant must be 'fp8' or 'nvfp4', got {quant!r}")

        self.checkpoint_path = checkpoint_path
        self.device = device
        self._user_max_seq = int(max_seq)
        self._quant_format = quant
        self._tokenizer = None
        self._prompt_ids = None
        self._pipeline: Nexn2Pipeline | None = None
        self.latency_records: list[float] = []

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
        t0 = time.perf_counter()
        logits = self._pipeline.forward(self._prompt_ids)
        self.latency_records.append(time.perf_counter() - t0)
        return logits

    def generate(self, max_new_tokens: int, *, do_sample: bool = False):
        """Autoregressive generate over the current prompt. Phase-1 -> HF."""
        if self._prompt_ids is None:
            raise ValueError('call set_prompt(...) before generate()')
        return self._pipeline.generate(
            self._prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
