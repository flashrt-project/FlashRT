"""FlashRT -- Qwen3.6-27B ROCm frontend."""

from __future__ import annotations

import time
from typing import Any

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class Qwen36TorchFrontendRocm:
    """ROCm Qwen3.6 frontend skeleton with raw official-FP8 weight loading.

    This frontend validates and owns the AMD
    weight ABI, then later phases replace the placeholder forward with the
    kernelized mixed linear/full-attention pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        max_seq: int = 2048,
        max_layers: int | None = None,
        load_weights: bool = True,
        weight_mode: str = "official_fp8",
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("Qwen36TorchFrontendRocm requires ROCm PyTorch")

        from flash_rt.frontends.torch.qwen36_rocm_weights import (
            extract_weights_qwen36_fp8_rocm,
            summarize_qwen36_rocm_weights,
        )

        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.max_seq = int(max_seq)
        self.latency_records: list[float] = []
        self._prompt_ids = None

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self.weights = extract_weights_qwen36_fp8_rocm(
            self.checkpoint_path,
            device=self.device,
            max_layers=max_layers,
            load_weights=load_weights,
            weight_mode=weight_mode,
        )
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.config_summary = summarize_qwen36_rocm_weights(self.weights)
        self.config_summary.update({"max_seq": self.max_seq, "load_s": self.load_s})

    def set_prompt(self, text: str) -> None:
        enc = self.tokenizer(text, return_tensors="pt")
        self._prompt_ids = enc.input_ids.to(self.device)
        if self._prompt_ids.shape[1] > self.max_seq:
            raise ValueError(
                f"prompt length {self._prompt_ids.shape[1]} exceeds max_seq={self.max_seq}"
            )

    def infer(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.6 ROCm forward is staged: raw FP8 ABI is loaded, "
            "kernelized mixed linear/full-attention forward lands next."
        )
