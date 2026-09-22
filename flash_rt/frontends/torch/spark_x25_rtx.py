"""FlashRT -- Spark-X2.5-4B torch frontend for RTX (SM120).

Spark-X2.5-4B is a hybrid-attention chat model: 36 layers, 27 with a 512-token
sliding window and 9 full-attention, NVFP4 weights, 1M-token position range.
Batch is 1 and every steady-state shape is fixed, so the whole decode loop is
captured into one CUDA Graph and replayed with no framework dispatch.

This frontend owns the checkpoint contract and the generation loop. The layer
sequence itself is in ``flash_rt.models.spark_x25.pipeline_rtx``; the kernels it
drives are the separate ``flash_rt_sparkx25`` module (SM120 only) plus
``flash_rt_kernels`` for the NVFP4 GEMMs and ``flash_rt_fa2`` for prefill
attention.

    from flash_rt.frontends.torch.spark_x25_rtx import SparkX25TorchFrontendRtx
    fe = SparkX25TorchFrontendRtx("/models/Spark-X2.5-4B", max_seq=131072)
    out = fe.generate(prompt_ids, max_new_tokens=128)

The frontend is constructed directly rather than through ``load_model``:
``load_model`` wraps models in the VLA surface, which a text decoder does not
have. Qwen3.6 and Qwen3-VL are constructed the same way.

See ``docs/spark_x25_usage.md`` for the parameter reference and
``docs/spark_x25_rtx.md`` for the measurements and the KV residency modes.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from flash_rt.models.spark_x25.config import SparkX25Config, load_config
from flash_rt.models.spark_x25.pipeline_rtx import SparkX25Runtime

__all__ = ["SparkX25TorchFrontendRtx", "validate_spark_x25_checkpoint"]


def validate_spark_x25_checkpoint(path: str) -> SparkX25Config:
    """Check that ``path`` is a Spark-X2.5-4B checkpoint this frontend can run.

    Returns the parsed config. Raises ValueError on a mismatch, so a wrong
    checkpoint fails here rather than as a shape error deep in a capture.
    """
    cfg = load_config(path)
    if cfg.num_hidden_layers != 36 or cfg.head_dim != 256:
        raise ValueError(
            f"not a Spark-X2.5-4B checkpoint: {cfg.num_hidden_layers} layers, "
            f"head_dim {cfg.head_dim} (expected 36 and 256)")
    if cfg.num_attention_heads // cfg.num_key_value_heads != 4:
        raise ValueError(
            "expected a 4:1 GQA group; the attention kernels assume four query "
            f"heads per KV head, got {cfg.num_attention_heads}:"
            f"{cfg.num_key_value_heads}")
    if cfg.sliding_window != 512:
        raise ValueError(f"expected a 512-token sliding window, got "
                         f"{cfg.sliding_window}")
    return cfg


class SparkX25TorchFrontendRtx:
    """Spark-X2.5-4B on RTX SM120. Text in, token ids out.

    ``max_seq`` sizes the KV caches and is the one knob that changes behaviour
    rather than cost: the full layers keep an exact bf16 KV plus an E4M3 copy
    for decode while both fit, and E4M3 alone with a bf16 staging buffer for
    prefill when they do not (around 217k on a 16 GB part). The first mode
    keeps the long-context logit cosine at ~0.999, the second at ~0.99 and is
    what makes a 262k window run at a usable rate. The choice is made at
    construction from the free device memory; ``kv8_only`` reports it.
    """

    def __init__(self, checkpoint: str, *, max_seq: int = 32768,
                 prefill_cap: int | None = None, prefill_chunk: int | None = None,
                 device: str = "cuda",
                 attn_splits: int | None = None,
                 attn_splits_slide: int | None = None) -> None:
        self.checkpoint = checkpoint
        self.config = validate_spark_x25_checkpoint(checkpoint)
        self.max_seq = int(max_seq)
        self.device = device
        self.runtime = SparkX25Runtime(
            checkpoint,
            max_seq=self.max_seq,
            prefill_cap=int(prefill_cap or min(self.max_seq, 8192)),
            prefill_chunk=prefill_chunk,
            device=device,
            attn_splits=attn_splits,
            attn_splits_slide=attn_splits_slide,
        )
        self._tokenizer = None
        self._prompt_len = 0

    # ── tokenizer ────────────────────────────────────────────────────────
    @property
    def tokenizer(self) -> Any:
        """Lazy, so importing the frontend does not require transformers."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint, trust_remote_code=True)
        return self._tokenizer

    @property
    def kv8_only(self) -> bool:
        """True when the full layers keep E4M3 alone (see the class docstring)."""
        return bool(self.runtime.kv8_only)

    @property
    def attn_splits(self) -> int:
        """Full-layer decode KV split count baked into the captured graph."""
        return int(self.runtime.attn_nsplit)

    @property
    def attn_splits_slide(self) -> int:
        """Sliding-layer decode KV split count baked into the captured graph."""
        return int(self.runtime.attn_nsplit_slide)

    # ── inference ────────────────────────────────────────────────────────
    def set_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill ``input_ids`` and return the last row's logits."""
        ids = input_ids.to(self.device).reshape(-1)
        self._prompt_len = int(ids.numel())
        with torch.no_grad():
            self.runtime.forward(ids, pos=0)
        return self.runtime.logits[:1]

    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int = 128,
                 graph_steps: int | None = None) -> torch.Tensor:
        """Greedy decode. Returns the prompt followed by the new tokens.

        The decode loop is captured once and replayed; ``graph_steps`` sets how
        many steps that graph covers (defaults to ``max_new_tokens``).
        """
        logits = self.set_prompt(input_ids)
        self.runtime.next_token.fill_(int(logits[-1].argmax()))
        steps = int(graph_steps or max_new_tokens)
        with torch.no_grad():
            new = self.runtime.decode_loop(self._prompt_len, steps)
        ids = input_ids.to(self.device).reshape(-1)
        return torch.cat([ids, new[:max_new_tokens]])

    def generate_text(self, prompt: str, *, max_new_tokens: int = 128,
                      enable_thinking: bool = False) -> str:
        """Convenience wrapper: renders the chat template, decodes the result."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking)
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        out = self.generate(ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[self._prompt_len:].tolist(),
                                     skip_special_tokens=True)
