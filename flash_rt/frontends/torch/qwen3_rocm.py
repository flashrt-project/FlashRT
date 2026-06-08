"""FlashRT -- PyTorch frontend for Qwen3 dense models on ROCm.

This is the AMD ROCm baseline for official BF16 Qwen3 checkpoints such as
``Qwen/Qwen3-8B``. It deliberately keeps the public surface close to the RTX
Qwen3 server-facing frontend while using Hugging Face's BF16 model as the
correctness anchor. Kernelized ROCm BF16/FP8 paths can be swapped in behind this
class once their owned-buffer contracts are ready.
"""

from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class Qwen3TorchFrontendRocm:
    """Qwen3-8B-class ROCm frontend using the official BF16 checkpoint.

    Public surface mirrors the early RTX Qwen3 frontend where practical:

    - ``set_prompt(text)``
    - ``generate(max_new_tokens=..., do_sample=...)``
    - ``infer({"prompt": ...})`` for simple usage
    - ``get_latency_stats()``

    This first ROCm path is intentionally not quantized. It is a reliable
    baseline for later hipBLASLt/CK/FlashAttention graph work.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda:0",
        max_seq: int = 2048,
        max_q_seq: int = 1,
        alloc_own_forward_buffers: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
        **_: Any,
    ) -> None:
        del max_q_seq, alloc_own_forward_buffers
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("Qwen3TorchFrontendRocm requires ROCm PyTorch")

        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.max_seq = int(max_seq)
        self._prompt_ids = None
        self.latency_records: list[float] = []
        self._last_output_ids = None

        dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }.get(str(torch_dtype).lower())
        if dtype is None:
            raise ValueError(f"unsupported torch_dtype={torch_dtype!r}")
        self.torch_dtype = dtype

        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        ).eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0

        cfg = self._model.config
        self._cfg = {
            "model_type": getattr(cfg, "model_type", None),
            "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0)),
            "hidden_size": int(getattr(cfg, "hidden_size", 0)),
            "intermediate_size": int(getattr(cfg, "intermediate_size", 0)),
            "num_attention_heads": int(getattr(cfg, "num_attention_heads", 0)),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", 0)),
            "head_dim": int(getattr(cfg, "head_dim", 0)),
            "vocab_size": int(getattr(cfg, "vocab_size", 0)),
            "torch_dtype": str(dtype).replace("torch.", ""),
            "attn_implementation": attn_implementation,
        }

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config_summary(self) -> dict[str, Any]:
        return dict(self._cfg)

    def reset_state(self) -> None:
        self._last_output_ids = None

    def set_prompt(self, text: str) -> None:
        import torch

        enc = self._tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        if input_ids.shape[1] > self.max_seq:
            raise ValueError(
                f"prompt length {input_ids.shape[1]} exceeds max_seq={self.max_seq}"
            )
        self._prompt_ids = input_ids
        if "attention_mask" in enc:
            self._prompt_attention_mask = enc.attention_mask.to(self.device)
        else:
            self._prompt_attention_mask = torch.ones_like(input_ids)

    def generate(
        self,
        prompt: str | None = None,
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        return_ids: bool = False,
    ) -> dict[str, Any]:
        import torch

        if prompt is not None:
            self.set_prompt(prompt)
        if self._prompt_ids is None:
            raise ValueError("prompt is required before generate()")

        gen_kwargs: dict[str, Any] = {
            "input_ids": self._prompt_ids,
            "attention_mask": self._prompt_attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(
                temperature=max(float(temperature), 1.0e-6),
                top_p=float(top_p),
            )
            if top_k and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

        with torch.inference_mode():
            t0 = time.perf_counter()
            out = self._model.generate(**gen_kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.latency_records.append(elapsed_ms)
        self._last_output_ids = out
        prompt_len = int(self._prompt_ids.shape[1])
        new_tokens = max(0, int(out.shape[1]) - prompt_len)
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        result: dict[str, Any] = {
            "text": text,
            "new_tokens": new_tokens,
            "generate_ms": elapsed_ms,
            "tok_per_s": (1000.0 * new_tokens / elapsed_ms) if elapsed_ms else 0.0,
            "prompt_tokens": prompt_len,
        }
        if return_ids:
            result["output_ids"] = out
        return result

    def infer(self, request: dict | str, debug: bool = False) -> dict[str, Any]:
        if isinstance(request, str):
            prompt = request
            max_new_tokens = 32
        else:
            prompt = request.get("prompt") or request.get("text")
            max_new_tokens = int(request.get("max_new_tokens", 32))
        if prompt is None and self._prompt_ids is None:
            raise ValueError("request must include 'prompt' on first call")
        result = self.generate(prompt, max_new_tokens=max_new_tokens)
        if debug:
            result["debug"] = {
                "backend": "rocm",
                "checkpoint": self.checkpoint_path,
                "config": self.config_summary,
                "load_s": self.load_s,
            }
        return result

    def get_latency_stats(self) -> dict[str, float]:
        if not self.latency_records:
            return {"count": 0}
        vals = list(self.latency_records)
        return {
            "count": len(vals),
            "mean_ms": sum(vals) / len(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
        }
