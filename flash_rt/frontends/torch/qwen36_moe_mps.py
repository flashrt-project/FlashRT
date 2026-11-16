"""FlashRT -- Qwen3.6-35B-A3B (qwen3_5_moe) MLX affine int4 export on Apple Silicon (native Metal plan).

The text decoder is a 40-layer MoE (256 experts, top-8 softmax routing plus a gated shared expert) whose layers
alternate three Gated DeltaNet (linear attention) layers with one gated softmax-attention layer. This frontend
reads the MLX affine export (4-bit codes with BF16 scale/bias per 64 inputs, 8-bit router) with FlashRT's own
safetensors reader, copies the weight bytes unchanged into one Metal buffer (plus a BF16 copy of the routers for
the prefill), and runs FlashRT's native Metal programs: a chunked prefill and a resident decode step that produces up to eight tokens per dispatch. No MLX or
framework code runs at inference time; a one-token step reproduces mlx-lm's logits bit for bit.

    from flash_rt.frontends.torch.qwen36_moe_mps import Qwen36MoeTorchFrontendMps
    fe = Qwen36MoeTorchFrontendMps("<path-to-Qwen3.6-35B-A3B-4bit>", max_seq=32768)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Text only: the checkpoint's vision tower is not loaded. Constructed directly rather than through ``load_model``
(a text decoder has no VLA ``predict`` surface). The module sits with the other frontends; PyTorch is not
imported. See ``docs/qwen36_moe_apple_usage.md``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from flash_rt.frontends.torch._qwen36_moe_mps_runtime import Qwen36MoeRuntime
# The tokenizer.json / chat-template reader is model independent (it lives with the Flint model code).
from flash_rt.models.flint.tokenizer import FlintTokenizer as _JsonTokenizer
from flash_rt.models.qwen36_moe.mlx_checkpoint import Qwen35MoeConfig
from flash_rt.models.qwen36_moe.pipeline_mps import BLOCK

__all__ = ["Qwen36MoeTorchFrontendMps"]


class Qwen36MoeTorchFrontendMps:
    """Greedy or top-k/top-p generation over the native qwen3_5_moe Metal runtime (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 32768, chat_template: Optional[str] = None,
                 library: Optional[str] = None):
        self.rt = Qwen36MoeRuntime(checkpoint_dir, max_seq=max_seq, library=library)
        self.cfg: Qwen35MoeConfig = self.rt.cfg
        self.tokenizer = _JsonTokenizer.from_directory(checkpoint_dir, chat_template=chat_template)
        self.eog = set(self.cfg.eos_token_ids)
        # generation_config.json's sampling settings; generation is greedy unless they are passed to stream()
        self.sampling_defaults = dict(temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k)
        self._state_ids: List[int] = []   # tokens the recurrent state and the KV cache currently hold
        self.last_reused = 0

    @staticmethod
    def _sample_config(temperature: float, top_p: float, top_k: int):
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if temperature == 0:
            return "greedy", 0
        if not 1 <= top_k <= 32:
            raise ValueError("sampling (temperature > 0) needs 1 <= top_k <= 32: the fused sampler keeps the "
                             "32 largest logits")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        return "sampled", top_k

    def stream(self, prompt_ids: Sequence[int], max_new_tokens: int = 256, stop_ids: Optional[Sequence[int]] = None,
               reuse_prefix: bool = False, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 20,
               seed: int = 0) -> Iterator[List[int]]:
        """Continuation of ``prompt_ids`` as it is produced, in blocks of up to eight tokens. Stops after an
        end-of-generation token (included); ``stop_ids=()`` disables stopping. The decode step itself stops at
        the first four stop ids; further ones are checked per block, and the tokens after them are dropped.
        ``temperature=0`` (the default) is greedy; otherwise top-k (1..32) / top-p sampling with Gumbel noise
        drawn on the GPU from ``seed`` and the token position (``sampling_defaults`` holds the checkpoint's
        ``generation_config.json`` settings).

        ``reuse_prefix``: when the prompt extends everything the model state already holds (the previous
        prompt and its generated tokens, e.g. the next turn of a chat), prefill only the new tokens. The
        DeltaNet state exists only at that position, so any other prompt is recomputed from the start."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(t < 0 or t >= self.cfg.vocab_size for t in prompt_ids):
            raise ValueError("prompt token id outside the checkpoint vocabulary")
        n = len(prompt_ids)
        if n + max_new_tokens + BLOCK > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens exceeds max_seq {self.rt.max_seq}")
        program, k = self._sample_config(temperature, top_p, top_k)
        stop = list(self.eog if stop_ids is None else stop_ids)
        ids = list(prompt_ids)
        held = self._state_ids
        start = len(held) if reuse_prefix and held and len(held) < n and ids[:len(held)] == held else 0
        self._state_ids = []
        first = self.rt.prefill(ids, start=start, program=program, top_k=k, top_p=top_p, temperature=temperature,
                                seed=seed)
        self.last_reused = start
        generated = [first]
        yield [first]
        if first in stop or max_new_tokens == 1:
            self._state_ids = ids
            return
        token, position = first, n
        while len(generated) < max_new_tokens:
            steps = min(BLOCK, max_new_tokens - len(generated))
            block = self.rt.decode(token, position, steps, program=program, top_k=k, top_p=top_p,
                                   temperature=temperature, eos=stop[:4], seed=seed)
            generated.extend(block)
            position += len(block)
            token = block[-1]
            hit = next((i for i, t in enumerate(block) if t in stop), None)
            if hit is not None:
                yield block[:hit + 1]
                break
            yield block
        # the state holds every token that was fed: the prompt and all generated tokens but the last
        self._state_ids = ids + generated[:-1] if self.rt.pos == n + len(generated) - 1 else []

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None, **sampling) -> List[int]:
        out: List[int] = []
        for block in self.stream(prompt_ids, max_new_tokens, stop_ids, **sampling):
            out.extend(block)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False, **sampling) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens, **sampling)
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)
