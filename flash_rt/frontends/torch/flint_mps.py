"""FlashRT -- Flint (RecursiveSI/flint) MLX affine int4 export on Apple Silicon (native Metal plan).

Flint is a 49-layer MoE chat model (256 experts, top-6 sigmoid routing plus a shared expert,
sliding-window/global attention with sinks). This frontend reads the MLX affine export (4-bit codes with
BF16 scale/bias per 64 inputs, 8-bit embedding) with FlashRT's own safetensors reader, copies the weight
bytes unchanged into one Metal buffer, and runs FlashRT's native Metal programs: a resident decode step
that produces up to eight tokens per dispatch. No MLX or framework code runs at inference time.

    from flash_rt.frontends.torch.flint_mps import FlintTorchFrontendMps
    fe = FlintTorchFrontendMps("<path-to-flint-int4>", max_seq=32768)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Constructed directly rather than through ``load_model`` (a text decoder has no VLA ``predict``
surface). The name follows the RTX frontend; PyTorch is not imported. See ``docs/flint_apple_usage.md``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from flash_rt.frontends.torch._flint_mps_runtime import FlintRuntime
from flash_rt.models.flint.config import FlintConfig
from flash_rt.models.flint.pipeline_mps import BLOCK
from flash_rt.models.flint.tokenizer import FlintTokenizer

__all__ = ["FlintTorchFrontendMps"]


class FlintTorchFrontendMps:
    """Greedy or top-k/top-p generation over the native Flint Metal runtime (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 32768, chat_template: Optional[str] = None,
                 library: Optional[str] = None):
        self.rt = FlintRuntime(checkpoint_dir, max_seq=max_seq, library=library)
        self.cfg: FlintConfig = self.rt.cfg
        self.tokenizer = FlintTokenizer.from_directory(checkpoint_dir, chat_template=chat_template)
        self.eog = set(self.cfg.eog_token_ids)
        self.check_block = BLOCK
        self._kv_ids: List[int] = []
        self.min_reuse = 64
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
        end-of-generation token (included); ``stop_ids=()`` disables stopping. ``temperature=0`` is greedy;
        otherwise top-k (1..32, default 20 as in the reference engine) / top-p sampling with Gumbel noise
        drawn on the GPU from ``seed`` and the token position.

        ``reuse_prefix``: keep the KV cache of the longest common prefix with the previous sequence (its
        prompt and generated tokens, at least 64 tokens) and prefill only the rest as a continued chunk. If
        the sliding-window history needed at the reuse position has been overwritten, the full prompt is
        recomputed. Reused positions keep the KV they were given (by prefill, or by decode for generated
        tokens), so the continuation can differ in rounding from a full prefill of the same tokens."""
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
        start = 0
        if reuse_prefix and self._kv_ids:
            start = min(self.common_prefix(ids), n - 1)
            history_start = max(0, start - self.cfg.sliding_window + 1)
            if start < self.min_reuse or history_start < self.rt.swa_kv_start:
                start = 0
        self._kv_ids = []
        first = self.rt.prefill(ids, start=start, program=program, top_k=k, top_p=top_p, temperature=temperature,
                                seed=seed)
        self.last_reused = start
        generated = [first]
        yield [first]
        if first in stop or max_new_tokens == 1:
            self._kv_ids = ids
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
        self._kv_ids = ids + generated[:-1]

    def common_prefix(self, ids: Sequence[int]) -> int:
        kv = self._kv_ids
        n = min(len(kv), len(ids))
        i = 0
        while i < n and kv[i] == ids[i]:
            i += 1
        return i

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
