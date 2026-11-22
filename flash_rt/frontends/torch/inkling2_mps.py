"""FlashRT -- Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, ``inkling2_loop``) on Apple Silicon (native Metal plan).

The text decoder visits 48 layers per token (24 stored layers; layers 6..17 run three passes with message gates
between passes) with block attention residuals, sliding-window and global attention with relative position bias,
short convolutions and a 128-expert MoE. The released checkpoint is used as is: BF16 attention and LM head,
NVFP4 experts in layers 0-19 and MXFP8 in 20-23, copied byte for byte into one Metal buffer and run by FlashRT's
own Metal programs: chunked prefill row programs and a resident decode step that produces up to eight tokens per
dispatch and reproduces mlx-lm's logits bit for bit. With ``speculative=True`` the checkpoint's DSpark drafter
(``mtp.safetensors``) proposes blocks of tokens that the target verifies (greedy; the output is the target's
greedy continuation up to near-ties in the verification's rounding).

    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    fe = Inkling2TorchFrontendMps("<path-to-Inkling-2-Mini>", max_seq=8192)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Text only (the vision tower is not loaded), greedy decoding. Constructed directly rather than through
``load_model`` (a text decoder has no VLA ``predict`` surface); PyTorch is not imported. See
``docs/inkling2_apple_usage.md``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from flash_rt.frontends.torch._inkling2_mps_runtime import BLOCK, Inkling2Runtime
# The tokenizer.json reader is model independent (it lives with the Flint model code).
from flash_rt.models.flint.tokenizer import FlintTokenizer as _JsonTokenizer

__all__ = ["Inkling2TorchFrontendMps"]

PREFILL_CHUNKS = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


class Inkling2TorchFrontendMps:
    """Greedy generation over the native inkling2_loop Metal runtime (one sequence)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, speculative: bool = False,
                 chat_template: Optional[str] = None, library: Optional[str] = None):
        shapes = [(c, 1) for c in PREFILL_CHUNKS]
        self.speculative = bool(speculative)
        self.rt = Inkling2Runtime(checkpoint_dir, max_seq=max_seq, library=library, row_capacity=PREFILL_CHUNKS[0],
                                  row_shapes=[(1, 1)] if self.speculative else shapes,
                                  mtp_batches=[1] if self.speculative else (),
                                  mtp_prefill=PREFILL_CHUNKS if self.speculative else ())
        self.cfg = self.rt.cfg
        self.tokenizer = _JsonTokenizer.from_directory(checkpoint_dir, chat_template=chat_template)
        self.eog = set(self.cfg.eos_token_ids)
        self._template = None

    def stream(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
               stop_ids: Optional[Sequence[int]] = None) -> Iterator[List[int]]:
        """Greedy continuation of ``prompt_ids`` as it is produced (blocks of up to eight tokens; with
        ``speculative`` the accepted tokens of each draft / verify step). Stops after an end-of-generation token
        (included); ``stop_ids=()`` disables stopping."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(t < 0 or t >= self.cfg.unpadded_vocab_size for t in prompt_ids):
            raise ValueError("prompt token id outside the checkpoint vocabulary")
        n = len(prompt_ids)
        if n + max_new_tokens + BLOCK > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens exceeds max_seq {self.rt.max_seq}")
        stop = set(self.eog if stop_ids is None else stop_ids)
        produced = 0
        if self.speculative:
            for step in self.rt.mtp_steps([list(prompt_ids)], max_new_tokens):
                block = step[0][:max_new_tokens - produced]
                produced += len(block)
                hit = next((i for i, t in enumerate(block) if t in stop), None)
                if hit is not None:
                    yield block[:hit + 1]
                    return
                yield block
                if produced >= max_new_tokens:
                    return
            return
        first = self.rt.prefill(list(prompt_ids))
        yield [first]
        if first in stop or max_new_tokens == 1:
            return
        token, position, produced = first, n, 1
        eos = sorted(stop)[:4]
        while produced < max_new_tokens:
            steps = min(BLOCK, max_new_tokens - produced)
            block = self.rt.decode(token, position, steps, program="greedy", eos=eos)
            produced += len(block)
            position += len(block)
            token = block[-1]
            hit = next((i for i, t in enumerate(block) if t in stop), None)
            if hit is not None:
                yield block[:hit + 1]
                return
            yield block

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None) -> List[int]:
        out: List[int] = []
        for block in self.stream(prompt_ids, max_new_tokens, stop_ids):
            out.extend(block)
        return out

    def apply_chat_template(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                            add_generation_prompt: bool = True, **template_kwargs) -> str:
        """The checkpoint's chat template (``chat_template.jinja``; it takes e.g. ``reasoning_effort``)."""
        if self.tokenizer.chat_template is None:
            raise ValueError("no chat template (chat_template.jinja)")
        if self._template is None:
            import json
            from jinja2.ext import Extension
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            class GenerationBlock(Extension):
                """``{% generation %} ... {% endgeneration %}`` (transformers' assistant-mask markers): the body."""
                tags = {"generation"}

                def parse(self, parser):
                    next(parser.stream)
                    return parser.parse_statements(("name:endgeneration",), drop_needle=True)

            def raise_exception(message):
                raise ValueError(message)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[GenerationBlock, "jinja2.ext.loopcontrols"])
            env.filters["tojson"] = lambda v, ensure_ascii=True, **kw: json.dumps(v, ensure_ascii=ensure_ascii)
            env.globals["raise_exception"] = raise_exception
            self._template = env.from_string(self.tokenizer.chat_template)
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                     **template_kwargs)

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False, **template_kwargs) -> str:
        prompt = self.apply_chat_template(messages, tools=tools, **template_kwargs)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens)
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)
