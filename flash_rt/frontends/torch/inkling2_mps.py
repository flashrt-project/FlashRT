"""FlashRT -- Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, ``inkling2_loop``) on Apple Silicon (native Metal plan).

The text decoder visits 48 layers per token (24 stored layers; layers 6..17 run three passes with message gates
between passes) with block attention residuals, sliding-window and global attention with relative position bias,
short convolutions and a 128-expert MoE. The released checkpoint is used as is: BF16 attention and LM head,
NVFP4 experts in layers 0-19 and MXFP8 in 20-23, copied byte for byte into one Metal buffer and run by FlashRT's
own Metal programs: chunked prefill row programs and a resident decode step that produces up to eight tokens per
dispatch and reproduces mlx-lm's logits bit for bit. With ``speculative=True`` the checkpoint's DSpark drafter
(``mtp.safetensors``) proposes blocks of tokens that the target verifies in one resident step over the 8 rows
(greedy): ``verify="fast"`` (default) sums the products in its own order (half-precision code products, results
within the rounding of the decode step's); ``verify="exact"`` repeats the decode step's arithmetic on every row, so
the output is the greedy decode's continuation bit for bit; ``verify="rows"`` uses the row programs. With the fast
verification a step verifies 2-8 rows chosen from the drafts' probabilities, or 12 / 16 with lookup drafts (the
continuation of an earlier occurrence in the text: after the drafter's block, or alone where the text repeats;
``lookup=False``: none; ``adaptive=False``: always 8 rows).
``dflash2=<directory>`` drafts with the DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2) instead of DSpark.
``q8=True`` runs the attention projections and the LM head from Q8_0 copies made at load (faster, not the
checkpoint's numerics).

    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    fe = Inkling2TorchFrontendMps("<path-to-Inkling-2-Mini>", max_seq=8192)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Text only (the vision tower is not loaded); greedy or top-k / top-p sampling. Constructed directly rather than through
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
VERIFY = {"fast": "fast", "exact": "resident", "rows": "rows"}     # speculative verification -> runtime program


class Inkling2TorchFrontendMps:
    """Generation over the native inkling2_loop Metal runtime (one sequence)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, speculative: bool = False,
                 chat_template: Optional[str] = None, library: Optional[str] = None, verify: str = "fast",
                 q8: bool = False, adaptive: bool = True, dflash2: Optional[str] = None, lookup: bool = True):
        if verify not in VERIFY:
            raise ValueError(f"verify must be one of {sorted(VERIFY)}")
        if q8 and verify == "exact":
            raise ValueError("verify='exact' repeats the BF16 decode step; it is not available with q8")
        shapes = [(c, 1) for c in PREFILL_CHUNKS]
        self.speculative = bool(speculative)
        self.rt = Inkling2Runtime(checkpoint_dir, max_seq=max_seq, library=library, row_capacity=PREFILL_CHUNKS[0],
                                  row_shapes=[(1, 1)] if self.speculative else shapes,
                                  mtp_batches=[1] if self.speculative else (),
                                  mtp_prefill=PREFILL_CHUNKS if self.speculative else (), q8=q8,
                                  dflash2=dflash2 if self.speculative else None)
        self.rt.mtp_verify = VERIFY[verify]
        self.rt.mtp_adaptive = bool(adaptive)
        self.rt.mtp_lookup = bool(lookup)
        self.cfg = self.rt.cfg
        self.tokenizer = _JsonTokenizer.from_directory(checkpoint_dir, chat_template=chat_template)
        self.eog = set(self.cfg.eos_token_ids)
        self._template = None

    def stream(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
               stop_ids: Optional[Sequence[int]] = None, temperature: float = 0.0, top_p: float = 1.0,
               top_k: int = 20, seed: int = 0) -> Iterator[List[int]]:
        """Continuation of ``prompt_ids`` as it is produced (blocks of up to eight tokens; with ``speculative`` the
        accepted tokens of each draft / verify step). Stops after an end-of-generation token (included);
        ``stop_ids=()`` disables stopping. ``temperature=0`` (the default) is greedy; otherwise top-k (1..32) /
        top-p sampling with Gumbel noise drawn on the GPU from ``seed`` and the token position (speculative
        decoding is greedy only)."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(t < 0 or t >= self.cfg.unpadded_vocab_size for t in prompt_ids):
            raise ValueError("prompt token id outside the checkpoint vocabulary")
        n = len(prompt_ids)
        margin = 2 * BLOCK if self.speculative else BLOCK
        if n + max_new_tokens + margin > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens (+{margin}) exceeds max_seq {self.rt.max_seq}")
        stop = set(self.eog if stop_ids is None else stop_ids)
        sampled = temperature > 0
        if sampled and self.speculative:
            raise ValueError("speculative decoding is greedy (temperature 0)")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        sampling = dict(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed) if sampled else {}
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
        if sampled:
            first = self.rt.sample_logits(self.rt.rows_logits(0), n - 1, **sampling)
        yield [first]
        if first in stop or max_new_tokens == 1:
            return
        token, position, produced = first, n, 1
        eos = sorted(stop)[:4]
        while produced < max_new_tokens:
            steps = min(BLOCK, max_new_tokens - produced)
            block = self.rt.decode(token, position, steps, program="sampled" if sampled else "greedy", eos=eos,
                                   **sampling)
            produced += len(block)
            position += len(block)
            token = block[-1]
            hit = next((i for i, t in enumerate(block) if t in stop), None)
            if hit is not None:
                yield block[:hit + 1]
                return
            yield block

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None, **sampling) -> List[int]:
        out: List[int] = []
        for block in self.stream(prompt_ids, max_new_tokens, stop_ids, **sampling):
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

            def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                """transformers' filter: the template sorts tool declarations and arguments, compact separators."""
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                                  sort_keys=sort_keys)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[GenerationBlock, "jinja2.ext.loopcontrols"])
            env.filters["tojson"] = tojson
            env.globals["raise_exception"] = raise_exception
            self._template = env.from_string(self.tokenizer.chat_template)
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                     **template_kwargs)

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False, sampling: Optional[Dict] = None, **template_kwargs) -> str:
        """``sampling``: keyword arguments of ``stream`` (temperature, top_p, top_k, seed); greedy by default.
        Other keyword arguments go to the chat template (e.g. ``reasoning_effort``)."""
        prompt = self.apply_chat_template(messages, tools=tools, **template_kwargs)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens, **(sampling or {}))
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)
