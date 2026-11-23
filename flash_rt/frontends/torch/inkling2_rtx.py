"""FlashRT -- Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, ``inkling2_loop``) on RTX SM120.

The text decoder visits 48 layers per token (24 stored layers; layers 6..17 run three passes with message gates
between passes) with block attention residuals, sliding-window and global attention with relative position bias,
short convolutions and a 128-expert MoE. The released checkpoint is used at its own precision: BF16 attention and
LM head, NVFP4 experts in layers 0-19 and MXFP8 in 20-23, run by the ``flash_rt_inkling2`` kernels (block-scaled
FP4 / FP8 tensor-core instructions of SM120) from recorded programs replayed as CUDA graphs:

* decode: one token per graph replay, the greedy token chosen on the device;
* prefill: tensor-core chunks of up to 512 rows (``prefill="fast"``) or per-row chunks of eight rows that are
  bit-identical to decoding the prompt token by token (``prefill="exact"``);
* ``speculative=True``: the checkpoint's DSpark drafter (``mtp.safetensors``) proposes ``draft_width - 1`` tokens
  per step and the target verifies them in one multi-row forward whose rows run the decode step's arithmetic, so
  the output is exactly the greedy decode output.

    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    fe = Inkling2TorchFrontendRtx("<path-to-Inkling-2-Mini>", max_seq=8192, speculative=True)
    text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

Text only (the vision tower is not loaded), greedy, one sequence. Constructed directly rather than through
``load_model`` (a text decoder has no VLA ``predict`` surface). See ``docs/inkling2_rtx_usage.md``.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import torch

from flash_rt.frontends.torch._inkling2_rtx_runtime import Inkling2RtxRuntime, read_config, visit_count
from flash_rt.models.flint.tokenizer import FlintTokenizer as _JsonTokenizer   # model-independent tokenizer.json

__all__ = ["Inkling2TorchFrontendRtx", "compile_chat_template"]

PREFILL_MODES = ("fast", "exact")


def compile_chat_template(source: str):
    """The checkpoint's Jinja chat template in transformers' environment: ``{% generation %}`` blocks render their
    body, ``loopcontrols``, ``raise_exception`` and ``tojson`` with transformers' arguments and defaults."""
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
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[GenerationBlock, "jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


class Inkling2TorchFrontendRtx:
    """Greedy generation over the native Inkling 2 runtime (one sequence).

    ``max_seq``: prompt plus generated tokens (KV capacity). ``speculative``: load the DSpark drafter and use it by
    default (``stream(..., speculative=False)`` still decodes plainly). ``prefill``: ``"fast"`` (tensor-core
    chunks) or ``"exact"`` (bit-identical to token-by-token decoding). ``check_block`` decode steps or
    ``check_rounds`` speculative steps run between host reads of the generated tokens (a stop token is found at
    most that late)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, speculative: bool = False, draft_width: int = 4,
                 prefill: str = "fast", check_block: int = 16, check_rounds: int = 4, device: int = 0,
                 chat_template: Optional[str] = None):
        if prefill not in PREFILL_MODES:
            raise ValueError(f"prefill must be one of {PREFILL_MODES}")
        if check_block < 1 or check_rounds < 1:
            raise ValueError("check_block and check_rounds must be positive")
        if speculative:   # the drafter's configuration, before any weight is loaded
            from flash_rt.frontends.torch._inkling2_rtx_mtp import check_mtp_config
            raw, text = read_config(checkpoint_dir)
            check_mtp_config(raw, visit_count(text))
        self.rt = Inkling2RtxRuntime(checkpoint_dir, max_seq=max_seq, device=device)
        self.cfg = self.rt.c
        self.vocab = self.rt.vocab
        self.prefill_mode = prefill
        self.check_block, self.check_rounds = int(check_block), int(check_rounds)
        self.drafter = None
        if speculative:
            from flash_rt.frontends.torch._inkling2_rtx_mtp import Inkling2RtxDrafter
            self.drafter = Inkling2RtxDrafter(self.rt, width=draft_width)
        self.speculative = bool(speculative)
        self.tokenizer = _JsonTokenizer.from_directory(checkpoint_dir, chat_template=chat_template)
        eos = self.rt.raw.get("eos_token_id")
        self.eog = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        self._template = None

    def warmup(self, prompt_len: int = 512):
        """Record and capture the programs a prompt of ``prompt_len`` tokens and the decode loop use, so the first
        request does not pay for graph capture."""
        ids = [0] * max(1, int(prompt_len))
        for _ in self.stream(ids, max_new_tokens=2, stop_ids=()):
            pass
        if self.drafter is not None:
            for _ in self.stream(ids, max_new_tokens=2, stop_ids=(), speculative=False):
                pass

    def _margin(self, speculative: bool) -> int:
        """Positions beyond the requested tokens a request may write: speculative steps run past the last read
        (and verify ``width`` rows ahead); plain decode stops at the requested count."""
        if speculative:
            return (self.check_rounds + 1) * self.drafter.width + 8
        return 1

    def stream(self, prompt_ids: Sequence[int], max_new_tokens: int = 256, stop_ids: Optional[Sequence[int]] = None,
               speculative: Optional[bool] = None, prefill: Optional[str] = None) -> Iterator[List[int]]:
        """Greedy continuation of ``prompt_ids`` as it is produced: the first token after prefill (with
        ``speculative``, together with the first step's tokens), then blocks. Stops after an end-of-generation
        token (included); ``stop_ids=()`` disables stopping. Each call starts a new sequence."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(t < 0 or t >= self.vocab for t in prompt_ids):
            raise ValueError("prompt token id outside the checkpoint vocabulary")
        spec = self.speculative if speculative is None else bool(speculative)
        if spec and self.drafter is None:
            raise ValueError("speculative decoding needs the drafter (construct with speculative=True)")
        mode = self.prefill_mode if prefill is None else prefill
        if mode not in PREFILL_MODES:
            raise ValueError(f"prefill must be one of {PREFILL_MODES}")
        n = len(prompt_ids)
        margin = self._margin(spec)
        if n + max_new_tokens + margin > self.rt.max_seq:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens (+{margin}) exceeds max_seq {self.rt.max_seq}")
        stop = self.eog if stop_ids is None else set(stop_ids)
        rt = self.rt
        rt.reset()
        ids = [int(t) for t in prompt_ids]

        def cut(block):
            hit = next((i for i, t in enumerate(block) if t in stop), None)
            return (block, False) if hit is None else (block[:hit + 1], True)

        if spec:
            first = self.drafter.prefill(ids, exact=mode == "exact")
            for block in self.drafter.steps(first, max_new_tokens, check_every=self.check_rounds):
                block, hit = cut(block)
                yield block
                if hit:
                    return
            return
        first = rt.prefill(ids, exact=mode == "exact")
        block, hit = cut([first])
        yield block
        if hit or max_new_tokens == 1:
            return
        produced, position = 1, n          # history[p]: the token fed at position p
        while produced < max_new_tokens:
            steps = min(self.check_block, max_new_tokens - produced)
            for _ in range(steps):
                rt.run("decode")
            block = rt.history[position + 1: position + 1 + steps].tolist()
            produced += steps
            position += steps
            block, hit = cut(block)
            yield block
            if hit:
                return

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int = 256,
                 stop_ids: Optional[Sequence[int]] = None, **kw) -> List[int]:
        out: List[int] = []
        for block in self.stream(prompt_ids, max_new_tokens, stop_ids, **kw):
            out.extend(block)
        return out

    def apply_chat_template(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                            add_generation_prompt: bool = True, **template_kwargs) -> str:
        """The checkpoint's chat template (``chat_template.jinja``; it takes e.g. ``reasoning_effort``)."""
        if self.tokenizer.chat_template is None:
            raise ValueError("no chat template (chat_template.jinja)")
        if self._template is None:
            self._template = compile_chat_template(self.tokenizer.chat_template)
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                     **template_kwargs)

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_new_tokens: int = 1024,
             skip_special: bool = False, **template_kwargs) -> str:
        """Greedy reply; keyword arguments go to the chat template (e.g. ``reasoning_effort``)."""
        prompt = self.apply_chat_template(messages, tools=tools, **template_kwargs)
        ids = self.generate(self.tokenizer.encode(prompt), max_new_tokens=max_new_tokens)
        if ids and ids[-1] in self.eog:
            ids = ids[:-1]
        return self.tokenizer.decode(ids, skip_special=skip_special)

    def close(self):
        """Release the CUDA graphs, buffers and weights."""
        self.rt.graphs.clear()
        self.drafter = None
        self.rt = None
        torch.cuda.empty_cache()
