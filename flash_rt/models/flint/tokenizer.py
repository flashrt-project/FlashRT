"""Flint tokenizer and chat template, built from the GGUF metadata.

The GGUF carries the byte-level BPE vocabulary, merges, token types and the
Jinja chat template. The tokenizer is assembled with the ``tokenizers``
library (optional dependency, imported lazily): the "flint" pre-tokenizer is a
cascade of three regex splits followed by the GPT-2 byte-to-unicode mapping;
merges are applied without the whole-word shortcut. Control and user-defined
tokens are matched verbatim in text, so rendered chat prompts tokenize to the
same ids as the reference executor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from flash_rt.models.flint.gguf import GGUFFile

# Pre-tokenizer cascade for tokenizer.ggml.pre == "flint", applied in order.
_FLINT_SPLITS = (
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)

_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4


class FlintTokenizer:
    def __init__(self, gguf: GGUFFile):
        try:
            from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("the Flint tokenizer needs the `tokenizers` package") from e
        if gguf.require("tokenizer.ggml.model") != "gpt2" or gguf.require("tokenizer.ggml.pre") != "flint":
            raise ValueError("expected a byte-level BPE tokenizer with the 'flint' pre-tokenizer")
        tokens: List[str] = list(gguf.require("tokenizer.ggml.tokens"))
        merges = [tuple(m.split(" ", 1)) for m in gguf.require("tokenizer.ggml.merges")]
        types = [int(t) for t in gguf.require("tokenizer.ggml.token_type")]
        vocab = {t: i for i, t in enumerate(tokens)}
        tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False,
                                   ignore_merges=False))
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex(p), behavior="isolated") for p in _FLINT_SPLITS]
            + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()
        specials = [AddedToken(tokens[i], special=True, normalized=False)
                    for i, t in enumerate(types) if t in (_TOKEN_TYPE_CONTROL, _TOKEN_TYPE_USER_DEFINED)]
        tok.add_special_tokens(specials)
        self._tok = tok
        self.vocab_size = len(tokens)
        self.bos_id = int(gguf.require("tokenizer.ggml.bos_token_id"))
        self.eos_id = int(gguf.require("tokenizer.ggml.eos_token_id"))
        self.chat_template: Optional[str] = gguf.get("tokenizer.chat_template")
        self._tmpl = None

    def encode(self, text: str) -> List[int]:
        """Tokenize text; special tokens written in the text are recognized (no BOS is added)."""
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def apply_chat_template(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                            add_generation_prompt: bool = True) -> str:
        if self.chat_template is None:
            raise ValueError("GGUF has no tokenizer.chat_template")
        if self._tmpl is None:
            import json
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)  # HF chat-template convention
            env.filters["tojson"] = lambda v, ensure_ascii=True, **kw: json.dumps(v, ensure_ascii=ensure_ascii)
            self._tmpl = env.from_string(self.chat_template)
        return self._tmpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)
