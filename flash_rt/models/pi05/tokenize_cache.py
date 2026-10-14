"""Cached chat tokenization for the served DM05 prompt.

``ChatTokenization`` renders the Jinja chat template and tokenizes the whole ~570-token
prompt on the host for every request (~2.1 ms), although the token sequence only
depends on the prompt, the metadata and -- when ``add_state`` is on -- the discretized
state text. The served LIBERO config has ``add_state=False``, so the ids are a pure
function of the prompt and are fully cached. The state is its own text item, bracketed by the
image tokens before it and ``<end_of_turn>`` after it, so its tokens do not merge
with anything around it. This wrapper therefore tokenizes

    [prefix: template, meta, prompt, image placeholders]  -- cached per prompt/meta
    [state text]                                           -- per request
    [suffix: end of turn + generation prompt]              -- cached

and concatenates the ids. The first request for each key is also run through the
original transform and must match it token for token (and the token types), otherwise
the key falls back to the original path permanently; ``verify_every=True`` checks
every request.
"""
from __future__ import annotations

import torch


class CachedChatTokenization:
    def __init__(self, inner, verify_every: bool = False):
        self.inner = inner
        self.tok = inner.tokenizer
        self.proc = inner.processor
        self.verify_every = verify_every
        self.cache: dict = {}
        self.fallback: set = set()
        self.image_token_id = int(self.proc.tokenizer.convert_tokens_to_ids(self.proc.image_token))
        self.stats = {"hits": 0, "misses": 0, "fallbacks": 0, "verified": 0}

    def _state_text(self, data) -> str:
        if not self.inner.add_state:
            return ""                       # served config: no state text in the prompt
        return "States: " + " ".join(str(b) for b in self.inner.action_to_bin_tokens(data["state"], self.inner.n_bins))

    def _key(self, data):
        meta = data.get("meta_data", {}) or {}
        return (data.get("prompt"), tuple(sorted((k, str(v)) for k, v in meta.items())), len(data["images"]))

    def _eligible(self) -> bool:
        return not self.inner.is_history

    def __call__(self, data):
        if not self._eligible():
            return self.inner(data)
        key = self._key(data)
        if key in self.fallback:
            self.stats["fallbacks"] += 1
            return self.inner(data)
        stt = self._state_text(data)
        st_ids = self.tok(stt, add_special_tokens=False)["input_ids"] if stt else []
        entry = self.cache.get(key)
        if entry is None or self.verify_every:
            ref = self.inner(data)
            ids = ref["input_ids"][0].tolist()
            n_st = len(st_ids)
            if entry is None:
                # locate the state tokens: the suffix is whatever follows them
                pos = next((i for i in range(len(ids) - n_st, -1, -1) if ids[i:i + n_st] == st_ids), -1)
                if pos < 0:
                    self.fallback.add(key); self.stats["fallbacks"] += 1
                    return ref
                entry = (ids[:pos], ids[pos + n_st:])
                self.cache[key] = entry
                self.stats["misses"] += 1
            pv = self.proc.image_processor(images=data["images"], return_tensors="pt",
                                           **self.inner._processor_device_kwargs())["pixel_values"]
            out = self._assemble(entry, st_ids, pv, ref["input_ids"].device)
            same = (torch.equal(out["input_ids"], ref["input_ids"])
                    and torch.equal(out["token_type_ids"].to(ref["token_type_ids"].dtype), ref["token_type_ids"])
                    and torch.equal(out["attention_mask"], ref["attention_mask"])
                    and torch.equal(out["pixel_values"].to(ref["pixel_values"].device), ref["pixel_values"]))
            self.stats["verified"] += 1
            if not same:
                self.fallback.add(key); self.cache.pop(key, None)
                return ref
            return ref
        self.stats["hits"] += 1
        pv = self.proc.image_processor(images=data["images"], return_tensors="pt",
                                       **self.inner._processor_device_kwargs())["pixel_values"]
        return self._assemble(entry, st_ids, pv, pv.device)

    def _assemble(self, entry, st_ids, pixel_values, device):
        prefix, suffix = entry
        ids = torch.tensor([prefix + st_ids + suffix], dtype=torch.long, device=device)
        tt = (ids == self.image_token_id).to(torch.long)
        return {"action": None, "action_mask": None, "input_ids": ids,
                "attention_mask": torch.ones_like(ids), "pixel_values": pixel_values,
                "token_type_ids": tt, "history_pixel_values": None, "history_mask": None}
