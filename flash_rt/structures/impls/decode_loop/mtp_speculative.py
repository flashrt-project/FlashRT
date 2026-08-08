"""MTP speculative decode — the decode_loop family's second member.

Checkpoints in this family ship a one-layer DeepSeek-style draft head
(``mtp.safetensors``) that transformers hosts never use. This member
loads it, assembles the draft from the host's own module classes, and
runs the draft/verify loop around the whole-step form. Greedy spec
decode is exact by construction: the verify pass recomputes every
draft token with the main model, so the accepted stream is identical
to plain greedy decode — the gate checks token identity, not a band.

The draft head is carried in BF16 (its FP8 blocks are dequantised at
load); its attention layer gets one extra slot in the static cache.
The gated-delta states cannot roll back through a rejected suffix, so
the loop snapshots them before each verify and re-advances the
accepted prefix from the snapshot when a draft is cut short.
"""

from __future__ import annotations

import torch


def _load_mtp_tensors(ckpt_dir):
    """The draft's tensors, ``mtp.`` prefix stripped.

    Two shipping forms: a sidecar ``mtp.safetensors``, or ``mtp.*`` keys
    inside the main sharded checkpoint (the MoE hosts ship this way) —
    the index says which shards carry them.
    """
    import json
    import pathlib

    from safetensors import safe_open

    d = pathlib.Path(str(ckpt_dir))
    side = d / "mtp.safetensors"
    if side.is_file():
        f = safe_open(str(side), "pt")
        return {k[len("mtp."):]: f.get_tensor(k) for k in f.keys()}
    idx_path = d / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise ValueError(
            f"refused: {d} carries neither mtp.safetensors nor a "
            "sharded index with mtp.* keys")
    wmap = json.loads(idx_path.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in wmap.items():
        if key.startswith("mtp."):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise ValueError(
            f"refused: this checkpoint's index carries no mtp.* keys "
            "(the host ships no draft head)")
    t = {}
    for shard, keys in by_shard.items():
        f = safe_open(str(d / shard), "pt")
        for k in keys:
            t[k[len("mtp."):]] = f.get_tensor(k)
    return t


def _dequant_block_fp8(w, scale_inv):
    n, k = w.shape
    bn, bk = n // scale_inv.shape[0], k // scale_inv.shape[1]
    wf = w.float().view(scale_inv.shape[0], bn, scale_inv.shape[1], bk)
    wf = wf * scale_inv.float().view(scale_inv.shape[0], 1,
                                     scale_inv.shape[1], 1)
    return wf.view(n, k).to(torch.bfloat16)


#: the draft's precision axes and their measured arms. The draft answers
#: to acceptance length alone — greedy spec output is anchored by the
#: verify pass either way — so both axes may trade precision for cost,
#: and the defaults are the measured sweet spot on the record: a private
#: W8 view of the shared head (the model's own head must not change —
#: step and verify share its numeric family) and the BF16 expert bank
#: (the FP4 bank measured AL-equal; BF16 is the conservative default).
DRAFT_FORMATS = {
    "head": ("w8a16_static", "host"),
    "experts": ("bf16", "nvfp4_dynamic"),
}


def check_draft_formats(head_format: str, experts_format: str) -> None:
    if head_format not in DRAFT_FORMATS["head"]:
        raise ValueError(
            f"refused: unknown draft head format {head_format!r}; "
            f"measured arms: {', '.join(DRAFT_FORMATS['head'])}")
    if experts_format not in DRAFT_FORMATS["experts"]:
        raise ValueError(
            f"refused: unknown draft experts format {experts_format!r}; "
            f"measured arms: {', '.join(DRAFT_FORMATS['experts'])}")


class _GatherExpertsBf16(torch.nn.Module):
    """BF16 expert bank behind the host contract, sync-free.

    Routed slots gather device-side and run as one batched matmul per
    projection — the same fixed-shape, host-silent step the packed form
    takes, at the bank's own precision. Sized for a draft layer: one
    layer's bank, a few MB per routed slot.
    """

    def __init__(self, gate_up, down, act_fn):
        super().__init__()
        self.register_buffer("_gu", gate_up.contiguous())
        self.register_buffer("_dn", down.contiguous())
        self._act = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        t, h = hidden_states.shape
        k = top_k_index.shape[1]
        flat = top_k_index.reshape(-1)
        gu = self._gu.index_select(0, flat)          # [T*k, 2I, H]
        dn = self._dn.index_select(0, flat)          # [T*k, H, I]
        x = hidden_states.unsqueeze(1).expand(t, k, h).reshape(t * k, 1, h)
        y = torch.bmm(x, gu.transpose(1, 2))         # [T*k, 1, 2I]
        gate, up = y.chunk(2, dim=-1)
        inter = self._act(gate) * up
        d = torch.bmm(inter, dn.transpose(1, 2))     # [T*k, 1, H]
        out = (d.view(t, k, h).float()
               * top_k_weights[..., None].float()).sum(dim=1)
        return out.to(hidden_states.dtype)


class MtpDraftHead(torch.nn.Module):
    """fc + one host-class decoder layer + norms; embed/head shared."""

    def __init__(self, model, lm, ckpt_dir, layer_slot: int, *,
                 head_format: str = "w8a16_static",
                 experts_format: str = "bf16"):
        super().__init__()
        check_draft_formats(head_format, experts_format)

        cfg = getattr(model.config, "text_config", model.config)
        full_idx = next(i for i, t in enumerate(cfg.layer_types)
                        if t == "full_attention")
        layer_cls = type(lm.layers[full_idx])
        norm_cls = type(lm.norm)
        hidden = int(cfg.hidden_size)
        dev = lm.norm.weight.device

        t = _load_mtp_tensors(ckpt_dir)

        # assemble on CPU, move once — the draft loads while the host
        # still has headroom, and never doubles on the device
        self.layer = layer_cls(cfg, full_idx).to(torch.bfloat16)
        pre = "layers.0."
        with torch.no_grad():
            for name, p in self.layer.named_parameters():
                w = t.get(pre + name)
                if w is None:
                    continue
                s = t.get(pre + name + "_scale_inv")
                w = (_dequant_block_fp8(w, s) if s is not None
                     else w.to(torch.bfloat16))
                p.copy_(w)
            self.layer = self.layer.to(dev)
            # a draft layer whose MLP is an expert bank must run a
            # gather-then-fixed-shape form: the draft chain is captured,
            # and the host bank's routed loop syncs the host. The bank
            # stays BF16 — the scheme's draft default — because the
            # draft's whole value is its acceptance length, and BF16
            # gathers capture just as well as packed ones (the draft is
            # one layer; a routed slot is a handful of MB).
            mlp = getattr(self.layer, "mlp", None)
            bank = getattr(mlp, "experts", None) if mlp is not None else None
            self.formats = {"head": "host", "experts": None}
            if bank is not None and hasattr(bank, "gate_up_proj") \
                    and torch.is_tensor(bank.gate_up_proj) \
                    and bank.gate_up_proj.dim() == 3:
                if experts_format == "nvfp4_dynamic":
                    from ..moe_experts.nvfp4_dynamic import (
                        bind_experts_seam)
                    mlp.experts, self.experts_conversion = \
                        bind_experts_seam(
                            {"gate_up_proj": bank.gate_up_proj.detach(),
                             "down_proj": bank.down_proj.detach()},
                            bank.act_fn)
                else:
                    mlp.experts = _GatherExpertsBf16(
                        bank.gate_up_proj.detach().to(dev),
                        bank.down_proj.detach().to(dev), bank.act_fn)
                self.formats["experts"] = experts_format
            self.fc = torch.nn.Linear(2 * hidden, hidden, bias=False,
                                      device=dev, dtype=torch.bfloat16)
            self.fc.weight.copy_(t["fc.weight"].to(torch.bfloat16))
            self.norm_h = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_h.weight.copy_(
                t["pre_fc_norm_hidden.weight"].to(torch.bfloat16))
            self.norm_e = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_e.weight.copy_(
                t["pre_fc_norm_embedding.weight"].to(torch.bfloat16))
            self.norm_out = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_out.weight.copy_(t["norm.weight"].to(torch.bfloat16))
        self.slot = int(layer_slot)
        self._embed = lm.embed_tokens
        self._rotary = lm.rotary_emb
        # the draft carries its own W8 view of the shared head: a draft
        # step pays the full-vocab projection every token, its precision
        # is judged by acceptance length alone, and the model's own head
        # must NOT change — the step and the verify pass share that one,
        # and splitting their numeric family at the logits is what a
        # W8-swapped model head was measured to do
        self._head = model.lm_head
        if head_format == "w8a16_static" \
                and isinstance(model.lm_head, torch.nn.Linear):
            try:
                from ..linear_proj import w8a16_static
                self._head = w8a16_static.bind_proj_seam(
                    {"w": model.lm_head.weight.detach()},
                    original=model.lm_head)
                self.formats["head"] = "w8a16_static"
            except (ValueError, RuntimeError):
                self._head = model.lm_head
        self.eval()

    @torch.no_grad()
    def forward(self, prev_h, tok_ids, pos_t, cache, mask_row):
        """(logits, h_out) at ``pos_t``; writes the draft's KV slot."""
        e = self._embed(tok_ids)
        h = self.fc(torch.cat([self.norm_e(e), self.norm_h(prev_h)],
                              dim=-1))
        cache._cp = pos_t
        h = self.layer(h, position_embeddings=self._rotary(
                           h, pos_t.view(1, -1)),
                       attention_mask=mask_row,
                       past_key_values=_SlotView(cache, self.slot),
                       use_cache=True, cache_position=pos_t)
        h = self.norm_out(h)
        return self._head(h), h


class _SlotView:
    """Route the draft layer's cache traffic to its private slot."""

    def __init__(self, cache, slot):
        self._c = cache
        self._s = slot

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return self._c.update(k, v, self._s, cache_kwargs)

    def __getattr__(self, name):
        return getattr(self._c, name)
