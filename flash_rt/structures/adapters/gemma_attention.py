"""Attention adapter for Gemma-family denoise hosts (pi05 / pi_gemma).

Where the attention math runs is host-specific. In this family the
transformer's own forward calls ``modeling_gemma.eager_attention_forward``
directly, bypassing the config/interface dispatch entirely, so the seam
is that function, not a module. This adapter locates it by capturing one
denoise pass, binds an :mod:`..impls.attention_core` per layer from the
captured shapes and masks, and installs a function-level patch that
routes the fixed denoise shape to the packed-KV kernel while leaving
prefill and any other shape on the host path.

Registering this adapter lets ``autobuild`` pick up the attention_core
structure for this host family with no per-host scaffolding at the call
site — the host still just calls ``auto_swaps``.
"""

from __future__ import annotations

from ..impls.attention_core import bind_attention_core


class GemmaAttentionAdapter:
    """Recognise a Gemma-family denoise host and wire its fa2 seam."""

    __name__ = "gemma_attention"

    def __call__(self, model, forward, *, prefix_cadence: bool = False):
        """Wire the fa2 seam, or refuse when nobody will refresh its prefix.

        This structure keeps the attention prefix — the vision and language
        tokens — in a packed region and only rewrites the suffix per step.
        That is correct within one observation, and it is what the bind-time
        check proves: the prefix does not move across the denoise loop.

        It is *not* correct across observations. A new image produces a new
        prefix, and the packed region still holds the one calibration
        captured, so the model attends to the wrong frame. Measured on
        Pi0.5 over twelve unseen frames: output match 0.9957 with this seam
        against 0.9997 without it, and max deviation 0.113 against 0.035.

        The refresh exists (``bind_attention_core`` returns it) and the
        tick pipeline drives it at the observation cadence. A caller that
        cannot must not get this seam, so it is offered only when
        ``prefix_cadence`` says the refresh will be called.
        """
        try:
            import transformers.models.gemma.modeling_gemma as mg
        except ImportError:
            return None
        orig = mg.eager_attention_forward

        recs = {"q": None, "masks": [], "keys": [], "values": []}

        def record(module, query, key, value, attention_mask, **kw):
            if query.shape[2] < 128:      # denoise (short) vs prefill
                recs["q"] = query.detach()
                recs["masks"].append(
                    attention_mask.detach()
                    if attention_mask is not None else None)
                recs["keys"].append(key.detach().clone())
                recs["values"].append(value.detach().clone())
            return orig(module, query, key, value, attention_mask, **kw)

        mg.eager_attention_forward = record
        try:
            with __import__("torch").no_grad():
                forward()
        finally:
            mg.eager_attention_forward = orig
        if recs["q"] is None:
            return None      # host never called this seam — not our family
        if not prefix_cadence:
            # after the family check, not before it: a refusal recorded
            # against a host that never had this seam is misinformation
            raise ValueError(
                "attention_core: this seam holds the attention prefix "
                "across calls and is only correct while someone refreshes "
                "it when the observation changes. Pass prefix_cadence=True "
                "and call plan.updates on every new observation, or leave "
                "it unbound — unbound measured 0.9997 output match on "
                "Pi0.5 unseen frames against 0.9957 bound-and-stale")

        n_layers = _infer_layers(model)
        if n_layers == 0 or len(recs["keys"]) % n_layers != 0:
            return None
        steps = len(recs["keys"]) // n_layers
        captures = [{
            "q": recs["q"],
            "keys": [recs["keys"][i + s * n_layers] for s in range(steps)],
            "values": [recs["values"][i + s * n_layers]
                       for s in range(steps)],
            "mask": recs["masks"][i],
        } for i in range(n_layers)]

        bound = bind_attention_core(captures)
        if bound is None:
            return None      # head_dim unsupported → host keeps its path
        cores, prefix_update = bound
        seq_q = recs["q"].shape[2]
        expert, expert_path = _expert_layers_at(model)
        for i, layer in enumerate(expert):
            layer.self_attn._fa2_core = cores[i]

        # no isolated speed bench here: benching this kernel against a
        # standalone compiled attention says it loses, while the same
        # swap measured inside the assembled graph wins by 0.76ms
        # (10x the intra-process variance) and improves parity. An
        # isolated probe cannot see what the seam actually replaces;
        # the composed net-win gate is the one that can.
        def fa2_fn(module, query, key, value, attention_mask, **kw):
            # no Python-visible side effects in here: a counter or any
            # host-side bookkeeping forces dynamo to break the graph at
            # every attention call, which fragments the surrounding
            # compiled region and pushes its CPU-side ops onto the
            # capture stream
            if query.shape[2] != seq_q or not hasattr(module, "_fa2_core"):
                return orig(module, query, key, value, attention_mask, **kw)
            return module._fa2_core(query, key, value,
                                    scale=kw.get("scaling")), None

        mg.eager_attention_forward = fa2_fn
        self._seq_q = seq_q

        def enable() -> None:
            mg.eager_attention_forward = fa2_fn

        def disable() -> None:
            """Route attention back to the host without unbinding.

            The gate needs a baseline arm that is the host, and this seam
            is the one that cannot be turned off by restoring a module:
            it is a patched function, so it stays live through
            ``detach()`` of every swap around it. Without a toggle the
            "off" arm would still be running this kernel and the net-win
            measurement would be comparing the attachment against itself.
            The bound cores stay where they are — the patch is what
            routes to them, and rebuilding them per arm would recapture.
            """
            if mg.eager_attention_forward is fa2_fn:
                mg.eager_attention_forward = orig

        def revert() -> None:
            """Undo everything this adapter did to the host and to
            ``transformers``.

            The patch above is a module-level rebinding, so without this
            it outlives the attachment: ``handle.detach()`` would restore
            every swapped module and leave the attention seam patched, and
            the promise that detaching gives back the original model would
            be false for the one seam that is not a module. The core
            attributes go too — a core still hanging off the host would
            keep the routed path reachable and keep reporting itself as
            live.
            """
            if mg.eager_attention_forward is fa2_fn:
                mg.eager_attention_forward = orig
            for layer in expert or ():
                if getattr(layer.self_attn, "_fa2_core", None) is not None:
                    del layer.self_attn._fa2_core

        # the swap map is empty (the seam is a function, not a module);
        # the patch and the per-layer core buffers are the swap. They are
        # handed back as ``observed`` so the cores still appear in the
        # attachment's ledger: a seam that cannot be swapped at a path can
        # still be counted, and "the shape guard sent every call to the
        # host" has to be visible somewhere. The shape check inside
        # ``fa2_fn`` deliberately keeps no counter of its own (that is a
        # graph break per attention call); it shows up instead as a core
        # whose own call count stayed at zero.
        # note: no extra host forward is run to self-verify — replaying
        # the host mutates its state (cache growth, guard shapes) and
        # that changes what the stage then captures. The recording pass
        # above already proves the seam is live in this host.
        observed = {f"{expert_path}.{i}.self_attn::fa2_core": core
                    for i, core in enumerate(cores)}
        # the refresh goes back to the caller. Discarding it was the whole
        # defect: the prefix then had no way to follow the observation.
        return {}, prefix_update, {"revert": [revert], "observed": observed,
                                   "toggle": (enable, disable)}

    def sublayer(self, layer):
        """An attention sublayer for one host block, or ``None``.

        Offered to the ``decoder_block`` structure, which owns the
        boundary where the projections' layout meets the kernel's. This
        family is half-split rotary, and the core bound above is what
        the function patch would otherwise route to — so the sublayer
        replaces a routed call, not a host path, and returning ``None``
        simply leaves that routing in place.
        """
        from ..impls.decoder_block import bind_attn_sublayer

        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        return bind_attn_sublayer(attn, getattr(attn, "_fa2_core", None))


def _infer_layers(model) -> int:
    layers = _expert_layers(model)
    return len(layers) if layers is not None else 0


def _expert_layers(model):
    """Find the denoise decoder layers under either the model or a
    policy wrapper — callers hand us whichever root they hold."""
    return _expert_layers_at(model)[0]


def _expert_layers_at(model) -> tuple[object, str]:
    """The denoise decoder layers and the dotted path they were found at.

    The path matters to the receipt: this adapter's seam is a patched
    function rather than a swapped module, so the only way it can be named
    in a report is by the layers it attached its cores to.
    """
    for path in ("paligemma_with_expert.gemma_expert.model.layers",
                 "model.paligemma_with_expert.gemma_expert.model.layers"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        else:
            return node, path
    return None, ""
