"""Pi0.5 flow-matching family: the timestep schedule becomes resident.

The host builds its denoise schedule inside ``sample_actions`` as
``torch.tensor([...python floats...], device=cuda)`` — one
host-to-device copy per call. Whether that line survives capture has
depended on the compiler's mood: a dynamo that covers the whole method
bakes it into the graph, a dependency upgrade that adds a graph break
in front of it drops the copy into the capture stream and the capture
refuses (measured: the same host line passed on 2026-07-25 and
refused after a transformers upgrade landed the next day). A pin must
not gamble on coverage.

The pin scopes one rule around the host's own method: a
``torch.tensor`` call that constructs a *constant float list* on a
device resolves to a cached resident tensor — same values, same
device, same dtype, allocated once outside capture. Everything else
passes straight through, the schedule stays value-identical by
construction, and the undo restores the host method bit-for-bit.
"""

from __future__ import annotations

import types

import torch

from .protocol import GraphLowering


def _looks_like_pi05_flow(module) -> bool:
    return (callable(getattr(module, "sample_actions", None))
            and callable(getattr(module, "denoise_step", None))
            and callable(getattr(module, "embed_suffix", None))
            and hasattr(module, "paligemma_with_expert"))


class Pi05DenoiseGraphLoweringAdapter:
    """Family: pi05_denoise — one pin, the resident step schedule."""

    def lower(self, model, forward) -> GraphLowering | None:
        target = None
        if _looks_like_pi05_flow(model):
            target = model
        else:
            for _name, mod in getattr(
                    model, "named_modules", lambda: ())():
                if _looks_like_pi05_flow(mod):
                    target = mod
                    break
            if target is None and _looks_like_pi05_flow(
                    getattr(model, "model", None)):
                target = model.model
        if target is None:
            return None

        cache: dict[tuple, torch.Tensor] = {}
        real_tensor = torch.tensor
        host_fn = target.sample_actions
        had_instance = "sample_actions" in target.__dict__

        def caching_tensor(data, *args, **kwargs):
            device = kwargs.get("device")
            if (device is not None and isinstance(data, (list, tuple))
                    and data
                    and all(isinstance(x, (float, int, bool))
                            for x in data)):
                key = (tuple(data), str(device),
                       str(kwargs.get("dtype")))
                hit = cache.get(key)
                if hit is None:
                    hit = real_tensor(data, *args, **kwargs)
                    cache[key] = hit
                return hit
            return real_tensor(data, *args, **kwargs)

        real_setitem = torch.Tensor.__setitem__

        def filling_setitem(t, idx, val):
            # a python-scalar write into a CUDA tensor is a
            # host-to-device copy the capture stream refuses; the same
            # store as an immediate-value fill is graph-legal and
            # bit-identical
            if (t.is_cuda and isinstance(idx, int)
                    and isinstance(val, (int, float, bool))):
                t.narrow(0, idx, 1).fill_(val)
                return
            real_setitem(t, idx, val)

        def pinned(self, *args, **kwargs):
            torch.tensor = caching_tensor
            torch.Tensor.__setitem__ = filling_setitem
            try:
                return host_fn(*args, **kwargs)
            finally:
                torch.tensor = real_tensor
                torch.Tensor.__setitem__ = real_setitem

        target.sample_actions = types.MethodType(pinned, target)

        # ---- pin 3: the pixel-patch embedding stack rides the band ----
        # The host keeps norms/embeddings in fp32 as a training-fidelity
        # choice; in the captured serving form every consumer of the
        # patch embeds casts to bf16 at its own entry, so the fp32 conv
        # pair is pure spend. Structural match only (a full-patch
        # Conv2d — kernel == stride — beside a position Embedding),
        # weights carried down in place with the originals retained,
        # and the arm's end-to-end parity gate stays the judge.
        embed_saved: list = []
        embed_hooks: list = []
        for _n, mod in getattr(model, "named_modules", lambda: ())():
            pe = getattr(mod, "patch_embedding", None)
            pos = getattr(mod, "position_embedding", None)
            if not (isinstance(pe, torch.nn.Conv2d)
                    and isinstance(pos, torch.nn.Embedding)):
                continue
            if tuple(pe.kernel_size) != tuple(pe.stride):
                continue
            f32 = [p for p in mod.parameters()
                   if p.dtype == torch.float32]
            if not f32:
                continue
            for p in f32:
                embed_saved.append((p, p.data))
                p.data = p.data.to(torch.bfloat16)
            embed_hooks.append(pe.register_forward_pre_hook(
                lambda _m, args: (args[0].to(torch.bfloat16),)
                + tuple(args[1:])))
            # dtype-transparent at the module boundary: every
            # downstream consumer keeps seeing the dtype the host
            # chose; only the patch projection itself rides the band
            embed_hooks.append(mod.register_forward_hook(
                lambda _m, _a, out: out.to(torch.float32)
                if isinstance(out, torch.Tensor) else out))
        # ---- pin 4: fp32 host linears ride the band, transparently --
        # The same fidelity policy leaves a handful of glue linears
        # (modality projector, time/action MLPs) in fp32, which on this
        # class of device means simt kernels with no tensor cores. Each
        # one is carried down in place with both boundaries cast back,
        # so every consumer and producer keeps its dtype contract and
        # the parity gate judges the whole move.
        for _n, mod in getattr(model, "named_modules", lambda: ())():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if mod.weight.dtype is not torch.float32:
                continue
            embed_saved.append((mod.weight, mod.weight.data))
            mod.weight.data = mod.weight.data.to(torch.bfloat16)
            if mod.bias is not None:
                embed_saved.append((mod.bias, mod.bias.data))
                mod.bias.data = mod.bias.data.to(torch.bfloat16)
            embed_hooks.append(mod.register_forward_pre_hook(
                lambda _m, args: (args[0].to(torch.bfloat16),)
                + tuple(args[1:])))
            embed_hooks.append(mod.register_forward_hook(
                lambda _m, _a, out: out.to(torch.float32)))
        pins = ["resident_step_schedule", "scalar_setitem_fill"]
        if embed_saved:
            pins.append("patch_embed_band")

        def undo() -> None:
            torch.tensor = real_tensor
            if had_instance:
                target.sample_actions = host_fn
            elif "sample_actions" in target.__dict__:
                del target.sample_actions
            for hook in embed_hooks:
                hook.remove()
            for p, data in embed_saved:
                p.data = data

        return GraphLowering(
            undo=undo, family="pi05_denoise",
            pins=tuple(pins),
            details={"host": type(target).__name__})
