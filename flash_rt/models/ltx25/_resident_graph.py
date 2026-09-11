"""Resident-transformer support: unlock CUDA-graph capture on a single GPU.

Upstream refuses ``CompilationConfig(capture=True)`` with the single-GPU
builder because every stage call rebuilds the transformer onto fresh GPU
storages and ``gpu_model`` disposes it afterwards -- captured graphs would
replay against freed weight pointers.

``ResidentSwapBuilder`` closes that gap:

    * the first ``build`` delegates to the inner builder, applies the FlashRT
      swap installers, neuters ``dispose`` on the built model, and caches it;
    * every later ``build`` returns the cached model -- weight storages never
      move, so captured graphs stay valid;
    * ``keeps_gpu_resident_weights`` reports True, which satisfies the
      upstream capture precondition.

Memory contract: the transformer (plus repacked FFN weights) stays resident
(~14GB for the 48-block model). The text encoder (~26GB bf16) is loaded and
freed inside one prompt encode, and the two cannot coexist on a single
consumer part. So residency is not permanent by construction: it is a lease
that ``release`` ends, and every path that needs the encoder again (a prompt
outside the embedding cache) ends it first and lets the next build take a
fresh one. That is why both classes below are written around release rather
than around caching alone -- a cache that has no way to step aside turns a
second prompt into an out-of-memory error.
"""

from __future__ import annotations

import gc
import logging

import torch

from flash_rt.models.ltx25._nvfp4_ffn_swap import (
    SwapInstallingBuilder, uninstall_nvfp4_ffn)

logger = logging.getLogger(__name__)


_x0_dispose_patched = False


def _patch_x0_dispose() -> None:
    """Make X0Model.dispose a no-op when it wraps a resident velocity model.

    The Disposable mixin's dispose walks the wrapper's own named_parameters,
    so neutering dispose on the inner model alone does not protect its
    storages -- a fresh X0Model wraps it on every stage build.
    """
    global _x0_dispose_patched
    if _x0_dispose_patched:
        return
    from ltx_core.model.transformer.model import X0Model

    original = X0Model.dispose

    def dispose(self):
        inner = getattr(self, "velocity_model", None)
        if getattr(inner, "_flash_rt_resident", False):
            return
        original(self)

    X0Model.dispose = dispose
    _x0_dispose_patched = True


def _evict_cached_shell(builder) -> bool:
    """Drop the builder's cached model shells. Returns whether one was found.

    The shell cache is upstream's, and reusing a shell is normally right:
    ``dispose`` frees the weights and the next build reassigns fresh ones
    onto the same structure. That contract holds only while the shell's
    parameters keep their shapes, and the resident FFN swap deliberately
    breaks it -- it releases the upstream projections it has repacked, which
    leaves zero-sized parameters that the next load cannot copy into. So the
    shell we mutated is not fit for reuse and is evicted here rather than
    handed to a build that would fail on it.
    """
    seen = set()
    while builder is not None and id(builder) not in seen:
        seen.add(id(builder))
        registry = getattr(builder, "registry", None)
        clear = getattr(registry, "clear", None)
        if callable(clear):
            clear()
            return True
        builder = getattr(builder, "_inner", None)
    return False


class ResidentSwapBuilder(SwapInstallingBuilder):
    """SwapInstallingBuilder that builds once and keeps the model resident.

    The stage's ``_prepared_builder`` derives fresh rewrapped builders from
    the original on every stage call, so the cache lives in a mutable holder
    shared by reference across every rewrap -- the second stage must see the
    model the first stage built, not build a sibling.
    """

    def __init__(self, inner, installers, _holder=None) -> None:
        super().__init__(inner, installers)
        self._holder = _holder if _holder is not None else {}

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        return True

    def build(self, **kwargs):
        model = self._holder.get("model")
        if model is not None:
            return model
        model = super().build(**kwargs)
        # gpu_model() disposes the X0Model wrapper after every stage, and the
        # Disposable mixin walks named_parameters -- which reach through to
        # this model's storages. Mark the model resident and teach X0Model's
        # dispose to skip wrappers holding a resident velocity model.
        model._flash_rt_resident = True
        model.dispose = lambda: None
        _patch_x0_dispose()
        self._holder["model"] = model
        logger.info("[ltx25] transformer resident: %.1fGB allocated",
                    torch.cuda.memory_allocated() / 2 ** 30)
        return model

    def _rewrap(self, inner):
        return ResidentSwapBuilder(inner, self._installers, self._holder)

    @property
    def is_resident(self) -> bool:
        return self._holder.get("model") is not None

    def release(self) -> int:
        """End the residency lease. Idempotent; returns bytes freed.

        Undoes what ``build`` established, and dropping the reference is not
        enough to do it: the upstream builder caches model *shells* by
        structure and reuses them across builds, so the shell outlives every
        reference this class holds. Anything attached to the shell therefore
        has to be detached by name -- the swap's repacked FP4 weights, and
        the capture runner whose per-shape graphs hold a private memory pool
        (measured at ~3GB, which is the difference between a second prompt
        rendering and the host running out of memory). ``dispose`` frees the
        loaded weights and, by upstream's own contract, leaves the shell.

        Order: detach what we attached, then let the disposal run, then
        collect -- the graphs must be unreferenced before the allocator is
        asked to give their pool back.
        """
        model = self._holder.pop("model", None)
        if model is None:
            return 0
        before = torch.cuda.memory_allocated()
        uninstall_nvfp4_ffn(model)
        # The captured block loop is an instance attribute over the class's
        # own method; removing it drops the runner, its captures, and their
        # pool. Absent outside capture mode, where this is a no-op.
        model.__dict__.pop("_process_transformer_blocks", None)
        model._flash_rt_resident = False
        model.__dict__.pop("dispose", None)
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        del model
        _evict_cached_shell(self._inner)
        gc.collect()
        torch.cuda.empty_cache()
        freed = before - torch.cuda.memory_allocated()
        logger.info("[ltx25] resident transformer released: %.1fGB freed",
                    freed / 2 ** 30)
        return freed


class CachingPromptEncoder:
    """Wraps the pipeline's PromptEncoder with an embedding cache.

    Repeat prompts must not re-run the ~26GB encoder while a transformer is
    resident, and a cache serves that. What a cache cannot serve is the miss:
    a prompt nobody encoded yet needs the encoder loaded, which needs the
    residency lease to end first. ``on_miss`` is that call, made before the
    inner encoder runs and only when the cache has nothing -- so a repeat
    prompt keeps the resident model and a new prompt pays a rebuild instead
    of running the host out of memory.
    """

    def __init__(self, inner, on_miss=None) -> None:
        self._inner = inner
        self._cache: dict[tuple, object] = {}
        self._on_miss = on_miss

    def __call__(self, prompts, **kwargs):
        key = (tuple(prompts), tuple(sorted(kwargs.items())))
        hit = self._cache.get(key)
        if hit is None:
            if self._on_miss is not None:
                self._on_miss()
            hit = self._inner(prompts, **kwargs)
            self._cache[key] = hit
        return hit

    def clear(self) -> None:
        """Drop cached embeddings. Idempotent."""
        self._cache.clear()

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_inner"), item)
