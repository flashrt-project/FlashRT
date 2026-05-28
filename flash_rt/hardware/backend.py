"""FlashRT — AttentionBackend protocol.

Anchors the upstream refactor: every model pipeline in ``flash_rt.models.*``
that has more than one hardware implementation (Pi0.5, Pi0, GROOT, Pi0.6)
calls attention through this protocol, so the hardware-specific code lives
in exactly one place (``flash_rt.hardware.{thor,rtx}.attention_*``).

Scope
-----
This protocol covers only *attention*. Everything else hardware-specific
(FP8 GEMM dispatch, fused-epilogue availability, per-shape algorithm
selection for Pi0-FAST's cuBLASLt heuristic gap, etc.) is orthogonal and
NOT part of this protocol. Pi0-FAST in particular is explicitly excluded:
it keeps its single-file SM-runtime-branch pattern, because its hardware
differences are in GEMM dispatch not attention. See
``upstream_refactor_plan.md`` §5 for the rationale.

Two storage models
------------------
The existing Thor and rtx attention backends do NOT agree on who owns
Q/K/V/O memory:

* **Backend-owned** (rtx): The backend allocates Q/K/V as torch tensors
  in its ``__init__``. Modern vendored FA2 backends also own stable
  output tensors; legacy pip ``flash_attn_func`` fallback paths keep an
  output reference so the torch caching allocator pins the allocation
  across CUDA Graph capture + replay. The pipeline writes Q/K/V into
  the backend's pre-allocated slots via pointers returned from
  :meth:`AttentionBackend.get_slot_ptrs`.

* **Pipeline-owned** (Thor): The pipeline allocates its own ``attn_out``
  scratch (which stores Q *before* the attention call and the output
  *after*), plus the encoder/decoder K/V cache as part of the weights
  dict, plus logits scratch. The backend is a thin wrapper around
  ``fvk.attention_qkv_fp16`` / ``fvk.fmha_strided_full`` / etc. and does
  not own any buffers — it takes pointers at ``run()`` time.

Both storage models are supported. The protocol's :meth:`get_slot_ptrs`
method returns whatever pointers the backend has on file for (site,
layer_idx); those pointers are *guaranteed stable* across CUDA Graph
capture + replay but make no promise about who allocated them.

A pipeline for a model that runs on both hardware families must be
written to *not care* which model its backend uses. In practice that
means:

    # Both models
    ptrs = self.attn.get_slot_ptrs("qwen3", layer_idx)
    # Pipeline writes Q/K/V via fvk strided copies / projections into
    # ptrs["Q"], ptrs["K"], ptrs["V"]
    ...
    out_ptr = self.attn.run("qwen3", layer_idx,
                            q_seq=Se, kv_seq=None, stream=stream)
    # Pipeline reads output at out_ptr (may or may not alias ptrs["O"])

The **same source code** runs on both hardware; only the backend
instance differs.

Output pointer stability
-------------------------
``run(...)`` returns an ``int`` device pointer. The contract is:

1. The returned pointer is valid until the **next** ``run(same_site,
   same_layer_idx)`` call on the same backend, or until the backend is
   destroyed, whichever comes first.
2. For rtx's backend-owned model, the returned pointer is the data_ptr
   of an internally-held torch tensor; the backend holds a reference so
   the caching allocator does not reassign the slot across CUDA Graph
   capture + replay. Pipelines MUST hold a reference to the backend for
   at least as long as any captured graph uses it.
3. For Thor's pipeline-owned model, the returned pointer is whatever
   ``O`` buffer the backend was told about at construction time (or the
   ``attn_out`` buffer the pipeline passed to Thor's fvk call — same
   pointer).
4. Pipelines MUST NOT cache the returned pointer across different
   ``run()`` calls at the same site/layer, because both backends reserve
   the right to swap internal slots. In practice rtx always returns the
   same pointer once warmed up and Thor always returns the pre-fixed
   pointer, but pipelines should re-read after every ``run()`` to stay
   future-proof.

Variable sequence length
-------------------------
Most sites allocate Q/K/V slots for ``max_q_seq`` / ``max_kv_seq`` tokens
but run with a smaller active length (e.g. Pi0.5 at encoder_seq_max=600
but Se=286 for a given prompt). ``run(q_seq=X, kv_seq=Y)`` tells the
backend how many rows of the slot are actually active. The backend
slices / passes-through-to-kernel as appropriate.

For self-attention, pass ``kv_seq=None`` and the backend uses
``kv_seq = q_seq``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


# ══════════════════════════════════════════════════════════════════
#  Attention site specification
# ══════════════════════════════════════════════════════════════════


@dataclass
class SiteSpec:
    """Description of one attention site in a model.

    A *site* is a distinct attention shape that appears at multiple
    layers. GROOT has four sites (SigLIP, Qwen3, DiT self, DiT cross);
    Pi0.5 has three (SigLIP vision, PaliGemma encoder, 10-step decoder).
    Every layer at the same site shares the same shape parameters.

    Attributes
    ----------
    num_layers :
        Number of layers using this site. The backend pre-allocates
        per-layer Q/K/V slots (for backend-owned model) or per-layer
        metadata (for pipeline-owned model).
    num_q_heads :
        Query-side head count. For MHA: equal to ``num_kv_heads``.
        For GQA: ``num_q_heads > num_kv_heads`` (e.g. Qwen3 16Q/8KV).
    num_kv_heads :
        Key/Value-side head count. MHA → equal to ``num_q_heads``.
    head_dim :
        Dimension per attention head (same for Q, K, V).
    max_q_seq :
        Maximum Q sequence length this site will ever run with. Used
        for slot allocation sizes.
    max_kv_seq :
        Maximum K/V sequence length. ``None`` means self-attention
        (max_kv_seq == max_q_seq). For cross-attention (DiT cross,
        Pi0.5 decoder layers) this is the KV side and may differ from
        the Q side.
    batch_axis :
        Whether the site treats its leading dimension as a batch
        (multi-view SigLIP: batch = num_views; everything else: batch
        = 1). Affects how Q/K/V slots are shaped.
    sliding_window :
        If set, attention is restricted to a (left=``sliding_window``,
        right=0) window around each query position. Required for
        Gemma 3's 5:1 local:global layers (Pi0.6 — Stage 5). ``None``
        for models that do not use SWA.
    causal :
        Whether this attention is causal (strict lower-triangular
        mask). Default False for all current models. Reserved for
        future decoder-only paths.
    extra :
        Backend-specific hints the protocol does not need to know
        about. Discouraged but available as an escape hatch.
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_q_seq: int
    max_kv_seq: Optional[int] = None
    batch_axis: int = 1
    sliding_window: Optional[int] = None
    causal: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_kv_seq is None:
            self.max_kv_seq = self.max_q_seq
        if self.num_kv_heads > self.num_q_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be <= num_q_heads "
                f"({self.num_q_heads}) — use MHA or standard GQA layout"
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads}) for GQA"
            )


# ══════════════════════════════════════════════════════════════════
#  Full-model attention specification
# ══════════════════════════════════════════════════════════════════


@dataclass
class AttentionSpec:
    """Full attention structure of one model (all sites it needs).

    A model's pipeline.py exposes ``make_attention_spec(...)`` as a
    static method that builds an ``AttentionSpec`` from the model's
    hyperparameters; the frontend passes this spec to
    ``hardware.make_attention_backend(arch, spec)`` to get a live
    backend instance, which it then injects into the pipeline
    constructor.

    This decouples model-specific shape knowledge from
    hardware-specific allocation logic:

      * models/pi05/pipeline.py knows ``num_views``, ``chunk_size``,
        ``num_encoder_layers``, etc. and builds the spec.
      * hardware/thor/attention.py and hardware/rtx/attention_pi05.py
        know how to allocate slots and dispatch kernels; they consume
        the spec.
    """

    sites: dict[str, SiteSpec] = field(default_factory=dict)

    def add_site(self, name: str, **kwargs) -> "AttentionSpec":
        if name in self.sites:
            raise ValueError(f"site {name!r} already added to spec")
        self.sites[name] = SiteSpec(**kwargs)
        return self

    def site(self, name: str) -> SiteSpec:
        if name not in self.sites:
            raise KeyError(
                f"site {name!r} not in spec. Known sites: "
                f"{sorted(self.sites.keys())}"
            )
        return self.sites[name]


# ══════════════════════════════════════════════════════════════════
#  The AttentionBackend protocol
# ══════════════════════════════════════════════════════════════════


class AttentionBackend(Protocol):
    """Hardware provider for attention in a model pipeline.

    Lifetime and threading
    ----------------------
    * One backend instance serves one pipeline. Do not share across
      pipelines.
    * The backend is constructed before the pipeline, passed to the
      pipeline as a constructor argument, and lives as long as the
      pipeline does.
    * Backends are single-threaded w.r.t. CUDA. Calls from different
      Python threads are not supported and undefined.

    Slots and pointers
    ------------------
    A *slot* is a (site, layer_idx, role) triple where ``role`` is one
    of ``"Q"``, ``"K"``, ``"V"``, and (backend-dependent) ``"O"``,
    ``"scratch"``, ``"logits"``. Each slot has a fixed device pointer
    for the lifetime of the backend. ``get_slot_ptrs`` returns the
    current mapping.

    Not every backend exposes every role. The rtx backends expose
    ``Q``, ``K``, ``V`` per slot and return the output pointer from
    ``run()``; vendored FA2 paths usually keep ``O`` backend-owned, while
    legacy fallback paths may keep only a reference to the allocated
    output tensor. The Thor backend exposes ``Q`` (which is also the
    output buffer, aliased with ``O``), plus ``K`` and ``V`` per layer.
    Pipelines MUST only read keys they know the target backend provides
    for a given site.
    """

    def sites(self) -> tuple[str, ...]:
        """Return the tuple of site names this backend was configured for.

        Matches the keys of the ``AttentionSpec`` it was built from.
        Useful for pipeline-side asserts.
        """

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return raw int device pointers for all slots at (site, layer).

        Returned dict always has at least the keys ``"Q"``, ``"K"``,
        ``"V"``. Backend may include additional keys — see site-specific
        docstrings on concrete backend implementations.

        Pointers returned are valid for the lifetime of the backend and
        stable across CUDA Graph capture + replay. The pipeline may
        cache them (it is safe to call ``get_slot_ptrs`` once per layer
        per site at pipeline construction time and reuse the dict on
        every infer).
        """
        ...

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq: Optional[int] = None,
        stream: int = 0,
        state_nk: Optional[int] = None,
    ) -> int:
        """Execute attention for one (site, layer) and return the output ptr.

        Parameters
        ----------
        site :
            Site name as registered in the ``AttentionSpec``.
        layer_idx :
            Zero-based layer index within the site (``0 <= layer_idx
            < SiteSpec.num_layers``).
        q_seq :
            Active Q sequence length for this call. Must satisfy
            ``q_seq <= SiteSpec.max_q_seq``. Rows ``q_seq..max_q_seq``
            of the Q slot are ignored.
        kv_seq :
            Active K/V sequence length. ``None`` means self-attention
            (``kv_seq = q_seq``). For cross-attention (DiT cross-attn,
            Pi0.5 decoder cross-attn) the pipeline passes the KV-side
            length explicitly — it may differ from ``q_seq``.
        stream :
            CUDA stream pointer (0 = default stream). The backend
            launches its kernel on this stream.
        state_nk :
            Only honored when the site's ``SiteSpec.extra["kernel"] ==
            "state_masked"`` (Pi0 decoder). The first query row acts as
            a *state token* that may only attend to the first
            ``state_nk`` K/V positions — the attention kernel masks
            logits at columns ``[state_nk:]`` for the state row. All
            other query rows attend over the full ``kv_seq``. For
            standard-kernel sites this argument is ignored; for
            state-masked sites it is required and must satisfy
            ``0 < state_nk <= kv_seq``.

        Returns
        -------
        int
            Raw device pointer to the attention output. The semantic
            shape of the output is backend-specific but always
            contiguous row-major with the following row layout:

              * self-attention: ``(q_seq, num_q_heads, head_dim)`` flat
              * cross-attention: same ``(q_seq, num_q_heads, head_dim)``

            The pipeline is responsible for interpreting the output as
            ``(q_seq, num_q_heads * head_dim)`` for the downstream
            output projection GEMM.

        Notes
        -----
        * The returned pointer is stable across CUDA Graph capture and
          replay for the same (site, layer_idx). See
          ``Output pointer stability`` in the module docstring.
        * If the backend uses GQA (num_kv_heads < num_q_heads), it
          handles head repetition internally; the pipeline's GEMM
          output is always sized as if the result had ``num_q_heads``
          heads.
        """
        ...

    def head_dim(self, site: str) -> int:
        """Return ``head_dim`` for ``site``. Convenience accessor."""
        ...

    def num_q_heads(self, site: str) -> int:
        """Return ``num_q_heads`` for ``site``. Convenience accessor."""
        ...

    def num_kv_heads(self, site: str) -> int:
        """Return ``num_kv_heads`` for ``site``. Convenience accessor."""
        ...


# ══════════════════════════════════════════════════════════════════
#  Optional base class (default implementations for accessors)
# ══════════════════════════════════════════════════════════════════


class AttentionBackendBase:
    """Convenience base class providing accessor defaults.

    Concrete backends can subclass this and implement only
    :meth:`get_slot_ptrs` and :meth:`run`. The metadata accessors are
    derived from the ``AttentionSpec`` passed at construction time.

    Using this base class is optional — a backend that does not want
    to inherit can implement the protocol directly (duck-typed).
    """

    def __init__(self, spec: AttentionSpec) -> None:
        self._spec = spec

    def sites(self) -> tuple[str, ...]:
        return tuple(self._spec.sites.keys())

    def head_dim(self, site: str) -> int:
        return self._spec.site(site).head_dim

    def num_q_heads(self, site: str) -> int:
        return self._spec.site(site).num_q_heads

    def num_kv_heads(self, site: str) -> int:
        return self._spec.site(site).num_kv_heads

    # get_slot_ptrs and run: abstract — subclass must implement

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        raise NotImplementedError

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq: Optional[int] = None,
        stream: int = 0,
        state_nk: Optional[int] = None,
    ) -> int:
        raise NotImplementedError
