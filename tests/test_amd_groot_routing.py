"""GROOT N1.7 on AMD CDNA4 — dispatch-table, load_model gates, attn backend.

What this file protects:

  * the ``("groot_n17", "torch", "amd_cdna4")`` registration in
    ``flash_rt.hardware._PIPELINE_MAP`` — resolution is lazy, so a module
    or class rename breaks only at runtime on an MI350X unless the exact
    strings are pinned here;
  * the ``load_model`` precision arms for this triple, as they are ACTUALLY
    implemented in ``flash_rt/api.py`` (each assertion below was read out of
    the source, not assumed):
      - a bare ``use_fp8=False`` is refused rather than silently ignored —
        there is no BF16-only GROOT N1.7 tier on CDNA4;
      - ``use_fp16=True`` with the default ``use_fp8=True`` is refused by the
        generic "fp16 requires fp8=False" guard, which fires FIRST;
      - ``use_fp16=True, use_fp8=False`` reaches the CDNA4 arm and raises
        ``NotImplementedError`` — the full-FP16 reference is not ported;
  * :class:`~flash_rt.amd.hardware.cdna4.attn_backend_groot_n17.Cdna4GrootN17AttnBackend`
    as the single backend serving all five documented sites, and its
    argument validation — an unknown site or an out-of-range layer index
    must raise instead of silently returning another site's slot pointer
    (the pipeline writes Q/K/V through those raw pointers, so a wrong slot
    is a silent-wrong-answer bug, not a crash).

Environment matrix:
  - No GPU / NVIDIA CI: the map test and every attention-backend test run
    (the backend is constructed on CPU, see ``cpu_attn`` below); the
    load_model arms skip, because the ``arch == "amd_cdna4"`` extension gate
    in flash_rt/api.py raises ImportError before any of them is reached.
  - MI350X with the extension built: everything runs.
No checkpoint is needed anywhere in this file — every load_model call is
aimed at an error that fires before checkpoint loading.
"""

from __future__ import annotations

import pytest


_AMD_FRONTEND_MODULE = "flash_rt.amd.frontends.torch.groot_n17"
_AMD_FRONTEND_CLASS = "GrootN17TorchFrontendAmd"

# Deliberately nonexistent: every load_model assertion below must fire
# before the checkpoint is opened, so this path must never be read.
_BOGUS_CKPT = "/nonexistent/flashrt-amd-groot-test-ckpt"


def _amd_ext_importable() -> bool:
    """True iff the compiled AMD module is importable in this env."""
    try:
        import flash_rt.amd.flash_rt_amd_kernels  # noqa: F401
        return True
    except ImportError:
        return False


def _amd_groot_frontend_importable() -> bool:
    """True iff the AMD GROOT N1.7 frontend module imports here.

    Resolution of ("groot_n17","torch","amd_cdna4") imports the frontend,
    which pulls in the Thor base and its third-party helpers (transformers
    et al., declared under an optional extra rather than the base install).
    Tests that must run *past* resolution gate on this so a lean
    environment skips instead of erroring.
    """
    try:
        import importlib
        importlib.import_module(_AMD_FRONTEND_MODULE)
        return True
    except ImportError:
        return False


_EXT_SKIP = pytest.mark.skipif(
    not _amd_ext_importable(),
    reason="flash_rt_amd_kernels not built (the amd_cdna4 extension gate in "
           "flash_rt/api.py raises ImportError before this check is reached)")
_FRONTEND_SKIP = pytest.mark.skipif(
    not _amd_groot_frontend_importable(),
    reason="AMD GROOT N1.7 frontend not importable here (optional deps "
           "missing); this assertion lives past resolve_pipeline_class")


# ---------------------------------------------------------------------------
# _PIPELINE_MAP registration
# ---------------------------------------------------------------------------


def test_pipeline_map_has_amd_groot_n17_entry():
    """The AMD GROOT N1.7 dispatch entry must stay registered verbatim.

    load_model resolves lazily through this dict; a drifted tuple (module
    rename, class rename) breaks resolution only at runtime on an AMD box,
    so the exact strings are pinned.
    """
    from flash_rt.hardware import _PIPELINE_MAP

    key = ("groot_n17", "torch", "amd_cdna4")
    assert key in _PIPELINE_MAP, "AMD GROOT N1.7 dispatch entry disappeared"
    assert _PIPELINE_MAP[key] == (_AMD_FRONTEND_MODULE, _AMD_FRONTEND_CLASS)


@_FRONTEND_SKIP
def test_resolver_returns_the_amd_frontend_class():
    """Resolution must hand back the real class object, not just a tuple —
    this is what catches a renamed/removed class that the map still names."""
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("groot_n17", "torch", "amd_cdna4")
    assert cls.__name__ == _AMD_FRONTEND_CLASS
    assert cls.__module__ == _AMD_FRONTEND_MODULE


def test_resolver_rejects_unported_groot_n17_combos():
    """Only torch/amd_cdna4 is registered for GROOT N1.7 on AMD. A JAX
    request, or an unknown arch string, must fail at resolution rather than
    import half a backend."""
    from flash_rt.hardware import resolve_pipeline_class

    for config, framework, arch in [
            ("groot_n17", "jax", "amd_cdna4"),
            ("groot_n17", "torch", "amd_cdna3"),
            ("groot_n17", "torch", "not_a_real_arch")]:
        with pytest.raises(RuntimeError, match="no pipeline"):
            resolve_pipeline_class(config, framework, arch)


# ---------------------------------------------------------------------------
# load_model precision arms (verified against flash_rt/api.py, not assumed)
# ---------------------------------------------------------------------------


@_EXT_SKIP
@_FRONTEND_SKIP
def test_bare_use_fp8_false_is_refused_on_amd():
    """Verified behavior (flash_rt/api.py, groot_n17 + amd_cdna4 arm): a
    bare ``use_fp8=False`` raises ValueError — the CDNA4 tier is FP8
    backbone + bf16 DiT with no BF16-only fallback, so silently serving FP8
    for a caller who asked for BF16 would be a correctness lie about which
    numerics ran. The message must name the FP8 default and point at the
    full-FP16 reference form."""
    import flash_rt

    with pytest.raises(ValueError) as caught:
        flash_rt.load_model(_BOGUS_CKPT, framework="torch",
                            config="groot_n17", hardware="amd_cdna4",
                            use_fp8=False)
    msg = str(caught.value)
    assert "defaults to FP8" in msg
    assert "use_fp16=True" in msg


@_EXT_SKIP
def test_use_fp16_without_clearing_fp8_is_refused():
    """``use_fp8`` defaults to True, so a caller passing only
    ``use_fp16=True`` is asking for two tiers at once. Verified: the
    generic guard in flash_rt/api.py fires FIRST (before any arch-specific
    GROOT arm), so this is the error a user actually sees."""
    import flash_rt

    with pytest.raises(ValueError, match="use_fp16=True requires "
                                         "use_fp8=False"):
        flash_rt.load_model(_BOGUS_CKPT, framework="torch",
                            config="groot_n17", hardware="amd_cdna4",
                            use_fp16=True)


@_EXT_SKIP
@_FRONTEND_SKIP
def test_use_fp16_reference_is_not_implemented_on_amd():
    """Verified behavior: ``use_fp16=True, use_fp8=False`` passes the
    generic guard and the fp16 allow-set (the amd_cdna4 triple IS listed
    there), then reaches the CDNA4 branch of the fp16 routing block and
    raises NotImplementedError — the full-FP16 GROOT N1.7 reference is not
    ported to CDNA4. NotImplementedError (not ValueError) is the contract:
    the combination is legal, just unbuilt."""
    import flash_rt

    with pytest.raises(NotImplementedError, match="not yet.*ported"):
        flash_rt.load_model(_BOGUS_CKPT, framework="torch",
                            config="groot_n17", hardware="amd_cdna4",
                            use_fp16=True, use_fp8=False)


# ---------------------------------------------------------------------------
# Cdna4GrootN17AttnBackend — sites and argument validation
# ---------------------------------------------------------------------------
#
# The backend allocates its Q/K/V/O slots in __init__ (verified by reading
# attn_backend_groot_n17.py: every slot is a torch.empty on ``device``), but
# ``device`` is a constructor argument and the aiter import is deferred
# behind FVK_AMD_ATTN. Constructing on "cpu" with FVK_AMD_ATTN=sdpa
# therefore exercises the whole slot/validation surface with no GPU and no
# aiter — which is exactly the part these tests are about. Dispatch itself
# (``_attn``) is GPU/aiter territory and is covered by the model E2E suite.

_SA = 41                # 1 state token + 40 action tokens
_DIT_KV = 32
_NUM_CROSS_BLOCKS = 16  # DiT cross-attention blocks
_NUM_SELF_BLOCKS = 16   # DiT self-attention blocks (hardcoded in the backend)


@pytest.fixture
def cpu_attn(monkeypatch):
    """A CPU-resident backend with tiny sequence lengths.

    ``FVK_AMD_ATTN=sdpa`` keeps ``_resolve_mha_fwd`` (and therefore the
    aiter dependency) out of construction; the slot geometry, the site
    table and every range check are hardware-independent.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("FVK_AMD_ATTN", "sdpa")
    from flash_rt.amd.hardware.cdna4.attn_backend_groot_n17 import (
        Cdna4GrootN17AttnBackend,
    )
    return Cdna4GrootN17AttnBackend(
        num_vit_views=2, vit_seq=8, llm_seq=16, vl_self_attn_seq=16,
        sa=_SA, dit_kv_seq=_DIT_KV,
        num_dit_cross_blocks=_NUM_CROSS_BLOCKS, device="cpu",
        backbone_dtype=torch.float16, dit_dtype=torch.bfloat16)


def test_backend_declares_the_five_documented_sites(cpu_attn):
    """One backend serves the whole model on AMD. The pipeline addresses
    slots by these exact strings, so the tuple is the interface."""
    assert cpu_attn.sites() == (
        "vit", "llm", "vl_self_attn", "dit_self", "dit_cross")


def test_every_site_exposes_qkvo_slot_pointers(cpu_attn):
    """``get_slot_ptrs`` is the pipeline's only handle on the slots; each
    site must yield all four non-null pointers."""
    for site in cpu_attn.sites():
        ptrs = cpu_attn.get_slot_ptrs(site, 0)
        assert set(ptrs) == {"Q", "K", "V", "O"}, site
        assert all(isinstance(p, int) and p != 0 for p in ptrs.values()), site


def test_dit_cross_blocks_have_distinct_kv_slots(cpu_attn):
    """Cross K/V are per-block (computed once per prompt, read by every
    denoise step). If two blocks aliased, later blocks would attend over
    the wrong keys — a silent numeric bug, so distinctness is pinned."""
    kv = [(cpu_attn.get_slot_ptrs("dit_cross", i)["K"],
           cpu_attn.get_slot_ptrs("dit_cross", i)["V"])
          for i in range(_NUM_CROSS_BLOCKS)]
    assert len(set(kv)) == _NUM_CROSS_BLOCKS, "dit_cross K/V slots alias"
    # Q and O are shared across blocks by design (one query set per step).
    qo = {cpu_attn.get_slot_ptrs("dit_cross", i)["Q"]
          for i in range(_NUM_CROSS_BLOCKS)}
    assert len(qo) == 1


def test_llm_kv_slots_keep_the_native_gqa_head_count(cpu_attn):
    """AMD delta vs the RTX backend: aiter consumes GQA natively, so the
    llm K/V slots hold 8 KV heads, not 16 expanded ones, and the AMD
    forward must skip the head-repeat step. A silent re-widening here would
    double llm K/V traffic and desync the pipeline's pointer arithmetic."""
    assert cpu_attn.llm_Q.shape[1] == 16
    assert cpu_attn.llm_K.shape[1] == 8
    assert cpu_attn.llm_V.shape[1] == 8
    assert cpu_attn.llm_O.shape[1] == 16


@pytest.mark.parametrize("site", ["nope", "dit", "", "VIT"])
def test_unknown_site_is_rejected(cpu_attn, site):
    """An unknown site must raise, never fall through to another site's
    pointers — the pipeline writes raw Q/K/V through whatever comes back."""
    with pytest.raises(KeyError, match="unknown site"):
        cpu_attn.get_slot_ptrs(site, 0)
    with pytest.raises(KeyError, match="unknown site"):
        cpu_attn.run(site, 0, 4)


@pytest.mark.parametrize("site,num_blocks", [
    ("dit_self", _NUM_SELF_BLOCKS),
    ("dit_cross", _NUM_CROSS_BLOCKS),
])
@pytest.mark.parametrize("delta", [0, 1, 7])
def test_layer_index_out_of_range_is_rejected(cpu_attn, site, num_blocks,
                                              delta):
    """Layer indexing into the per-block slot lists must be bounds-checked
    on both ends. Python's negative indexing makes -1 a particularly nasty
    silent alias (it would return the LAST block's K/V), hence it is tested
    explicitly rather than trusting the list."""
    with pytest.raises(IndexError, match="out of range"):
        cpu_attn.get_slot_ptrs(site, num_blocks + delta)
    with pytest.raises(IndexError, match="out of range"):
        cpu_attn.get_slot_ptrs(site, -1 - delta)


def test_run_rejects_out_of_range_sequence_lengths(cpu_attn):
    """Slots are fixed-capacity; a q_seq/kv_seq past the allocation would
    read or write out of bounds. Verified: the backend raises ValueError
    before any dispatch, so these run on CPU without touching aiter."""
    with pytest.raises(ValueError, match="out of range"):
        cpu_attn.run("dit_self", 0, _SA + 1)
    with pytest.raises(ValueError, match="out of range"):
        cpu_attn.run("dit_self", 0, 0)
    with pytest.raises(ValueError, match="out of range"):
        cpu_attn.run("dit_cross", 0, _SA, kv_seq=_DIT_KV + 1)
    with pytest.raises(ValueError, match="out of range"):
        cpu_attn.run("llm", 0, 17)


def test_run_rejects_asymmetric_self_attention(cpu_attn):
    """The llm / dit_self sites are self-attention over one slot set; a
    kv_seq differing from q_seq means the caller believes it is running
    cross-attention and would silently read a truncated key set."""
    with pytest.raises(ValueError, match="kv_seq must equal q_seq"):
        cpu_attn.run("llm", 0, 8, kv_seq=4)
    with pytest.raises(ValueError, match="kv_seq must equal q_seq"):
        cpu_attn.run("dit_self", 0, 8, kv_seq=4)


def test_vit_requires_the_per_view_sequence_length(cpu_attn):
    """ViT attention is batched per view: the slot is (views * per_view)
    rows and ``run`` reshapes on that split, so only the per-view length is
    a legal q_seq. Passing the full multi-view length must raise, not
    silently attend across view boundaries."""
    per_view = 8 // 2
    with pytest.raises(ValueError, match="per-view"):
        cpu_attn.run("vit", 0, 8)
    with pytest.raises(ValueError, match="per-view"):
        cpu_attn.run("vit", 0, per_view, kv_seq=8)


@pytest.mark.parametrize("kwargs,match", [
    (dict(vit_seq=9), "not divisible"),          # views must tile the slot
    (dict(sa=0), "must be positive"),
    (dict(dit_kv_seq=0), "must be positive"),
])
def test_construction_rejects_inconsistent_geometry(monkeypatch, kwargs,
                                                    match):
    """Geometry errors must surface at construction, not as a wrong-shaped
    view thousands of kernel launches later."""
    pytest.importorskip("torch")
    monkeypatch.setenv("FVK_AMD_ATTN", "sdpa")
    from flash_rt.amd.hardware.cdna4.attn_backend_groot_n17 import (
        Cdna4GrootN17AttnBackend,
    )
    base = dict(num_vit_views=2, vit_seq=8, llm_seq=16, vl_self_attn_seq=16,
                sa=_SA, dit_kv_seq=_DIT_KV, device="cpu")
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Cdna4GrootN17AttnBackend(**base)
