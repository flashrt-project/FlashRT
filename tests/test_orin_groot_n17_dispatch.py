"""Dispatch + hardware-gate tests for GR00T N1.7 on Jetson Orin SM87.

No GPU, no checkpoint and no kernels required: these pin the registration and
the fail-fast behaviour so a wrong-arch or wrong-precision request is refused
at construction rather than at the first kernel launch inside ``set_prompt``.
"""

import pytest

torch = pytest.importorskip("torch")

import flash_rt.frontends.torch.groot_n17_orin as orin_mod
from flash_rt.hardware import _SM87_ALLOWED, resolve_pipeline_class

CLS = orin_mod.GrootN17TorchFrontendOrin
KEY = ("groot_n17", "torch", "rtx_sm87")


def test_dispatch_resolves_to_the_orin_frontend():
    try:
        cls = resolve_pipeline_class(*KEY)
    except ModuleNotFoundError as exc:
        if exc.name != "flash_rt.flash_rt_kernels":
            raise
        pytest.skip("flash_rt_kernels was not built")
    assert cls is CLS


def test_sm87_key_is_allowlisted():
    # The allowlist exists because Ampere has no FP8/FP4 tensor cores; without
    # the entry resolve_pipeline_class refuses the key before importing.
    assert KEY in _SM87_ALLOWED


def test_other_archs_still_resolve_to_their_own_frontend():
    # Registering SM87 must not disturb the FP8/FP4 tiers.
    assert resolve_pipeline_class(
        "groot_n17", "torch", "rtx_sm89").__name__ == "GrootN17TorchFrontendRtxSm89"
    assert resolve_pipeline_class(
        "groot_n17", "torch", "thor").__name__ == "GrootN17TorchFrontendThorFP8"


class _Probe:
    """Exercise _require_arch against mocked CUDA state."""

    def run(self, device="cuda:0"):
        return CLS._require_arch(object.__new__(CLS), device)


def test_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv(CLS._FORCE_ARCH_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe().run()


def test_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv(CLS._FORCE_ARCH_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _i: (12, 0))
    with pytest.raises(RuntimeError, match="requires Jetson Orin SM87"):
        _Probe().run()


def test_accepts_sm87(monkeypatch):
    monkeypatch.delenv(CLS._FORCE_ARCH_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _i: (8, 7))
    _Probe().run()  # must not raise


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv(CLS._FORCE_ARCH_ENV, "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe().run()


@pytest.mark.parametrize("kwarg", ["use_fp8", "use_fp4"])
def test_low_bit_tiers_rejected_before_any_cuda_or_checkpoint_work(kwarg):
    # SM87 has neither FP8 nor FP4 tensor cores. The refusal must precede the
    # arch probe and the checkpoint load, so a bogus path is never touched.
    with pytest.raises(RuntimeError, match="does not support"):
        CLS("/nonexistent/fake-ckpt", **{kwarg: True})


def _canon_key(key):
    from flash_rt.executors.torch_weights import Cat

    if isinstance(key, Cat):
        return ("Cat", tuple(key.keys), key.dim, key.dtype)
    return key


def _same_checkpoint_source(base_key, orin_key):
    """Keys must address the same checkpoint tensors.

    A ``Cat`` composite is allowed to differ in exactly one respect: its cast
    dtype, which the Orin derivation retargets from fp16 to bf16.
    """
    b, o = _canon_key(base_key), _canon_key(orin_key)
    if isinstance(b, tuple) and isinstance(o, tuple):
        assert b[0] == o[0] == "Cat"
        assert b[1:3] == o[1:3], f"Cat sources differ: {b[1:3]} vs {o[1:3]}"
        assert o[3] is torch.bfloat16, f"Cat still casts to {o[3]}"
        return True
    return b == o


def test_orin_spec_drops_every_fp8_quant_op():
    from flash_rt.executors.torch_weights import Quant, ToFp16
    from flash_rt.models.groot_n17.weight_spec import WEIGHT_SPEC
    from flash_rt.models.groot_n17.weight_spec_orin import ORIN_WEIGHT_SPEC

    def items(spec):
        for blk in spec.blocks:
            for _ in range(blk.num_layers):
                for it in blk.items:
                    yield it
        yield from spec.singletons

    base, orin = list(items(WEIGHT_SPEC)), list(items(ORIN_WEIGHT_SPEC))
    assert len(base) == len(orin), "item count drifted from the shared spec"

    quant_dropped = 0
    for b, o in zip(base, orin):
        assert b.name == o.name
        assert _same_checkpoint_source(b.key, o.key), (
            f"{o.name}: checkpoint source drifted")
        assert o.scale_into is None, f"{o.name} still declares a scale sink"
        assert not any(isinstance(op, Quant) for op in o.transforms), o.name
        assert not any(isinstance(op, ToFp16) for op in o.transforms), o.name
        quant_dropped += sum(isinstance(op, Quant) for op in b.transforms)
    # Every weight that the FP8 tiers quantize must have been de-quantized here;
    # if this count ever drops to 0 the derivation has stopped doing anything.
    assert quant_dropped > 0


def test_every_kernel_the_pipeline_calls_exists_in_this_build():
    """Pin the SM87 kernel surface the bf16 pipeline depends on.

    A missing symbol otherwise surfaces as an AttributeError at the first
    launch, deep inside ``set_prompt``. Scanning the AST (not the source text)
    keeps this from tripping over the module's own kernel-gap documentation.
    """
    import ast
    import pathlib

    import flash_rt.flash_rt_kernels as fvk

    path = pathlib.Path(orin_mod.__file__).parents[2] / (
        "models/groot_n17/pipeline_orin.py")
    tree = ast.parse(path.read_text())

    gemm_names, fvk_names = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(
                node.value, ast.Name):
            continue
        if node.value.id == "gemm":
            gemm_names.add(node.attr)
        elif node.value.id == "fvk":
            fvk_names.add(node.attr)

    assert gemm_names and fvk_names, "found no kernel calls; the scan is broken"
    runner = fvk.GemmRunner()
    for name in sorted(gemm_names):
        assert hasattr(runner, name), f"GemmRunner has no {name}"
    for name in sorted(fvk_names):
        assert hasattr(fvk, name), f"flash_rt_kernels has no {name}"


# ── INT8 DiT tier contract (no GPU, no checkpoint) ────────────────────────

class _Rec:
    """Recording stub for the kernel modules ``dit_forward`` is handed."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def fn(*a, **k):
            self.calls.append(name)
            return 0            # cutlass_int8_rowwise_bf16out's success status
        return fn


class _Attn:
    def get_slot_ptrs(self, site, j):
        return {"Q": 1, "K": 2, "V": 3, "O": 4}

    def run(self, site, j, *, q_seq, kv_seq, stream=0):
        return 0


def _dit_stub_args(int8: bool, drop: tuple = (), exclude: tuple = ()):
    """bufs/weights/dims shaped like the frontend hands them, as bare ints.

    ``exclude`` models ``dit_bf16_families``: those families ship no ``_w8``/
    ``_s`` and are declared in ``weights["bf16_families"]``, which is exactly
    the pairing ``dit_forward`` cross-checks.
    """
    D, FF, Sa = 1536, 6144, 41
    bufs = {"h": 10, "xn": 11, "o_proj_out": 12, "ff_proj_out": 13}
    if int8:
        for role in ("xn1_i8", "xn1_s", "xn2_i8", "xn2_s",
                     "o_i8", "o_s", "ff_i8", "ff_s"):
            bufs[role] = 20 + len(bufs)
    for role in drop:
        bufs.pop(role, None)
    fams = ("q", "k", "v", "o", "ff_proj", "ff_down")
    weights = {"scale_msa": [0] * 32, "shift_msa": [0] * 32}
    for f in fams:
        weights[f + "_w"] = [0] * 32
        weights[f + "_b"] = [0] * 32
        if int8 and f not in exclude:
            weights[f + "_w8"] = [0] * 32
            weights[f + "_s"] = [0] * 32
    if int8 and exclude:
        weights["bf16_families"] = list(exclude)
    dims = {"Sa": Sa, "D": D, "FF": FF, "Skv_text": 8, "Skv_image": 9}
    return bufs, weights, dims


def test_int8_dit_refuses_to_run_without_its_scratch_buffers():
    """A missing int8 scratch buffer must raise, never fall back to bf16.

    Silent fallback is the dangerous failure here: it is numerically plausible
    (bf16 is *more* accurate), so every INT8 precision and latency measurement
    would quietly be measuring the other tier. AGENTS.md red line #4.
    """
    from flash_rt.models.groot_n17 import pipeline_orin

    for role in ("xn1_i8", "ff_s", "o_i8"):
        bufs, weights, dims = _dit_stub_args(True, drop=(role,))
        fvk, gemm = _Rec(), _Rec()
        with pytest.raises(KeyError) as ei:
            pipeline_orin.dit_forward(gemm, fvk, bufs, weights, dims,
                                      attn=_Attn())
        assert role in str(ei.value)
        assert not fvk.calls and not gemm.calls, (
            f"it launched kernels before noticing {role} was missing")


def test_int8_dit_tier_selection_and_quantize_pass_count():
    """Pin the tier switch and the four-quantize-per-layer budget.

    ``xn`` after adaLN feeds Q and (on self layers) K and V, so it is quantized
    once and reused -- 4 passes per layer, not one per GEMM. If that regression
    ever lands, the extra 64 launches per step are pure loss, and this is the
    only place it would be caught without a profiler.
    """
    from flash_rt.models.groot_n17 import pipeline_orin

    # bf16 tier: no int8 kernel is touched at all.
    bufs, weights, dims = _dit_stub_args(False)
    fvk, gemm = _Rec(), _Rec()
    pipeline_orin.dit_forward(gemm, fvk, bufs, weights, dims, attn=_Attn())
    assert gemm.calls.count("bf16_nn") == 160       # 16 self x6 + 16 cross x4
    assert "cutlass_int8_rowwise_bf16out" not in fvk.calls
    assert "quantize_int8_rowwise" not in fvk.calls

    # INT8 tier: same 160 GEMMs, but through CUTLASS, plus 4 quantizes/layer.
    bufs, weights, dims = _dit_stub_args(True)
    fvk, gemm = _Rec(), _Rec()
    pipeline_orin.dit_forward(gemm, fvk, bufs, weights, dims, attn=_Attn())
    assert not gemm.calls, "the INT8 tier must not touch gemm.bf16_nn"
    assert fvk.calls.count("cutlass_int8_rowwise_bf16out") == 160
    assert fvk.calls.count("quantize_int8_rowwise") == 128   # 4 x 32 layers


def test_int8_dit_exempt_families_really_run_bf16():
    """``bf16_families`` must route those GEMMs to bf16_nn, not to INT8.

    Pinned because the first implementation defined the GEMM helper once per
    tier branch, so under the INT8 tier every site went INT8 regardless of the
    exemption list — and the exempt family's absent ``_w8`` surfaced as a
    KeyError at the first self layer instead of as a wrong number.

    The counts are the interesting part: ``k``/``v`` are consumed on the 16 self
    layers only, so exempting them moves **32 of the 160** GEMMs to bf16 and
    removes **no** quantize pass, because the post-adaLN activation is shared
    with ``q`` and ``q`` is still INT8.
    """
    from flash_rt.models.groot_n17 import pipeline_orin

    bufs, weights, dims = _dit_stub_args(True, exclude=("k", "v"))
    fvk, gemm = _Rec(), _Rec()
    pipeline_orin.dit_forward(gemm, fvk, bufs, weights, dims, attn=_Attn())
    assert fvk.calls.count("cutlass_int8_rowwise_bf16out") == 128   # 160 - 32
    assert gemm.calls.count("bf16_nn") == 32                        # 16 x {k,v}
    assert fvk.calls.count("quantize_int8_rowwise") == 128          # unchanged


def test_int8_dit_exemptions_validate_in_both_directions():
    """A family list that disagrees with the weight dict must raise.

    Silent disagreement is the dangerous case: whichever way it resolves, the
    GEMM runs in a tier nobody asked for and every INT8 measurement built on it
    is describing something else.
    """
    from flash_rt.models.groot_n17 import pipeline_orin

    # exempt but still shipping _w8
    bufs, weights, dims = _dit_stub_args(True)
    weights["bf16_families"] = ["k"]
    with pytest.raises(KeyError, match="exempt but ship one anyway"):
        pipeline_orin.dit_forward(_Rec(), _Rec(), bufs, weights, dims,
                                  attn=_Attn())

    # INT8 but missing _w8, with no exemption declared for it
    bufs, weights, dims = _dit_stub_args(True, exclude=("v",))
    del weights["bf16_families"]
    with pytest.raises(KeyError, match=r"ship no _w8/_s"):
        pipeline_orin.dit_forward(_Rec(), _Rec(), bufs, weights, dims,
                                  attn=_Attn())

    # unknown family name
    bufs, weights, dims = _dit_stub_args(True)
    weights["bf16_families"] = ["kv"]
    with pytest.raises(KeyError, match="not DiT weight families"):
        pipeline_orin.dit_forward(_Rec(), _Rec(), bufs, weights, dims,
                                  attn=_Attn())

    # exemption list on the bf16 tier, where it can only be a mistake
    bufs, weights, dims = _dit_stub_args(False)
    weights["bf16_families"] = ["k"]
    with pytest.raises(KeyError, match="already bf16"):
        pipeline_orin.dit_forward(_Rec(), _Rec(), bufs, weights, dims,
                                  attn=_Attn())


def test_int8_dit_is_the_shipped_default():
    """Pin the default tier so an accidental revert is a test failure.

    Read from the signature rather than restated: the point is that the shipped
    default and the docs' claim stay in step. INT8 is the default because it is
    1.36-1.42x faster and cleared every gate the bf16 tier is held to without
    a single threshold being relaxed (docs §6.9); the accepted cost is a
    worst-case decoded-action error of 0.494 deg on a joint range of roughly
    [-133, 152] deg -- and that error is the bf16 tier's too, i.e. it is the
    backbone's floor, not INT8's (§6.14).

    The K/V exemption is pinned separately because it is a *policy* default
    ("the DiT's K/V must not be quantized"), not a speed one: it costs 1.00 ms
    of the 33.23 ms DiT loop and buys frame 300's decoded error back from
    0.364 deg to 0.194 deg against bf16's own 0.187 deg.
    """
    import inspect

    sig = inspect.signature(orin_mod.GrootN17TorchFrontendOrin.__init__)
    assert sig.parameters["use_int8_dit"].default is True, (
        "use_int8_dit no longer defaults to True; if this is deliberate, "
        "docs/groot_n17_orin_sm87.md §0.1/§6.9.6 quote INT8 as the shipped "
        "default and must be updated in the same change")
    assert sig.parameters["dit_bf16_families"].default is None, (
        "dit_bf16_families should default to the None sentinel, so that "
        "use_int8_dit=False stays constructible without cancelling an "
        "exemption the caller never asked for")
    assert tuple(orin_mod.GrootN17TorchFrontendOrin._DIT_BF16_FAMILIES) == (
        "k", "v"), (
        "the DiT's K/V projections are no longer exempt from INT8 by default; "
        "if this is deliberate, docs §6.14 records the 1.00 ms / 0.364->0.194 "
        "deg trade that decided it and must be updated in the same change")
    assert orin_mod.GrootN17TorchFrontendOrin._DIT_QUANT == (
        "int8_rowwise(k,v=bf16)")


# ── backbone runtime: observation loading (no GPU, no checkpoint) ──────────

def _runtime_stub(monkeypatch, fuse: bool, Sv: int = 8, Se: int = 5):
    """A frontend shell plus a stand-in runtime holding only the input slots.

    The routing under test is a dict-key decision, so it is pinned on CPU: the
    GPU suite never builds an unfused frontend, which is how the bug this
    section exists for got through 31 passing tests.
    """
    fe = object.__new__(CLS)
    fe.device = "cpu"
    bf16 = torch.bfloat16
    rt = {"fuse": fuse, "Sv": Sv, "Se": Se, "loaded": {}}
    if fuse:
        rt["pv"] = torch.zeros(Sv, 1536, dtype=bf16)
    else:
        rt["pf"] = torch.zeros(Sv, 1024, dtype=bf16)
        rt["llm_in"] = torch.zeros(Se, 2048, dtype=bf16)
        # sentinel: the loader must never write the LLM's residual stream
        rt["llm_h"] = torch.full((Se, 2048), 7.0, dtype=bf16)
    monkeypatch.setattr(CLS, "_build_backbone_runtime", lambda self: rt)
    return fe, rt


def test_unfused_embeds_load_into_their_own_buffer_not_the_residual_stream(
        monkeypatch):
    """``llm_h`` is the forward's to write; loading into it is a silent bug.

    With persistent buffers, an observation loaded straight into ``llm_h`` is
    destroyed by the 16 LLM layers' in-place residual updates, so the *second*
    call starts from the first call's layer-15 output. Nothing raises and the
    cosine against the reference stays plausible — it just describes the wrong
    input.
    """
    fe, rt = _runtime_stub(monkeypatch, fuse=False)
    embeds = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    feats = torch.randn(1, 8, 1024, dtype=torch.bfloat16)
    fe._kbb_load_inputs({"llm_input_embeds": embeds, "pixel_features": feats})

    assert torch.equal(rt["llm_in"], embeds.reshape(5, 2048))
    assert torch.equal(rt["pf"], feats.reshape(8, 1024))
    assert bool((rt["llm_h"] == 7.0).all()), (
        "the loader wrote the LLM residual stream; the next observation would "
        "start from this one's output")


def test_fused_pixel_values_load_into_the_persistent_patch_buffer(monkeypatch):
    fe, rt = _runtime_stub(monkeypatch, fuse=True)
    pv = torch.randn(1, 8, 1536, dtype=torch.bfloat16)
    fe._kbb_load_inputs({"pixel_values": pv})
    assert torch.equal(rt["pv"], pv.reshape(8, 1536))


def test_unchanged_observation_is_not_recopied_but_a_new_one_is(monkeypatch):
    """Both directions of the skip-if-unchanged fast path.

    The skip is what keeps the per-call host-to-device move and bf16 conversion
    (~1.5 ms measured on a 512-patch frame) off the steady-state path. Skipping
    too *eagerly* is the failure mode that matters, so a new tensor object and
    an in-place edit of the same object must both force a reload.
    """
    fe, rt = _runtime_stub(monkeypatch, fuse=False)
    embeds = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    aux = {"llm_input_embeds": embeds,
           "pixel_features": torch.randn(1, 8, 1024, dtype=torch.bfloat16)}

    fe._kbb_load_inputs(aux)
    rt["llm_in"].zero_()
    fe._kbb_load_inputs(aux)
    assert bool((rt["llm_in"] == 0).all()), "an unchanged source was re-copied"

    embeds.mul_(2.0)                      # same object, mutated in place
    fe._kbb_load_inputs(aux)
    assert torch.equal(rt["llm_in"], embeds.reshape(5, 2048)), (
        "an in-place edit of the loaded observation was skipped")

    rt["llm_in"].zero_()
    fe._kbb_load_inputs(dict(aux, llm_input_embeds=embeds.clone()))
    assert torch.equal(rt["llm_in"], embeds.reshape(5, 2048)), (
        "a different source tensor was skipped")


@pytest.mark.parametrize("fuse,aux,missing", [
    # pixel_features loads first, so it has to be a real tensor for the loader
    # to reach the absent key rather than failing inside .to()
    (False, {"pixel_features": torch.zeros(1, 8, 1024)}, "llm_input_embeds"),
    (True, {}, "pixel_values"),
])
def test_a_missing_observation_key_raises(monkeypatch, fuse, aux, missing):
    fe, _ = _runtime_stub(monkeypatch, fuse=fuse)
    with pytest.raises(KeyError, match=missing):
        fe._kbb_load_inputs(aux)


# ── continuous inference: the per-observation contract ─────────────────────

#: Tiny stand-ins for the DiT's K/V projections. Only shape *consistency*
#: matters here, but ``_HEADS * _HD == _D_DIT`` must hold so the attention
#: backend's 3-D slot geometry is exercised rather than flattened away.
_D_BB, _D_DIT, _HEADS, _HD = 4, 6, 2, 3


class _FakeAttn:
    """Slot geometry only. The refresh writes through these and nothing else.

    The slots are 3-D ``[max(Skv_text, Skv_image), heads, head_dim]`` exactly
    like ``RtxFlashAttnBackendGrootN17``'s: ``_kv_slot_copy`` flattens with
    ``view(shape[0], -1)``, and a 2-D stub would turn that view into a no-op
    and hide an indexing bug.
    """

    def __init__(self, seq: int, n: int = 16):
        self.dit_cross_K = [torch.zeros(seq, _HEADS, _HD) for _ in range(n)]
        self.dit_cross_V = [torch.zeros(seq, _HEADS, _HD) for _ in range(n)]


def _obs_stub(fuse: bool = True, Se: int = 9, n_image: int = 4):
    """A frontend shell holding only what the contract and the refresh read."""
    fe = object.__new__(CLS)
    fe.device = "cpu"
    fe._fuse_image_embeds = fuse
    fe.Se = Se
    g = torch.Generator().manual_seed(0)

    mask = torch.zeros(Se, dtype=torch.bool)
    mask[:n_image] = True
    fe._visual_pos_masks = mask
    fe._backbone_features = torch.randn(1, Se, _D_BB, generator=g)
    for fam in ("k", "v"):
        setattr(fe, f"_dit_{fam}_w",
                [torch.randn(_D_BB, _D_DIT, generator=g) for _ in range(32)])
        setattr(fe, f"_dit_{fam}_b",
                [torch.randn(_D_DIT, generator=g) for _ in range(32)])
    fe._dit_attn = _FakeAttn(max(n_image, Se - n_image))
    #: The serving projection is a bf16 tensor-core kernel (``gemm.bf16_nn`` +
    #: ``add_bias_bf16``), which needs CUDA and a ``GemmRunner``. These tests pin
    #: slot and graph *bookkeeping*, not that arithmetic — the GPU precision
    #: suite and the three-tier A/B (docs §6.19) do — so the stub shadows the
    #: instance attribute with the fp32 reference arm, which yields identical
    #: shapes and dtypes. Shadowing on the instance leaves the class untouched.
    fe._project_dit_cross_kv = fe._project_dit_cross_kv_fp32
    return fe


def _obs_aux(fuse: bool = True, Se: int = 9, Sv: int = 8):
    """A minimal bundle carrying every key the contract looks at."""
    g = torch.Generator().manual_seed(1)
    masks = torch.zeros(1, Se, dtype=torch.bool)
    masks[0, :4] = True
    aux = {
        "grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "visual_pos_masks": masks,
        "rope_cos": torch.randn(1, Se, 32, 128, generator=g),
        "rope_sin": torch.randn(1, Se, 32, 128, generator=g),
    }
    if fuse:
        aux["pixel_values"] = torch.randn(1, Sv, 1536, generator=g)
        aux["input_ids"] = torch.arange(Se).reshape(1, Se)
    else:
        aux["pixel_features"] = torch.randn(1, Sv, 1024, generator=g)
        aux["llm_input_embeds"] = torch.randn(1, Se, 2048, generator=g)
    return aux


def _flip(t):
    """A copy of ``t`` with one element changed, for any dtype."""
    out = t.clone()
    flat = out.flatten()
    flat[0] = (not bool(flat[0])) if out.dtype == torch.bool else flat[0] + 1
    return out


def _seeded(fe, fuse=True):
    aux = _obs_aux(fuse)
    fe._observation_contract = fe._snapshot_observation_contract(aux)
    return aux


def test_observation_slots_follow_the_fusion_mode():
    fe = object.__new__(CLS)
    fe._fuse_image_embeds = True
    assert fe._observation_slots() == ("pixel_values",)
    fe._fuse_image_embeds = False
    assert fe._observation_slots() == ("pixel_features", "llm_input_embeds")


@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("drop", ["grid_thw", "visual_pos_masks",
                                  "rope_cos", "rope_sin"])
def test_the_snapshot_requires_every_baked_metadata_key(fuse, drop):
    fe = _obs_stub(fuse)
    aux = _obs_aux(fuse)
    del aux[drop]
    with pytest.raises(ValueError, match=drop):
        fe._snapshot_observation_contract(aux)


@pytest.mark.parametrize("fuse,drop", [
    (True, "pixel_values"), (True, "input_ids"),
    (False, "pixel_features"), (False, "llm_input_embeds"),
])
def test_the_snapshot_requires_the_observation_slots(fuse, drop):
    fe = _obs_stub(fuse)
    aux = _obs_aux(fuse)
    del aux[drop]
    with pytest.raises(ValueError, match=drop):
        fe._snapshot_observation_contract(aux)


def test_input_ids_is_pinned_only_where_the_fusion_reads_it():
    """Fused mode bakes the token ids; unfused mode must not constrain them.

    ``set_prompt`` copies ``input_ids`` into ``_fus_ids`` once and the fusion
    reads that buffer by pointer ever after, so a changed token layout has to
    be refused rather than ignored. The unfused path takes its embeds from the
    caller and never looks at the ids, so pinning them there would reject
    legitimate observations for a key the runtime does not consume.
    """
    fused = _obs_stub(True)
    _seeded(fused, True)
    assert "input_ids" in fused._observation_contract

    plain = _obs_stub(False)
    aux = _seeded(plain, False)
    assert "input_ids" not in plain._observation_contract
    plain._validate_observation_contract(
        dict(aux, input_ids=torch.zeros(1, 9, dtype=torch.long)))


@pytest.mark.parametrize("fuse", [True, False])
def test_an_unchanged_observation_is_accepted_whether_reused_or_rebuilt(fuse):
    fe = _obs_stub(fuse)
    aux = _seeded(fe, fuse)
    fe._validate_observation_contract(aux)              # identical objects
    fe._validate_observation_contract(_obs_aux(fuse))   # equal, rebuilt


@pytest.mark.parametrize("key", ["grid_thw", "visual_pos_masks",
                                 "rope_cos", "rope_sin", "input_ids"])
def test_mutated_baked_metadata_is_refused_not_silently_served(key):
    fe = _obs_stub(True)
    aux = _seeded(fe, True)
    with pytest.raises(ValueError, match=key):
        fe._validate_observation_contract(dict(aux, **{key: _flip(aux[key])}))


def test_an_in_place_edit_of_an_already_validated_tensor_is_refused():
    """The identity + ``_version`` fast path must not become a hole.

    A caller that reuses one aux dict across frames and writes into it in place
    presents the *same object* every time, so identity alone waves it through
    and the frontend keeps serving the prompt it was built for. PyTorch's
    mutation counter is what catches that, and this is the only test that
    exercises the second branch of the fast path.
    """
    fe = _obs_stub(True)
    aux = _seeded(fe, True)
    fe._validate_observation_contract(aux)
    aux["rope_cos"].mul_(0.0)
    with pytest.raises(ValueError, match="rope_cos"):
        fe._validate_observation_contract(aux)


# ── inference tensors on the per-observation path ──────────────────────────
#
# ``torch.inference_mode()`` is the recommended way to drive the official
# ``Gr00tPolicy.get_action``, and a live integration that hands FlashRT the
# tensors it captured there passes *inference tensors*. These pin that the two
# identity fast paths degrade to "always re-check" on them rather than either
# crashing or waving an unverifiable tensor through.

def test_mutation_version_reads_the_counter_and_survives_an_inference_tensor():
    """``getattr``'s default does not fire, because the property *raises*."""
    from flash_rt.frontends.torch.groot_n17_orin import _mutation_version

    t = torch.zeros(3)
    assert _mutation_version(t) == 0
    t.mul_(1.0)
    assert _mutation_version(t) == 1

    with torch.inference_mode():
        inf = torch.zeros(3)
    assert inf.is_inference()
    with pytest.raises(RuntimeError, match="version counter"):
        getattr(inf, "_version", None)
    assert _mutation_version(inf) is None


def test_an_inference_tensor_mutation_is_refused_not_memoized():
    """A cached ``None`` version must not match the next ``None``.

    ``_validate_observation_contract`` memoizes ``(validated_source,
    validated_version)`` so a second presentation of an already-checked tensor
    skips the host-side compare. If an inference tensor's unreadable version
    were allowed to satisfy that memo, an in-place edit made inside the
    caller's ``inference_mode()`` block — same object, no readable counter —
    would be served as if it were the prompt it was built from.
    """
    fe = _obs_stub(True)
    aux = _seeded(fe, True)
    with torch.inference_mode():
        inf = aux["rope_cos"].clone()
    assert inf.is_inference() and torch.equal(inf, aux["rope_cos"])
    aux = dict(aux, rope_cos=inf)

    fe._validate_observation_contract(aux)          # checked by value, memoized
    with torch.inference_mode():
        inf.mul_(0.0)
    with pytest.raises(ValueError, match="rope_cos"):
        fe._validate_observation_contract(aux)


def test_an_inference_tensor_observation_is_always_recopied(monkeypatch):
    """The loader's skip must not be decided by object identity alone.

    A caller that reuses one aux buffer across observations presents the same
    object every frame. With a readable counter the in-place edit is caught; on
    an inference tensor nothing is, so the copy must happen unconditionally or
    the backbone replays the first frame's bytes forever.
    """
    fe, rt = _runtime_stub(monkeypatch, fuse=False)
    with torch.inference_mode():
        embeds = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
        feats = torch.randn(1, 8, 1024, dtype=torch.bfloat16)
    assert embeds.is_inference()
    aux = {"llm_input_embeds": embeds, "pixel_features": feats}

    fe._kbb_load_inputs(aux)
    rt["llm_in"].zero_()
    fe._kbb_load_inputs(aux)
    assert torch.equal(rt["llm_in"], embeds.reshape(5, 2048)), (
        "an inference tensor was skipped on identity alone")


def test_the_slowdown_is_announced_once(monkeypatch, caplog):
    """Losing the fast path costs ~1.5 ms/observation; say so, once.

    Red line #4 is about not degrading silently. The degradation here is
    performance rather than correctness, but a caller who cannot see it has no
    way to learn that cloning outside the ``inference_mode`` block buys it back.
    """
    import logging

    import flash_rt.frontends.torch.groot_n17_orin as mod
    monkeypatch.setattr(mod, "_warned_inference_tensor", False)

    fe, rt = _runtime_stub(monkeypatch, fuse=False)
    with torch.inference_mode():
        embeds = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
        feats = torch.randn(1, 8, 1024, dtype=torch.bfloat16)
    aux = {"llm_input_embeds": embeds, "pixel_features": feats}

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        fe._kbb_load_inputs(aux)
        fe._kbb_load_inputs(aux)
        fe._kbb_load_inputs(aux)
    hits = [r for r in caplog.records if "inference tensor" in r.getMessage()]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"
    assert "_kbb_load_inputs" in hits[0].getMessage()


def test_the_dit_graph_bypass_is_announced_once_with_its_cost(monkeypatch,
                                                             caplog):
    """Bypassing the DiT graphs costs 1.66-1.69x per observation; say so, once.

    The correctness half of this is gated on the GPU
    (``test_a_changed_denoising_parameter_bypasses_the_graphs``), which is where
    the replay counter lives. What is pinned here is the part that needs no
    checkpoint: that the bypass is *announced*, that the announcement names both
    parameter triples so a caller can see which knob moved, that it carries the
    measured cost rather than a vague "slower", and that it does not repeat every
    frame of a long run -- a warning per observation would bury the log the
    caller is trying to read.
    """
    import logging

    import flash_rt.frontends.torch.groot_n17_orin as mod
    monkeypatch.setattr(mod, "_warned_dit_graph_params", False)

    captured = (4, 40, 1000)
    want = (4, 20, 1000)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        for _ in range(3):
            mod._note_dit_graph_bypass(captured, want)

    hits = [r for r in caplog.records
            if "DiT CUDA graphs were captured for" in r.getMessage()]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"
    msg = hits[0].getMessage()
    assert str(captured) in msg and str(want) in msg, (
        "the warning does not name both triples, so a caller cannot tell which "
        "parameter moved")
    assert "1.66-1.69x" in msg and "2305.50 MiB" in msg, (
        "the warning does not carry the measured cost; a caller cannot weigh it")
    assert "use_dit_graph=False" in msg, (
        "the warning does not say how to make the slower arm explicit")


@pytest.mark.parametrize("fuse", [True, False])
def test_a_reshaped_observation_is_refused(fuse):
    fe = _obs_stub(fuse)
    aux = _seeded(fe, fuse)
    slot = fe._observation_slots()[0]
    shape = list(aux[slot].shape)
    shape[1] += 1
    with pytest.raises(ValueError, match=slot):
        fe._validate_observation_contract(
            dict(aux, **{slot: torch.zeros(*shape)}))


# ── continuous inference: the cross-K/V refresh ────────────────────────────

def test_kv_slot_copy_writes_only_the_rows_it_is_given():
    """The slot is sized to ``max(Skv_text, Skv_image)``.

    The shorter family therefore leaves a tail of stale rows, which FA2 never
    reads because it is handed the real ``kv_seq``. Writing the tail too would
    be harmless; *not* writing the head would be silent corruption.
    """
    slot = torch.full((6, _HEADS, _HD), 9.0)
    src = torch.arange(4 * _D_DIT, dtype=torch.float32).reshape(4, _D_DIT)
    CLS._kv_slot_copy(slot, src)
    flat = slot.view(6, -1)
    assert torch.equal(flat[:4], src)
    assert bool((flat[4:] == 9.0).all()), "the unused tail was touched"


def test_kv_slot_copy_refuses_a_source_that_does_not_fit():
    slot = torch.zeros(3, _HEADS, _HD)
    with pytest.raises(RuntimeError, match="does not fit"):
        CLS._kv_slot_copy(slot, torch.zeros(4, _D_DIT))
    with pytest.raises(RuntimeError, match="does not fit"):
        CLS._kv_slot_copy(slot, torch.zeros(2, _D_DIT + 1))


def test_refresh_rewrites_the_slots_in_place_and_keeps_the_graphs():
    """Both halves of the continuous path's whole reason to exist.

    The four DiT graphs captured the slots' ``data_ptr()``s, so a refresh that
    allocated fresh tensors would leave every graph reading the *previous*
    observation's K/V — and would still replay successfully, producing actions
    that simply stop tracking the camera. Hence: the slot contents must change,
    and the graphs, the backend object, and the pointers must all survive.
    """
    fe = _obs_stub()
    K, V = fe._project_dit_cross_kv()
    fe._dit_cross_K, fe._dit_cross_V = K, V
    for j in range(16):
        CLS._kv_slot_copy(fe._dit_attn.dit_cross_K[j], K[j])
        CLS._kv_slot_copy(fe._dit_attn.dit_cross_V[j], V[j])

    attn = fe._dit_attn
    graphs = [object() for _ in range(4)]
    fe._dit_graphs = graphs
    ptrs = [t.data_ptr() for t in attn.dit_cross_K]
    before = [t.clone() for t in attn.dit_cross_K]

    fe._backbone_features = torch.randn(
        1, fe.Se, _D_BB, generator=torch.Generator().manual_seed(7))
    fe._refresh_dit_cross_kv()

    assert not any(torch.equal(b, a)
                   for b, a in zip(before, attn.dit_cross_K)), (
        "the slots still hold the previous observation's K/V")
    K2, _ = fe._project_dit_cross_kv()
    for j in range(16):
        got = attn.dit_cross_K[j].view(attn.dit_cross_K[j].shape[0], -1)
        assert torch.equal(got[: K2[j].shape[0]], K2[j])
    assert fe._dit_graphs is graphs, "the captured DiT graphs were dropped"
    assert fe._dit_attn is attn, "the attention backend was rebuilt"
    assert [t.data_ptr() for t in attn.dit_cross_K] == ptrs, (
        "the slots moved, so every captured graph now reads freed memory")


def test_precompute_invalidates_the_graphs_that_refresh_preserves():
    """The two entry points must not be swapped.

    ``_precompute_dit_cross_kv`` is the one-shot path and *has* to invalidate:
    it allocates fresh K/V that no captured graph points at, so leaving the
    graphs alive would strand them exactly as the refresh must not.
    """
    fe = _obs_stub()
    fe._dit_cross_K, fe._dit_cross_V = fe._project_dit_cross_kv()
    fe._dit_graphs = [object()]
    fe._precompute_dit_cross_kv()
    assert not hasattr(fe, "_dit_attn")
    assert not hasattr(fe, "_dit_graphs")


def test_refresh_falls_back_to_precompute_when_nothing_is_captured():
    fe = _obs_stub()
    del fe._dit_attn
    calls = []
    fe._precompute_dit_cross_kv = lambda: calls.append(1)
    fe._refresh_dit_cross_kv()
    assert calls == [1], "with no graphs to preserve the cheaper path was skipped"


def test_refresh_refuses_a_shape_change_rather_than_serving_it():
    """Skv_text/Skv_image are baked into the captured graphs via ``_dit_dims``.

    A new mask splits the same tokens into different text/image lengths, so the
    graphs would read the wrong span. Refuse — and refuse cleanly, leaving the
    frontend usable rather than half-invalidated.
    """
    fe = _obs_stub()
    fe._dit_cross_K, fe._dit_cross_V = fe._project_dit_cross_kv()
    graphs = [object()]
    fe._dit_graphs = graphs
    fe._visual_pos_masks[:6] = True
    with pytest.raises(RuntimeError, match="cross-K/V shapes"):
        fe._refresh_dit_cross_kv()
    assert fe._dit_graphs is graphs
    assert fe._dit_attn is not None



# ─────────────────────────────────────────────────────────────────────────
# Inherited-surface audit: what this frontend does NOT offer, and whether the
# absence is explained rather than crashing.
#
# GrootN17TorchFrontendOrin inherits Orin -> RtxFP16 -> Rtx -> Thor, so the
# whole Thor public API is visible on it. Two of those inherited members
# describe capabilities SM87 does not have; both are pinned here so the gap
# stays *explained* instead of turning into an AttributeError at a call site.
# ─────────────────────────────────────────────────────────────────────────

#: FP8 per-stage act-scale alphas that Thor's ``_snapshot_precision_spec``
#: reads unconditionally. None of them exist on this frontend, which is why
#: ``calibrate`` is overridden to refuse.
_THOR_ALPHA_ATTRS = ("_vit_alpha_q", "_vit_alpha_o", "_dsm_alpha_fc1",
                     "_llm_alpha_qkv", "_vlsa_alpha_q")


def test_the_fp8_alpha_machinery_really_is_absent():
    """Guard the guard: the refusal must not paper over a working capability.

    If a future change gives this frontend FP8 alphas, ``calibrate`` should be
    re-enabled rather than left refusing — and this test is what says so.
    """
    fe = _obs_stub()
    present = [a for a in _THOR_ALPHA_ATTRS if hasattr(fe, a)]
    assert not present, (
        f"this frontend now has FP8 alphas {present}; the calibrate() refusal "
        "in groot_n17_orin.py is stale and should be reconsidered")
    with pytest.raises(AttributeError):
        CLS._snapshot_precision_spec(fe, method="single_frame", n=1,
                                     percentile=None)


def test_calibrate_refuses_with_a_reason_instead_of_crashing():
    """Inherited, ``calibrate`` raised a bare AttributeError.

    The public API advertises calibration; on SM87 there is nothing to
    calibrate (no FP8/FP4, the INT8 DiT tier's scales are dynamic per-row and
    recomputed every forward, the backbone is bf16). Red line #4: refuse loudly
    with the reason, not with an attribute error from somebody else's code path.

    ``NotImplementedError`` specifically, because ``flash_rt/api.py`` documents
    that "unsupported frontends raise a clear NotImplementedError from their
    calibrate() method". It subclasses ``RuntimeError``, so callers already
    catching the latter keep working.
    """
    fe = _obs_stub()
    assert issubclass(NotImplementedError, RuntimeError)
    with pytest.raises(NotImplementedError, match="nothing to calibrate"):
        fe.calibrate([{}, {}])
    with pytest.raises(NotImplementedError, match="dynamic per-row"):
        fe.calibrate({})


def test_precision_spec_stays_none_and_is_inherited_unchanged():
    """``None`` means "no calibration has run" — on Orin it never can."""
    assert CLS.precision_spec.fget is not None
    fe = _obs_stub()
    assert fe.precision_spec is None


def test_the_backbone_graph_arm_is_opt_out_and_routed_from_infer():
    """Lever #11 landed, so the old "known gap" tripwire is now a contract.

    ``run_backbone_graph`` exists for parity with ``GrootN17TorchFrontendThorFP8``
    and the RTX FP8 mixin. It is tied to the per-observation entry, not to
    ``set_prompt``: the capture costs 262.1 ms against 4.54 ms saved per
    observation (§6.13.2), so under the one-shot contract it is a pure loss and
    the one-shot arm must keep running eager.
    """
    import inspect

    assert hasattr(CLS, "run_backbone_graph")
    from flash_rt.frontends.torch import groot_n17_thor_fp8
    assert hasattr(groot_n17_thor_fp8.GrootN17TorchFrontendThorFP8,
                   "run_backbone_graph")

    sig = inspect.signature(CLS.__init__)
    assert sig.parameters["use_backbone_graph"].default is True, (
        "the graph arm should default on: infer(aux=...) implies a stream of "
        "observations, which is the only regime where the capture pays")

    src = inspect.getsource(CLS.infer)
    assert "run_backbone_graph" in src
    assert "_run_kernel_backbone" in src, "the eager arm must remain reachable"
    assert "_use_backbone_graph" in src


def test_exactly_one_contract_validation_happens_per_aux_call():
    """``run_backbone_graph`` validates internally, so ``infer`` must not too.

    Two passes would be harmless numerically but would double the per-frame host
    cost this path exists to reduce — and, worse, a future edit could leave the
    eager arm unvalidated. Pin the shape of both arms instead of trusting it.
    """
    import inspect

    infer_src = inspect.getsource(CLS.infer)
    graph_src = inspect.getsource(CLS.run_backbone_graph)
    assert graph_src.index("_validate_observation_contract") < \
        graph_src.index("replay()"), \
        "the graph must validate the contract before it replays"

    # On the graph arm infer delegates validation; on the eager arm it does it.
    branch = infer_src.split("if self._use_backbone_graph:")[1]
    graph_arm, eager_arm = branch.split("else:", 1)
    assert "_validate_observation_contract" not in graph_arm
    assert "_validate_observation_contract" in eager_arm


def test_the_backbone_graph_captures_once_and_only_from_the_aux_path():
    """Lazy capture, one graph per frontend, and never from ``set_prompt``."""
    import inspect

    cap = inspect.getsource(CLS._capture_backbone_graph)
    # Side-stream warmup before capture: without it the first execution of each
    # kernel resolves its workspace lazily *inside* the capture. Count call
    # sites, not bare mentions — the docstring names _kbb_forward twice.
    assert cap.count("self._kbb_forward(") == 2, \
        "expected warmup runs plus exactly one captured run"
    assert "range(3)" in cap
    assert "capture_begin" in cap and "capture_end" in cap
    assert "cuda.Stream" in cap

    run = inspect.getsource(CLS.run_backbone_graph)
    assert 'hasattr(self, "_kbb_graph")' in run, "capture must be lazy"
    assert run.index("_kbb_load_inputs") < run.index("_capture_backbone_graph"), \
        "load the observation before capturing, so warmup sees real values"

    # set_prompt must not capture: the one-shot contract is one frame, where the
    # 262.1 ms capture is a pure loss.
    assert "_capture_backbone_graph" not in inspect.getsource(CLS.set_prompt)


def test_the_per_observation_entry_point_is_orin_only_and_contract_checked():
    """Pin what Orin adds over the inherited ``infer``.

    The RTX FP8 mixin also accepts ``aux=``, but it refreshes only
    ``_backbone_features`` and leaves ``_dit_cross_K``/``_dit_cross_V`` pointing
    at the prompted observation — Thor's ``infer`` recomputes them only
    ``if not hasattr(self, "_dit_cross_K")``. Orin's override is the one that
    refreshes them in place *and* validates the graph-baked metadata first.
    """
    import inspect

    sig = inspect.signature(CLS.infer)
    assert "aux" in sig.parameters
    assert sig.parameters["aux"].default is None

    from flash_rt.frontends.torch import groot_n17_rtx_fp8
    mixin = groot_n17_rtx_fp8._GrootN17FP8BackboneMixin
    assert "aux" in inspect.signature(mixin.infer).parameters
    src = inspect.getsource(mixin.infer)
    assert "_dit_cross" not in src, (
        "the RTX FP8 mixin now touches the DiT cross-K/V; the §8 🔴 item about "
        "its stale cross-K/V may be fixed — re-check and update the doc")
    orin_src = inspect.getsource(CLS.infer)
    assert "_validate_observation_contract" in orin_src
    assert "_refresh_dit_cross_kv" in orin_src


# ─────────────────────────────────────────────────────────────────────────
# The bf16 tensor-core cross-KV projection (docs §6.18 / §6.19).
#
# The arithmetic itself is gated on the GPU (precision suite + the three-tier
# A/B). What is pinned here is the *contract*, which is where the silent bugs
# live: GemmRunner.bf16_nn takes (A, B, D, M, N, K) -- note N before K -- and a
# swap produces garbage rather than an error, because the operands are raw
# pointers with no shape to check against.
# ─────────────────────────────────────────────────────────────────────────

class _RecordingGemm:
    """Records the (M, N, K) triples and zero-fills the output."""

    def __init__(self):
        self.calls = []

    def bf16_nn(self, A, B, D, M, N, K, stream=0):
        self.calls.append(("bf16_nn", M, N, K))


class _RecordingFvk:
    def __init__(self):
        self.calls = []

    def add_bias_bf16(self, out, bias, rows, cols, stream=0):
        self.calls.append(("add_bias_bf16", rows, cols))


def _kernel_stub(n_image=4, Se=9):
    """An _obs_stub wired for the *serving* (kernel) arm, on CPU."""
    fe = _obs_stub()
    # Undo the fp32 shadow so the real serving method runs.
    del fe._project_dit_cross_kv
    fe._gemm = _RecordingGemm()
    fe._fvk = _RecordingFvk()
    return fe


def test_the_serving_projection_passes_M_N_K_in_the_kernel_order():
    """``bf16_nn(A, B, D, M, N, K)`` -- N before K, and both from the weights.

    The stub's weights are (_D_BB, _D_DIT) = (4, 6), i.e. K=4 and N=6, and the
    two source families have M = n_image and Se - n_image. A swapped N/K would
    still "work" on raw pointers and silently produce the wrong matrix.
    """
    fe = _kernel_stub(n_image=4, Se=9)
    K_list, V_list = fe._project_dit_cross_kv()
    assert len(K_list) == 16 and len(V_list) == 16

    gemm_calls = fe._gemm.calls
    assert len(gemm_calls) == 32, "16 blocks x {k, v}"
    # li = 2j, and li % 4 == 0 selects the text family => j even => text.
    for j in range(16):
        want_m = (9 - 4) if (j % 2 == 0) else 4
        for slot in (2 * j, 2 * j + 1):
            name, M, N, K = gemm_calls[slot]
            assert name == "bf16_nn"
            assert (M, N, K) == (want_m, _D_DIT, _D_BB), (
                f"call {slot}: got (M,N,K)=({M},{N},{K}), expected "
                f"({want_m},{_D_DIT},{_D_BB})")

    bias_calls = fe._fvk.calls
    assert len(bias_calls) == 32
    for j in range(16):
        want_m = (9 - 4) if (j % 2 == 0) else 4
        for slot in (2 * j, 2 * j + 1):
            assert bias_calls[slot] == ("add_bias_bf16", want_m, _D_DIT)


def test_the_serving_projection_emits_bf16_at_the_right_shapes():
    """The slots and the fp32 arm both expect bf16 (M, _D_DIT)."""
    fe = _kernel_stub(n_image=4, Se=9)
    K_list, V_list = fe._project_dit_cross_kv()
    for j, (k, v) in enumerate(zip(K_list, V_list)):
        want_m = (9 - 4) if (j % 2 == 0) else 4
        for t in (k, v):
            assert t.dtype == torch.bfloat16
            assert tuple(t.shape) == (want_m, _D_DIT)
            assert t.is_contiguous()


def test_a_weight_whose_K_disagrees_with_the_features_is_refused():
    """Red line #4: a checkpoint/backbone mismatch must not be served.

    The shapes are read off the weights rather than off a constant, so this is
    the check that catches a wrong spec or a wrong checkpoint.
    """
    fe = _kernel_stub()
    fe._dit_k_w[0] = torch.randn(_D_BB + 1, _D_DIT)
    with pytest.raises(RuntimeError, match="do not match"):
        fe._project_dit_cross_kv()


def test_the_fused_bias_epilogue_is_not_used():
    """``bf16_nn_bias`` returns cuBLAS code=15 on SM87 -- at *both* M=20 and
    M=128, so it is not the M-alignment case ``pipeline_orin.py`` documents.
    Pin the workaround so a future "simplification" does not reintroduce it."""
    import inspect

    src = inspect.getsource(CLS._project_dit_cross_kv)
    # Match the *call* form: the docstring names bf16_nn_bias while explaining
    # why it is avoided, so a bare substring search would match prose.
    assert "bf16_nn_bias(" not in src
    assert "add_bias_bf16(" in src


def test_the_dead_fp32_weight_cache_is_gone():
    """403 MB of promoted weights that the bf16 arm does not need."""
    assert not hasattr(CLS, "_dit_cross_kv_weights")
    import inspect

    assert "_dit_kv_w32" not in inspect.getsource(CLS._project_dit_cross_kv)
    # The reference arm promotes inline instead, and is documented as such.
    ref = inspect.getsource(CLS._project_dit_cross_kv_fp32)
    assert "Reference only" in ref
    assert ".float()" in ref


# ─────────────────────────────────────────────────────────────────────────
# Fused bf16 rotate-half RoPE (lever #9)
#
# The shim this replaced is *bit-identical* to the kernel at every real shape
# (docs §6.22), so no numerical gate can catch a wiring mistake here -- the
# output would be identical right up until it silently is not. What is pinned is
# the contract, where the silent bugs live:
#
#   * the kernel indexes cos/sin at HALF width, so a full-width table would
#     rotate the second half of every head with the wrong frequencies;
#   * ``q_heads`` and ``k_heads`` are separate (the LLM site is GQA 16 vs 8) and
#     a swap rotates K with the wrong stride;
#   * absent kernel or absent half-table must fall back to the shim, not crash
#     and not silently take a wrong path.
# ─────────────────────────────────────────────────────────────────────────

import flash_rt.models.groot_n17.pipeline_orin as _P  # noqa: E402


class _RecordingRope:
    """Records the fused-kernel argument tuple and touches nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, q_in, k_in, cos, sin, q_out, k_out,
                 rows, q_heads, k_heads, head_dim, stream=0):
        self.calls.append((q_in, k_in, cos, sin, q_out, k_out,
                           rows, q_heads, k_heads, head_dim, stream))


def _dup_table(rows, hd, seed=0):
    """A ``cat(emb, emb)`` rope table, i.e. what this port actually builds."""
    g = torch.Generator().manual_seed(seed)
    half = torch.randn(rows, hd // 2, generator=g)
    return torch.cat((half, half), dim=-1)


def test_the_half_table_is_the_first_half_contiguous_and_a_real_copy():
    """The kernel reads ``cos_tab[row * (hd/2) + d]`` -- a strided view would
    silently read the wrong rows, so the slice has to be materialized."""
    t = _dup_table(5, 8)
    out = _P._rope_half_table(t, "test")
    assert out is not None
    assert tuple(out.shape) == (5, 4)
    assert torch.equal(out, t[:, :4])
    assert out.is_contiguous()
    assert out.data_ptr() != t.data_ptr(), "returned a view, not a copy"


@pytest.mark.parametrize("bad", [
    pytest.param(_dup_table(5, 8).index_copy_(
        1, torch.tensor([5]), torch.randn(5, 1)), id="halves-differ"),
    pytest.param(torch.randn(5, 7), id="odd-head-dim"),
    pytest.param(torch.randn(2, 5, 8), id="not-2d"),
])
def test_a_table_the_kernel_cannot_use_falls_back_instead_of_rotating_wrong(bad):
    """No exception, no wrong math: ``None`` routes the site to the shim."""
    assert _P._rope_half_table(bad, "test") is None


def test_the_half_table_refusal_is_announced_once(monkeypatch, caplog):
    """Losing the fused kernel costs ~11.9 ms/backbone; say so, once."""
    import logging

    monkeypatch.setattr(_P, "_warned_full_width_table", False)
    t = _dup_table(5, 8)
    t[:, 5] += 1.0                                # break the duplication
    with caplog.at_level(logging.WARNING, logger=_P.logger.name):
        for _ in range(3):
            assert _P._rope_half_table(t, "rope_cos") is None
    hits = [r for r in caplog.records if "fused rotate-half kernel" in r.getMessage()]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"


def _rope_stub(rows=6, q_heads=16, k_heads=8, hd=8, with_half=True):
    g = torch.Generator().manual_seed(1)
    Q = torch.randn(rows, q_heads, hd, generator=g)
    K = torch.randn(rows, k_heads, hd, generator=g)
    cos, sin = _dup_table(rows, hd), _dup_table(rows, hd, seed=2)
    tbufs = {"cos": cos, "sin": sin,
             "cos_half": _P._rope_half_table(cos, "c") if with_half else None,
             "sin_half": _P._rope_half_table(sin, "s") if with_half else None}
    return Q, K, tbufs


def test_the_fused_call_passes_gqa_head_counts_in_order_and_rotates_in_place(
        monkeypatch):
    """``(rows, q_heads, k_heads, head_dim)`` with 16 != 8, and in == out.

    A q/k head swap does not raise: it rotates K with Q's stride, which reads
    past the end of K's rows or reuses them, and the cosine simply drops.
    """
    rec = _RecordingRope()
    monkeypatch.setattr(_P, "_rope_neox_qk_kernel", lambda: rec)
    Q, K, tbufs = _rope_stub(q_heads=16, k_heads=8)
    _P._rope_qk(Q, K, tbufs, 6, 16, 8, 8, 7)

    assert len(rec.calls) == 1, "one fused launch replaces two shim calls"
    q_in, k_in, c, s, q_out, k_out, rows, qh, kh, hd, stream = rec.calls[0]
    assert (rows, qh, kh, hd) == (6, 16, 8, 8)
    assert qh != kh, "the GQA case must be distinguishable or the pin is vacuous"
    assert stream == 7
    assert (q_in, q_out) == (Q.data_ptr(), Q.data_ptr()), "Q must rotate in place"
    assert (k_in, k_out) == (K.data_ptr(), K.data_ptr()), "K must rotate in place"
    assert c == tbufs["cos_half"].data_ptr()
    assert s == tbufs["sin_half"].data_ptr()


@pytest.mark.parametrize("kernel,half", [
    pytest.param(None, True, id="kernel-absent"),
    pytest.param("rec", False, id="half-table-absent"),
])
def test_the_shim_is_the_fallback_and_runs_twice(monkeypatch, kernel, half):
    """Two shim calls (Q then K) -- the pre-fusion shape, kept as the fallback."""
    n = {"shim": 0}
    real = _P._rope_rotate_half
    monkeypatch.setattr(_P, "_rope_rotate_half",
                        lambda *a, **k: n.__setitem__("shim", n["shim"] + 1))
    rec = _RecordingRope()
    monkeypatch.setattr(_P, "_rope_neox_qk_kernel",
                        lambda: (rec if kernel else None))
    Q, K, tbufs = _rope_stub(with_half=half)
    _P._rope_qk(Q, K, tbufs, 6, 16, 8, 8, 0)
    assert n["shim"] == 2
    assert rec.calls == [], "the kernel must not run on the fallback path"
    assert real is not None


def test_the_kernel_resolver_caches_its_answer(monkeypatch):
    """Resolved once per process, not per layer (40 rope sites per backbone)."""
    hits = {"n": 0}

    def fake_import():
        hits["n"] += 1
        return None

    monkeypatch.setattr(_P, "_VLK_ROPE_QK", _P._VLK_UNRESOLVED)
    import builtins
    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name == "flash_rt" or "qwen3_vl_kernels" in name:
            fake_import()
            raise ImportError("simulated absent extension")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(_P, "_warned_no_rope_kernel", False)
    assert _P._rope_neox_qk_kernel() is None
    assert _P._rope_neox_qk_kernel() is None
    assert _P._rope_neox_qk_kernel() is None
    assert hits["n"] == 1, f"resolver re-imported {hits['n']} times"


def test_set_prompt_builds_all_four_half_tables():
    """Both rope sites (ViT and LLM) need their own pair."""
    import inspect

    src = inspect.getsource(CLS.set_prompt)
    for name in ("_mrope_cos_half", "_mrope_sin_half",
                 "_vit_cos_half", "_vit_sin_half"):
        assert f"self.{name} = _rope_half_table(" in src, name


def test_both_tbufs_carry_the_half_tables():
    """The wiring: a tbufs without them silently takes the shim path."""
    import inspect

    src = inspect.getsource(CLS)
    assert src.count('"cos_half": self._vit_cos_half') == 1
    assert src.count('"cos_half": self._mrope_cos_half') == 1
    assert src.count('"sin_half": self._vit_sin_half') == 1
    assert src.count('"sin_half": self._mrope_sin_half') == 1


def test_the_two_call_sites_pass_their_own_head_geometry():
    """ViT is MHA (NH, NH); the LLM is GQA (NHQ, NHKV) -- and not the reverse.

    The GPU A/B settles this decisively (a swap breaks bit-identity against the
    shim, and it held at ``torch.equal == True``), but the CPU suite has to be
    able to catch it too, since it is what runs where there is no SM87 board.
    """
    import inspect

    vit = inspect.getsource(_P.qwen3vl_vit_forward)
    llm = inspect.getsource(_P.qwen3vl_llm_forward)
    assert "_rope_qk(Q_t, K_t, tbufs, S, NH, NH, HD, stream)" in vit
    assert "_rope_qk(Q_t, K_t, tbufs, S, NHQ, NHKV, HD, stream)" in llm
    assert "NHKV, NHQ" not in llm, "GQA head counts swapped"


def test_the_pipeline_has_no_remaining_direct_shim_call_sites():
    """Both rope sites go through ``_rope_qk``; only its fallback calls the shim.

    A direct ``_rope_rotate_half(Q_t, ...)`` left in a layer loop would keep
    working, stay bit-identical, and cost the 11.9 ms this lever exists to
    recover -- invisible to every numerical gate.
    """
    import inspect

    src = inspect.getsource(_P)
    assert src.count("_rope_rotate_half(Q_t") == 1, "only inside _rope_qk"
    assert src.count("_rope_rotate_half(K_t") == 1, "only inside _rope_qk"
    for fn in (_P.qwen3vl_vit_forward, _P.qwen3vl_llm_forward):
        body = inspect.getsource(fn)
        assert "_rope_qk(" in body, fn.__name__
        assert "_rope_rotate_half(" not in body, fn.__name__


# ─────────────────────────────────────────────────────────────────────────
# The SM87 backbone attention backend
#
# ``OrinGrootN17BackboneAttn`` is the BF16 sibling of the RTX backend, and no
# numerical gate in the repo can see it go wrong: every failure mode here is a
# dtype or head-count misrouting, and a kernel handed bf16 pointers through an
# fp16 entry -- or 16 KV heads where the slots hold 8 -- does not raise. It
# reinterprets the bytes and returns finite, plausible, wrong numbers. So the
# contract is pinned at the argument level, with FA2 replaced by recorders and
# nothing launched.
# ─────────────────────────────────────────────────────────────────────────

import flash_rt  # noqa: E402
from flash_rt.hardware.rtx.attn_backend_groot_n17_backbone import (  # noqa: E402
    RtxGrootN17BackboneAttn, _LLM_HD, _LLM_NH, _VIT_HD, _VIT_NH, _VLSA_HD,
    _VLSA_NH,
)
from flash_rt.hardware.rtx.attn_backend_groot_n17_orin import (  # noqa: E402
    OrinGrootN17BackboneAttn, _LLM_NHKV,
)

#: The *real* FA2 module, resolved at import time. Every test below swaps
#: ``flash_rt.flash_rt_fa2`` for a recorder, so a test that wants to make a
#: claim about the build (rather than about the stub) has to hold this handle.
try:
    from flash_rt import flash_rt_fa2 as _REAL_FA2
except ImportError:                              # pragma: no cover
    _REAL_FA2 = None


class _RecordingFa2:
    """Stands in for ``flash_rt.flash_rt_fa2`` and records entry selection.

    Carries exactly the three entries this build ships -- ``fwd_fp16``,
    ``fwd_bf16``, ``fwd_bf16_causal``; there is no ``fwd_fp16_causal``, which
    is why the causal arm below is unconditional rather than dtype-dispatched.
    """

    ENTRIES = ("fwd_fp16", "fwd_bf16", "fwd_bf16_causal")

    def __init__(self):
        self.calls = []
        for name in self.ENTRIES:
            setattr(self, name, self._recorder(name))

    def _recorder(self, name):
        def call(**kw):
            self.calls.append((name, kw))
        return call

    @property
    def names(self):
        return [c[0] for c in self.calls]


#: Slot lengths for the stub backend. Deliberately distinct from one another
#: and from ``_vit_per`` so a slot mix-up between sites cannot cancel out.
_BB_LLMS, _BB_VITV, _BB_VITS, _BB_VLSA = 16, 2, 8, 12


def _attn_backend(monkeypatch, **kwargs):
    """Build the real backend on CPU with FA2 and the device probe stubbed.

    ``__init__`` runs for real -- the slot geometry and the ``del`` contract
    are themselves under test -- so only the two things that would launch a
    kernel or need a GPU are replaced: the FA2 module (recorders, above) and
    the device-properties call that reads the SM count.

    ``flash_rt.flash_rt_kernels`` is deliberately *not* stubbed. Putting a
    stand-in in ``sys.modules`` looks harmless and is not: the package resolves
    a missing submodule attribute through ``flash_rt.__getattr__`` ->
    ``_extensions.require`` -> ``present``, and ``present`` calls
    ``importlib.util.find_spec``, which raises ``ValueError`` on a
    ``sys.modules`` entry that has no ``__spec__``. ``present`` catches that as
    "not built", so the stub makes the real, built extension look missing and
    the parent's ``import flash_rt.flash_rt_kernels`` fails with build
    instructions. The real module is what this file already depends on at
    ``test_every_kernel_the_pipeline_calls_exists_in_this_build``.
    """
    import types

    if _REAL_FA2 is None:
        pytest.skip("flash_rt_fa2 was not built")
    fa2 = _RecordingFa2()
    monkeypatch.setattr(flash_rt, "flash_rt_fa2", fa2, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _dev: types.SimpleNamespace(multi_processor_count=16))
    kwargs.setdefault("device", "cpu")
    return OrinGrootN17BackboneAttn(
        num_vit_views=kwargs.pop("num_vit_views", _BB_VITV),
        vit_seq=kwargs.pop("vit_seq", _BB_VITS),
        llm_seq=kwargs.pop("llm_seq", _BB_LLMS),
        vl_self_attn_seq=kwargs.pop("vl_self_attn_seq", _BB_VLSA), **kwargs)


@pytest.fixture
def be(monkeypatch):
    return _attn_backend(monkeypatch)


def test_the_llm_kv_slots_hold_native_gqa_heads(be):
    """SM87's first forced difference from the RTX parent, pinned as geometry.

    FA2 takes ``num_heads_q`` and ``num_heads_kv`` separately, so the parent's
    GQA pre-expansion (``gpu_repeat_interleave_heads``, FP16-only) is dropped
    and K/V hold Qwen3-VL-2B's native 8 heads instead of 16. Q stays at 16.
    """
    assert _LLM_NHKV == 8, "Qwen3-VL-2B is 16 query heads / 8 KV heads"
    assert be.llm_K.shape == (_BB_LLMS, 8, _LLM_HD)
    assert be.llm_V.shape == be.llm_K.shape
    assert be.llm_Q.shape == (_BB_LLMS, _LLM_NH, _LLM_HD)
    assert be.llm_O.shape == be.llm_Q.shape
    for t in (be.llm_Q, be.llm_K, be.llm_V, be.llm_O):
        assert t.dtype == torch.bfloat16
    # the LSE scratch is fp32 and padded to a 128 multiple, as FA2 expects
    assert be._llm_lse.dtype == torch.float32
    assert be._llm_lse.shape == (1, _LLM_NH, 128)


def test_a_fallthrough_into_the_parent_llm_arm_fails_loudly(be):
    """``del self._llm_logits, self._llm_ctx``'s stated purpose, pinned as stated.

    The FP16 cuBLAS-MHA scaffolding is deleted, not merely unused, so a routing
    mistake that lands in the parent's llm arm raises instead of reading the
    bf16 slots through an fp16 kernel and returning finite, plausible, wrong
    numbers. Calling the parent's ``run`` directly is what such a mistake looks
    like -- it is reachable, because ``super().run`` is one line away.
    """
    assert not hasattr(be, "_llm_logits")
    assert not hasattr(be, "_llm_ctx")
    with pytest.raises(AttributeError, match="_llm_logits"):
        RtxGrootN17BackboneAttn.run(be, "llm", 0, 8)
    assert be._fa2.calls == [], "it launched a kernel before failing"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_slots_are_refused_at_construction(monkeypatch, dtype):
    """Refuse before any slot is handed out, not at the first launch."""
    with pytest.raises(ValueError, match="requires bf16 slots"):
        _attn_backend(monkeypatch, slot_dtype=dtype)


def test_the_llm_arm_runs_causal_fa2_with_native_gqa_head_counts(be):
    """The kernel-boundary view of "no GQA pre-expansion".

    ``num_heads_q``/``num_heads_kv`` = 16/8 is what tells FA2 to apply the
    group of 2 internally; passing 16/16 against 8-head slots would read past
    the end of K and V.
    """
    ptr = be.run("llm", 3, 9)
    assert ptr == be.llm_O.data_ptr()
    assert be._fa2.names == ["fwd_bf16_causal"]
    kw = be._fa2.calls[0][1]
    assert (kw["num_heads_q"], kw["num_heads_kv"]) == (_LLM_NH, _LLM_NHKV)
    assert kw["seqlen_q"] == kw["seqlen_k"] == 9
    assert kw["batch"] == 1 and kw["head_dim"] == _LLM_HD
    assert kw["softmax_scale"] == pytest.approx(1.0 / _LLM_HD ** 0.5)
    assert kw["Q"] == be.llm_Q.data_ptr()
    assert kw["K"] == be.llm_K.data_ptr(), "K is not the native-GQA slot"
    assert kw["V"] == be.llm_V.data_ptr(), "V is not the native-GQA slot"
    assert kw["O"] == be.llm_O.data_ptr()


def test_the_llm_arm_refuses_a_cross_attention_request(be):
    """``llm`` is self-attention; a differing ``kv_seq`` is a caller bug."""
    with pytest.raises(ValueError, match="kv_seq must equal q_seq"):
        be.run("llm", 0, 8, kv_seq=7)
    assert be._fa2.calls == [], "it launched before refusing"
    # kv_seq=None means "same as q_seq", and an equal one is accepted
    be.run("llm", 0, 8)
    be.run("llm", 0, 8, kv_seq=8)
    assert be._fa2.names == ["fwd_bf16_causal"] * 2


@pytest.mark.parametrize("seq", [0, -1, 17])
def test_an_llm_seq_outside_the_captured_range_is_refused(be, seq):
    """The slots were allocated at ``llm_seq``; beyond it is out of bounds."""
    with pytest.raises(ValueError, match="out of range"):
        be.run("llm", 0, seq)
    assert be._fa2.calls == []


def test_vit_and_vl_self_attn_delegate_to_the_parent(be):
    """Orin overrides only the ``llm`` arm; the other two inherit.

    What must survive the inheritance is the parent's slot reshaping -- the
    ``vit`` site is a multi-view batched view, which is where a per-view length
    bug would show up -- while the FA2 entry selection is Orin's, so both are
    pinned here together.
    """
    be.run("vit", 0, _BB_VITS // _BB_VITV)
    be.run("vl_self_attn", 0, 5)
    assert be._fa2.names == ["fwd_bf16", "fwd_bf16"], (
        "bf16 slots must not reach fwd_fp16")

    vit = be._fa2.calls[0][1]
    assert vit["batch"] == _BB_VITV
    assert vit["seqlen_q"] == vit["seqlen_k"] == _BB_VITS // _BB_VITV
    assert vit["num_heads_q"] == vit["num_heads_kv"] == _VIT_NH
    assert vit["head_dim"] == _VIT_HD

    vlsa = be._fa2.calls[1][1]
    assert vlsa["batch"] == 1 and vlsa["seqlen_q"] == 5
    assert vlsa["num_heads_q"] == vlsa["num_heads_kv"] == _VLSA_NH
    assert vlsa["head_dim"] == _VLSA_HD


def test_the_vit_arm_refuses_a_q_seq_that_is_not_one_view(be):
    with pytest.raises(ValueError, match="per-view len"):
        be.run("vit", 0, _BB_VITS)          # the total, not the per-view length


def test_an_unknown_site_is_refused(be):
    with pytest.raises(KeyError, match="unknown site"):
        be.run("dit_cross", 0, 4)


@pytest.mark.parametrize("dtype,expected", [
    (torch.bfloat16, "fwd_bf16"),
    # unreachable through the shipped slots (which __init__ forces to bf16),
    # but the dispatch expression is what routes the ViT and vl_self_attn sites
    # and a silent inversion there is exactly the failure this section exists
    # to make visible, so pin both arms of it
    (torch.float16, "fwd_fp16"),
])
def test_run_fa2_picks_the_entry_matching_q_dtype(be, dtype, expected):
    q = torch.zeros(1, 4, _VIT_NH, _VIT_HD, dtype=dtype)
    k = torch.zeros(1, 4, _VIT_NH, _VIT_HD, dtype=dtype)
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    lse = torch.zeros(1, _VIT_NH, 128, dtype=torch.float32)
    assert be._run_fa2(q, k, v, o, lse, _VIT_HD, 0) == o.data_ptr()
    assert be._fa2.names == [expected]


def test_the_causal_arm_is_unconditional_and_its_reason_still_holds(be):
    """``_run_fa2_causal`` does not dispatch on dtype, because it cannot.

    This build has no ``fwd_fp16_causal``, so the causal entry is bf16-only and
    the bf16 slot requirement in ``__init__`` is what keeps that sound. If a
    future FA2 build adds an fp16 causal entry, this test goes red on purpose:
    the unconditional choice would then be a decision to re-make, not a fact.
    """
    q = torch.zeros(1, 4, _LLM_NH, _LLM_HD, dtype=torch.float16)
    k = torch.zeros(1, 4, _LLM_NHKV, _LLM_HD, dtype=torch.float16)
    o = torch.zeros(1, 4, _LLM_NH, _LLM_HD, dtype=torch.float16)
    lse = torch.zeros(1, _LLM_NH, 128, dtype=torch.float32)
    be._run_fa2_causal(q, k, torch.zeros_like(k), o, lse, _LLM_HD, 0)
    assert be._fa2.names == ["fwd_bf16_causal"]

    if _REAL_FA2 is None:
        pytest.skip("flash_rt_fa2 was not built")
    assert hasattr(_REAL_FA2, "fwd_bf16_causal")
    assert not hasattr(_REAL_FA2, "fwd_fp16_causal"), (
        "this build gained an fp16 causal entry; _run_fa2_causal's "
        "unconditional choice needs re-examining")


# ─────────────────────────────────────────────────────────────────────────
# The image→embeds fusion's position-embedding interpolation
#
# ``fast_pos_embed_interpolate`` is shared by the Orin frontend and (as a
# private copy) by Thor FP8, and the fusion's whole claim to trust rests on its
# output being *bit-identical* to HF's, not merely close. So the pin is
# ``torch.equal`` against an independently transcribed bilinear reference, and
# the negative control is the arithmetic the module's own note says is wrong:
# compute in fp32, round once at the end. That variant measures max|d| = 0.125
# (1 bf16 ULP at this table's ~32 magnitude), which was observed to ride
# through the 24-layer ViT tower and drop vit_block_17 from 0.998156 to
# 0.995355 and backbone_features from 0.999729 to 0.996150.
# ─────────────────────────────────────────────────────────────────────────

from flash_rt.frontends.torch._groot_n17_fusion import (  # noqa: E402
    fast_pos_embed_interpolate,
)

#: Real geometry, both verified against the checkpoint and a real prompt:
#: ``visual.pos_embed.weight`` is ``(2304, 1024)`` (side 48) and
#: ``image_grid_thw`` is ``[[1,16,16],[1,16,16]]`` (docs §6.16).
_PE_SIDE, _PE_DIM = 48, 1024
_PE_ROWS = _PE_SIDE * _PE_SIDE
_PE_GRIDS = [(1, 16, 16), (1, 16, 16)]


@pytest.fixture(scope="module")
def pe_table():
    """A stand-in table with the real one's shape and magnitude spread.

    ``visual.pos_embed.weight`` measures absmax 32.75 / std 0.6019 -- heavy
    tailed, and the tail is load-bearing here: 1 bf16 ULP is 0.125 at |x| ~ 32
    but only ~0.004 at |x| ~ 0.6, so a plain ``randn`` stand-in would put the
    fp32-vs-bf16 negative control below the level of evidence. The reference
    below was also checked bit-identical against the real checkpoint table
    (max|d| = 0.0, and 0.125 for the fp32 variant), so the synthetic one is a
    convenience, not a weaker claim.
    """
    g = torch.Generator().manual_seed(0)
    t = torch.randn(_PE_ROWS, _PE_DIM, generator=g) * 0.6
    t[torch.randperm(_PE_ROWS, generator=g)[:_PE_ROWS // 100]] *= 12.0
    return t.to(torch.bfloat16)


def _ref_pos_embed(pos_embed, grid_thw, *, merge=2, compute_dtype=None):
    """Independent transcription of the bilinear interpolation.

    Written from the definition rather than from the implementation: explicit
    ``(row, col)`` indexing for the four-corner gather, and an explicit nested
    loop for the merge-block reordering. Those are the two places a
    transposition could hide, and neither shares code with the thing under
    test. ``torch.linspace`` is used verbatim because it is torch's, not the
    implementation's, arithmetic.

    ``compute_dtype`` is where the gather, the weighting and the 4-way sum
    happen. ``None`` means "the table's own dtype" -- what HF does, and what
    the implementation must match. Passing ``torch.float32`` reproduces the
    *wrong* variant the module's note warns about.
    """
    cd = compute_dtype if compute_dtype is not None else pos_embed.dtype
    table = pos_embed.to(cd)
    side = int(round(pos_embed.shape[0] ** 0.5))
    outs = []
    for t, h, w in grid_thw:
        ys = torch.linspace(0, side - 1, h)
        xs = torch.linspace(0, side - 1, w)
        y0, x0 = ys.long(), xs.long()
        y1 = (y0 + 1).clamp(max=side - 1)
        x1 = (x0 + 1).clamp(max=side - 1)
        fy, fx = ys - y0, xs - x0
        raster = torch.empty(h, w, table.shape[1], dtype=cd)
        for r in range(h):
            for c in range(w):
                p00 = table[y0[r] * side + x0[c]]
                p01 = table[y0[r] * side + x1[c]]
                p10 = table[y1[r] * side + x0[c]]
                p11 = table[y1[r] * side + x1[c]]
                # weights are formed in fp32 and rounded to cd once, then the
                # products and the 4-way sum run in cd -- HF's order
                a = ((1 - fy[r]) * (1 - fx[c])).to(cd)
                b = ((1 - fy[r]) * fx[c]).to(cd)
                e = (fy[r] * (1 - fx[c])).to(cd)
                f = (fy[r] * fx[c]).to(cd)
                raster[r, c] = ((a * p00.to(cd)) + (b * p01.to(cd))
                                + (e * p10.to(cd)) + (f * p11.to(cd)))
        blocks = [raster[bh * merge + i, bw * merge + j]
                  for bh in range(h // merge)
                  for bw in range(w // merge)
                  for i in range(merge)
                  for j in range(merge)]
        outs.append(torch.stack(blocks).repeat(t, 1))
    return torch.cat(outs)


def test_the_interpolation_is_bit_identical_to_an_independent_reference(
        pe_table):
    """The equality the fusion is gated on, reproduced on CPU.

    ``torch.equal``, not a tolerance: the GPU-side gate
    (``test_fusion_reproduces_hf_embeds``) already pins this against HF on real
    data, and it needs a checkpoint. This is the same claim where it can be
    pinned cheaply and where a dtype or ordering regression shows up without
    one.
    """
    got = fast_pos_embed_interpolate(pe_table, _PE_GRIDS, device="cpu")
    want = _ref_pos_embed(pe_table, _PE_GRIDS)
    assert got.shape == (sum(t * h * w for t, h, w in _PE_GRIDS), _PE_DIM)
    assert got.dtype == pe_table.dtype == torch.bfloat16
    assert torch.equal(got, want), (
        f"max|d| {(got.float() - want.float()).abs().max().item():.4g}")


def test_computing_in_fp32_and_rounding_once_is_not_the_same(pe_table):
    """Negative control for the pin above, and the module's recorded number.

    Without this, "bit-identical" could be satisfied by an implementation that
    quietly promoted to fp32 -- which is *more* accurate in isolation and
    measurably worse end to end, because HF does the gather and the sum in bf16
    and the ViT's 24 residual layers amplify the 1-ULP disagreement.
    """
    got = fast_pos_embed_interpolate(pe_table, _PE_GRIDS, device="cpu")
    wrong = _ref_pos_embed(pe_table, _PE_GRIDS,
                           compute_dtype=torch.float32).to(torch.bfloat16)
    assert not torch.equal(got, wrong), (
        "the implementation now matches fp32-then-round, i.e. it stopped "
        "doing the arithmetic in the table's dtype")
    max_d = (got.float() - wrong.float()).abs().max().item()
    assert max_d == 0.125, (
        f"max|d| {max_d} -- 1 bf16 ULP at this table's magnitude is 0.125; a "
        "different value means the reference and the implementation disagree "
        "about something other than the compute dtype")


def test_the_output_is_in_merge_block_order_not_raster_order(pe_table):
    """The merger later views this as ``(Sv // merge**2, D * merge**2)``.

    Raster and merge-block order are a permutation of one another -- same rows,
    same multiset -- which is exactly why a gate on the fused ViT *input* would
    still pass with the order wrong and only the merger's output would degrade.
    """
    got = fast_pos_embed_interpolate(pe_table, _PE_GRIDS, device="cpu")
    blocks = _ref_pos_embed(pe_table, _PE_GRIDS, merge=2)
    raster = _ref_pos_embed(pe_table, _PE_GRIDS, merge=1)
    assert torch.equal(got, blocks)
    assert not torch.equal(got, raster), "the output is in raster order"

    # hand-checkable form of the permutation: block 0 is raster rows
    # (0,0) (0,1) (1,0) (1,1), and block 1 starts at raster (0,2) -- not (2,0)
    h = w = _PE_GRIDS[0][1]
    for out_i, (r, c) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        assert torch.equal(got[out_i], raster[r * w + c])
    assert torch.equal(got[4], raster[0 * w + 2])
    assert not torch.equal(got[4], raster[4])


def test_the_temporal_axis_repeats_one_grid(pe_table):
    """``t`` frames of the same image share one interpolated grid.

    A grid row is a property of the camera setup and resolution, so the fusion
    computes it once per prompt and a CUDA graph can bake the pointer in.
    """
    once = fast_pos_embed_interpolate(pe_table, [(1, 16, 16)], device="cpu")
    twice = fast_pos_embed_interpolate(pe_table, [(2, 16, 16)], device="cpu")
    n = once.shape[0]
    assert twice.shape == (2 * n, _PE_DIM)
    assert torch.equal(twice[:n], once)
    assert torch.equal(twice[n:], once)


def test_the_result_follows_the_table_dtype_and_the_requested_device(pe_table):
    """``out_dtype`` is the table's, and the caller's device is honoured."""
    fp32 = fast_pos_embed_interpolate(
        pe_table.float(), _PE_GRIDS, device="cpu")
    assert fp32.dtype == torch.float32
    assert torch.equal(fp32, _ref_pos_embed(pe_table.float(), _PE_GRIDS))

    got = fast_pos_embed_interpolate(pe_table, _PE_GRIDS, device="cpu")
    assert got.device.type == "cpu"
    assert got.is_contiguous()


def test_a_table_whose_row_count_is_not_a_perfect_square_is_refused():
    """Refuse rather than guess the side length.

    An interpolated position embedding of the wrong shape would still be added
    to every patch, and would only ever show up as a degraded cosine -- the
    module says so, so pin that it does refuse.
    """
    for rows in (2305, 2303, 1000):
        with pytest.raises(ValueError, match="not a perfect square"):
            fast_pos_embed_interpolate(
                torch.zeros(rows, 8, dtype=torch.bfloat16), [(1, 4, 4)],
                device="cpu")
