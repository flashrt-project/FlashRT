"""Contracts of the sage executable forms of ``attention_core``.

Both forms qualify a site from the family's own captures and answer the
family's way: ``None`` for a site they do not claim, an exception when the
calibration is inconsistent or the package will not serve the device. These
cases need no device and no kernel package -- what they check is the
qualification walk and the selection semantics, which is where the two forms
join the public family and therefore where a mistake reaches other hosts.

Numerical and latency evidence for the forms themselves is a hardware
qualification and lives with the model's benchmark runs, not here.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from flash_rt.structures.impls import attention_core  # noqa: E402
from flash_rt.structures.impls.attention_core import (  # noqa: E402
    sage2_blackwell, sage3_blackwell)

FORMS = (("sage2", sage2_blackwell), ("sage3", sage3_blackwell))


def _captures(b=1, heads=32, seq=2688, dim=128, kv_seq=None,
              kv_heads=None, dtype=torch.bfloat16, mask=None, n=1):
    """Family-shaped captures: per-call dicts in host SDPA layout."""
    kv_seq = seq if kv_seq is None else kv_seq
    kv_heads = heads if kv_heads is None else kv_heads
    row = {
        "q": torch.empty(b, heads, seq, dim, dtype=dtype, device="meta"),
        "key": torch.empty(b, kv_heads, kv_seq, dim, dtype=dtype,
                           device="meta"),
        "value": torch.empty(b, kv_heads, kv_seq, dim, dtype=dtype,
                             device="meta"),
        "mask": mask,
    }
    return [dict(row) for _ in range(n)]


# --------------------------------------------------------------------
# the family contract: what a binder is handed and what it answers
# --------------------------------------------------------------------

@pytest.mark.parametrize("name,module", FORMS)
def test_binder_takes_the_family_capture_convention(name, module):
    """A sequence of capture dicts, not a shape-carrying object.

    This is the whole reason these forms are reachable from the family at
    all, and it is the kind of mismatch that stays invisible until a host
    actually routes through it: the earlier version of these binders read
    attributes off a single object and raised AttributeError the first time
    the family called them.
    """
    captures = _captures(dim=96)          # a dim no artifact advertises
    assert module.bind_dense_attention(captures) is None


@pytest.mark.parametrize("name,module", FORMS)
def test_empty_captures_are_a_caller_error(name, module):
    with pytest.raises(ValueError, match="no captures"):
        module.bind_dense_attention([])


@pytest.mark.parametrize("name,module", FORMS)
def test_masked_sites_are_not_claimed(name, module):
    """A mask has no form these kernels accept, and the packed-KV plan of
    the BF16 form does not transfer -- the quantizers consume dense NHD."""
    mask = torch.zeros(1, 1, 8, 8, device="meta")
    assert module.bind_dense_attention(_captures(mask=mask)) is None


@pytest.mark.parametrize("name,module", FORMS)
def test_non_bf16_sites_are_not_claimed(name, module):
    assert module.bind_dense_attention(
        _captures(dtype=torch.float16)) is None


@pytest.mark.parametrize("name,module", FORMS)
def test_moving_shape_within_one_calibration_raises(name, module):
    captures = _captures(n=2)
    captures[1]["q"] = torch.empty(1, 32, 1344, 128, dtype=torch.bfloat16,
                                   device="meta")
    with pytest.raises(ValueError, match="moved within one"):
        module.bind_dense_attention(captures)


@pytest.mark.parametrize("name,module", FORMS)
def test_a_mask_appearing_mid_calibration_raises(name, module):
    captures = _captures(n=2)
    captures[1]["mask"] = torch.zeros(1, 1, 8, 8, device="meta")
    with pytest.raises(ValueError, match="mask appeared"):
        module.bind_dense_attention(captures)


def test_sage3_does_not_claim_cross_attention():
    """The artifact's attention is self-attention: one shape for Q/K/V."""
    assert sage3_blackwell.bind_dense_attention(
        _captures(kv_seq=1024)) is None


def test_sage3_does_not_claim_grouped_query_sites():
    assert sage3_blackwell.bind_dense_attention(
        _captures(kv_heads=8)) is None


def test_sage2_does_not_claim_mismatched_kv():
    assert sage2_blackwell.bind_dense_attention(
        _captures(kv_seq=1024, kv_heads=8)) is None


# --------------------------------------------------------------------
# selection: preferring a form is a decision, never a default
# --------------------------------------------------------------------

def test_quantized_forms_are_absent_from_the_default_order(monkeypatch):
    """The default ladder must stay precision-first for every host.

    These forms trade a bounded numerical error for speed. That is a
    deployment decision, so it cannot ride in on a family default: a host
    that never asked for it must keep the numerics-preserving order it has
    receipts for.
    """
    tried = []

    def spy(name):
        def binder(captures):
            tried.append(name)
            return None
        return binder

    monkeypatch.setattr(attention_core, "bind_dense_attention", spy("fa2"))
    for name, module in FORMS:
        monkeypatch.setattr(module, "bind_dense_attention", spy(name))
    attention_core.bind_dense_attention_best(_captures())
    assert "fa2" in tried
    assert "sage2" not in tried and "sage3" not in tried


@pytest.mark.parametrize("name", ["sage2", "sage3"])
def test_a_preferred_form_is_tried_before_the_order(name, monkeypatch):
    tried = []

    def spy(label):
        def binder(captures):
            tried.append(label)
            return None
        return binder

    monkeypatch.setattr(attention_core, "bind_dense_attention", spy("fa2"))
    for form_name, module in FORMS:
        monkeypatch.setattr(module, "bind_dense_attention", spy(form_name))
    attention_core.bind_dense_attention_best(_captures(), prefer=(name,))
    assert tried[0] == name, tried


def test_preference_order_is_the_callers_order(monkeypatch):
    tried = []

    def spy(label):
        def binder(captures):
            tried.append(label)
            return None
        return binder

    monkeypatch.setattr(attention_core, "bind_dense_attention", spy("fa2"))
    for form_name, module in FORMS:
        monkeypatch.setattr(module, "bind_dense_attention", spy(form_name))
    attention_core.bind_dense_attention_best(
        _captures(), prefer=("sage3", "sage2"))
    assert tried[:2] == ["sage3", "sage2"], tried


def test_an_unknown_preferred_form_is_an_error():
    """Silently ignoring it would report the default ladder's result as
    though the caller's choice had been honoured."""
    with pytest.raises(ValueError, match="unknown preferred form"):
        attention_core.bind_dense_attention_best(
            _captures(), prefer=("sage9",))


# --------------------------------------------------------------------
# the host-family adapter
# --------------------------------------------------------------------

def _attention_module(rope_type="interleaved", gate=True, heads=32):
    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(4, 4)
            self.to_k = torch.nn.Linear(4, 4)
            self.to_v = torch.nn.Linear(4, 4)
            self.norm_q = torch.nn.LayerNorm(4)
            self.norm_k = torch.nn.LayerNorm(4)
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Dropout(0.0)])
            self.heads = heads
            self.rope_type = rope_type
            self.to_gate_logits = torch.nn.Linear(4, 4) if gate else None

    return _Attn()


class _GatedRotaryProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, query_rotary_emb=None,
                 key_rotary_emb=None):
        return hidden_states


class _SingleRotaryProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        return hidden_states


def test_adapter_recognises_the_gated_dual_rotary_contract():
    from flash_rt.structures.adapters.diffusers_gated_rotary_attention import (
        _compatible_site)

    module = _attention_module()
    ok, reason = _compatible_site(module, _GatedRotaryProcessor())
    assert ok, reason


@pytest.mark.parametrize("mutate,expected", [
    (lambda m: setattr(m, "to_gate_logits", None) or m, True),
    (lambda m: delattr(m, "to_gate_logits") or m, False),
    (lambda m: setattr(m, "rope_type", "unknown") or m, False),
    (lambda m: setattr(m, "heads", 0) or m, False),
    (lambda m: setattr(m, "norm_q", None) or m, False),
])
def test_adapter_negative_recognition(mutate, expected):
    """Recognition is structural, and each missing slot refuses by name."""
    from flash_rt.structures.adapters.diffusers_gated_rotary_attention import (
        _compatible_site)

    module = mutate(_attention_module())
    ok, reason = _compatible_site(module, _GatedRotaryProcessor())
    assert ok is expected
    if not ok:
        assert reason


def test_adapter_declines_a_single_rotary_processor():
    """The sibling rotary family has its own adapter; this one must not
    claim its sites just because the module slots look alike."""
    from flash_rt.structures.adapters.diffusers_gated_rotary_attention import (
        _compatible_site)

    ok, reason = _compatible_site(_attention_module(),
                                  _SingleRotaryProcessor())
    assert not ok
    assert "rotary_emb" in reason, reason


def test_adapter_returns_nothing_on_a_host_without_such_sites():
    """A host that is not this family gets no adapter result at all."""
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert DiffusersGatedRotaryAttentionAdapter()(
        model, lambda: None) is None


def test_adapter_passes_the_preference_to_the_family(monkeypatch):
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)
    import flash_rt.structures.adapters.diffusers_gated_rotary_attention \
        as adapter_module

    seen = {}

    def fake_bind(captures, *, prefer=()):
        seen["prefer"] = prefer
        return None

    monkeypatch.setattr(adapter_module, "bind_dense_attention_best",
                        fake_bind)

    module = _attention_module(heads=2)   # head_dim 2 over the 4-wide stub
    module.processor = _GatedRotaryProcessor()
    model = torch.nn.Module()
    model.attn = module

    def forward():
        module.processor(module, torch.zeros(1, 2, 4))

    DiffusersGatedRotaryAttentionAdapter(("sage2",))(model, forward)
    assert seen["prefer"] == ("sage2",)


def test_adapter_default_prefers_nothing():
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)

    assert DiffusersGatedRotaryAttentionAdapter().prefer == ()


# --------------------------------------------------------------------
# the precision axis: a scheme names the forms, nothing else does
# --------------------------------------------------------------------

def test_default_schemes_name_no_attention_forms():
    """Every existing profile must keep the published order.

    This is the property that makes the new attribute safe to add: a host
    that selected any scheme before this change gets exactly the ladder it
    was measured against.
    """
    from flash_rt.structures import schemes

    for name in schemes.names():
        if name == "nvfp4_balance_sage":
            continue
        scheme = schemes.get(name)
        assert getattr(scheme, "attention_forms", ()) == (), name


def test_the_quantized_profile_names_them():
    from flash_rt.structures import schemes

    scheme = schemes.get("nvfp4_balance_sage")
    assert scheme.attention_forms == ("sage2", "sage3")
    assert scheme.name == "nvfp4_balance_sage"


def test_adapter_takes_the_forms_from_the_scheme(monkeypatch):
    from flash_rt.structures import schemes
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)
    import flash_rt.structures.adapters.diffusers_gated_rotary_attention \
        as adapter_module

    seen = {}

    def fake_bind(captures, *, prefer=()):
        seen["prefer"] = prefer
        return None

    monkeypatch.setattr(adapter_module, "bind_dense_attention_best",
                        fake_bind)
    module = _attention_module(heads=2)
    module.processor = _GatedRotaryProcessor()
    model = torch.nn.Module()
    model.attn = module

    def forward():
        module.processor(module, torch.zeros(1, 2, 4))

    DiffusersGatedRotaryAttentionAdapter()(
        model, forward, scheme=schemes.get("nvfp4_balance_sage"))
    assert seen["prefer"] == ("sage2", "sage3")


def test_a_scheme_without_forms_leaves_the_order_alone(monkeypatch):
    from flash_rt.structures import schemes
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)
    import flash_rt.structures.adapters.diffusers_gated_rotary_attention \
        as adapter_module

    seen = {}

    def fake_bind(captures, *, prefer=()):
        seen["prefer"] = prefer
        return None

    monkeypatch.setattr(adapter_module, "bind_dense_attention_best",
                        fake_bind)
    module = _attention_module(heads=2)
    module.processor = _GatedRotaryProcessor()
    model = torch.nn.Module()
    model.attn = module

    def forward():
        module.processor(module, torch.zeros(1, 2, 4))

    DiffusersGatedRotaryAttentionAdapter()(
        model, forward, scheme=schemes.get("nvfp4_balance"))
    assert seen["prefer"] == ()


# --------------------------------------------------------------------
# the shared staging pool
# --------------------------------------------------------------------

def test_staging_is_shared_per_shape(monkeypatch):
    """Seams with the same form and shape must share one scratch set.

    The scratch is the size of Q/K/V at the bound shape, and a transformer
    reaches this seam once per block: at long sequence lengths, private
    scratch per seam does not fit on a consumer part at all. It is safe to
    share because nothing survives a call -- every buffer is written at the
    top of the forward and read before it returns.
    """
    from flash_rt.structures.impls.attention_core import sage2_blackwell as s2

    made = []

    class _Art:
        def capabilities(self):
            return {"head_dims": (128,)}

        def allocate_workspace(self, q, k, v, **kw):
            made.append((tuple(q.shape), kw.get("fp8v")))
            return object()

    monkeypatch.setattr(s2, "_artifact", lambda: _Art())
    monkeypatch.setattr(s2, "_STAGING", {})

    a = s2._staging_for((1, 8, 2, 128), (1, 8, 2, 128), torch.bfloat16,
                        "meta", "pv_fp8", "per_warp")
    b = s2._staging_for((1, 8, 2, 128), (1, 8, 2, 128), torch.bfloat16,
                        "meta", "pv_fp8", "per_warp")
    assert a is b
    assert len(made) == 1, "the second seam must not allocate again"


@pytest.mark.parametrize("differs", [
    {"q_shape": (2, 8, 2, 128)},
    {"variant": "pv_fp16"},
    {"granularity": "per_thread"},
    {"dtype": torch.float16},
])
def test_a_different_form_or_shape_gets_its_own_staging(differs, monkeypatch):
    """Sharing is keyed, not global: nothing may hand back a buffer of the
    wrong size, precision, or layout."""
    from flash_rt.structures.impls.attention_core import sage2_blackwell as s2

    class _Art:
        def capabilities(self):
            return {"head_dims": (128,)}

        def allocate_workspace(self, q, k, v, **kw):
            return object()

    monkeypatch.setattr(s2, "_artifact", lambda: _Art())
    monkeypatch.setattr(s2, "_STAGING", {})

    base = dict(q_shape=(1, 8, 2, 128), kv_shape=(1, 8, 2, 128),
                dtype=torch.bfloat16, device="meta", variant="pv_fp8",
                granularity="per_warp")
    first = s2._staging_for(**base)
    second = s2._staging_for(**{**base, **differs,
                                **({"kv_shape": differs["q_shape"]}
                                   if "q_shape" in differs else {})})
    assert first is not second


def test_staging_is_not_registered_as_module_state(monkeypatch):
    """Shared scratch must not appear in a seam's state_dict.

    Registering it would have every bound seam claim the same storage as
    its own parameter state, which is both wrong and silently divergent
    once two seams share one pool.
    """
    from flash_rt.structures.impls.attention_core import sage2_blackwell as s2

    class _Art:
        def capabilities(self):
            return {"head_dims": (128,)}

        def allocate_workspace(self, q, k, v, **kw):
            return object()

        sage2_prefill_fp8v_bf16_d128 = staticmethod(lambda *a, **k: None)
        sage2_prefill_f16_bf16_d128 = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(s2, "_artifact", lambda: _Art())
    monkeypatch.setattr(s2, "_STAGING", {})
    core = s2.DenseAttentionSage2((1, 8, 2, 128), (1, 8, 2, 128),
                                  torch.bfloat16, "meta")
    assert core.state_dict() == {}


def test_an_explicit_argument_outranks_the_profile(monkeypatch):
    """A caller tuning one seam should not have to register a scheme.

    The override is the same statement the profile makes, so it reaches
    adapters the same way; what it must not do is silently lose to the
    profile the caller left at its default.
    """
    from flash_rt.structures import autobuild, schemes
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)
    import flash_rt.structures.adapters.diffusers_gated_rotary_attention \
        as adapter_module

    override = autobuild._AttentionOverride(["sage3"])
    assert override.attention_forms == ("sage3",)

    seen = {}

    def fake_bind(captures, *, prefer=()):
        seen["prefer"] = prefer
        return None

    monkeypatch.setattr(adapter_module, "bind_dense_attention_best",
                        fake_bind)
    module = _attention_module(heads=2)
    module.processor = _GatedRotaryProcessor()
    model = torch.nn.Module()
    model.attn = module

    def forward():
        module.processor(module, torch.zeros(1, 2, 4))

    DiffusersGatedRotaryAttentionAdapter()(model, forward, scheme=override)
    assert seen["prefer"] == ("sage3",)


def test_the_front_door_takes_attention_forms():
    """The argument has to exist where a user actually calls in."""
    import inspect

    from flash_rt import structures
    from flash_rt.structures import autobuild, frontdoor

    for fn in (frontdoor.attach, autobuild.auto_swaps):
        assert "attention_forms" in inspect.signature(fn).parameters, fn
    # the public name is a lazy passthrough; what matters is that it does
    # not filter the argument out on the way
    params = inspect.signature(structures.attach).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


# --------------------------------------------------------------------
# a declined form must not keep its working set
# --------------------------------------------------------------------

def test_release_routed_drops_the_forms_and_keeps_revert_callable():
    """Reverting a routed seam is not releasing it.

    The host processor goes back either way; what differs is whether the
    bound form is still reachable. It has to be while the gate might yet
    activate it, and must not be once the gate has declined -- otherwise an
    attachment costs more memory than the host it replaced, for a form that
    never runs.
    """
    from flash_rt.structures.autobuild import AutoPlan

    routes = [("module", "original", "routed")]
    reverted = []

    plan = AutoPlan()
    plan.observed["site.processor"] = object()
    plan.toggles.append((lambda: None, lambda: reverted.append("off")))
    plan.revert.append(lambda: reverted.append("revert"))
    plan.releases.append(routes.clear)

    plan.release_routed()
    assert routes == [], "the adapter's own hold must be dropped"
    assert plan.observed == {} and plan.toggles == []
    assert reverted == ["off"], "release disables, it does not revert"

    plan.release_routed()          # idempotent
    assert plan.releases == []
    for undo in plan.revert:       # still callable, nothing left to undo
        undo()
    assert reverted == ["off", "revert"]


def test_adapter_publishes_a_release(monkeypatch):
    """The adapter has to offer the hold it wants dropped."""
    from flash_rt.structures.adapters import (
        DiffusersGatedRotaryAttentionAdapter)
    import flash_rt.structures.adapters.diffusers_gated_rotary_attention \
        as adapter_module

    class _Core(torch.nn.Module):
        def forward(self, q, k, v, **kw):
            return q

    monkeypatch.setattr(adapter_module, "bind_dense_attention_best",
                        lambda captures, **kw: _Core())
    module = _attention_module(heads=2)
    module.processor = _GatedRotaryProcessor()
    model = torch.nn.Module()
    model.attn = module

    def forward():
        module.processor(module, torch.zeros(1, 2, 4))

    _, _, extras = DiffusersGatedRotaryAttentionAdapter()(model, forward)
    assert extras.get("release"), "no way to give the bound forms back"
    assert extras.get("observed")
    for release in extras["release"]:
        release()
    # the toggles still work after a release: they close over the list,
    # not over what was in it
    enable, disable = extras["toggle"]
    enable(); disable()
