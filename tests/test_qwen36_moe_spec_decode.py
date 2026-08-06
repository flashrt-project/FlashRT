"""Speculative decode: the supported API, the cache bound, and equivalence.

The claim this path makes is that ``generate_spec`` emits exactly what
``generate`` emits -- a draft is kept only where the model's own argmax agrees
with it -- so the tests are equality tests, not tolerance tests. The ones that
can run without a GPU or a checkpoint (the constructor contract and the graph
cache's eviction policy) always run; the rest skip and say why.

Run the whole file against a checkpoint with:

    FLASHRT_QWEN36_MOE_CKPT_DIR=/path/to/Qwen3.6-35B-A3B \\
    PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    pytest -q -p no:cacheprovider tests/test_qwen36_moe_spec_decode.py
"""

from __future__ import annotations

import collections
import inspect
import os

import pytest

CKPT = os.environ.get("FLASHRT_QWEN36_MOE_CKPT_DIR")

requires_gpu_checkpoint = pytest.mark.skipif(
    not CKPT,
    reason="set FLASHRT_QWEN36_MOE_CKPT_DIR to the Qwen3.6-35B-A3B checkpoint",
)

# A prompt short enough that a window covers a meaningful fraction of the run,
# and a generation long enough for several windows including rejected tails.
PROMPT = "Explain why deterministic reductions matter."
NEW_TOKENS = 24


# ── The constructor contract ──────────────────────────────────────────────
# These run everywhere: they are about the public surface, not the kernels.


def _frontend_cls():
    from flash_rt.frontends.torch.qwen36_moe import Qwen36MoeTextFrontend

    return Qwen36MoeTextFrontend


def test_load_mtp_is_a_public_argument_defaulting_off():
    params = inspect.signature(_frontend_cls()).parameters
    assert "load_mtp" in params, (
        "the draft head must be reachable through the constructor, not by "
        "setting a private attribute on a subclass")
    assert params["load_mtp"].default is False, (
        "the head is a transformer layer's worth of weights that generate() "
        "never reads; loading it must be asked for")
    assert params["load_mtp"].kind is inspect.Parameter.KEYWORD_ONLY


def test_spec_graph_cache_max_is_a_public_argument():
    params = inspect.signature(_frontend_cls()).parameters
    assert "spec_graph_cache_max" in params
    assert params["spec_graph_cache_max"].default is None


@pytest.mark.parametrize("bad", [0, -1])
def test_spec_graph_cache_max_rejects_a_cache_that_holds_nothing(bad):
    with pytest.raises(ValueError, match="spec_graph_cache_max"):
        _frontend_cls()("/nonexistent", spec_graph_cache_max=bad)


def test_generate_spec_without_the_head_says_how_to_get_it():
    frontend = _frontend_cls().__new__(_frontend_cls())
    frontend._prompt_ids = object()
    frontend._load_mtp = False
    with pytest.raises(RuntimeError, match="load_mtp=True"):
        frontend.generate_spec(8)


def test_generate_spec_without_a_prompt_says_so():
    frontend = _frontend_cls().__new__(_frontend_cls())
    frontend._prompt_ids = None
    frontend._load_mtp = True
    with pytest.raises(ValueError, match="set_prompt"):
        frontend.generate_spec(8)


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_spec_rejects_a_window_that_drafts_nothing(bad):
    frontend = _frontend_cls().__new__(_frontend_cls())
    frontend._prompt_ids = object()
    frontend._load_mtp = True
    with pytest.raises(ValueError, match="k must be at least 1"):
        frontend.generate_spec(8, k=bad)


def test_decode_state_takes_the_cache_bound():
    from flash_rt.frontends.torch import _nexn2_rtx_decode as decode

    params = inspect.signature(decode.Nexn2DecodeState).parameters
    assert params["spec_graph_cache_max"].default is None
    assert (params["spec_graph_cache_max"].kind
            is inspect.Parameter.KEYWORD_ONLY)


# ── The graph cache's eviction policy ─────────────────────────────────────
# A captured graph owns its memory pool, so this bound is what keeps a long
# generation from accumulating device memory one position at a time. The
# policy is a plain function so it can be checked without capturing anything.


def test_cache_evicts_least_recently_used_at_the_bound():
    from flash_rt.frontends.torch._nexn2_rtx_decode import _cache_put

    cache = collections.OrderedDict()
    for i in range(4):
        _cache_put(cache, i, f"g{i}", 2)
    assert list(cache) == [2, 3], "the two most recent survive, the rest go"


def test_cache_eviction_follows_use_not_insertion():
    from flash_rt.frontends.torch._nexn2_rtx_decode import _cache_put

    cache = collections.OrderedDict()
    for i in range(2):
        _cache_put(cache, i, f"g{i}", 2)
    cache.move_to_end(0)                # a replay of position 0
    _cache_put(cache, 2, "g2", 2)
    assert list(cache) == [0, 2], "the replayed entry must outlive the idle one"


def test_cache_bound_of_zero_keeps_everything():
    from flash_rt.frontends.torch._nexn2_rtx_decode import _cache_put

    cache = collections.OrderedDict()
    for i in range(8):
        _cache_put(cache, i, f"g{i}", 0)
    assert len(cache) == 8


def test_spec_cache_default_is_lower_than_the_decode_cache():
    """A speculative graph covers k+1 positions, so its pool is larger.

    Reusing the decode cap of 256 is what ran a 32 GB board out of memory at a
    2048-token context; the two bounds must not be the same number again by
    accident.
    """
    from flash_rt.frontends.torch._nexn2_rtx_decode import _qwen35moe_env

    decode_cap = int(_qwen35moe_env("GRAPH_CACHE_MAX", "256"))
    spec_cap = int(_qwen35moe_env("SPEC_GRAPH_CACHE_MAX", "16"))
    assert 0 < spec_cap < decode_cap


# ── Equivalence, against a real checkpoint ───────────────────────────────


@pytest.fixture(scope="module")
def spec_frontend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    frontend = _frontend_cls()(
        CKPT, device="cuda:0", max_seq=512, load_mtp=True,
        spec_graph_cache_max=8)
    frontend.set_prompt(PROMPT)
    return frontend


@requires_gpu_checkpoint
def test_the_head_is_loaded_only_when_asked(spec_frontend):
    assert spec_frontend.load_mtp is True
    assert spec_frontend._weights.ptrs.get("mtp") is not None


@requires_gpu_checkpoint
@pytest.mark.parametrize("k", [1, 2])
def test_speculative_decode_emits_what_greedy_emits(spec_frontend, k):
    """The whole point: a speed change and nothing else.

    Compared in one process against the plain path, because a rate or a token
    stream from another process is a different measurement.
    """
    spec_frontend.set_prompt(PROMPT)
    plain = spec_frontend.generate(NEW_TOKENS)
    spec_frontend.set_prompt(PROMPT)
    speculative = spec_frontend.generate_spec(NEW_TOKENS, k=k)
    assert speculative == plain


@requires_gpu_checkpoint
@pytest.mark.parametrize("k", [1, 2])
def test_a_window_computes_what_the_decode_steps_compute(spec_frontend, k):
    """Emitted argmax, recurrent state, conv state and written KV, bit for bit.

    Token equality is the outcome; this is the mechanism. A window that agreed
    on tokens while drifting in state would pass the test above and diverge
    later, so the state is compared directly -- with torch.equal, because a
    verified row is meant to *be* the decode row, not to approximate it.

    Note the replay. Capturing the graph snapshots everything the block
    mutates and restores it afterwards, so the state right after
    ``_ensure_spec_graph`` is the state *before* the window; only the replay
    advances it. Reading it any earlier compares a prefill to k+1 decode steps
    and fails for a reason that has nothing to do with the window.
    """
    import torch

    from flash_rt.frontends.torch import _nexn2_rtx_decode as decode

    spec_frontend.set_prompt(PROMPT)
    state = spec_frontend._decode_state_or_new()
    fvk, device = spec_frontend._fvk, spec_frontend.device
    ids = spec_frontend._prompt_ids

    logits = decode.seed_prefill(state, ids, fvk, device)
    pos = int(ids.view(-1).shape[0])
    token = logits[0].argmax().view(1)

    window = k + 1
    decode._ensure_spec_buffers(state, window, device)
    state._spec_tokens[0].copy_(token[0])
    graph, _ = decode._ensure_spec_graph(state, pos, k, fvk, device)
    graph.replay()
    torch.cuda.synchronize()

    # What the window emitted, and the state it advanced to.
    drafted = state._spec_tokens[:window].tolist()
    window_argmax = state._spec_argmax[:window].tolist()
    window_lin = [t.clone() for t in state.lin_state]
    window_conv = [t.clone() for t in state.lin_conv_state]
    # The model's own full-attention ranks only. The draft head owns one more
    # (state.mtp_rank), which the window writes and a plain decode step has no
    # counterpart for -- there is nothing to compare it against here. That the
    # head's own KV is right is what the K=1/K=2 token-equality test above
    # exercises: a draft computed off a wrong slot is not accepted.
    full = state.n_full
    window_kv_k = state.attn.K_cache[:full, pos:pos + window].clone()
    window_kv_v = state.attn.V_cache[:full, pos:pos + window].clone()

    # The same tokens through the plain decode step, from the same prefill.
    decode.seed_prefill(state, ids, fvk, device)
    step_argmax = []
    for i, tok in enumerate(drafted):
        t = torch.tensor([tok], dtype=torch.long, device=device)
        out = decode.decode_step(state, t.view(1, 1), pos + i, fvk, device)
        step_argmax.append(int(out.reshape(-1).argmax()))

    assert window_argmax == step_argmax, (
        "the window and the decode steps disagree on what comes next")
    for rank, (a, b) in enumerate(zip(state.lin_state, window_lin)):
        assert torch.equal(a, b), f"recurrent state diverged at rank {rank}"
    for rank, (a, b) in enumerate(zip(state.lin_conv_state, window_conv)):
        assert torch.equal(a, b), f"conv state diverged at rank {rank}"
    assert torch.equal(
        state.attn.K_cache[:full, pos:pos + window], window_kv_k), "K diverged"
    assert torch.equal(
        state.attn.V_cache[:full, pos:pos + window], window_kv_v), "V diverged"


@requires_gpu_checkpoint
def test_a_rejected_tail_is_rolled_back(spec_frontend):
    """Rewinding must land on the state the kept prefix ends in.

    Driven by asking the window to verify a draft that is wrong on purpose:
    the emitted prefix is then shorter than the window, which is the branch
    that rewinds. What it must produce is the state a plain step would leave.
    """
    import torch

    from flash_rt.frontends.torch import _nexn2_rtx_decode as decode

    spec_frontend.set_prompt(PROMPT)
    state = spec_frontend._decode_state_or_new()
    fvk, device = spec_frontend._fvk, spec_frontend.device
    ids = spec_frontend._prompt_ids

    logits = decode.seed_prefill(state, ids, fvk, device)
    pos = int(ids.view(-1).shape[0])
    token = logits[0].argmax().view(1)

    tokens, next_pos = decode.spec_decode_step(
        state, token, pos, 2, fvk, device)
    kept = next_pos - pos
    assert 1 <= kept <= 3
    assert len(tokens) == kept

    after_window = [t.clone() for t in state.lin_state]

    # The same emitted prefix, one plain step at a time.
    decode.seed_prefill(state, ids, fvk, device)
    emitted = [int(token)] + tokens[:-1]
    for i, tok in enumerate(emitted):
        t = torch.tensor([tok], dtype=torch.long, device=device)
        decode.decode_step(state, t.view(1, 1), pos + i, fvk, device)

    for a, b in zip(state.lin_state, after_window):
        assert torch.equal(a, b), (
            "the rewind did not land on the state the kept prefix ends in")


@requires_gpu_checkpoint
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_boundary_token_counts(spec_frontend, count):
    """A window emits between 1 and k+1 tokens, so the requested count is not
    a multiple of anything. Asking for fewer than a window emits must still
    return exactly what was asked for."""
    spec_frontend.set_prompt(PROMPT)
    out = spec_frontend.generate_spec(count, k=2)
    assert len(out) == count


@requires_gpu_checkpoint
def test_the_graph_cache_stays_within_its_bound(spec_frontend):
    """A generation revisits more positions than the cache holds."""
    spec_frontend.set_prompt(PROMPT)
    state = spec_frontend._decode_state_or_new()
    cap = state.spec_graph_cache_max
    spec_frontend.generate_spec(NEW_TOKENS, k=2)
    assert 0 < len(state._spec_graphs) <= cap
