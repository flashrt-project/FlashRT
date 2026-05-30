"""Optional GPU test for the serving-layer capsule pin/restore policy.

The frontend capsule API is gated bit-exact in test_qwen36_agent_capsule.py; this
gates the *serving wiring* (CapsuleStore + flashrt_pin_prefix + the "restore"
PrefixPlan action): a request that restores a pinned shared-prefix capsule must
produce the same tokens as a cold full prefill of the same prompt, while
re-prefilling only the suffix after the chunk-aligned pin boundary.

Run manually with:

    FLASHRT_QWEN36_NVFP4_CKPT_DIR=... \
    FLASHRT_QWEN36_MTP_CKPT_DIR=... \
    pytest -q tests/test_qwen36_agent_capsule_serving.py -s
"""

from __future__ import annotations

import os

import pytest


CKPT = os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR", "")
MTP = os.environ.get("FLASHRT_QWEN36_MTP_CKPT_DIR", "")


pytestmark = pytest.mark.skipif(
    not CKPT or not MTP,
    reason=(
        "set FLASHRT_QWEN36_NVFP4_CKPT_DIR and "
        "FLASHRT_QWEN36_MTP_CKPT_DIR to run Qwen3.6 capsule-serving tests"
    ),
)


def _service(monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("FLASHRT_QWEN36_LONG_KV_CACHE", "fp8")
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx
    from serving.qwen36_agent.qwen36_engine import Qwen36FrontendAgentEngine
    from serving.qwen36_agent.service import AgentService
    from serving.qwen36_agent.session import SessionRegistry, CapsuleStore

    fe = Qwen36TorchFrontendRtx(CKPT, quant="nvfp4", device="cuda", max_seq=32768)
    if getattr(fe, "_long_ctx_mode", False):
        fe._long_ctx_route_min_seq = 0
    if not fe._long_ctx_mode or getattr(fe, "_long_kv_cache_mode", "") != "fp8":
        pytest.skip("long fp8-KV route required for capsule serving")
    eng = Qwen36FrontendAgentEngine(fe)
    if not eng.supports_capsule():
        pytest.skip("frontend has no capsule API")
    store = CapsuleStore(budget_bytes=8 << 30)

    def serve(messages, *, pin):
        svc = AgentService(eng, sessions=SessionRegistry())
        svc.capsules = store
        from serving.qwen36_agent.service import AgentRequest
        return svc.complete(AgentRequest(
            messages=messages, max_tokens=32, session_id="t", K=6,
            pin_prefix=pin))

    return fe, serve, store


def _shared_system(fe, approx_tokens: int = 6000) -> str:
    unit = ("FlashRT is a latency-first inference runtime. Capsules freeze a "
            "committed execution boundary so a shared prefix is restored, not "
            "re-prefilled. ")
    ids = fe._tokenizer(unit * 200, add_special_tokens=False).input_ids
    return fe._tokenizer.decode(ids[:approx_tokens])


def test_serving_capsule_restore_matches_cold_full_prefill(monkeypatch):
    fe, serve, store = _service(monkeypatch)
    sys_text = _shared_system(fe, 6000)
    pin_n = 6000  # aligned down to a chunk multiple (>= one 2048 chunk)

    msgs_a = [{"role": "system", "content": sys_text},
              {"role": "user", "content": "Summarize the text in one sentence."}]
    msgs_b = [{"role": "system", "content": sys_text},
              {"role": "user", "content": "List two ideas from the text."}]

    r_pin = serve(msgs_a, pin=pin_n)          # cold + pin the aligned prefix
    assert r_pin.prefix_plan.action == "pin"
    assert store.footprint() > 0

    r_restore = serve(msgs_b, pin=pin_n)      # restore + append only the suffix
    assert r_restore.prefix_plan.action == "restore"
    assert r_restore.stats.cached_tokens > 0
    assert r_restore.stats.cached_tokens % fe.long_prefill_chunk_size() == 0
    # only the suffix after the pin boundary is re-prefilled
    assert r_restore.stats.new_prefill_tokens < r_pin.stats.new_prefill_tokens

    r_cold = serve(msgs_b, pin=None)          # cold full prefill of the same prompt
    assert r_restore.text == r_cold.text
    assert (r_restore.usage["completion_tokens"]
            == r_cold.usage["completion_tokens"])


def test_serving_capsule_disabled_by_default(monkeypatch):
    """With no budget, flashrt_pin_prefix is inert and the path is unchanged."""
    fe, _, _ = _service(monkeypatch)
    from serving.qwen36_agent.qwen36_engine import Qwen36FrontendAgentEngine
    from serving.qwen36_agent.service import AgentService, AgentRequest
    from serving.qwen36_agent.session import SessionRegistry

    eng = Qwen36FrontendAgentEngine(fe)
    svc = AgentService(eng, sessions=SessionRegistry())  # budget 0
    assert not svc.capsules.enabled
    res = svc.complete(AgentRequest(
        messages=[{"role": "user", "content": "Hello."}],
        max_tokens=8, session_id="t", K=6, pin_prefix=4096))
    assert res.prefix_plan.action != "restore"
    assert res.prefix_plan.action != "pin"
