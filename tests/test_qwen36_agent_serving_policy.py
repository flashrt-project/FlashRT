import pytest

from serving.qwen36_agent.openai_stream import sse_from_events
from serving.qwen36_agent.prefix import longest_common_prefix
from serving.qwen36_agent.qwen36_engine import Qwen36FrontendAgentEngine
from serving.qwen36_agent.engine import DecodeChunk
from serving.qwen36_agent.service import (
    AgentRequest,
    AgentService,
    request_from_openai,
    result_to_openai,
)
from serving.qwen36_agent.session import SessionRegistry
from serving.qwen36_agent.tool_stream import ToolCallStreamParser


def test_prefix_match_distinguishes_append_truncate_and_diverge():
    assert longest_common_prefix([1, 2], [1, 2, 3]).append_only
    assert longest_common_prefix([1, 2], [1, 2]).exact
    assert longest_common_prefix([1, 2, 3], [1, 2]).matched == 2
    assert longest_common_prefix([1, 9], [1, 2]).divergent


def test_session_registry_plans_incremental_agent_turns():
    reg = SessionRegistry(max_sessions=2)
    rec, plan0 = reg.plan_request("s1", [10, 11, 12])
    assert plan0.action == "append"
    assert plan0.cached_tokens == 0
    assert plan0.new_prefill_tokens == 3

    rec.commit([10, 11, 12])
    rec2, plan1 = reg.plan_request("s1", [10, 11, 12, 13, 14])
    assert rec2 is rec
    assert plan1.action == "append"
    assert plan1.cached_tokens == 3
    assert plan1.new_prefill_tokens == 2

    _, plan2 = reg.plan_request("s1", [10, 11])
    assert plan2.action == "truncate"
    assert plan2.cached_tokens == 2

    _, plan3 = reg.plan_request("s1", [10, 99])
    assert plan3.action == "rebuild"
    assert plan3.cached_tokens == 0


def test_session_registry_lru_eviction_keeps_hot_session():
    reg = SessionRegistry(max_sessions=2)
    reg.create(session_id="a")
    reg.create(session_id="b")
    reg.mark_hot("a")
    reg.create(session_id="c")
    snap = reg.snapshot()
    ids = [s["session_id"] for s in snap["sessions"]]
    assert "a" in ids
    assert "c" in ids
    assert "b" not in ids


def test_tool_stream_parser_holds_partial_tags_and_json():
    p = ToolCallStreamParser()
    out = p.feed("hello <tool")
    assert [(e.kind, e.payload) for e in out] == [("text", "hello ")]

    out = p.feed('_call>{"name":"search","arguments":{"q":"x"}}')
    assert out == []

    out = p.feed("</tool_call> done")
    assert len(out) == 2
    assert out[0].kind == "tool_call"
    tc = out[0].payload
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == '{"q":"x"}'
    assert (out[1].kind, out[1].payload) == ("text", " done")

    tail = p.finish()
    assert tail == []


def test_tool_stream_parser_accepts_qwen_xml_function_calls():
    p = ToolCallStreamParser()

    out = p.feed(
        "<tool_call>\n"
        "<function=write>\n"
        "<parameter=path>\n"
        "merge_sort.py\n"
        "</parameter>\n"
        "<parameter=content>\n"
        "print('ok')\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>")

    assert len(out) == 1
    assert out[0].kind == "tool_call"
    tc = out[0].payload
    assert tc["function"]["name"] == "write"
    assert tc["function"]["arguments"] == (
        "{\"path\":\"merge_sort.py\",\"content\":\"print('ok')\"}")


def test_sse_stream_contains_role_tool_call_and_done():
    p = ToolCallStreamParser()
    events = []
    events.extend(p.feed('x<tool_call>{"name":"run","arguments":{}}</tool_call>'))
    events.extend(p.finish())
    chunks = list(sse_from_events("chatcmpl-test", "qwen", events,
                                  finish_reason="tool_calls"))
    joined = "".join(chunks)
    assert '"role":"assistant"' in joined
    assert '"tool_calls"' in joined
    assert '"finish_reason":"tool_calls"' in joined
    assert chunks[-1] == "data: [DONE]\n\n"


class FakeAgentEngine:
    model_name = "fake-qwen36"
    max_seq = 262208

    def __init__(self):
        self.prefills = []
        self.outputs = [
            DecodeChunk((ord("h"),), "h", 1),
            DecodeChunk((ord("i"),), "i", 1),
        ]

    def tokenize_chat(self, messages, tools=None, *, enable_thinking=False):
        del tools, enable_thinking
        out = []
        for msg in messages:
            out.extend(ord(ch) for ch in (msg.get("content") or ""))
            if msg.get("role") != "assistant":
                out.append(0)
        return out

    def prefill(self, token_ids, *, cached_tokens=0, max_tokens=1, K=6):
        self.prefills.append((list(token_ids), cached_tokens, max_tokens, K))

    def generate_stream(self, *, max_tokens, K):
        del K
        yield from self.outputs[:max_tokens]


def test_agent_service_reuses_exact_session_prefix_when_history_is_returned():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    req0 = AgentRequest(
        session_id="agent-1",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=2,
    )
    res0 = svc.complete(req0)
    assert res0.stats.cached_tokens == 0
    assert res0.stats.new_prefill_tokens == 4
    assert engine.prefills[-1][1:] == (0, 2, 6)
    assert res0.text == "hi"
    assert res0.finish_reason == "stop"

    req1 = AgentRequest(
        session_id="agent-1",
        messages=[
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "de"},
        ],
        max_tokens=1,
    )
    res1 = svc.complete(req1)
    assert res1.prefix_plan.action == "append"
    assert res1.stats.cached_tokens == 6
    assert res1.stats.new_prefill_tokens == 3
    assert engine.prefills[-1][1:] == (6, 1, 6)


def test_agent_service_uses_message_append_when_visible_history_hides_tokens():
    class HiddenEngine(FakeAgentEngine):
        def __init__(self):
            super().__init__()
            self.outputs = [DecodeChunk((999, ord("h")), "h", 2)]

        def append_suffix_tokens_for_messages(
                self, previous, incoming, *, tools=None,
                enable_thinking=False):
            del tools, enable_thinking
            assert previous == [
                {"role": "user", "content": "abc"},
                {"role": "assistant", "content": "h"},
            ]
            assert incoming[:len(previous)] == previous
            return [42, 43]

    engine = HiddenEngine()
    svc = AgentService(engine)
    res0 = svc.complete(AgentRequest(
        session_id="agent-hidden",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=1,
    ))
    assert res0.text == "h"

    res1 = svc.complete(AgentRequest(
        session_id="agent-hidden",
        messages=[
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "h"},
            {"role": "user", "content": "next"},
        ],
        max_tokens=1,
    ))

    assert res1.prefix_plan.action == "message_append"
    assert res1.stats.cached_tokens == 6
    assert res1.stats.new_prefill_tokens == 2
    assert engine.prefills[-1][0][:6] == [ord("a"), ord("b"), ord("c"), 0, 999, ord("h")]
    assert engine.prefills[-1][0][6:] == [42, 43]
    assert engine.prefills[-1][1:] == (6, 1, 6)


def test_agent_service_rebuilds_token_journal_without_hot_state():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    rec = svc.sessions.create(session_id="cold")
    rec.commit([ord("a"), 0])

    res = svc.complete(AgentRequest(
        session_id="cold",
        messages=[{"role": "user", "content": "a"}],
        max_tokens=1,
    ))
    assert res.prefix_plan.action == "activate_rebuild"
    assert res.stats.cached_tokens == 0
    assert engine.prefills[-1][1:] == (0, 1, 6)


def test_agent_service_rebuilds_truncate_without_rollback():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    rec = svc.sessions.create(session_id="hot")
    rec.commit([ord("a"), 0, ord("h"), ord("i")])
    svc.sessions.mark_hot("hot")

    res = svc.complete(AgentRequest(
        session_id="hot",
        messages=[{"role": "user", "content": "a"}],
        max_tokens=1,
    ))
    assert res.prefix_plan.action == "activate_rebuild"
    assert res.stats.cached_tokens == 0
    assert engine.prefills[-1][1:] == (0, 1, 6)


def test_agent_service_parses_tool_calls_from_generated_stream():
    engine = FakeAgentEngine()
    engine.outputs = [
        DecodeChunk((1000,), "hello ", 1),
        DecodeChunk((1001,), '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>', 1),
    ]
    svc = AgentService(engine)
    res = svc.complete(AgentRequest(
        session_id="agent-tools",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=2,
    ))
    assert res.text == "hello "
    assert res.finish_reason == "tool_calls"
    assert res.tool_calls[0]["function"]["name"] == "lookup"


def test_agent_service_stops_decode_after_tool_call_to_keep_session_hot():
    engine = FakeAgentEngine()
    engine.outputs = [
        DecodeChunk((1001,),
                    '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>',
                    1),
        DecodeChunk((999, ord("x")), "", 0, stop=True, state_lookahead=2),
    ]
    svc = AgentService(engine)

    res = svc.complete(AgentRequest(
        session_id="agent-tools-hot",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=8,
    ))

    assert res.finish_reason == "tool_calls"
    assert res.usage["completion_tokens"] == 1
    assert svc.sessions.hot_session_id == "agent-tools-hot"
    msg = svc.sessions.get("agent-tools-hot").visible_messages[-1]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "lookup"


def test_agent_service_stream_stops_decode_after_tool_call_to_keep_session_hot():
    engine = FakeAgentEngine()
    engine.outputs = [
        DecodeChunk((1001,),
                    '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>',
                    1),
        DecodeChunk((999, ord("x")), "", 0, stop=True, state_lookahead=2),
    ]
    svc = AgentService(engine)

    chunks = list(svc.stream_openai(AgentRequest(
        session_id="agent-tools-stream-hot",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=8,
    ), model=engine.model_name))

    joined = "".join(chunks)
    assert '"tool_calls"' in joined
    assert '"completion_tokens":1' in joined
    assert svc.sessions.hot_session_id == "agent-tools-stream-hot"


def test_agent_service_stream_tool_call_close_after_delta_keeps_hot_session():
    engine = FakeAgentEngine()
    engine.outputs = [
        DecodeChunk((1001,),
                    '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>',
                    1),
        DecodeChunk((999, ord("x")), "", 0, stop=True, state_lookahead=2),
    ]
    svc = AgentService(engine)

    gen = svc.stream_openai(AgentRequest(
        session_id="agent-tools-close-hot",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=8,
    ), model=engine.model_name)
    next(gen)                 # role chunk
    tool_delta = next(gen)    # tool_call delta, emitted after commit/mark hot
    assert '"tool_calls"' in tool_delta
    assert svc.sessions.hot_session_id == "agent-tools-close-hot"
    gen.close()
    assert svc.sessions.hot_session_id == "agent-tools-close-hot"


def test_agent_service_stream_openai_yields_live_sse_chunks():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    chunks = list(svc.stream_openai(AgentRequest(
        session_id="agent-stream",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=2,
    ), model=engine.model_name))
    joined = "".join(chunks)
    assert '"role":"assistant"' in joined
    assert '"content":"h"' in joined
    assert '"content":"i"' in joined
    assert '"completion_tokens":2' in joined
    assert chunks[-1] == "data: [DONE]\n\n"
    assert svc.sessions.hot_session_id == "agent-stream"


def test_agent_service_stream_openai_reuses_hot_session_prefix():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    list(svc.stream_openai(AgentRequest(
        session_id="agent-stream-cache",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=2,
    ), model=engine.model_name))

    list(svc.stream_openai(AgentRequest(
        session_id="agent-stream-cache",
        messages=[
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "z"},
        ],
        max_tokens=1,
    ), model=engine.model_name))

    assert engine.prefills[-1][1:] == (6, 1, 6)


def test_agent_service_pin_prefix_fails_without_capsule_budget():
    engine = FakeAgentEngine()
    svc = AgentService(engine)

    with pytest.raises(ValueError, match="capsule-budget"):
        svc.complete(AgentRequest(
            session_id="pin-no-budget",
            messages=[{"role": "user", "content": "abc"}],
            max_tokens=1,
            pin_prefix=1,
        ))


def test_agent_service_pin_prefix_fails_when_long_route_unavailable():
    class CapsuleButNoLongRouteEngine(FakeAgentEngine):
        def supports_capsule(self):
            return True

        def capsule_aligned_len(self, prompt_len, max_tokens):
            del prompt_len, max_tokens
            return 0

    engine = CapsuleButNoLongRouteEngine()
    svc = AgentService(engine, capsule_budget_bytes=1024)

    with pytest.raises(ValueError, match="long FP8-KV route"):
        svc.complete(AgentRequest(
            session_id="pin-no-long-route",
            messages=[{"role": "user", "content": "abc"}],
            max_tokens=1,
            pin_prefix=1,
        ))


def test_qwen36_agent_fastapi_non_stream_and_stream_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from serving.qwen36_agent.server import build_app

    engine = FakeAgentEngine()
    app = build_app(AgentService(engine))
    client = TestClient(app)

    model_resp = client.get("/v1/models")
    assert model_resp.status_code == 200
    assert model_resp.json()["data"][0]["id"] == "fake-qwen36"

    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "abc"}],
        "max_completion_tokens": 2,
        "flashrt_session_id": "http-session",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["flashrt"]["prefix_action"] == "append"

    stream_resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "def"}],
        "max_tokens": 2,
        "stream": True,
        "flashrt_session_id": "http-stream",
    })
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")
    text = stream_resp.text
    assert '"role":"assistant"' in text
    assert '"content":"h"' in text
    assert "data: [DONE]" in text


def test_qwen36_agent_fastapi_rejects_output_above_service_cap():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from serving.qwen36_agent.server import build_app

    app = build_app(AgentService(
        FakeAgentEngine(), default_max_tokens=4, max_output_tokens=8))
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "abc"}],
        "max_tokens": 9,
    })

    assert resp.status_code == 400
    assert "max_tokens must be <= 8" in resp.text


def test_agent_service_applies_default_session_id_for_openai_clients():
    svc = AgentService(FakeAgentEngine(), default_session_id="local-agent")

    req = svc.request_from_openai({
        "messages": [{"role": "user", "content": "abc"}],
        "max_tokens": 1,
    })

    assert req.session_id == "local-agent"


def test_agent_service_keeps_explicit_session_over_default_session_id():
    svc = AgentService(FakeAgentEngine(), default_session_id="local-agent")

    req = svc.request_from_openai({
        "messages": [{"role": "user", "content": "abc"}],
        "max_tokens": 1,
        "flashrt_session_id": "explicit",
    })

    assert req.session_id == "explicit"


def test_openai_request_and_response_include_flashrt_cache_metrics():
    engine = FakeAgentEngine()
    svc = AgentService(engine)
    req = request_from_openai({
        "messages": [{"role": "user", "content": "a"}],
        "max_completion_tokens": 1,
        "stream": "false",
        "temperature": 0.7,
        "top_p": 0.9,
        "tool_choice": "auto",
        "flashrt_session_id": "s",
    })
    assert req.max_tokens == 1
    res = svc.complete(req)
    body = result_to_openai(res, model=engine.model_name)
    assert body["model"] == "fake-qwen36"
    assert body["flashrt"]["session_id"] == "s"
    assert body["flashrt"]["new_prefill_tokens"] == 2


def test_openai_request_uses_configured_default_and_output_cap():
    req = request_from_openai({
        "messages": [{"role": "user", "content": "a"}],
    }, default_max_tokens=123, max_output_tokens=256)
    assert req.max_tokens == 123

    req = request_from_openai({
        "messages": [{"role": "user", "content": "a"}],
        "max_completion_tokens": 77,
    }, default_max_tokens=123, max_output_tokens=256)
    assert req.max_tokens == 77

    with pytest.raises(ValueError, match="max_tokens must be <= 256"):
        request_from_openai({
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 257,
        }, default_max_tokens=123, max_output_tokens=256)


class FakeTokenizer:
    def __init__(self):
        self.template_kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        self.messages = messages
        return "|".join((m.get("content") or "") for m in messages)

    def __call__(self, prompt, add_special_tokens=False):
        del add_special_tokens

        class Encoded:
            input_ids = [ord(ch) for ch in prompt]

        return Encoded()

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(i) for i in ids)


class FakeFrontend:
    device = "cpu"
    _user_max_seq = 128
    _long_ctx_mode = False

    def __init__(self):
        self._tokenizer = FakeTokenizer()
        self.prefill_args = None
        self.append_args = None
        self.long_prefill_args = None
        self.long_append_args = None

    def prefill_own_speculative_nvfp4_agent(self, input_ids, *,
                                            max_new_tokens, K):
        self.prefill_args = (input_ids.tolist(), max_new_tokens, K)

    def append_own_speculative_nvfp4_agent(self, input_ids, *,
                                           start_pos, max_new_tokens, K):
        self.append_args = (
            input_ids.tolist(), start_pos, max_new_tokens, K)

    def prefill_long_ctx_nvfp4_agent(self, input_ids, *,
                                     max_new_tokens, K):
        self.long_prefill_args = (input_ids.tolist(), max_new_tokens, K)

    def append_long_ctx_nvfp4_agent(self, input_ids, *,
                                    start_pos, max_new_tokens, K):
        self.long_append_args = (
            input_ids.tolist(), start_pos, max_new_tokens, K)

    def decode_own_speculative_nvfp4_committed_stream(self, *,
                                                      max_new_tokens, K):
        del K
        for i in range(max_new_tokens):
            yield (ord("a") + i,)

    def decode_long_ctx_nvfp4_committed_stream(self, *, max_new_tokens, K):
        del K
        for i in range(max_new_tokens):
            yield (ord("x") + i,)


def test_qwen36_frontend_agent_engine_wires_short_committed_split():
    fe = FakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe, model_name="fake")

    ids = engine.tokenize_chat(
        [{"role": "assistant", "content": None},
         {"role": "user", "content": "go"}],
        enable_thinking=True,
    )
    assert ids == [ord("|"), ord("g"), ord("o")]
    assert fe._tokenizer.template_kwargs["enable_thinking"] is True

    engine.prefill(ids, cached_tokens=0, max_tokens=2, K=4)
    assert fe.prefill_args == ([[ord("|"), ord("g"), ord("o")]], 2, 4)

    chunks = list(engine.generate_stream(max_tokens=2, K=4))
    assert [c.token_ids for c in chunks] == [(ord("a"),), (ord("b"),)]
    assert "".join(c.text for c in chunks) == "ab"


def test_qwen36_frontend_agent_engine_hides_think_tags_by_default():
    class ThinkTokenizer(FakeTokenizer):
        def decode(self, ids, skip_special_tokens=False):
            del ids, skip_special_tokens
            return "<think>\n\n</think>\n\nanswer"

    class ThinkFrontend(FakeFrontend):
        def __init__(self):
            super().__init__()
            self._tokenizer = ThinkTokenizer()

        def decode_own_speculative_nvfp4_committed_stream(self, *,
                                                          max_new_tokens, K):
            del max_new_tokens, K
            yield (1,)

    engine = Qwen36FrontendAgentEngine(ThinkFrontend())
    engine.tokenize_chat([{"role": "user", "content": "go"}])

    chunks = list(engine.generate_stream(max_tokens=1, K=4))

    assert chunks[0].text == "answer"


def test_qwen36_frontend_agent_engine_normalizes_openai_tool_call_history():
    fe = FakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe)

    engine.render_chat([{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": "{\"path\":\"x.txt\",\"content\":\"ok\"}",
            },
        }],
    }])

    tool_call = fe._tokenizer.messages[0]["tool_calls"][0]
    assert tool_call["function"]["arguments"] == {
        "path": "x.txt",
        "content": "ok",
    }


def test_qwen36_frontend_agent_engine_stops_visible_text_at_im_end():
    class StopTokenizer(FakeTokenizer):
        eos_token_id = 999
        pad_token_id = None

        def convert_tokens_to_ids(self, token):
            return 999 if token == "<|im_end|>" else -1

        def decode(self, ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(i) for i in ids if i != 999)

    class StopFrontend(FakeFrontend):
        def __init__(self):
            super().__init__()
            self._tokenizer = StopTokenizer()

        def decode_own_speculative_nvfp4_committed_stream(self, *,
                                                          max_new_tokens, K):
            del max_new_tokens, K
            yield (ord("o"), ord("k"), 999, ord("u"))
            yield (ord("n"),)

    engine = Qwen36FrontendAgentEngine(StopFrontend())
    engine.tokenize_chat([{"role": "user", "content": "go"}])

    chunks = list(engine.generate_stream(max_tokens=8, K=4))

    assert len(chunks) == 1
    # Only the visible tokens are committed to the journal; the stop token (999)
    # and the post-stop tail ('u') are reported as state lookahead, not journaled.
    assert chunks[0].token_ids == (ord("o"), ord("k"))
    assert chunks[0].text == "ok"
    assert chunks[0].stop is True
    assert chunks[0].state_lookahead == 2


def test_qwen36_frontend_agent_engine_wires_short_append_split():
    fe = FakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe)

    engine.prefill([1, 2, 3], cached_tokens=2, max_tokens=1, K=4)
    assert fe.append_args == ([[1, 2, 3]], 2, 1, 4)


def test_qwen36_frontend_agent_engine_wires_long_cold_split():
    class LongFakeFrontend(FakeFrontend):
        _long_ctx_mode = True

        def _should_use_long_ctx_route(self, prompt_len, max_tokens):
            return prompt_len + max_tokens > 4

    fe = LongFakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe)

    engine.prefill([1, 2, 3, 4], cached_tokens=0, max_tokens=2, K=5)
    assert fe.long_prefill_args == ([[1, 2, 3, 4]], 2, 5)
    chunks = list(engine.generate_stream(max_tokens=2, K=5))
    assert [c.token_ids for c in chunks] == [(ord("x"),), (ord("y"),)]


def test_qwen36_frontend_agent_engine_wires_long_append_split():
    class LongFakeFrontend(FakeFrontend):
        _long_ctx_mode = True

        def _should_use_long_ctx_route(self, prompt_len, max_tokens):
            return True

    fe = LongFakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe)
    engine.prefill([1, 2, 3], cached_tokens=2, max_tokens=1, K=4)
    assert fe.long_append_args == ([[1, 2, 3]], 2, 1, 4)


def test_qwen36_frontend_agent_engine_warmup_runs_committed_stream():
    fe = FakeFrontend()
    engine = Qwen36FrontendAgentEngine(fe)

    warmed = engine.warmup_committed_stream(
        [(4, 3)], K=4, committed_max_prompt=8)

    assert len(warmed) == 1
    assert warmed[0]["route"] == "short"
    assert fe.prefill_args[1:] == (3, 4)


def test_qwen36_frontend_agent_engine_warmup_uses_long_graph_hooks():
    class LongWarmFrontend(FakeFrontend):
        _long_ctx_mode = True
        _user_max_seq = 4096

        def __init__(self):
            super().__init__()
            self.decode_warm_shapes = None
            self.prefill_warm_shapes = None

        def _should_use_long_ctx_route(self, prompt_len, max_tokens):
            return prompt_len >= 16

        def warmup_long_ctx_decode_graphs(self, shapes, K=6):
            self.decode_warm_shapes = (list(shapes), K)
            return [(16, 4, K)]

        def warmup_long_ctx_prefill_graphs(self, shapes):
            self.prefill_warm_shapes = list(shapes)
            return [(0, 16, "last")]

    fe = LongWarmFrontend()
    engine = Qwen36FrontendAgentEngine(fe)

    warmed = engine.warmup_committed_stream(
        [(16, 4)],
        K=5,
        committed_max_prompt=8,
        long_prefill_graphs=True,
    )

    assert warmed[0]["route"] == "long_graphs"
    assert warmed[0]["prefill_graphs"] == 1
    assert warmed[0]["decode_graphs"] == 1
    assert fe.prefill_warm_shapes == [(16, 4)]
    assert fe.decode_warm_shapes == ([(16, 4)], 5)


def test_qwen36_engine_trims_visible_tokens_and_flags_state_lookahead():
    """A stop token mid-chunk: commit only visible tokens, report the post-stop
    tokens (eos + verified tail) as state lookahead."""
    class StopTokenizer(FakeTokenizer):
        eos_token_id = 7

        def convert_tokens_to_ids(self, token):
            del token
            return -1

    class StopFrontend(FakeFrontend):
        def __init__(self):
            super().__init__()
            self._tokenizer = StopTokenizer()

        def decode_own_speculative_nvfp4_committed_stream(self, *,
                                                          max_new_tokens, K):
            del max_new_tokens, K
            yield (ord("a"), ord("b"), 7, ord("c"))   # eos at index 2, extra 'c'

    engine = Qwen36FrontendAgentEngine(StopFrontend(), model_name="fake")
    chunks = list(engine.generate_stream(max_tokens=16, K=3))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.token_ids == (ord("a"), ord("b"))   # visible only
    assert chunk.text == "ab"
    assert chunk.stop is True
    assert chunk.state_lookahead == 2                # eos + 'c' committed to KV


def test_agent_service_stop_lookahead_trims_journal_and_forces_rebuild():
    """Post-stop committed tokens must not enter the journal, and the session
    must not be hot-appendable (the GPU state leads the visible transcript)."""
    class StopEngine(FakeAgentEngine):
        def generate_stream(self, *, max_tokens, K):
            del max_tokens, K
            yield DecodeChunk((ord("h"),), "h", 1)
            yield DecodeChunk((ord("i"),), "i", 1, stop=True, state_lookahead=2)

    svc = AgentService(StopEngine())
    res = svc.complete(AgentRequest(
        session_id="s", messages=[{"role": "user", "content": "abc"}],
        max_tokens=4))

    assert res.text == "hi"
    session = svc.sessions.get("s")
    # journal = prompt (abc + role sep) + visible generated; no post-stop tokens
    assert session.token_ids == [ord("a"), ord("b"), ord("c"), 0,
                                 ord("h"), ord("i")]
    assert svc.sessions.hot_session_id is None       # not hot -> next turn rebuilds

    res2 = svc.complete(AgentRequest(
        session_id="s",
        messages=[{"role": "user", "content": "abc"},
                  {"role": "assistant", "content": "hi"},
                  {"role": "user", "content": "x"}],
        max_tokens=1))
    assert res2.prefix_plan.action in ("rebuild", "activate_rebuild")
    assert res2.stats.cached_tokens == 0


def test_agent_service_serializes_concurrent_requests():
    """The service lock prevents two requests from interleaving on the shared
    mutable frontend state."""
    import threading
    import time

    state = {"active": 0, "max": 0}

    class SlowEngine(FakeAgentEngine):
        def generate_stream(self, *, max_tokens, K):
            del max_tokens, K
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            state["active"] -= 1
            yield DecodeChunk((ord("h"),), "h", 1)

    svc = AgentService(SlowEngine())
    start = threading.Barrier(2)

    def call():
        start.wait()
        svc.complete(AgentRequest(
            session_id=None, messages=[{"role": "user", "content": "a"}],
            max_tokens=1))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max"] == 1   # serialized: never two generators at once


def test_stream_disconnect_clears_hot_session():
    """A client disconnect closes the stream generator before the final commit;
    the GPU state advanced, so the hot session must be cleared (force rebuild)."""
    svc = AgentService(FakeAgentEngine())
    svc.sessions.hot_session_id = "stale-prev"   # a hot session from a previous turn     # a hot session from a previous turn

    gen = svc.stream_openai(
        AgentRequest(session_id="s",
                     messages=[{"role": "user", "content": "a"}],
                     max_tokens=2),
        model="m")
    next(gen)                                # start: prefill + first SSE chunk
    gen.close()                              # simulate mid-stream disconnect

    assert svc.sessions.hot_session_id is None


def test_complete_exception_clears_hot_session():
    """If generation raises mid-request, the hot session is cleared so the next
    turn rebuilds instead of appending onto advanced GPU state."""
    class BoomEngine(FakeAgentEngine):
        def generate_stream(self, *, max_tokens, K):
            del max_tokens, K
            raise RuntimeError("decode blew up")
            yield  # pragma: no cover

    svc = AgentService(BoomEngine())
    svc.sessions.hot_session_id = "stale-prev"   # a hot session from a previous turn

    with pytest.raises(RuntimeError):
        svc.complete(AgentRequest(
            session_id="s", messages=[{"role": "user", "content": "a"}],
            max_tokens=1))

    assert svc.sessions.hot_session_id is None


def test_stream_full_consume_keeps_hot_session():
    """A normally completed stream still marks the session hot (no false clear)."""
    svc = AgentService(FakeAgentEngine())
    chunks = list(svc.stream_openai(
        AgentRequest(session_id="s",
                     messages=[{"role": "user", "content": "a"}],
                     max_tokens=2),
        model="m"))
    assert chunks[-1] == "data: [DONE]\n\n"
    assert svc.sessions.hot_session_id == "s"
