from serving.qwen36_agent.openai_stream import sse_from_events
from serving.qwen36_agent.prefix import longest_common_prefix
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
