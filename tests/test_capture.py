from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from uqguard import CaptureMiddleware, ModelFallbackMiddleware, read_trace

BOOK_CALL = {"name": "book_flight", "args": {"flight_id": "F1"}, "id": "x", "type": "tool_call"}


def _req(messages=None, model=None):
    return ModelRequest(model=model, messages=messages or [HumanMessage("book it")])


def test_capture_k_samples_and_tool_result_backfill(tmp_path):
    mw = CaptureMiddleware(k=3, trace_dir=tmp_path, run_id="t")
    mw.new_thread("task1")

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="booking", tool_calls=[BOOK_CALL])])

    resp = mw.wrap_model_call(_req(), handler)
    assert calls == 3  # k samples
    assert isinstance(resp, ModelResponse)

    # next model call carries the tool result -> backfills and flushes step 1
    followup = [
        HumanMessage("book it"),
        AIMessage(content="", tool_calls=[BOOK_CALL]),
        ToolMessage("Booked F1, confirmation B1.", tool_call_id="x"),
    ]
    mw.wrap_model_call(_req(followup), lambda r: ModelResponse(result=[AIMessage("done")]))
    mw.after_agent({"messages": [*followup, AIMessage("done")]}, None)

    steps = read_trace(tmp_path / "t.jsonl")
    assert len(steps) == 2
    assert [c.tool_name for c in steps[0].candidates] == ["book_flight"] * 3
    assert steps[0].chosen.args == {"flight_id": "F1"}
    assert steps[0].tool_result == "Booked F1, confirmation B1."
    assert steps[0].step_id == "task1/0" and steps[0].thread_id == "task1"
    # final step: answered in text, no tool, no tool result
    assert steps[1].chosen.tool_name == "__none__"
    assert steps[1].tool_result is None


def test_new_thread_flushes_pending(tmp_path):
    mw = CaptureMiddleware(k=1, trace_dir=tmp_path, run_id="t")
    mw.new_thread("a")
    mw.wrap_model_call(_req(), lambda r: ModelResponse(result=[AIMessage("hi")]))
    mw.new_thread("b")  # must flush a's pending step
    steps = read_trace(tmp_path / "t.jsonl")
    assert len(steps) == 1 and steps[0].thread_id == "a"


def test_async_capture_samples_concurrently(tmp_path):
    import asyncio

    mw = CaptureMiddleware(k=3, trace_dir=tmp_path, run_id="t")
    mw.new_thread("task1")
    in_flight, peak = 0, 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ModelResponse(result=[AIMessage(content="", tool_calls=[BOOK_CALL])])

    resp = asyncio.run(mw.awrap_model_call(_req(), handler))
    assert isinstance(resp, ModelResponse)
    assert peak == 3  # k samples truly in flight together, not sequential
    assert len(mw.pending_of("task1").candidates) == 3


def test_threads_keyed_by_langgraph_thread_id(tmp_path, monkeypatch):
    # two conversations interleaving on one agent must not share state
    import langgraph.config

    current = {"tid": "conv-a"}
    monkeypatch.setattr(langgraph.config, "get_config",
                        lambda: {"configurable": {"thread_id": current["tid"]}})
    mw = CaptureMiddleware(k=1, trace_dir=tmp_path, run_id="t")

    def act(request):
        return ModelResponse(result=[AIMessage(content="", tool_calls=[BOOK_CALL])])

    mw.wrap_model_call(_req(), act)  # conv-a step 0, left pending
    current["tid"] = "conv-b"
    mw.wrap_model_call(_req(), act)  # conv-b step, must not flush conv-a's pending

    a, b = mw.pending_of("conv-a"), mw.pending_of("conv-b")
    assert a is not None and b is not None and a is not b
    assert a.thread_id == "conv-a" and b.thread_id == "conv-b"
    assert mw.history_of("conv-a") == [] and mw.history_of("conv-b") == []


def test_eviction_never_drops_a_thread_awaiting_human(tmp_path, monkeypatch):
    import uqguard.capture as cap

    monkeypatch.setattr(cap, "_MAX_THREADS", 2)
    mw = CaptureMiddleware(k=1, trace_dir=tmp_path, run_id="t")
    mw.new_thread("interrupted")

    def act(request):
        return ModelResponse(result=[AIMessage(content="", tool_calls=[BOOK_CALL])])

    mw.wrap_model_call(_req(), act)
    pending = mw.pending_of("interrupted")
    pending.gate = "ESCALATE"  # a human decision is outstanding

    for name in ("x1", "x2", "x3"):
        mw._state(name)  # churn threads to force eviction
    assert mw.pending_of("interrupted") is pending  # survived the cap
    from uqguard.capture import to_candidate

    # no AIMessage in the response at all
    none = to_candidate(ModelResponse(result=[]))
    assert none.tool_name == "__none__" and none.args == {}
    # parallel tool calls are all captured
    second = {"name": "refund", "args": {"booking_id": "B1"}, "id": "y", "type": "tool_call"}
    multi = to_candidate(ModelResponse(
        result=[AIMessage(content="", tool_calls=[BOOK_CALL, second])]))
    assert multi.tool_name == "book_flight"
    assert multi.extra_calls == [{"name": "refund", "args": {"booking_id": "B1"}}]


def test_fallback_swaps_model_and_sticks():
    fallback_sentinel = object()
    mw = ModelFallbackMiddleware(fallback_sentinel)
    models_seen = []

    def flaky(request):
        models_seen.append(request.model)
        if len(models_seen) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return ModelResponse(result=[AIMessage("ok")])

    mw.wrap_model_call(_req(model="primary"), flaky)
    assert models_seen == ["primary", fallback_sentinel]

    mw.wrap_model_call(_req(model="primary"), flaky)  # sticky: primary never retried
    assert models_seen[-1] is fallback_sentinel and len(models_seen) == 3
