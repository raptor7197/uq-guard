import pytest
from langchain_core.messages import ToolMessage

import uqguard.gate
from uqguard import AgentStep, CandidateAction, GateMiddleware, ThresholdPolicy
from uqguard.scorers import arg_agreement


def _cand(tool="book_flight", **args):
    return CandidateAction(tool_name=tool, args=args, raw_text="")


def _step(candidates):
    return AgentStep(step_id="t/0", thread_id="t", candidates=candidates,
                     chosen=candidates[0])


def test_agreement_scores():
    assert arg_agreement(_step([_cand(flight_id="F1")] * 5)) == 1.0
    split = [_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")]
    assert arg_agreement(_step(split)) == 0.8
    # dict key order normalized
    a = CandidateAction(tool_name="s", args={"a": 1, "b": 2}, raw_text="")
    b = CandidateAction(tool_name="s", args={"b": 2, "a": 1}, raw_text="")
    assert arg_agreement(_step([a, b])) == 1.0
    # answering in text vs acting counts as disagreement
    mixed = [_cand(flight_id="F1")] * 3 + [CandidateAction(tool_name="__none__", args={}, raw_text="?")] * 2
    assert arg_agreement(_step(mixed)) == 0.6


def test_agreement_normalizes_values():
    # case, numeric strings, and date formats don't count as disagreement
    a = _cand(origin="NYC", date="2026-03-03", price=450)
    b = _cand(origin="nyc", date="03/03/2026", price="450")
    assert arg_agreement(_step([a, b])) == 1.0
    # genuinely different values still disagree
    c = _cand(origin="NYC", date="2026-03-04", price=450)
    assert arg_agreement(_step([a, c])) == 0.5


def test_agreement_scores_the_chosen_action_not_the_majority():
    # the agent executes sample 1; when it's the minority, the score must say so
    cands = [_cand(flight_id="F6")] + [_cand(flight_id="F7")] * 4
    assert arg_agreement(_step(cands)) == 0.2  # not the majority's 0.8


def test_agreement_covers_parallel_calls():
    one = CandidateAction(tool_name="book_flight", args={"flight_id": "F1"}, raw_text="",
                          extra_calls=[{"name": "refund", "args": {"booking_id": "B1"}}])
    two = CandidateAction(tool_name="book_flight", args={"flight_id": "F1"}, raw_text="",
                          extra_calls=[{"name": "refund", "args": {"booking_id": "B2"}}])
    assert arg_agreement(_step([one, one.model_copy()])) == 1.0
    assert arg_agreement(_step([one, two])) == 0.5  # batch differs in the 2nd call


def test_policy_sets_signals_and_gate():
    step = _step([_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")])
    assert ThresholdPolicy(threshold=1.0).decide(step) == "ESCALATE"
    assert step.signals == {"arg_agreement": 0.8, "tool_churn": 1.0}
    assert step.confidence == 0.8
    step2 = _step([_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")])
    assert ThresholdPolicy(threshold=0.7).decide(step2) == "PROCEED"


def test_policy_escalates_on_empty_or_failing_signals():
    step = _step([_cand(flight_id="F1")])
    assert ThresholdPolicy(scorers=()).decide(step) == "ESCALATE"
    assert step.confidence == 0.0

    def broken(step, history=()):
        raise RuntimeError("judge outage")
    broken.name = "broken"

    step2 = _step([_cand(flight_id="F1")])
    p = ThresholdPolicy(threshold=1.0, scorers=("arg_agreement", broken))
    assert p.decide(step2) == "PROCEED"  # failed scorer is missing, not fatal
    assert step2.signals == {"arg_agreement": 1.0}

    step3 = _step([_cand(flight_id="F1")])
    assert ThresholdPolicy(scorers=(broken,)).decide(step3) == "ESCALATE"


class FakeTrace:
    def __init__(self):
        self.written = []

    def write(self, step):
        self.written.append(step)


class FakeCapture:
    def __init__(self, step):
        self.step = step
        self.history = []
        self.trace = FakeTrace()

    def current_thread(self):
        return "t"

    def pending_of(self, thread_id):
        return self.step

    def history_of(self, thread_id):
        return self.history


class FakeRequest:
    tool_call = {"name": "book_flight", "args": {"flight_id": "F7"}, "id": "tc1"}


@pytest.fixture
def executed():
    calls = []

    def handler(request):
        calls.append(request)
        return ToolMessage(content="Booked.", tool_call_id="tc1")

    return calls, handler


def test_gate_proceeds_without_interrupt(executed, monkeypatch):
    calls, handler = executed
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: pytest.fail("interrupted"))
    step = _step([_cand(flight_id="F7")] * 5)
    mw = GateMiddleware(FakeCapture(step))
    mw.wrap_tool_call(FakeRequest(), handler)
    assert len(calls) == 1 and step.gate == "PROCEED"


def test_gate_escalates_approve_executes(executed, monkeypatch):
    calls, handler = executed
    payloads = []
    monkeypatch.setattr(uqguard.gate, "interrupt",
                        lambda p: payloads.append(p) or {"action": "approve"})
    step = _step([_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")])
    mw = GateMiddleware(FakeCapture(step))
    mw.wrap_tool_call(FakeRequest(), handler)
    assert step.gate == "ESCALATE" and step.human_action == "approve"
    assert len(calls) == 1
    assert payloads[0]["confidence"] == 0.8 and len(payloads[0]["candidates"]) == 5


def test_gate_escalates_reject_blocks(executed, monkeypatch):
    calls, handler = executed
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: {"action": "reject"})
    step = _step([_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")])
    mw = GateMiddleware(FakeCapture(step))
    msg = mw.wrap_tool_call(FakeRequest(), handler)
    assert calls == []  # tool never executed
    assert isinstance(msg, ToolMessage) and msg.status == "error"
    assert step.human_action == "reject"


def test_gate_passes_through_without_capture(executed):
    calls, handler = executed
    mw = GateMiddleware(FakeCapture(None))
    mw.wrap_tool_call(FakeRequest(), handler)
    assert len(calls) == 1


def test_gate_flushes_partial_evidence_on_decision(executed, monkeypatch):
    calls, handler = executed
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: {"action": "approve"})
    step = _step([_cand(flight_id="F7")] * 4 + [_cand(flight_id="F6")])
    capture = FakeCapture(step)
    GateMiddleware(capture).wrap_tool_call(FakeRequest(), handler)
    # evidence written before the interrupt could lose it
    assert len(capture.trace.written) == 1
    written = capture.trace.written[0]
    assert written.partial and written.gate == "ESCALATE"
    assert not step.partial  # only the trace copy is marked


def test_gate_decides_once_under_parallel_tool_calls(monkeypatch):
    import threading
    import time

    decides = []

    class SlowPolicy:
        def decide(self, step, history=()):
            decides.append(1)
            time.sleep(0.05)  # widen the race window
            step.confidence = 1.0
            return "PROCEED"

    step = _step([_cand(flight_id="F7")] * 2)
    capture = FakeCapture(step)
    mw = GateMiddleware(capture, SlowPolicy())

    def call():
        mw.wrap_tool_call(FakeRequest(), lambda r: ToolMessage("ok", tool_call_id="tc1"))

    threads = [threading.Thread(target=call) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(decides) == 1  # one decision, one judge cost, one partial line
    assert len(capture.trace.written) == 1


def test_gate_async_paths(executed, monkeypatch):
    import asyncio

    calls, _ = executed

    async def ahandler(request):
        calls.append(request)
        return ToolMessage(content="Booked.", tool_call_id="tc1")

    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: {"action": "reject"})
    ok_step = _step([_cand(flight_id="F7")] * 2)
    asyncio.run(GateMiddleware(FakeCapture(ok_step)).awrap_tool_call(FakeRequest(), ahandler))
    assert len(calls) == 1 and ok_step.gate == "PROCEED"

    split = _step([_cand(flight_id="F7"), _cand(flight_id="F6")])
    msg = asyncio.run(GateMiddleware(FakeCapture(split)).awrap_tool_call(FakeRequest(), ahandler))
    assert len(calls) == 1 and msg.status == "error"  # rejected, tool not run
