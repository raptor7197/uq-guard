import pytest
from langchain_core.messages import ToolMessage

import uqguard.gate
from uqguard import (
    AgentStep,
    CandidateAction,
    GateMiddleware,
    LogisticFusion,
    RoutedPolicy,
    WeightedSum,
    accepted_error,
    conformal_threshold,
    ece,
    risk_coverage,
)


def _cand(tool="book_flight", **args):
    return CandidateAction(tool_name=tool, args=args, raw_text="")


def _step(candidates=None):
    candidates = candidates or [_cand(flight_id="F1")]
    return AgentStep(step_id="t/0", thread_id="t", candidates=candidates,
                     chosen=candidates[0])


def test_conformal_threshold_bounds_wrong_acceptance():
    # 10 wrong steps with confidences 0.05..0.95; alpha=0.2 must put the
    # threshold high enough that at most ~20% of wrong steps clear it
    conf = [i / 20 + 0.025 for i in range(20)]
    correct = [i % 2 == 1 for i in range(20)]
    t = conformal_threshold(conf, correct, alpha=0.2)
    wrong_accepted = sum(1 for c, ok in zip(conf, correct, strict=True) if not ok and c >= t)
    assert wrong_accepted / 10 <= 0.2
    err, n = accepted_error(conf, correct, t)
    assert n == sum(1 for c in conf if c >= t)


def test_conformal_no_wrong_examples():
    assert conformal_threshold([0.9, 0.8], [True, True], alpha=0.1) == 0.0


def test_conformal_threshold_excludes_ties():
    # unanimous-but-wrong steps all score 1.0; with >= acceptance a threshold
    # of exactly 1.0 would accept 100% of them against alpha=0.1
    conf = [1.0, 1.0, 1.0, 1.0, 0.9]
    correct = [False, False, False, False, True]
    t = conformal_threshold(conf, correct, alpha=0.1)
    wrong_accepted = sum(1 for c, ok in zip(conf, correct, strict=True) if not ok and c >= t)
    assert wrong_accepted == 0


def test_fusion_imputes_missing_signals_neutrally():
    data = [({"a": 1.0, "j": 1.0}, True), ({"a": 0.9, "j": 0.8}, True),
            ({"a": 0.2, "j": 0.1}, False), ({"a": 0.1, "j": 0.3}, False)]
    fusion = LogisticFusion().fit([d for d, _ in data], [y for _, y in data])
    # a missing judge signal reads as 0.5 (neutral), not 1.0 (approval)
    assert fusion({"a": 1.0}) == fusion({"a": 1.0, "j": 0.5})
    assert fusion({"a": 1.0}) < fusion({"a": 1.0, "j": 1.0})


def test_weighted_sum_empty_signals_zero():
    assert WeightedSum()({}) == 0.0  # no evidence is not full confidence


def test_risk_coverage_and_ece_shapes():
    conf = [0.9, 0.8, 0.3, 0.2]
    ok = [True, True, False, False]
    cov, risk = risk_coverage(conf, ok)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert risk[-1] == 0.5 and risk[0] == 0.0
    assert ece(conf, ok) < ece(conf, [not o for o in ok])  # calibrated < anti-calibrated


def test_fusion_weighted_sum_and_logistic():
    assert WeightedSum()({"a": 1.0, "b": 0.0}) == 0.5
    assert WeightedSum({"a": 3.0})({"a": 1.0, "b": 0.0}) == 0.75
    data = [({"s": 1.0}, True), ({"s": 0.9}, True), ({"s": 0.2}, False), ({"s": 0.1}, False)]
    fusion = LogisticFusion().fit([d for d, _ in data], [y for _, y in data])
    assert fusion({"s": 0.95}) > fusion({"s": 0.15})
    assert "s" in fusion.coefficients()
    with pytest.raises(RuntimeError):
        LogisticFusion()({"s": 1.0})


def test_routed_policy_routes_by_weakest_signal():
    def fixed(name, value):
        def scorer(step, history=()):
            return value
        scorer.name = name
        return scorer

    step = _step()
    p = RoutedPolicy(threshold=0.8, scorers=(fixed("options_set", 0.0), fixed("tool_churn", 1.0)))
    assert p.decide(step) == "CLARIFY"
    step = _step()
    p = RoutedPolicy(threshold=0.8, scorers=(fixed("tool_churn", 0.3), fixed("arg_agreement", 0.9)))
    assert p.decide(step) == "ESCALATE"
    step = _step()
    p = RoutedPolicy(threshold=0.8, scorers=(fixed("arg_agreement", 0.5),))
    assert p.decide(step) == "RETRY"
    step = _step()
    p = RoutedPolicy(threshold=0.4, scorers=(fixed("arg_agreement", 0.5),))
    assert p.decide(step) == "PROCEED"


class FakeTrace:
    def write(self, step):
        pass


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
    tool_call = {"name": "book_flight", "args": {"flight_id": "F1"}, "id": "tc1"}


def _handler_recorder():
    calls = []

    def handler(request):
        calls.append(request)
        return ToolMessage(content="Booked.", tool_call_id="tc1")

    return calls, handler


class RetryPolicy:
    def decide(self, step, history=()):
        step.confidence = 0.5
        return "RETRY"


def test_gate_retry_bounded_then_escalates(monkeypatch):
    calls, handler = _handler_recorder()
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: {"action": "approve"})
    mw = GateMiddleware(FakeCapture(None), RetryPolicy(), max_retries=1)

    step1 = _step()
    mw.capture.step = step1
    msg = mw.wrap_tool_call(FakeRequest(), handler)
    assert step1.gate == "RETRY" and msg.status == "error" and calls == []

    mw.capture.history.append(step1)  # capture flushes the step on the next model call
    step2 = _step()
    mw.capture.step = step2
    mw.wrap_tool_call(FakeRequest(), handler)
    assert step2.gate == "ESCALATE"  # budget spent -> escalate
    assert step2.human_action == "approve" and len(calls) == 1


def test_gate_retry_budget_resets_with_history(monkeypatch):
    # a new conversation (fresh history) gets a fresh budget -- no leaked state
    calls, handler = _handler_recorder()
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: pytest.fail("interrupted"))
    mw = GateMiddleware(FakeCapture(None), RetryPolicy(), max_retries=1)
    step1 = _step()
    mw.capture.step = step1
    mw.wrap_tool_call(FakeRequest(), handler)
    assert step1.gate == "RETRY"

    mw.capture.history = []  # new_thread() cleared the conversation
    step2 = _step()
    mw.capture.step = step2
    mw.wrap_tool_call(FakeRequest(), handler)
    assert step2.gate == "RETRY"  # budget is back


class ClarifyPolicy:
    def decide(self, step, history=()):
        step.confidence = 0.0
        return "CLARIFY"


def test_gate_clarify_blocks_with_answer(monkeypatch):
    calls, handler = _handler_recorder()
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda p: {"answer": "the 09:00 one"})
    step = _step()
    mw = GateMiddleware(FakeCapture(step), ClarifyPolicy())
    msg = mw.wrap_tool_call(FakeRequest(), handler)
    assert calls == [] and msg.status == "error"
    assert "the 09:00 one" in msg.content
    assert step.human_action == "clarified: the 09:00 one"


def test_guard_facade_builds_middleware(tmp_path):
    from uqguard import Guard

    guard = Guard(k=3, threshold=0.7, trace_dir=tmp_path)
    assert guard.middleware == [guard.capture, guard.gate]
    assert guard.gate.policy.threshold == 0.7
    guard.new_thread("x")
    assert guard.capture.current_thread() == "x"
