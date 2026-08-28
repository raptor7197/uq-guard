"""Per-tool risk config: scorers, thresholds, and routing scoped per tool."""

from uqguard import AgentStep, CandidateAction, Guard, RoutedPolicy, ThresholdPolicy, ToolConfig


def _fixed(name, value):
    def scorer(step, history=()):
        return value

    scorer.name = name
    return scorer


def _step(tool="book_flight", **args):
    c = CandidateAction(tool_name=tool, args=args, raw_text="")
    return AgentStep(step_id="t/0", thread_id="t", candidates=[c], chosen=c)


def test_threshold_policy_per_tool_threshold():
    p = ThresholdPolicy(
        threshold=1.0,
        scorers=(_fixed("arg_agreement", 0.6),),
        tool_config={"book_flight": ToolConfig(threshold=0.5)},
    )
    # book_flight cleared its per-tool threshold; another tool stays strict
    assert p.decide(_step(tool="book_flight")) == "PROCEED"
    assert p.decide(_step(tool="search_flights")) == "ESCALATE"


def test_routed_policy_per_tool_scorers():
    # judge scorer runs ONLY on book_flight; search steps never pay for it
    p = RoutedPolicy(
        threshold=0.8,
        scorers=(_fixed("arg_agreement", 0.9),),
        tool_config={
            "book_flight": ToolConfig(
                scorers=(_fixed("arg_agreement", 0.9), _fixed("options_set", 0.0)),
            ),
        },
    )
    book = _step(tool="book_flight")
    p.decide(book)
    assert "options_set" in book.signals  # judge ran for book_flight
    search = _step(tool="search_flights")
    p.decide(search)
    assert "options_set" not in search.signals  # and not for search


class _InstanceScorer:
    """A scorer instance (e.g. OptionsSetScorer) has .name but, unlike a
    plain function, no __name__ -- regression fixture for the crash below."""

    name = "custom_instance"

    def __call__(self, step, history=()):
        return 0.9


def test_score_accepts_scorer_instance_lacking_dunder_name():
    # getattr(entry, "name", entry.__name__) evaluates entry.__name__ eagerly
    # as the default, regardless of whether "name" is found -- crashing on
    # any scorer instance even though .name exists. Every --judge caller
    # (demo_agent.py, integrate_existing_agent.py, eval/calibrate.py) passes
    # OptionsSetScorer this way.
    p = ThresholdPolicy(threshold=0.5, scorers=(_InstanceScorer(),))
    step = _step()
    assert p.decide(step) == "PROCEED"
    assert step.signals["custom_instance"] == 0.9


def test_routed_policy_per_tool_threshold_and_routes():
    p = RoutedPolicy(
        threshold=0.8,
        scorers=(_fixed("arg_agreement", 0.5),),
        tool_config={
            "search_flights": ToolConfig(threshold=0.4),  # read-only: lenient
            "refund": ToolConfig(routes={"arg_agreement": "ESCALATE"}),
        },
    )
    assert p.decide(_step(tool="search_flights")) == "PROCEED"  # 0.5 >= 0.4
    assert p.decide(_step(tool="book_flight")) == "RETRY"  # default routing
    assert p.decide(_step(tool="refund")) == "ESCALATE"  # overridden route


def test_routed_policy_per_tool_assigns_gate_via_middleware(monkeypatch):
    # decide() returns the gate; GateMiddleware is what records it on the step
    import uqguard.gate

    from langchain_core.messages import ToolMessage

    from uqguard import GateMiddleware

    p = RoutedPolicy(
        threshold=0.8,
        scorers=(_fixed("arg_agreement", 0.5),),
        tool_config={"refund": ToolConfig(routes={"arg_agreement": "ESCALATE"})},
    )

    class FakeCapture:
        history = []

        def __init__(self, step):
            self.step = step

        def current_thread(self):
            return "t"

        def pending_of(self, thread_id):
            return self.step

        def history_of(self, thread_id):
            return self.history

        class trace:
            @staticmethod
            def write(step):
                pass

    step = _step(tool="refund")
    mw = GateMiddleware(FakeCapture(step), p)
    monkeypatch.setattr(uqguard.gate, "interrupt", lambda payload: {"action": "approve"})
    request = type("R", (), {"tool_call": {"name": "refund", "args": {}, "id": "tc1"}})()
    mw.wrap_tool_call(request, lambda r: ToolMessage(content="ok", tool_call_id="tc1"))
    assert step.gate == "ESCALATE"


def test_tool_config_none_fields_fall_back():
    p = RoutedPolicy(
        threshold=0.8,
        scorers=(_fixed("arg_agreement", 0.5),),
        tool_config={"refund": ToolConfig(threshold=0.9)},
    )  # only threshold set
    assert p.decide(_step(tool="refund")) == "RETRY"  # route defaults apply
    assert p.decide(_step(tool="book_flight")) == "RETRY"  # unconfigured tool -> defaults


def test_tool_config_none_fields_fall_back_global_threshold():
    p = ThresholdPolicy(
        threshold=0.9,
        scorers=(_fixed("arg_agreement", 0.95),),
        tool_config={"refund": ToolConfig(threshold=1.0)},
    )
    assert p.decide(_step(tool="refund")) == "ESCALATE"  # per-tool 1.0: 0.95 < 1.0
    assert p.decide(_step(tool="book_flight")) == "PROCEED"  # global 0.9: 0.95 >= 0.9


def test_guard_facade_accepts_tool_config(tmp_path):
    guard = Guard(
        k=3, threshold=0.8, trace_dir=tmp_path, tool_config={"refund": ToolConfig(threshold=1.0)}
    )
    assert guard.policy.tool_config["refund"].threshold == 1.0
    assert guard.policy.threshold == 0.8  # policy default untouched


def test_tool_config_accepts_plain_dict():
    # the natural dict form must be coerced, not crash deep in decide()
    p = ThresholdPolicy(
        threshold=1.0,
        scorers=(_fixed("arg_agreement", 0.6),),
        tool_config={"book_flight": {"threshold": 0.5}},
    )
    assert p.decide(_step(tool="book_flight")) == "PROCEED"
    assert p.decide(_step(tool="search_flights")) == "ESCALATE"

    p2 = RoutedPolicy(
        threshold=0.8,
        scorers=(_fixed("arg_agreement", 0.5),),
        tool_config={"refund": {"routes": {"arg_agreement": "ESCALATE"}}},
    )
    assert p2.decide(_step(tool="refund")) == "ESCALATE"


def test_tool_config_rejects_unknown_types():
    import pytest

    p = ThresholdPolicy(tool_config={"book_flight": 42})
    with pytest.raises(TypeError, match="ToolConfig or dict"):
        p.decide(_step(tool="book_flight"))
