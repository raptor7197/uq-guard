from uqguard import AgentStep, CandidateAction


def test_agentstep_json_roundtrip():
    c = CandidateAction(tool_name="book_flight", args={"flight_id": "F1"}, raw_text="book F1")
    step = AgentStep(step_id="s1", thread_id="t1", candidates=[c, c, c], chosen=c)
    again = AgentStep.model_validate_json(step.model_dump_json())
    assert again == step
    assert again.gate is None and again.signals == {}
    assert again.chosen.logprob is None


def test_defaults_not_shared_between_instances():
    a = AgentStep(step_id="a", thread_id="t", candidates=[], chosen=_c())
    b = AgentStep(step_id="b", thread_id="t", candidates=[], chosen=_c())
    a.signals["x"] = 1.0
    a.retrieval_context.append("doc")
    assert b.signals == {} and b.retrieval_context == []


def _c():
    return CandidateAction(tool_name="__none__", args={}, raw_text="")
