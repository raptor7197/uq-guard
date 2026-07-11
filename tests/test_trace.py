import pytest

from uqguard import AgentStep, CandidateAction, TraceWriter, read_trace


def _step(step_id="t/0", **kw):
    c = CandidateAction(tool_name="book_flight", args={"flight_id": "F1"}, raw_text="")
    return AgentStep(step_id=step_id, thread_id="t", candidates=[c], chosen=c, **kw)


def test_write_read_roundtrip(tmp_path):
    w = TraceWriter(tmp_path, run_id="r")
    w.write(_step("t/0"))
    w.write(_step("t/1"))
    steps = read_trace(tmp_path / "r.jsonl")
    assert [s.step_id for s in steps] == ["t/0", "t/1"]


def test_read_keeps_last_line_per_step_id(tmp_path):
    w = TraceWriter(tmp_path, run_id="r")
    w.write(_step("t/0", partial=True, gate="ESCALATE"))
    w.write(_step("t/1"))
    w.write(_step("t/0", tool_result="Booked.", gate="ESCALATE", human_action="approve"))
    steps = read_trace(tmp_path / "r.jsonl")
    assert [s.step_id for s in steps] == ["t/0", "t/1"]  # order of first appearance
    assert not steps[0].partial and steps[0].human_action == "approve"


def test_partial_only_step_survives(tmp_path):
    # crash after the gate decision: the partial line is all we have
    w = TraceWriter(tmp_path, run_id="r")
    w.write(_step("t/0", partial=True, gate="ESCALATE", signals={"arg_agreement": 0.4}))
    (steps,) = read_trace(tmp_path / "r.jsonl")
    assert steps.partial and steps.gate == "ESCALATE"


def test_redact_hook(tmp_path):
    def scrub(step):
        return step.model_copy(update={"retrieval_context": ["[redacted]"]})

    w = TraceWriter(tmp_path, run_id="r", redact=scrub)
    w.write(_step("t/0", retrieval_context=["my card is 4111-1111"]))
    (step,) = read_trace(tmp_path / "r.jsonl")
    assert step.retrieval_context == ["[redacted]"]


def test_read_skips_blank_lines(tmp_path):
    w = TraceWriter(tmp_path, run_id="r")
    w.write(_step("t/0"))
    with w.path.open("a") as f:
        f.write("\n")
    assert len(read_trace(w.path)) == 1


def test_read_survives_torn_line(tmp_path):
    # crash mid-write leaves a torn final line; the rest must stay readable
    w = TraceWriter(tmp_path, run_id="r")
    w.write(_step("t/0"))
    with w.path.open("a") as f:
        f.write('{"step_id": "t/1", "thread_id": "t", "cand')  # torn
    steps = read_trace(w.path)
    assert [s.step_id for s in steps] == ["t/0"]


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_trace(tmp_path / "nope.jsonl")


def test_concurrent_writes_stay_line_atomic(tmp_path):
    import threading

    w = TraceWriter(tmp_path, run_id="r")
    big = _step().model_copy(update={"retrieval_context": ["x" * 20_000]})

    def burst(i):
        for j in range(20):
            w.write(big.model_copy(update={"step_id": f"t{i}/{j}"}))

    threads = [threading.Thread(target=burst, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(read_trace(w.path)) == 80  # no interleaved/corrupt lines
