"""Pure-function tests for eval/run_when2call.py -- no model/network calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from run_when2call import (  # noqa: E402
    _to_candidate,
    build_row,
    no_tool_breakdown,
    to_tool_specs,
)

from langchain_core.messages import AIMessage  # noqa: E402


def test_to_tool_specs_converts_bfcl_dict_type():
    raw = json.dumps(
        {
            "name": "search",
            "description": "search things",
            "parameters": {
                "type": "dict",
                "required": ["q"],
                "properties": {"q": {"type": "string"}},
            },
        }
    )
    specs = to_tool_specs([raw])
    assert specs == [
        {
            "name": "search",
            "description": "search things",
            "parameters": {
                "type": "object",
                "required": ["q"],
                "properties": {"q": {"type": "string"}},
            },
        }
    ]


def test_to_tool_specs_converts_nested_dict_type():
    raw = json.dumps(
        {
            "name": "update",
            "parameters": {
                "type": "dict",
                "properties": {
                    "prefs": {"type": "dict", "properties": {"size": {"type": "string"}}}
                },
            },
        }
    )
    specs = to_tool_specs([raw])
    assert specs[0]["parameters"]["properties"]["prefs"]["type"] == "object"


def test_to_tool_specs_missing_description_defaults_empty():
    raw = json.dumps({"name": "noop", "parameters": {"type": "dict", "properties": {}}})
    specs = to_tool_specs([raw])
    assert specs[0]["description"] == ""


def test_to_candidate_tool_call():
    msg = AIMessage(content="", tool_calls=[{"name": "book", "args": {"id": "F1"}, "id": "x"}])
    c = _to_candidate(msg)
    assert c.tool_name == "book"
    assert c.args == {"id": "F1"}


def test_to_candidate_no_tool_call():
    msg = AIMessage(content="I can't help with that.")
    c = _to_candidate(msg)
    assert c.tool_name == "__none__"
    assert c.raw_text == "I can't help with that."


def test_to_candidate_extra_calls():
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "a", "args": {}, "id": "1"},
            {"name": "b", "args": {"x": 1}, "id": "2"},
        ],
    )
    c = _to_candidate(msg)
    assert c.tool_name == "a"
    assert c.extra_calls == [{"name": "b", "args": {"x": 1}}]


class _FakeBound:
    def __init__(self, messages):
        self._messages = iter(messages)

    def invoke(self, _):
        return next(self._messages)


class _FakeModel:
    def __init__(self, messages):
        self._messages = messages

    def bind_tools(self, _specs):
        return _FakeBound(self._messages)


def test_build_row_no_tools_skips_model_call_and_predicts_no_tool():
    example = {"question": "hi", "tools": [], "correct_answer": "cannot_answer"}
    row = build_row(0, example, model=None, k=3)
    assert row["predicted"] == "no_tool"
    assert row["correct"] is True
    assert row["signals"]["arg_agreement"] == 1.0


def test_build_row_correct_when_model_calls_expected_tool():
    tool = json.dumps({"name": "book", "parameters": {"type": "dict", "properties": {}}})
    example = {"question": "book it", "tools": [tool], "correct_answer": "tool_call"}
    messages = [
        AIMessage(content="", tool_calls=[{"name": "book", "args": {}, "id": str(i)}])
        for i in range(3)
    ]
    row = build_row(0, example, model=_FakeModel(messages), k=3)
    assert row["predicted"] == "call"
    assert row["correct"] is True
    assert row["signals"]["arg_agreement"] == 1.0


def test_build_row_wrong_when_model_declines_a_callable_tool():
    tool = json.dumps({"name": "book", "parameters": {"type": "dict", "properties": {}}})
    example = {"question": "book it", "tools": [tool], "correct_answer": "tool_call"}
    messages = [AIMessage(content="I can't do that.") for _ in range(3)]
    row = build_row(0, example, model=_FakeModel(messages), k=3)
    assert row["predicted"] == "no_tool"
    assert row["correct"] is False


def test_no_tool_breakdown_only_counts_no_tool_predictions():
    rows = [
        {"predicted": "no_tool", "ground_truth": "cannot_answer"},
        {"predicted": "no_tool", "ground_truth": "cannot_answer"},
        {"predicted": "no_tool", "ground_truth": "request_for_info"},
        {"predicted": "call", "ground_truth": "tool_call"},
    ]
    assert no_tool_breakdown(rows) == {"cannot_answer": 2, "request_for_info": 1}
