"""Core data model. Everything in uqguard hangs off AgentStep (SPEC.md §2)."""

from typing import Literal

from pydantic import BaseModel

Gate = Literal["PROCEED", "RETRY", "CLARIFY", "ESCALATE"]


class CandidateAction(BaseModel):
    tool_name: str  # "__none__" when a sample answered in text instead of calling a tool
    args: dict
    raw_text: str
    logprob: float | None = None  # None for black-box models
    extra_calls: list[dict] = []  # parallel tool calls beyond the first: [{"name", "args"}]


class AgentStep(BaseModel):
    step_id: str
    thread_id: str
    candidates: list[CandidateAction]  # k samples
    chosen: CandidateAction
    retrieval_context: list[str] = []
    tool_result: str | None = None  # filled post-execution
    signals: dict[str, float] = {}  # scorer name -> score
    confidence: float | None = None  # fused
    gate: Gate | None = None
    human_action: str | None = None  # approve / reject / "clarified: ..."
    partial: bool = False  # early flush (gate evidence saved before the step completed)
