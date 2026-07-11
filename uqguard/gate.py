"""Gate middleware: pause the graph before an uncertain tool call executes.

Composes after CaptureMiddleware (which owns the pending AgentStep for the
current model turn). On ESCALATE it uses LangGraph's `interrupt` -- the graph
checkpoints, surfaces the evidence, and resumes on `Command(resume={"action":
"approve" | "reject"})`. Requires a checkpointer on the agent.

The moment a gate decision is made, the pending step is flushed to the trace
with partial=True, so the evidence survives a crash or a never-resumed
interrupt (the final complete line supersedes it on read).

The retry budget is derived from the thread's step history (steps already
gated RETRY), so it resets with the conversation and holds no state of its
own.

In-process state only: an interrupt survives across resume within one process
(same middleware instances); it does not survive a process restart.

Parallel tool calls in one step run as concurrent tasks; the decision is made
once under a lock and covers the batch. Calls that arrive while an ESCALATE
is unresolved may each surface their own interrupt for the same decision --
after one approve/reject resumes, the rest observe the recorded human_action.
"""

import logging
import threading

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from uqguard.policy import ThresholdPolicy
from uqguard.step import AgentStep

log = logging.getLogger("uqguard")


def _blocked(request, content: str) -> ToolMessage:
    tc = request.tool_call
    return ToolMessage(content=content, name=tc["name"], tool_call_id=tc["id"], status="error")


def _evidence(step: AgentStep) -> dict:
    return {
        "step_id": step.step_id,
        "tool": step.chosen.tool_name,
        "args": step.chosen.args,
        "extra_calls": step.chosen.extra_calls,
        "confidence": step.confidence,
        "signals": step.signals,
        "candidates": [{"tool": c.tool_name, "args": c.args, "extra_calls": c.extra_calls}
                       for c in step.candidates],
    }


class GateMiddleware(AgentMiddleware):
    def __init__(self, capture, policy=None, max_retries: int = 1):
        super().__init__()
        self.capture = capture
        self.policy = policy or ThresholdPolicy()
        self.max_retries = max_retries
        # ponytail: one lock for all threads; per-conversation locks if a slow
        # judge call blocking unrelated conversations ever matters
        self._decide_lock = threading.Lock()
        self._warned_threads = set()

    def _gate(self, request) -> ToolMessage | None:
        """Decide once per step; return a blocking ToolMessage or None to execute."""
        tid = self.capture.current_thread()
        step = self.capture.pending_of(tid)
        if step is None:  # no captured step (gate used without capture) -> pass through
            if tid not in self._warned_threads:  # a misplumbed capture must not fail silently
                self._warned_threads.add(tid)
                log.warning("thread %s: no captured step; gate passing tool calls through "
                            "(is CaptureMiddleware before GateMiddleware in the list?)", tid)
            return None

        history = self.capture.history_of(tid)
        with self._decide_lock:  # parallel tool calls race the check-then-decide
            if step.gate is None:  # score once per step, even with parallel tool calls
                step.gate = self.policy.decide(step, history)
                if step.gate == "RETRY":
                    used = sum(1 for s in history if s.gate == "RETRY")
                    if used >= self.max_retries:
                        step.gate = "ESCALATE"  # retries exhausted, a human decides
                log.info("%s: gate=%s confidence=%.2f signals=%s",
                         step.step_id, step.gate, step.confidence, step.signals)
                # evidence must survive a crash or unresumed interrupt (issue: lost steps)
                self.capture.trace.write(step.model_copy(update={"partial": True}))

        if step.gate == "RETRY":
            conf = step.confidence if step.confidence is not None else 0.0
            return _blocked(request, (
                f"Low-confidence action (confidence {conf:.2f}); not executed. "
                "Re-read the latest tool results. Only act if the results uniquely satisfy "
                "the user's request; otherwise say what information is missing."
            ))

        if step.gate == "CLARIFY" and step.human_action is None:
            answer = interrupt({"type": "clarify", **_evidence(step)})
            answer = (answer or {}).get("answer", "")
            step.human_action = f"clarified: {answer}"
            log.info("%s: user clarified: %s", step.step_id, answer)
            return _blocked(request, (
                f"Tool not executed. The request was ambiguous; the user clarified: "
                f"{answer!r}. Act on the clarified request."
            ))

        if step.gate == "ESCALATE" and step.human_action is None:
            decision = interrupt({"type": "approve", **_evidence(step)})
            step.human_action = (decision or {}).get("action", "reject")
            log.info("%s: human decision: %s", step.step_id, step.human_action)

        if step.human_action == "reject":
            return _blocked(request, (
                f"Human reviewer rejected the call to `{request.tool_call['name']}`. The tool "
                "was not executed. Do not retry unless the user explicitly asks."
            ))
        return None

    def wrap_tool_call(self, request, handler):
        blocked = self._gate(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        blocked = self._gate(request)
        return blocked if blocked is not None else await handler(request)
