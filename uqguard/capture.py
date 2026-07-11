"""k-sampling capture middleware (SPEC Phase 1).

Wraps the agent's model node: every model call is executed k times, all k
candidate actions are recorded as an AgentStep, and the first sample is what
the agent actually acts on (identical to unguarded behavior). The step is
flushed to the trace once its tool result arrives with the next model call.

Per-conversation state (pending step, step history) is keyed by the LangGraph
thread id when the middleware runs inside a graph, so concurrent conversations
on one agent don't interleave. Outside a graph (or without a thread_id in the
config) it falls back to the name set via new_thread().

Sync invoke samples sequentially; ainvoke/astream sample the k candidates
concurrently via asyncio.gather.

The wrapped model needs temperature > 0 or all k samples collapse to one.
"""

import asyncio
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from uqguard.step import AgentStep, CandidateAction
from uqguard.trace import TraceWriter

log = logging.getLogger("uqguard")

_MAX_THREADS = 256  # ponytail: insertion-order eviction; real LRU if hot threads get evicted


def _short(x, n=100):
    s = repr(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def to_candidate(response: ModelResponse) -> CandidateAction:
    msg = next((m for m in response.result if isinstance(m, AIMessage)), None)
    if msg is None:
        return CandidateAction(tool_name="__none__", args={}, raw_text="")
    text = msg.text if isinstance(msg.text, str) else str(msg.content)
    if msg.tool_calls:
        first, *rest = msg.tool_calls
        return CandidateAction(
            tool_name=first["name"], args=first["args"], raw_text=text,
            extra_calls=[{"name": t["name"], "args": t["args"]} for t in rest],
        )
    # answered in text instead of acting -- that disagreement is itself signal
    return CandidateAction(tool_name="__none__", args={}, raw_text=text)


def _trailing_tool_results(messages):
    """Contents of the ToolMessages at the tail of the conversation, if any."""
    out = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            out.append(str(m.content))
        elif isinstance(m, AIMessage):
            break
    return "\n".join(reversed(out)) or None


@dataclass
class _Thread:
    pending: AgentStep | None = None  # last step, waiting for its tool result
    history: list[AgentStep] = field(default_factory=list)  # flushed steps (scorer context)


class CaptureMiddleware(AgentMiddleware):
    def __init__(self, k: int = 5, trace_dir="runs", run_id: str | None = None,
                 redact: Callable[[AgentStep], AgentStep] | None = None):
        super().__init__()
        self.k = k
        self.trace = TraceWriter(trace_dir, run_id=run_id, redact=redact)
        self._threads: dict[str, _Thread] = {}
        self._fallback_thread = "t0"
        self._counter = itertools.count()  # global: step ids stay unique across rethreads

    # -- thread state ---------------------------------------------------------

    def current_thread(self) -> str:
        """LangGraph thread id when running inside a graph, else the
        new_thread() name (single-conversation fallback)."""
        try:
            from langgraph.config import get_config

            tid = (get_config() or {}).get("configurable", {}).get("thread_id")
            if tid is not None:
                return str(tid)
        except Exception:  # outside a runnable context
            pass
        return self._fallback_thread

    @staticmethod
    def _awaiting_human(st: _Thread) -> bool:
        # evicting a thread mid-interrupt would make the resumed tool call find
        # no pending step and execute unapproved -- never evict these
        return (st.pending is not None and st.pending.gate in ("CLARIFY", "ESCALATE")
                and st.pending.human_action is None)

    def _state(self, thread_id: str) -> _Thread:
        st = self._threads.get(thread_id)
        if st is None:
            while len(self._threads) >= _MAX_THREADS:
                victim = next((t for t, s in self._threads.items()
                               if not self._awaiting_human(s)), None)
                if victim is None:
                    break  # everything is mid-interrupt; grow past the cap instead
                self._flush_thread(victim, None)
                del self._threads[victim]
            st = self._threads[thread_id] = _Thread()
        return st

    def pending_of(self, thread_id: str) -> AgentStep | None:
        st = self._threads.get(thread_id)
        return st.pending if st else None

    def history_of(self, thread_id: str) -> list[AgentStep]:
        st = self._threads.get(thread_id)
        return st.history if st else []

    @property
    def history(self) -> list[AgentStep]:
        """Current thread's flushed steps (single-conversation convenience)."""
        return self.history_of(self.current_thread())

    def new_thread(self, name) -> None:
        """Call at the start of each conversation/task sharing this middleware.
        Flushes leftovers and resets state (and retry budget) for the name."""
        for tid in {self._fallback_thread, str(name)}:
            if tid in self._threads:
                if self._awaiting_human(self._threads[tid]):
                    log.warning("thread %s reset while a gate decision was awaiting a human; "
                                "resuming that interrupt will find no pending step", tid)
                self._flush_thread(tid, None)
                del self._threads[tid]
        self._fallback_thread = str(name)

    # -- capture --------------------------------------------------------------

    def _flush_thread(self, thread_id: str, tool_result: str | None) -> None:
        st = self._threads.get(thread_id)
        if st is None or st.pending is None:
            return
        st.pending.tool_result = tool_result
        if tool_result is not None:
            log.info("%s: tool result: %s", st.pending.step_id, _short(tool_result))
        self.trace.write(st.pending)
        st.history.append(st.pending)
        st.pending = None

    def _flush(self, tool_result: str | None) -> None:
        self._flush_thread(self.current_thread(), tool_result)

    def _pre(self, request: ModelRequest) -> tuple[str, str]:
        tid = self.current_thread()
        self._flush_thread(tid, _trailing_tool_results(request.messages))
        step_id = f"{tid}/{next(self._counter)}"
        log.info("%s: sampling %d candidates", step_id, self.k)
        return tid, step_id

    def _record(self, request: ModelRequest, tid: str, step_id: str,
                responses: list[ModelResponse]) -> ModelResponse:
        candidates = []
        for i, r in enumerate(responses):
            c = to_candidate(r)
            candidates.append(c)
            log.info("%s: sample %d/%d -> %s %s", step_id, i + 1, self.k, c.tool_name,
                     _short(c.args) if c.args else _short(c.raw_text, 60))
        distinct = len({(c.tool_name, repr(sorted(c.args.items()))) for c in candidates})
        log.info("%s: acting on %s %s (%d distinct action(s) across samples)",
                 step_id, candidates[0].tool_name, _short(candidates[0].args), distinct)
        user_msgs = [m for m in request.messages if isinstance(m, HumanMessage)]
        self._state(tid).pending = AgentStep(
            step_id=step_id,
            thread_id=tid,
            candidates=candidates,
            chosen=candidates[0],
            # the user request is the context every action must be grounded in
            retrieval_context=[str(user_msgs[-1].content)] if user_msgs else [],
        )
        return responses[0]

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        tid, step_id = self._pre(request)
        responses = [handler(request) for _ in range(self.k)]  # sync handler: sequential
        return self._record(request, tid, step_id, responses)

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        tid, step_id = self._pre(request)
        responses = await asyncio.gather(*(handler(request) for _ in range(self.k)))
        return self._record(request, tid, step_id, list(responses))

    def after_agent(self, state, runtime) -> None:
        self._flush_thread(self.current_thread(), _trailing_tool_results(state["messages"]))

    async def aafter_agent(self, state, runtime) -> None:
        self.after_agent(state, runtime)
