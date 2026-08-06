"""The ten-line integration surface.

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from uqguard import Guard

    guard = Guard(k=5, threshold=0.8)
    agent = create_agent(model, tools=tools, middleware=guard.middleware,
                         checkpointer=InMemorySaver())
    # per conversation: guard.new_thread(thread_id); handle interrupts on invoke

Escalations surface as LangGraph interrupts with an evidence payload; resume
with Command(resume={"action": "approve" | "reject"}) or, for clarify
interrupts, Command(resume={"answer": "..."}).

Works with invoke and ainvoke/astream. Concurrent conversations are kept
apart by the LangGraph thread_id in each invocation's config; new_thread()
is the single-conversation fallback (and resets scorer history + retry
budget for that name).
"""

import logging

from uqguard.capture import CaptureMiddleware
from uqguard.gate import GateMiddleware
from uqguard.policy import RoutedPolicy

log = logging.getLogger("uqguard")


class Guard:
    """Passing `policy=` overrides `threshold`/`scorers`/`fusion`/`tool_config`."""

    def __init__(
        self,
        k=5,
        scorers=("arg_agreement", "tool_churn"),
        threshold=0.8,
        fusion=None,
        policy=None,
        tool_config=None,
        trace_dir="runs",
        max_retries=1,
        redact=None,
    ):
        if k < 1:
            raise ValueError(
                f"Guard(k={k}): k is the number of candidate samples per model call "
                "and must be >= 1 (k=0 would leave the step with no candidates)"
            )
        if k < 2:
            log.warning(
                "Guard(k=%d): consistency scorers need k >= 2 samples to disagree; "
                "only judge-type scorers will produce signal",
                k,
            )
        self.capture = CaptureMiddleware(k=k, trace_dir=trace_dir, redact=redact)
        self.policy = policy or RoutedPolicy(
            threshold=threshold,
            scorers=tuple(scorers),
            fusion=fusion,
            tool_config=tool_config or {},
        )
        self.gate = GateMiddleware(self.capture, self.policy, max_retries=max_retries)

    @property
    def middleware(self):
        return [self.capture, self.gate]

    def new_thread(self, name):
        self.capture.new_thread(name)

    @property
    def trace_path(self):
        return self.capture.trace.path
