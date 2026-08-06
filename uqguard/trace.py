"""JSONL trace: one AgentStep per line, append-only.

A step may appear twice: once with partial=True (flushed early so gate
evidence survives a crash or a never-resumed interrupt) and once complete.
read_trace keeps the last line per step_id.

Traces persist raw user text, tool args and tool results. If those carry
PII/secrets, pass `redact` -- it receives each AgentStep before writing and
returns the (possibly scrubbed) step to persist.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from uqguard.step import AgentStep

log = logging.getLogger("uqguard")


class TraceWriter:
    """Reusing an explicit run_id across process restarts appends to the same
    file with a reset step counter -- step ids can collide and read_trace will
    merge them. Use a fresh run_id per process."""

    def __init__(
        self,
        trace_dir="runs",
        run_id: str | None = None,
        redact: Callable[[AgentStep], AgentStep] | None = None,
    ):
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(trace_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact = redact
        self._lock = threading.Lock()  # concurrent conversations share one file

    def write(self, step: AgentStep) -> None:
        if self.redact:
            step = self.redact(step)
        with self._lock, self.path.open("a") as f:
            f.write(step.model_dump_json() + "\n")


def read_trace(path) -> list[AgentStep]:
    steps: dict[str, AgentStep] = {}  # last line per step_id wins, insertion order kept
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        try:
            step = AgentStep.model_validate_json(line)
        except ValidationError:
            # a crash mid-write leaves a torn line; the readable evidence around
            # it is the whole point of the trace -- skip, don't die
            log.warning("%s:%d: skipping unparseable trace line", path, n)
            continue
        steps[step.step_id] = step
    return list(steps.values())
