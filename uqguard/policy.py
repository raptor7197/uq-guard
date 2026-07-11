"""Gating policies: scores -> gate decision.

ThresholdPolicy: min-of-signals vs threshold, PROCEED/ESCALATE. Simple, no labels.
RoutedPolicy: fused confidence vs (conformally calibrated) threshold, and the
weakest signal routes the outcome -- the signals measure different failure
modes, so the failing signal names the right intervention:
  options_set low -> the REQUEST is ambiguous -> CLARIFY (ask the end user)
  tool_churn low  -> the agent is flailing    -> ESCALATE (human review)
  agreement low   -> the model is unsure      -> RETRY (re-decide with feedback)

A scorer that raises is treated as missing, not fatal (the graph must not
crash because a judge call failed). No signals at all -> no evidence the
action is safe -> ESCALATE.
"""

import logging
from dataclasses import dataclass, field

from uqguard.scorers import get_scorer
from uqguard.step import AgentStep, Gate

log = logging.getLogger("uqguard")


def _score(step: AgentStep, history, scorers) -> None:
    for entry in scorers:
        fn = get_scorer(entry) if isinstance(entry, str) else entry  # name or instance
        name = entry if isinstance(entry, str) else getattr(entry, "name", entry.__name__)
        try:
            step.signals[name] = fn(step, history)
        except Exception:
            log.warning("scorer %r failed on %s; treating signal as missing",
                        name, step.step_id, exc_info=True)


@dataclass
class ThresholdPolicy:
    threshold: float = 1.0  # strict default: any candidate disagreement escalates
    scorers: tuple = ("arg_agreement", "tool_churn")

    def decide(self, step: AgentStep, history=()) -> Gate:
        _score(step, history, self.scorers)
        if not step.signals:  # every scorer failed or none configured
            step.confidence = 0.0
            return "ESCALATE"
        step.confidence = min(step.signals.values())
        return "PROCEED" if step.confidence >= self.threshold else "ESCALATE"


@dataclass
class RoutedPolicy:
    """Full outcome set with pluggable fusion. threshold usually comes from
    uqguard.conformal.conformal_threshold on a calibration run."""

    threshold: float = 0.8
    scorers: tuple = ("arg_agreement", "tool_churn")
    fusion: object = None  # callable signals_dict -> float; None = min()
    routes: dict = field(default_factory=lambda: {
        "options_set": "CLARIFY",
        "tool_churn": "ESCALATE",
        "arg_agreement": "RETRY",
    })

    def decide(self, step: AgentStep, history=()) -> Gate:
        _score(step, history, self.scorers)
        if not step.signals:  # every scorer failed or none configured
            step.confidence = 0.0
            return "ESCALATE"
        step.confidence = (
            self.fusion(step.signals) if self.fusion else min(step.signals.values())
        )
        if step.confidence >= self.threshold:
            return "PROCEED"
        weakest = min(step.signals, key=step.signals.get)
        return self.routes.get(weakest, "ESCALATE")
