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

Both policies accept a `tool_config` dict: per-tool overrides (scorer list,
threshold, routing) for the tools whose risk profile differs from the
default. This is the mechanism the audit called for to scope the judge to
state-changing tools and to make destructive tools stricter than read-only
ones without re-specifying the whole policy.
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
class ToolConfig:
    """Per-tool policy overrides. Any field left None falls back to the
    policy-level default, so a single tool can be tightened without
    re-specifying everything:

        RoutedPolicy(threshold=0.8, tool_config={
            "book_flight": ToolConfig(threshold=1.0),      # strict: any doubt escalates
            "refund": ToolConfig(scorers=("arg_agreement", "options_set")),
        })

    scorers: tool-specific scorer list (e.g. judge only on destructive tools)
    threshold: tool-specific confidence threshold
    routes: tool-specific weakest-signal routing (RoutedPolicy only)
    """

    scorers: tuple | None = None
    threshold: float | None = None
    routes: dict | None = None


def _cfg(tool_config, tool_name) -> ToolConfig:
    """Per-tool override for a step's tool; unset fields are None and the
    caller falls back to policy defaults. Only None means 'fall back': an
    explicitly empty tuple/dict is honored as-is."""
    cfg = tool_config.get(tool_name)
    return cfg if cfg is not None else ToolConfig()


@dataclass
class ThresholdPolicy:
    threshold: float = 1.0  # strict default: any candidate disagreement escalates
    scorers: tuple = ("arg_agreement", "tool_churn")
    tool_config: dict = field(default_factory=dict)  # tool_name -> ToolConfig

    def decide(self, step: AgentStep, history=()) -> Gate:
        cfg = _cfg(self.tool_config, step.chosen.tool_name)
        scorers = cfg.scorers if cfg.scorers is not None else self.scorers
        threshold = cfg.threshold if cfg.threshold is not None else self.threshold
        _score(step, history, scorers)
        if not step.signals:  # every scorer failed or none configured
            step.confidence = 0.0
            return "ESCALATE"
        step.confidence = min(step.signals.values())
        return "PROCEED" if step.confidence >= threshold else "ESCALATE"


@dataclass
class RoutedPolicy:
    """Full outcome set with pluggable fusion. threshold usually comes from
    uqguard.conformal.conformal_threshold on a calibration run.
    tool_config: tool_name -> ToolConfig per-tool overrides."""

    threshold: float = 0.8
    scorers: tuple = ("arg_agreement", "tool_churn")
    fusion: object = None  # callable signals_dict -> float; None = min()
    routes: dict = field(default_factory=lambda: {
        "options_set": "CLARIFY",
        "tool_churn": "ESCALATE",
        "arg_agreement": "RETRY",
    })
    tool_config: dict = field(default_factory=dict)  # tool_name -> ToolConfig

    def decide(self, step: AgentStep, history=()) -> Gate:
        cfg = _cfg(self.tool_config, step.chosen.tool_name)
        scorers = cfg.scorers if cfg.scorers is not None else self.scorers
        threshold = cfg.threshold if cfg.threshold is not None else self.threshold
        routes = cfg.routes if cfg.routes is not None else self.routes
        _score(step, history, scorers)
        if not step.signals:  # every scorer failed or none configured
            step.confidence = 0.0
            return "ESCALATE"
        step.confidence = (
            self.fusion(step.signals) if self.fusion else min(step.signals.values())
        )
        if step.confidence >= threshold:
            return "PROCEED"
        weakest = min(step.signals, key=step.signals.get)
        return routes.get(weakest, "ESCALATE")
