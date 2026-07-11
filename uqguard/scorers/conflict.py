"""Tool-churn scorer: doom-loop detector (taxonomy class 3).

An agent that re-calls the same tool with different args after fruitless
results is guessing, not progressing -- within-step agreement stays perfect
while the trajectory flails. Confidence decays with each fruitless retry.
Pure logic on step history; no models, no deps.

Fruitless detection is a word-boundary heuristic tuned to not fire on
legitimate content that merely contains 'none' or 'no' mid-sentence.
Pass a custom `fruitless` predicate for domain-specific tool outputs;
NLI result-vs-expectation checking is the documented upgrade path.
"""

import re

from uqguard.scorers.base import register_scorer

# empty containers, or failure phrases at word boundaries
_FRUITLESS = re.compile(
    r"^\s*$"
    r"|^\s*(\[\]|\{\}|null|none)\s*\.?\s*$"
    r"|\berror\b"
    r"|\bnot found\b"
    r"|\bno (results?|matches?|flights?|bookings?|records?|data|items?)\b"
    r"|\bnothing (found|matched)\b",
    re.IGNORECASE,
)


def default_fruitless(result: str) -> bool:
    return bool(_FRUITLESS.search(result))


class ToolChurn:
    name = "tool_churn"

    def __init__(self, fruitless=None):
        self.fruitless = fruitless or default_fruitless

    def __call__(self, step, history=()) -> float:
        prior_flails = [
            s for s in history
            if s.chosen.tool_name == step.chosen.tool_name
            and s.chosen.args != step.chosen.args
            and s.tool_result is not None
            and self.fruitless(s.tool_result)
        ]
        return 1.0 / (1 + len(prior_flails))


tool_churn = register_scorer("tool_churn")(ToolChurn())
