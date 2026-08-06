"""Argument-agreement scorer: do the k sampled candidates agree on the action?

Catches underspecification splits (taxonomy class 1): split tool choices,
fabricated/varying arguments. Blind to unanimous tie-breaks and doom-loops
(classes 2-3, other scorers). Values are normalized before comparison --
dict key order, string case, numeric strings and common date formats don't
count as disagreement ('NYC' == 'nyc', '450' == 450, '03/03/2026' ==
'2026-03-03'). Parallel tool calls beyond the first are part of the key, so
a batch that differs anywhere scores as disagreement.
"""

import json
from collections import Counter
from datetime import datetime

from uqguard.scorers.base import register_scorer

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%B %d, %Y",
)


def normalize_value(v):
    """Canonicalize one arg value: case-fold strings, unify numbers, ISO dates."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        try:
            return float(s)
        except ValueError:
            return s.casefold()
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    return v


class ArgAgreement:
    """Fraction of candidates matching the CHOSEN action (tool_name, args,
    extra_calls). The agent executes step.chosen, so that is the action whose
    support matters: samples [B, A, A, A, A] act on B and must score 0.2, not
    the majority's 0.8 -- scoring the majority would wave through the exact
    minority action the samples disagreed with.

    `normalize` is pluggable per deployment: callable(tool_name, args) -> args,
    applied before the default value normalization (e.g. to drop volatile
    fields or canonicalize domain-specific formats for specific tools).
    """

    name = "arg_agreement"

    def __init__(self, normalize=None):
        self.normalize = normalize

    def _key(self, candidate) -> tuple[str, str]:
        args = candidate.args
        if self.normalize:
            args = self.normalize(candidate.tool_name, args)
        calls = [{"name": candidate.tool_name, "args": args}, *candidate.extra_calls]
        return candidate.tool_name, json.dumps(normalize_value(calls), sort_keys=True, default=str)

    def __call__(self, step, history=()) -> float:
        if not step.candidates:
            return 1.0
        counts = Counter(self._key(c) for c in step.candidates)
        return counts[self._key(step.chosen)] / len(step.candidates)


arg_agreement = register_scorer("arg_agreement")(ArgAgreement())
