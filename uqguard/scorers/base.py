"""Scorer registry. A scorer is any callable AgentStep -> float in [0, 1],
higher = more confident."""

SCORERS = {}


def register_scorer(name):
    def deco(fn):
        SCORERS[name] = fn
        return fn

    return deco


def get_scorer(name):
    try:
        return SCORERS[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; registered: {sorted(SCORERS)}") from None
