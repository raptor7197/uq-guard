"""Property-based tests: invariants that must hold for ANY input.

Hypothesis shrinks failures to minimal counterexamples, so these catch
edge cases a hand-written example would miss (boundary ties, empty inputs,
degenerate distributions). Invariants asserted:

- conformal_threshold always lands in [0, 1] (the no-lockout guarantee),
  is 0.0 when no wrong step exists, is monotone in alpha, and -- outside
  the documented cap-at-1.0 caveat -- keeps wrong-acceptance at or under
  the finite-sample bound floor((n+1)*alpha).
- normalize_value is idempotent on arbitrary JSON-ish values.
- ArgAgreement and WeightedSum always score in [0, 1]; unanimous
  candidates score exactly 1.0.
- ece/accepted_error/risk_coverage outputs are valid rates.
- to_candidate never raises, whatever the response looks like.
"""

import math

import pytest
from hypothesis import given, settings, strategies as st
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage

from uqguard import (
    AgentStep,
    CandidateAction,
    WeightedSum,
    accepted_error,
    conformal_threshold,
    ece,
    risk_coverage,
)
from uqguard.capture import to_candidate
from uqguard.scorers import ArgAgreement, normalize_value

CONF = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
PAIRS = st.lists(st.tuples(CONF, st.booleans()), max_size=15)
ALPHA = st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False)
MAX_EXAMPLES = 100


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS, ALPHA)
def test_conformal_threshold_always_in_unit_interval(data, alpha):
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    t = conformal_threshold(conf, ok, alpha)
    assert 0.0 <= t <= 1.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS)
def test_conformal_no_wrong_steps_accepts_everything(data):
    if all(o for _, o in data):
        conf = [c for c, _ in data]
        assert conformal_threshold(conf, [True] * len(conf), 0.1) == 0.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS, ALPHA)
def test_conformal_wrong_acceptance_respects_finite_sample_bound(data, alpha):
    if not data:
        return
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    wrong = [c for c, o in zip(conf, ok, strict=True) if not o]
    if not wrong or max(wrong) == 1.0:
        return  # documented caveat: a wrong step at max confidence can't be gated out
    t = conformal_threshold(conf, ok, alpha)
    wrong_accepted = sum(1 for c, o in zip(conf, ok, strict=True) if not o and c >= t)
    assert wrong_accepted <= math.floor((len(data) + 1) * alpha)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS, ALPHA, ALPHA)
def test_conformal_threshold_monotone_in_alpha(data, a1, a2):
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    if a2 <= a1:
        return
    # looser risk budget never demands a stricter threshold
    assert conformal_threshold(conf, ok, a2) <= conformal_threshold(conf, ok, a1)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS, CONF)
def test_accepted_error_reports_consistent_counts(data, threshold):
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    err, n = accepted_error(conf, ok, threshold)
    assert n == sum(1 for c in conf if c >= threshold)
    if n:
        assert 0.0 <= err <= 1.0
        assert err * n == pytest.approx(
            sum(1 for c, o in zip(conf, ok, strict=True) if c >= threshold and not o)
        )


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS)
def test_ece_is_a_rate(data):
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    if not conf:
        return
    assert 0.0 <= ece(conf, ok) <= 1.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(PAIRS)
def test_risk_coverage_curve_shape(data):
    if not data:
        return
    conf = [c for c, _ in data]
    ok = [o for _, o in data]
    coverage, risk = risk_coverage(conf, ok)
    n = len(conf)
    assert list(coverage) == [i / n for i in range(1, n + 1)]
    assert all(0.0 <= r <= 1.0 for r in risk)
    # the first accepted step is the highest-confidence one (ties: whatever
    # argsort lands on -- replicate the implementation's ordering)
    import numpy as np

    order = np.argsort(-np.asarray(conf, dtype=float))
    first = int(order[0])
    assert risk[0] == (0.0 if ok[first] else 1.0)
    assert risk[-1] == pytest.approx(sum(1 for o in ok if not o) / n)


def _finite_float_string(s):
    try:
        return math.isfinite(float(s))
    except ValueError:
        return True


JSONISH = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.text(max_size=40).filter(_finite_float_string),
    ),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=10), children, max_size=5)
    ),
    max_leaves=25,
)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(JSONISH)
def test_normalize_value_idempotent(v):
    once = normalize_value(v)
    assert normalize_value(once) == once


TOOL = st.sampled_from(["book_flight", "search_flights", "refund", "__none__"])
ARGS = st.dictionaries(
    st.sampled_from(["flight_id", "origin", "price", "booking_id"]),
    st.one_of(st.text(max_size=10), CONF),
    max_size=3,
)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(TOOL, st.lists(ARGS, min_size=1, max_size=8))
def test_arg_agreement_score_in_unit_interval(tool, arg_list):
    cands = [CandidateAction(tool_name=tool, args=a, raw_text="") for a in arg_list]
    step = AgentStep(step_id="p/0", thread_id="p", candidates=cands, chosen=cands[0])
    score = ArgAgreement()(step)
    assert 0.0 <= score <= 1.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(TOOL, ARGS)
def test_arg_agreement_unanimous_is_one(tool, args):
    c = CandidateAction(tool_name=tool, args=args, raw_text="")
    step = AgentStep(step_id="p/0", thread_id="p", candidates=[c] * 4, chosen=c)
    assert ArgAgreement()(step) == 1.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    st.dictionaries(
        st.sampled_from(["a", "b", "c"]), st.floats(0.1, 5.0, allow_nan=False), max_size=3
    ),
    st.dictionaries(st.sampled_from(["a", "b", "c"]), CONF, max_size=3),
)
def test_weighted_sum_score_in_unit_interval(weights, signals):
    if not signals:
        return
    score = WeightedSum(weights)(signals)
    assert 0.0 <= score <= 1.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    st.dictionaries(st.sampled_from(["a", "b", "c"]), CONF, max_size=3),
    CONF,
)
def test_weighted_sum_with_names_bounds_imputation(signals, missing):
    if not signals:
        return
    names = {"a", "b", "c"}
    score = WeightedSum(names=names, missing=missing)(signals)
    assert 0.0 <= score <= 1.0


TOOL_CALL = st.fixed_dictionaries(
    {
        "name": st.text(max_size=8),
        "args": st.dictionaries(st.text(max_size=5), st.integers(), max_size=2),
        "id": st.text(max_size=8),
    }
)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    st.lists(
        st.builds(
            AIMessage,
            content=st.one_of(st.text(max_size=20), st.lists(st.text(max_size=5), max_size=3)),
            tool_calls=st.lists(TOOL_CALL, max_size=3),
        ),
        max_size=4,
    )
)
def test_to_candidate_never_raises_and_keeps_parallel_calls(messages):
    cand = to_candidate(ModelResponse(result=messages))
    assert isinstance(cand, CandidateAction)
    # to_candidate only looks at the FIRST AIMessage, and only its own
    # tool_calls; anything else resolves to a text-only step
    first = messages[0] if messages else None
    if first and first.tool_calls:
        assert len(cand.extra_calls) == len(first.tool_calls) - 1
    else:
        assert cand.tool_name == "__none__"
