"""Split-conformal acceptance threshold + calibration diagnostics.

Guarantee (stated precisely, do not oversell): with calibration data
exchangeable with deployment steps, the threshold bounds P(accept | wrong
action) <= alpha. It does NOT directly bound P(wrong | accepted) -- report
the empirical accepted-error rate alongside. Agent steps within a trajectory
are not i.i.d.; calibrate per step-type where possible and always report
empirical coverage (see docs/SPEC.md risks).
"""

import math

import numpy as np


def conformal_threshold(confidences, correct, alpha=0.1):
    """Smallest confidence t such that at most an alpha fraction of WRONG
    calibration steps score >= t, with the (n+1) finite-sample correction.

    Acceptance everywhere in uqguard is `confidence >= threshold`, and the
    signals are discrete (agreement in {i/k}, churn in {1/(1+m)}), so ties at
    the quantile are the norm, not the exception. The returned threshold is
    nudged just above the quantile value so tied wrong steps do NOT count as
    accepted -- otherwise a calibration set whose wrong steps all score 1.0
    (unanimous-but-wrong, the tie-break class) would yield t=1.0 and accept
    100% of them under >=."""
    wrong = np.asarray(
        [c for c, ok in zip(confidences, correct, strict=True) if not ok], dtype=float
    )
    if wrong.size == 0:
        return 0.0  # never saw a wrong step: accept everything (and say so)
    q = min(1.0, math.ceil((wrong.size + 1) * (1 - alpha)) / wrong.size)
    t = float(np.quantile(wrong, q, method="higher"))
    return float(np.nextafter(t, np.inf))


def accepted_error(confidences, correct, threshold):
    """Empirical P(wrong | accepted) at a threshold. The honest headline number."""
    pairs = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if c >= threshold]
    if not pairs:
        return 0.0, 0
    errs = sum(1 for _, ok in pairs if not ok)
    return errs / len(pairs), len(pairs)


def risk_coverage(confidences, correct):
    """Sweep thresholds: returns (coverage, risk) arrays for the risk-coverage curve."""
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)  # accept highest-confidence first
    errs = np.cumsum(~ok[order])
    n_accepted = np.arange(1, len(conf) + 1)
    coverage = n_accepted / len(conf)
    risk = errs / n_accepted
    return coverage, risk


def ece(confidences, correct, bins=10):
    """Expected calibration error over equal-width confidence bins."""
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - ok[mask].mean())
    return float(total)
