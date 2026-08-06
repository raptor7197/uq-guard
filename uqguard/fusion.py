"""Signal fusion: per-signal scores -> one confidence.

WeightedSum needs no labels. LogisticFusion fits on labeled steps (signals
dict + correct/wrong) and is what the calibration script trains. Both are
callables signals_dict -> float so RoutedPolicy can take either.

A signal that is absent at inference time (scorer raised, judge skipped) is
imputed as `missing` (default 0.5, neutral) -- NOT 1.0: "the scorer never
ran" must not read as "the scorer approved". Judge outages in the live gate
score an explicit 0.0 (fail-closed) rather than going missing; see
scorers/options.py.

LogisticFusion learns the canonical signal set from its fit data; WeightedSum
is label-free so it takes the canonical set explicitly via `names=` -- when
`names` is None it has no way to know a signal is missing and scores only the
signals present (re-normalized).
"""


class WeightedSum:
    """Weighted mean of signals; needs no labels.

    missing: value imputed for a signal absent at inference time. Neutral 0.5,
             NOT 1.0 -- "the scorer never ran" must not read as approval.
    names:   the canonical signal set. Signals in `names` missing from a call
             impute at `missing`; signals outside `names` are ignored. Pass
             names for judge-bearing signal sets so an outage stays neutral.
    """

    def __init__(self, weights=None, missing: float = 0.5, names: set | None = None):
        self.weights = weights or {}
        self.missing = missing
        self.names = names

    def __call__(self, signals):
        if not signals:
            return 0.0  # no evidence is not full confidence
        names = self.names if self.names is not None else set(signals)
        num = sum(self.weights.get(n, 1.0) * signals.get(n, self.missing) for n in names)
        den = sum(self.weights.get(n, 1.0) for n in names)
        return num / den


class LogisticFusion:
    def __init__(self, missing: float = 0.5):
        self.model = None
        self.names = None
        self.missing = missing

    def fit(self, signal_dicts, labels):
        from sklearn.linear_model import LogisticRegression

        self.names = sorted({k for d in signal_dicts for k in d})
        x = [[d.get(n, self.missing) for n in self.names] for d in signal_dicts]
        self.model = LogisticRegression(max_iter=1000).fit(x, labels)
        return self

    def __call__(self, signals):
        if self.model is None:
            raise RuntimeError("LogisticFusion is unfitted; call fit() or use WeightedSum")
        x = [[signals.get(n, self.missing) for n in self.names]]
        return float(self.model.predict_proba(x)[0, 1])

    def coefficients(self):
        return dict(zip(self.names, self.model.coef_[0], strict=True))
