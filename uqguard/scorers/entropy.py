"""Semantic-entropy scorer (deferred in Phase 3, revisited for text answers).

The Phase-3 deferral reason: tool-call responses have empty `raw_text`, so
text entropy added nothing over arg_agreement. That stays true for tool-call
steps (text-free candidates score neutral 1.0 and never drag the min down).
This scorer exists for the case that tool calls can't see: final answers and
RAG generations, where the sampled `raw_text`s are the decision being made.

Scoring: cluster the k sampled raw texts by semantic equivalence, entropy
over cluster fractions normalized by log(k) -> 1 - H/Hmax in [0, 1].
Unanimous text -> entropy 0 -> 1.0; k distinct phrasings -> 1.0 entropy -> 0.0.

Default equivalence is normalized exact match (stdlib). Pass `equivalence`
(a, b) -> bool for NLI-based clustering (sentence-transformers / cross-encoder
deberta) -- the documented heavy-dependency upgrade path. Cost: free.
"""

import math
import re

from uqguard.scorers.base import register_scorer

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.strip()).casefold()


def default_equivalence(a: str, b: str) -> bool:
    return _normalize(a) == _normalize(b)


class SemanticEntropy:
    name = "semantic_entropy"

    def __init__(self, equivalence=None):
        # callable (raw_a, raw_b) -> bool; None = normalized exact match
        self.equivalence = equivalence or default_equivalence

    def __call__(self, step, history=()) -> float:
        texts = [c.raw_text for c in step.candidates if c.raw_text.strip()]
        if len(texts) < 2:
            # text-free step (pure tool calls): no textual signal -- neutral,
            # must not drag confidence down (agreement/churn own these steps)
            return 1.0
        clusters: list[list[str]] = []
        for t in texts:
            for cl in clusters:
                if self.equivalence(t, cl[0]):
                    cl.append(t)
                    break
            else:
                clusters.append([t])
        k = len(texts)
        entropy = -sum((len(c) / k) * math.log(len(c) / k) for c in clusters)
        hmax = math.log(k)
        return 1.0 - (entropy / hmax if hmax else 0.0)


semantic_entropy = register_scorer("semantic_entropy")(SemanticEntropy())
