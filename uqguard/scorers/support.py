"""Retrieval-support scorer (deferred in Phase 3, revisited for RAG).

Phase-3 deferral reason: the travel demo has no retrieval corpus to test
grounding against. This scorer is the RAG-era version: is the chosen action
grounded in what the agent actually saw (the retrieval context and the tool
results), or is it unsupported?

Scoring, lexical default (stdlib, no deps): the action's *claims* are the
tokens of its argument values (or, for a text/final-answer step, the tokens
of the raw_text). Support = fraction of claim tokens that appear in the
grounding text (retrieval_context + the conversation's tool results).
Fabricated ids/answers share no tokens with the evidence -> score near 0.

Pass `embed` (callable text -> vector, e.g. sentence-transformers BGE) for
embedding-similarity scoring instead of token overlap -- the documented
upgrade path. Cost: free.
"""

import re

from uqguard.scorers.base import register_scorer

_WORD = re.compile(r"[a-z0-9]+")

# tokens that carry no grounding signal; keep the list tiny and honest
_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "with", "is", "are",
    "was", "were", "be", "it", "i", "you", "we", "they", "that", "this", "do",
    "does", "did", "please", "my", "your", "our", "me", "us", "what", "how",
})


class RetrievalSupport:
    name = "retrieval_support"

    def __init__(self, embed=None, stopwords=_STOPWORDS):
        self.embed = embed  # callable text -> vector; None = lexical overlap
        self.stopwords = stopwords

    def _tokens(self, text: str) -> set[str]:
        return {w for w in _WORD.findall(text.casefold()) if w not in self.stopwords}

    def _claims(self, step) -> tuple[str, set[str]]:
        """The action's claim text and tokens. For text answers the claim IS
        the raw_text; for tool calls it is the argument values."""
        if step.chosen.tool_name == "__none__":
            text = step.chosen.raw_text
        else:
            text = " ".join(str(v) for v in step.chosen.args.values())
        return text, self._tokens(text)

    def _grounding(self, step, history) -> list[str]:
        texts = list(step.retrieval_context)
        texts.extend(s.tool_result for s in history if s.tool_result)
        return [t for t in texts if t]

    def __call__(self, step, history=()) -> float:
        claim_text, claims = self._claims(step)
        if not claims:
            return 1.0  # nothing claimed, nothing unsupported
        ground = self._grounding(step, history)
        if not ground:
            return 0.0  # no evidence at all -> the action is unsupported
        if self.embed is not None:
            return self._embed_score(claim_text, " ".join(ground))
        ground_tokens = set().union(*(self._tokens(g) for g in ground))
        hits = sum(1 for t in claims if t in ground_tokens)
        return hits / len(claims)

    def _embed_score(self, claim_text: str, ground_text: str) -> float:
        import numpy as np

        c = np.asarray(self.embed(claim_text))
        g = np.asarray(self.embed(ground_text))
        denom = float(np.linalg.norm(c) * np.linalg.norm(g))
        if denom == 0:
            return 0.0
        return float(max(0.0, c @ g / denom))


retrieval_support = register_scorer("retrieval_support")(RetrievalSupport())
