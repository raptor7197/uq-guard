import math

from uqguard import AgentStep, CandidateAction, SemanticEntropy
from uqguard.scorers import RetrievalSupport, get_scorer, retrieval_support, semantic_entropy


def _step(texts, tool="__none__", args=None, ctx=(), result=None, history=()):
    cands = [CandidateAction(tool_name=tool, args=args or {}, raw_text=t) for t in texts]
    step = AgentStep(step_id="t/0", thread_id="t", candidates=cands, chosen=cands[0],
                     retrieval_context=list(ctx), tool_result=result)
    return step, history


def test_semantic_entropy_unanimous_is_one():
    step, _ = _step(["Refunded in full.", "Refunded in full.", "Refunded in full."])
    assert semantic_entropy(step) == 1.0


def test_semantic_entropy_max_dispersion_is_zero():
    step, _ = _step(["yes", "no", "maybe", "other"])
    assert semantic_entropy(step) == 0.0


def test_semantic_entropy_partial_split_matches_entropy():
    step, _ = _step(["a", "a", "b"])
    p2, p1 = 2 / 3, 1 / 3
    h = -(p2 * math.log(p2) + p1 * math.log(p1)) / math.log(3)
    assert math.isclose(semantic_entropy(step), 1 - h, abs_tol=1e-9)


def test_semantic_entropy_neutral_on_text_free_steps():
    # pure tool-call steps have empty raw_text: no textual signal, must not drag
    step, _ = _step(["", "", ""], tool="book_flight", args={"flight_id": "F1"})
    assert semantic_entropy(step) == 1.0


def test_semantic_entropy_custom_equivalence():
    # case/whitespace-insensitive grouping via a custom equivalence
    scorer = SemanticEntropy(equivalence=lambda a, b: a.casefold() == b.casefold())
    step, _ = _step(["Flight F1", "flight f1", "other"])
    assert math.isclose(scorer(step), 1 - ((2 / 3) * math.log(3 / 2) + (1 / 3) * math.log(3)) / math.log(3),
                        abs_tol=1e-9)


def test_semantic_entropy_registered():
    assert get_scorer("semantic_entropy") is semantic_entropy


def test_retrieval_support_grounded_args():
    step, _ = _step([""], tool="book_flight", args={"flight_id": "F3"},
                    ctx=["Book the F3 flight on 2026-03-03."])
    assert retrieval_support(step) == 1.0


def test_retrieval_support_ungrounded_args():
    # F9 never appears in the request or any tool result -> unsupported
    step, _ = _step([""], tool="book_flight", args={"flight_id": "F9"},
                    ctx=["Book the F3 flight on 2026-03-03."])
    assert retrieval_support(step) == 0.0


def test_retrieval_support_grounds_in_tool_results():
    # the flight id came from the search results, not the user request
    search, _ = _step([""], tool="search_flights", args={"origin": "NYC"},
                      result="[{'id': 'F4', 'time': '09:00'}]")
    book, _ = _step([""], tool="book_flight", args={"flight_id": "F4"},
                    ctx=["Book the 09:00 flight."])
    assert retrieval_support(book, [search]) == 1.0


def test_retrieval_support_text_answer_grounded():
    # RAG-style: final answer paraphrases the retrieved doc (history tool result)
    retrieval, _ = _step([""], tool="search_docs",
                         result="[refund-policy] cancelled within 48 hours receive a 50% credit")
    answer, _ = _step(["You get a 50 percent travel credit when cancelling within 48 hours."],
                      ctx=["What is the refund policy?"])
    assert retrieval_support(answer, [retrieval]) >= 0.5


def test_retrieval_support_no_evidence_is_zero():
    step, _ = _step(["50 percent credit"], ctx=[])
    assert retrieval_support(step) == 0.0


def test_retrieval_support_no_claims_is_neutral():
    step, _ = _step([""], tool="book_flight", args={}, ctx=["anything"])
    assert retrieval_support(step) == 1.0


def test_retrieval_support_embed_path():
    # fake embedder: one-hot over a fixed vocabulary -> exact cosine
    vocab = ["f3", "f9", "book", "flight", "please"]

    def embed(text):
        import numpy as np

        v = np.zeros(len(vocab))
        for w in text.split():
            if w.casefold() in vocab:
                v[vocab.index(w.casefold())] = 1.0
        return v

    scorer = RetrievalSupport(embed=embed)
    grounded, _ = _step([""], tool="book_flight", args={"flight_id": "F3"},
                        ctx=["Book flight F3 please."])
    ungrounded, _ = _step([""], tool="book_flight", args={"flight_id": "F9"},
                          ctx=["Book flight F3 please."])
    assert scorer(grounded) > 0.5  # "F3" is in the grounding
    assert scorer(ungrounded) == 0.0  # "F9" is nowhere
    assert scorer(grounded) > scorer(ungrounded)


def test_retrieval_support_registered():
    assert get_scorer("retrieval_support") is retrieval_support
