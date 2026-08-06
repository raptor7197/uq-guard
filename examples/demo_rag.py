"""RAG demo: the two scorers deferred in Phase 3, now exercised.

Phase 3 deferred semantic_entropy and retrieval_support with explicit reasons:
tool-call responses have empty raw_text (so text entropy added nothing over
arg_agreement) and there was no retrieval corpus to test grounding against.
This example is that corpus: a small policy-document store, a search_docs
tool, and prompts that are either answered by the docs or *not in the docs at
all* (the model must fabricate -- the failure the two deferred scorers exist
for).

The final answers are text steps (tool_name="__none__"), so raw_text finally
carries signal: semantic_entropy measures dispersion across the k sampled
answers, retrieval_support measures how much of the answer is grounded in the
retrieved document. A fabricated answer scores low on both.

Run:  uv run python examples/demo_rag.py [--task N] [--k 3] [--model ...]
"""

import argparse
import logging

log = logging.getLogger("demo_rag")

DOCS = {
    "refund-policy": (
        "Refunds: bookings cancelled more than 48 hours before departure are "
        "refunded in full. Cancellations within 48 hours receive a 50% travel "
        "credit, no cash refund."
    ),
    "baggage-policy": (
        "Baggage: each passenger may check one bag up to 23kg free of charge. "
        "Extra checked bags cost $40 each. Carry-on must fit the overhead bin."
    ),
    "loyalty-program": (
        "Loyalty: members earn 1 point per dollar on flights, 2 points per "
        "dollar on hotels. Points expire after 12 months of inactivity."
    ),
}

# grounded prompts are answered by exactly one doc; the last one matches nothing
TASKS = [
    {
        "id": 1,
        "grounded": True,
        "doc": "refund-policy",
        "prompt": "What is the refund policy for bookings cancelled within 48 hours?",
    },
    {
        "id": 2,
        "grounded": True,
        "doc": "baggage-policy",
        "prompt": "How many bags can I check and what does an extra bag cost?",
    },
    {
        "id": 3,
        "grounded": True,
        "doc": "loyalty-program",
        "prompt": "How many loyalty points do I earn per dollar spent on hotels?",
    },
    {
        "id": 4,
        "grounded": False,
        "doc": None,
        "prompt": "What is the compensation policy for weather-delayed flights?",
    },
]


def build_agent(model, middleware=()):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def search_docs(query: str) -> str:
        """Search the policy document store; returns the best-matching policy."""
        q = query.casefold()
        best, best_score = None, 0
        for title, text in DOCS.items():
            score = sum(1 for w in q.split() if w in text.casefold() or w in title)
            if score > best_score:
                best, best_score = f"[{title}] {text}", score
        if best is None or best_score == 0:
            return "No policy document matches that question."
        return best

    return create_agent(
        model,
        tools=[search_docs],
        system_prompt=(
            "You are a customer-support assistant. Answer the user's question "
            "using the retrieved policy document. If the document does not "
            "answer the question, say so clearly. Never invent policy details."
        ),
        middleware=list(middleware),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, help="run a single task id (default: all)")
    ap.add_argument("--model", help="override model, e.g. openai:gpt-4o-mini")
    ap.add_argument("--k", type=int, default=3, help="candidate samples per step")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from demo_agent import resolve_model
    from uqguard import CaptureMiddleware
    from uqguard.scorers import RetrievalSupport, SemanticEntropy

    entropy = SemanticEntropy()  # default normalized-exact equivalence
    support = RetrievalSupport()  # lexical grounding, stdlib

    tasks = [t for t in TASKS if args.task is None or t["id"] == args.task]
    for task in tasks:
        capture = CaptureMiddleware(k=args.k, trace_dir="runs", run_id=f"rag-{task['id']}")
        agent = build_agent(resolve_model(args.model), middleware=[capture])
        capture.new_thread(f"rag{task['id']}")
        agent.invoke(
            {"messages": [{"role": "user", "content": task["prompt"]}]}, {"recursion_limit": 6}
        )
        capture._flush(None)

        # score every flushed step; the last one is the final answer
        history: list = []
        lines = []
        for step in capture.history:
            ent = entropy(step, history)
            sup = support(step, history)
            history.append(step)
            lines.append((step, ent, sup))
        print(f"\ntask {task['id']} (grounded: {task['grounded']}): {task['prompt']!r}")
        for step, ent, sup in lines:
            kind = "ANSWER" if step.chosen.tool_name == "__none__" else step.chosen.tool_name
            print(
                f"  {step.step_id:>14} [{kind:>6}] "
                f"semantic_entropy={ent:5.2f} retrieval_support={sup:5.2f}"
            )
        if not lines:  # no steps flushed (e.g. the run errored): nothing to score
            print("  -> no steps captured (run errored?)")
            continue
        final = lines[-1]
        flag = min(final[1], final[2]) < 0.5
        print(
            f"  -> final answer confidence=min({final[1]:.2f},{final[2]:.2f}) "
            f"= {min(final[1], final[2]):.2f} {'FLAGGED for review' if flag else 'OK'}"
        )
    if args.task is None:
        print(
            "\nnote: trace per task in runs/rag-<id>.jsonl; answer text at "
            "step.chosen.raw_text via read_trace()"
        )


if __name__ == "__main__":
    main()
