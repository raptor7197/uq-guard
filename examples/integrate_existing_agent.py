"""Bolt uqguard onto an EXISTING agent without touching its internals.

demo_agent.py builds the guard into the agent from the start; this example
starts with a plain, pre-existing customer-support agent (search orders,
refund orders, cancel subscriptions) and shows the four steps to wrap it:

1. Build the agent exactly as you do today -- build_agent() below is a
   normal langchain.agents.create_agent call, unchanged by the integration.
2. At create_agent time, pass middleware=guard.middleware and a
   checkpointer (interrupts require one).
3. Handle interrupts in your invoke loop: ESCALATE -> approve/reject,
   CLARIFY -> free-text answer. Resume with Command(resume=...).
4. Keep conversations apart: guard.new_thread(thread_id) at the start of
   each conversation, and pass thread_id in the invocation config.

Plus the config knobs: per-tool strictness via ToolConfig (destructive
tools get threshold=1.0 so any doubt reaches a human), the options_set
judge scorer (enables the CLARIFY route), PII redaction for the JSONL
trace, and a multi-turn task showing the full dialog reaching the scorers.

Run:  uv run python examples/integrate_existing_agent.py [--k 5] [--gate] [--judge]
      # non-interactive answers: UQGUARD_HUMAN=approve UQGUARD_CLARIFY="order O-1042"
"""

import argparse
import logging
import os
import re

from demo_agent import resolve_model  # same model-resolution chain as the other demos

log = logging.getLogger("integrate")

ORDERS = [
    {"id": "O-1041", "customer": "C-7", "date": "2026-02-11", "amount": 89.0, "item": "headset"},
    {"id": "O-1042", "customer": "C-7", "date": "2026-03-04", "amount": 210.0, "item": "keyboard"},
    {"id": "O-1043", "customer": "C-7", "date": "2026-03-27", "amount": 45.0, "item": "mouse"},
    {"id": "O-1050", "customer": "C-9", "date": "2026-05-02", "amount": 620.0, "item": "monitor"},
]

SUBSCRIPTIONS = [
    {"id": "S-101", "customer": "C-7", "tier": "pro", "active": True},
    {"id": "S-102", "customer": "C-7", "tier": "basic", "active": True},
]

# clear tasks have exactly one acceptable action; ambiguous ones have several
# equally-valid readings (or none the model could justify) -- the silent wrong
# guesses the guard exists to catch. Task 2 is multi-turn.
TASKS = [
    {
        "id": 1,
        "turns": ["Refund order O-1042."],
        "ambiguous": False,
        "accept": {"O-1042"},
    },
    {
        "id": 2,
        "turns": ["Show me my orders.", "Refund the one from March."],
        "ambiguous": True,  # O-1042 and O-1043 are both from March
        "accept": {"O-1042", "O-1043"},
    },
    {
        "id": 3,
        "turns": ["I lost my monitor order from May. Cancel it."],
        "ambiguous": False,
        "accept": {"O-1050"},
    },
    {
        "id": 4,
        "turns": ["Cancel my subscription."],
        "ambiguous": True,  # C-7 has two active subscriptions
        "accept": {"S-101", "S-102"},
    },
]


class FakeSupportAPI:
    """In-memory 'world' with checkable ground truth: .refunded / .cancelled."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.refunded = set()
        self.cancelled = set()

    def search_orders(self, customer_id: str) -> str:
        hits = [o for o in ORDERS if o["customer"] == customer_id.upper()]
        return str([{k: o[k] for k in ("id", "date", "amount", "item")} for o in hits])

    def refund_order(self, order_id: str) -> str:
        if order_id.upper() not in {o["id"] for o in ORDERS}:
            return f"Error: unknown order {order_id}."
        self.refunded.add(order_id.upper())
        return f"Refunded {order_id.upper()}."

    def cancel_subscription(self, subscription_id: str) -> str:
        if subscription_id.upper() not in {s["id"] for s in SUBSCRIPTIONS}:
            return f"Error: unknown subscription {subscription_id}."
        self.cancelled.add(subscription_id.upper())
        return f"Cancelled {subscription_id.upper()}."


def build_agent(api, model=None, middleware=(), checkpointer=None):
    """The EXISTING agent. No uqguard imports here: this is what the project
    already has, and the integration does not change a single line of it."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def search_orders(customer_id: str) -> str:
        """List orders for a customer id, e.g. C-7."""
        return api.search_orders(customer_id)

    @tool
    def refund_order(order_id: str) -> str:
        """Refund a single order by order id, e.g. O-1042."""
        return api.refund_order(order_id)

    @tool
    def cancel_subscription(subscription_id: str) -> str:
        """Cancel a subscription by subscription id, e.g. S-101."""
        return api.cancel_subscription(subscription_id)

    return create_agent(
        resolve_model(model),
        tools=[search_orders, refund_order, cancel_subscription],
        system_prompt=(
            "You are a customer-support assistant. Use the tools to complete "
            "the user's request, then briefly confirm what you did. Never ask "
            "the user questions; if a request is open to interpretation, use "
            "your best judgment and act."
        ),
        middleware=list(middleware),
        checkpointer=checkpointer,
    )


def redact_pii(step):
    """Scrub PII before a step is written to the JSONL trace: order and
    subscription ids become hashes, so raw customer data never leaves the
    process. TraceWriter calls this on every write."""
    step = step.model_copy(deep=True)
    for cand in [step.chosen, *step.candidates]:
        cand.args = {
            k: _hash(v) if re.fullmatch(r"(?:O|S)-\d+", str(v)) else v for k, v in cand.args.items()
        }
    return step


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode()).hexdigest()[:8]


def guarded_agent(api, args):
    """The whole integration in one function. Every call site in an existing
    project that builds an agent picks this up by changing `build_agent(...)`
    to `build_agent(..., middleware=guard.middleware, checkpointer=...)`."""
    from langgraph.checkpoint.memory import InMemorySaver

    from uqguard import Guard, ToolConfig

    scorers: list[str | object] = ["arg_agreement", "tool_churn"]
    if args.judge:
        from uqguard.scorers import OptionsSetScorer

        # one judge call per gated step; scope it to the destructive tools
        scorers.append(
            OptionsSetScorer(
                resolve_model(args.model), tools=("refund_order", "cancel_subscription")
            )
        )

    guard = Guard(
        k=args.k,
        scorers=scorers,
        threshold=args.threshold,
        tool_config={
            # destructive tools: any doubt goes to a human, never to a retry
            "refund_order": ToolConfig(threshold=1.0, routes={"arg_agreement": "ESCALATE"}),
            "cancel_subscription": ToolConfig(threshold=1.0),
        },
        trace_dir=args.trace_dir,
        redact=redact_pii,  # traces carry order/subscription ids
    )
    agent = build_agent(
        api,
        model=args.model,
        middleware=guard.middleware,  # [CaptureMiddleware, GateMiddleware]
        checkpointer=InMemorySaver(),  # interrupts (ESCALATE/CLARIFY) need one
    )
    return guard, agent


def human_review(payload):
    """Surface an interrupt to the human. Non-interactive answers via
    UQGUARD_HUMAN=approve|reject and UQGUARD_CLARIFY=<text>; else stdin."""
    kind = payload.get("type", "approve")
    print(f"\n=== {'CLARIFICATION NEEDED' if kind == 'clarify' else 'ESCALATED'} ===")
    print(f"  step:       {payload['step_id']}")
    print(f"  action:     {payload['tool']} {payload['args']}")
    print(f"  confidence: {payload['confidence']:.2f}  signals: {payload['signals']}")
    print("  candidates:")
    for c in payload["candidates"]:
        print(f"    - {c['tool']} {c['args']}")
    if kind == "clarify":
        if forced := os.environ.get("UQGUARD_CLARIFY"):
            print(f"  clarification (via UQGUARD_CLARIFY): {forced}")
            return {"answer": forced}
        return {"answer": input("  your clarification> ").strip()}
    if forced := os.environ.get("UQGUARD_HUMAN"):
        print(f"  decision (via UQGUARD_HUMAN): {forced}")
        return {"action": forced}
    while (ans := input("  approve/reject> ").strip().lower()) not in ("approve", "reject"):
        pass
    return {"action": ans}


def converse(agent, config, user_text):
    """Send one user message and drive any interrupts to resolution."""
    from langgraph.types import Command

    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]}, config)
    while result.get("__interrupt__"):
        decision = human_review(result["__interrupt__"][0].value)
        result = agent.invoke(Command(resume=decision), config)
    return result


def check(task, api):
    """True iff the agent did exactly one acceptable state-changing action."""
    if "Cancel" in task["turns"][0] or task["id"] == 4:
        done = api.cancelled
    else:
        done = api.refunded
    return len(done) == 1 and done <= task["accept"], done


def run_task(guard, agent, api, task, args):
    thread = f"task{task['id']}"
    config = {"configurable": {"thread_id": thread}}
    if guard:
        guard.new_thread(thread)  # fresh scorer history + retry budget
    api.reset()
    for turn in task["turns"]:
        converse(agent, config, turn)
    ok, done = check(task, api)
    tag = "AMBIG" if task["ambiguous"] else "CLEAR"
    verdict = "OK" if ok else ("WRONG" if done else "NO-ACTION")
    print(f"task {task['id']:>2} [{tag}] -> {sorted(done) or '[]'} {verdict}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, help="run a single task id (default: all)")
    ap.add_argument("--model", help="override model, e.g. openai:gpt-4o-mini")
    ap.add_argument("--k", type=int, default=5, help="candidate samples per model call")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="gate tool calls on candidate agreement (needs --k >= 2)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="escalate when fused confidence < threshold",
    )
    ap.add_argument(
        "--judge",
        action="store_true",
        help="add the options_set judge scorer (one extra model call per gated step)",
    )
    ap.add_argument("--trace-dir", default="runs", help="where runs/*.jsonl traces land")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress step-level logs")
    args = ap.parse_args()
    if args.gate and args.k < 2:
        ap.error("--gate needs --k >= 2 (agreement over one sample is meaningless)")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api = FakeSupportAPI()
    if args.gate:
        guard, agent = guarded_agent(api, args)
        print(f"guarded agent: {len(TASKS)} tasks, k={args.k}, threshold={args.threshold}")
    else:
        guard, agent = None, build_agent(api, model=args.model)
        print("unguarded agent (add --gate to run the guard)")

    tasks = [t for t in TASKS if args.task in (None, t["id"])]
    ok = sum(run_task(guard, agent, api, t, args) for t in tasks)
    print(f"\n{ok}/{len(tasks)} tasks acceptable")
    if guard:
        print(f"trace: {guard.trace_path}")


if __name__ == "__main__":
    main()
