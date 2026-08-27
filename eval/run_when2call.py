"""When2Call eval (SPEC P5, When2Call track).

Single-turn tool-decision benchmark: `nvidia/When2Call` (HF dataset, config
"test", split "mcq"). Each example gives a question + a tool set (BFCL-style
JSON schemas) and a 3-way ground truth: "tool_call" / "request_for_info" /
"cannot_answer".

No agent graph is needed here (single-turn, not multi-step) -- k candidates
come straight from `model.bind_tools(...).invoke(...)`, reusing the same
tool_calls-vs-text parsing branch as uqguard.capture.to_candidate. The
arg_agreement signal (do the k samples agree on the same action?) is fed
through the exact fit/cal/test split + conformal calibration in
eval/calibrate.py's evaluate() -- rows here are one-example-per-"task", which
that function already handles correctly.

Scope, stated honestly: UQ-Guard's existing architecture decides a binary
question (call a tool or not -- PROCEED vs not). That binary is the headline
metric. Disambiguating "request_for_info" from "cannot_answer" within the
no-tool bucket is a separate text-classification problem nothing in this
codebase attempts; it's reported only as an unscored breakdown, not folded
into the accuracy number.

Run: uv run python eval/run_when2call.py --n 60 --k 3 [--model ...]
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from demo_agent import resolve_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import _TRANSIENT, evaluate, plots  # noqa: E402

from uqguard.scorers import arg_agreement  # noqa: E402
from uqguard.step import AgentStep, CandidateAction  # noqa: E402

log = logging.getLogger("when2call")
OUT = Path(__file__).parent / "out" / "when2call"


def to_tool_specs(tool_json_strs):
    """BFCL-style tool JSON (`"type": "dict"`) -> flat name/description/parameters
    specs `.bind_tools()` accepts. Recurses because nested properties can also
    carry `"type": "dict"`."""

    def fix_types(node):
        if isinstance(node, dict):
            if node.get("type") == "dict":
                node = {**node, "type": "object"}
            return {k: fix_types(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fix_types(x) for x in node]
        return node

    specs = []
    for s in tool_json_strs:
        d = json.loads(s)
        specs.append(
            {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": fix_types(d.get("parameters", {"type": "object", "properties": {}})),
            }
        )
    return specs


def _to_candidate(msg):
    text = msg.text if isinstance(msg.text, str) else str(msg.content)
    if msg.tool_calls:
        first, *rest = msg.tool_calls
        return CandidateAction(
            tool_name=first["name"],
            args=first["args"],
            raw_text=text,
            extra_calls=[{"name": t["name"], "args": t["args"]} for t in rest],
        )
    return CandidateAction(tool_name="__none__", args={}, raw_text=text)


def _invoke_with_retry(bound, question, attempts=6):
    for attempt in range(attempts):
        try:
            return bound.invoke([{"role": "user", "content": question}])
        except Exception as e:
            s = str(e)
            if not any(m in s for m in _TRANSIENT) or attempt == attempts - 1:
                raise
            log.warning("transient error (%.60s...), waiting 25s", s)
            time.sleep(25)


def build_row(idx, example, model, k):
    specs = to_tool_specs(example["tools"])
    if specs:
        bound = model.bind_tools(specs)
        candidates = [
            _to_candidate(_invoke_with_retry(bound, example["question"])) for _ in range(k)
        ]
    else:
        # no tool could possibly apply -- skip the model call, "no tool" is certain
        candidates = [CandidateAction(tool_name="__none__", args={}, raw_text="")] * k

    step = AgentStep(
        step_id=f"w2c/{idx}", thread_id=f"w2c/{idx}", candidates=candidates, chosen=candidates[0]
    )
    predicted = "call" if step.chosen.tool_name != "__none__" else "no_tool"
    actual = "call" if example["correct_answer"] == "tool_call" else "no_tool"
    return {
        "task": idx,
        "kind": example["correct_answer"],
        "step": step.step_id,
        "tool": predicted,
        "signals": {"arg_agreement": arg_agreement(step, [])},
        "correct": predicted == actual,
        "predicted": predicted,
        "ground_truth": example["correct_answer"],
    }


def no_tool_breakdown(rows):
    """Unscored diagnostic: among examples we predicted "no_tool", how do they
    split across the two ground-truth labels that bucket conflates? Not a
    metric UQ-Guard's binary decision claims to solve -- see module docstring."""
    return dict(Counter(r["ground_truth"] for r in rows if r["predicted"] == "no_tool"))


def run(n, k, model_name, seed):
    from datasets import load_dataset

    ds = load_dataset("nvidia/When2Call", "test", split="mcq")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    model = resolve_model(model_name)

    rows = []
    for i, idx in enumerate(idxs):
        example = ds[idx]
        row = build_row(i, example, model, k)
        rows.append(row)
        log.info(
            "example %d/%d [%s]: predicted=%s correct=%s",
            i + 1,
            len(idxs),
            row["kind"],
            row["predicted"],
            row["correct"],
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    OUT.mkdir(parents=True, exist_ok=True)

    rows = run(args.n, args.k, args.model, args.seed)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1, default=str))

    results, (conf, labels) = evaluate(rows, args.alpha, seed=args.seed, per_tool=False)
    results["no_tool_breakdown"] = no_tool_breakdown(rows)
    plots(conf, labels, out_dir=OUT)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
