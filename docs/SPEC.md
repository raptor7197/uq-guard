# UQ-Guard — Build Spec (v1)

> `prd.md` = research deep-dive + landscape. This file = what to build, with what, in what order.
> Scope guardrail: LangGraph-only, one primary benchmark, no probe training. Anything not in a phase below is v2.

---

## 1. Product in Five Lines

`pip install uqguard`. Wrap an existing LangGraph agent in ≤10 lines. Before each tool call executes, UQ-Guard samples k candidate actions, scores uncertainty from multiple signals, fuses them into one calibrated confidence, and gates: **PROCEED** / **RETRY_WITH_CONTEXT** / **CLARIFY_USER** / **ESCALATE_HUMAN** (via LangGraph `interrupt`). Conformal calibration bounds the error rate of accepted actions by user-set α. Every decision lands in a JSONL trace viewable in an audit UI.

```python
from uqguard import Guard, policies

guard = Guard(
    scorers=["arg_agreement", "semantic_entropy", "tool_conflict", "retrieval_support"],
    policy=policies.Conformal(alpha=0.1),
    on_escalate=my_review_handler,
)
app = guard.wrap(my_langgraph_app)
```

## 2. Core Data Model (build first, everything hangs off it)

```python
class CandidateAction(BaseModel):
    tool_name: str
    args: dict
    raw_text: str
    logprob: float | None          # None for black-box

class AgentStep(BaseModel):
    step_id: str
    thread_id: str
    candidates: list[CandidateAction]   # k samples
    chosen: CandidateAction
    retrieval_context: list[str]        # docs visible to the model this step
    tool_result: str | None             # filled post-execution
    signals: dict[str, float]           # scorer_name -> score
    confidence: float | None            # fused
    gate: Literal["PROCEED", "RETRY", "CLARIFY", "ESCALATE"] | None
    human_action: str | None            # approve / edit / reject
```

`GateDecision` = (gate, confidence, per-signal breakdown, threshold used). One JSONL line per step.

## 3. Tech Stack

| Layer | Choice | Why / notes |
|---|---|---|
| Language / tooling | Python 3.11+, `uv`, hatchling, ruff, pytest | Zero-config modern packaging |
| Agent framework | `langgraph` (>=1.0) + `langchain` | HITL middleware + `interrupt` already do pause/approve/edit/reject — drive it, don't rebuild it |
| Schemas | pydantic v2 | AgentStep/GateDecision above |
| LLM access | Any LangChain chat model; demo on an OpenAI-compatible endpoint | Works with hosted APIs and local (Ollama/vLLM) unchanged. k-sampling = k async calls at temperature ~0.7 (the `n` param is unreliable with tool calls — don't depend on it) |
| Arg-agreement scorer | stdlib only | Normalized exact-match / majority vote across k sampled tool calls. No ML deps. Ship first |
| Semantic entropy scorer | `uqlm` (black-box scorers) OR own NLI clustering | UQLM is LangChain-native and light — default. `lm-polygraph` = optional extra (`uqguard[whitebox]`), it drags torch/research deps |
| NLI (conflict + clustering) | `cross-encoder/nli-deberta-v3-base` via `transformers` | CPU-viable, standard |
| Embeddings (retrieval support) | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | Small on purpose; swap for -large later |
| Conformal | ~20 lines of numpy (split conformal = quantile of calibration nonconformity scores) | MAPIE/crepes are overkill for thresholding one scalar; keep `crepes` in dev-deps as a cross-check |
| Fusion | `sklearn` LogisticRegression | Weighted-sum fallback when no labels |
| Metrics / plots | scikit-learn (AUROC), matplotlib (reliability, risk-coverage) | `netcal` only if ECE-by-hand annoys you |
| Eval harness | `deeplearning-wisc/agentuq` (τ²-bench) + `nvidia/When2Call` (HF datasets) | agentuq's logprob AUROC/AUARC = published baseline to beat |
| Audit UI | Streamlit reading the JSONL trace | One file. Not React, not a platform |
| Tracking | JSONL first; W&B optional flag | |
| CI | GitHub Actions: ruff + pytest on 3.11/3.12 | |

## 4. Repo Layout

```
uqguard/
  step.py                    # AgentStep, CandidateAction, GateDecision
  capture.py                 # k-sampling wrapper around the model node
  scorers/
    base.py                  # Scorer protocol + registry (register_scorer)
    agreement.py             # Phase 2
    entropy.py               # Phase 3
    conflict.py              # Phase 3
    support.py               # Phase 3
  fusion.py                  # Phase 4
  conformal.py               # Phase 4
  policy.py                  # gate thresholds -> PROCEED/RETRY/CLARIFY/ESCALATE
  integrations/langgraph.py  # Guard.wrap(); interrupt wiring
  trace.py                   # JSONL writer/reader
examples/demo_agent.py       # toy travel-booking agent, 3 tools, seeded ambiguous tasks
eval/run_tau2.py             # Phase 5
eval/run_when2call.py        # Phase 5
ui/audit.py                  # Phase 6 (Streamlit)
tests/
```

---

## 5. Phases

Sized for a side project (~6–10 h/wk). Each phase ends with a runnable exit test — don't start the next phase until it passes. Phases 0–2 produce the demoable core; everything after deepens it.

### Phase 0 — Scaffold + toy agent (1 weekend)
**Build:** `uv init`; package skeleton; `examples/demo_agent.py` — LangGraph agent with 3 tools (`search_flights`, `book_flight`, `refund`) over a fake in-memory API with checkable ground truth; 10 seeded tasks, half deliberately ambiguous ("book me the cheap one" when two match). CI.
**Exit test:** `uv run python examples/demo_agent.py --task 3` completes a booking end-to-end; pytest green in CI.
**Why the ambiguous tasks matter:** they are your eval fixture for every later phase — a gate that never fires on task 1–5 and always fires on 6–10 is the smoke test forever.

### Phase 1 — Capture layer (1 weekend)
**Build:** `capture.py` — wrap the model node: fire k async completions (k configurable, default 5, temp 0.7), parse each into `CandidateAction`, pick chosen (first sample or majority), record `AgentStep`, write JSONL via `trace.py`. Capture retrieval context and (post-hoc) tool result.
**Exit test:** run demo → `runs/<id>.jsonl` has one line per step, each with 5 parsed candidate tool calls. `pytest tests/test_capture.py` asserts schema round-trip.
**Gotchas:** tool-call parsing across k samples must tolerate a sample that answers in text instead of calling a tool (that itself is signal — record it as `tool_name="__none__"`); cap k-sampling to the model-node only, never the tool node (cost).

### Phase 2 — First scorer + working gate ← the "it works" moment (1–2 weekends)
**Build:**
- `scorers/agreement.py`: normalized exact-match agreement over `(tool_name, args)` across candidates; score = fraction agreeing with majority. Field-level, so "same tool, different date" scores low. Stdlib only.
- `policy.py`: hardcoded threshold → PROCEED / ESCALATE (two outcomes only).
- `integrations/langgraph.py`: `Guard.wrap()` — on ESCALATE, trigger LangChain HITL middleware / `interrupt` with the GateDecision as payload; resume on approve/edit/reject.
**Exit test:** demo task "book the 9am flight" → proceeds untouched. Task "book me the cheap one" (two matches) → pauses, prints candidates + agreement score, human approves, resumes. This is the 30-second demo GIF.
**Do NOT:** add more scorers, fusion, or conformal yet. One signal, one threshold, working interrupt = the skeleton everything bolts onto.

### Phase 3 — Full scorer suite (2 weekends)
**Build:**
- `scorers/base.py`: `Scorer` protocol (`score(step: AgentStep) -> float`) + registry; user-custom scorers via `@register_scorer`.
- `entropy.py`: semantic entropy over candidate raw_texts — NLI-cluster then entropy (or delegate to UQLM's consistency scorers; pick one, benchmark later).
- `conflict.py`: post-execution NLI between agent's stated expectation and tool result (contradiction prob = conflict). Note: this scorer fires *after* execution — it gates the *next* step and feeds RETRY, it can't block the call it scores.
- `support.py`: max cosine sim between chosen action's args/claim and retrieval context chunks; low sim = unsupported.
- Lazy-load models once per process; cache NLI pairs.
**Exit test:** trace lines now carry all 4 signals; on the 10 seeded tasks, ambiguous ones separably score worse on ≥1 signal (assert in test).
**Gotcha:** k=5 NLI clustering is O(k²) pairs = 10 NLI calls per step — fine on CPU; don't prematurely optimize.

### Phase 4 — Fusion + conformal calibration (2 weekends)
**Build:**
- Labels: run demo agent over ~200 generated tasks with programmatic ground truth (fake API makes correctness checkable) → per-step correct/wrong labels.
- `fusion.py`: LogisticRegression on the 4 signals → confidence. Fallback `WeightedSum` for label-free users.
- `conformal.py`: split conformal — hold out calibration steps, nonconformity = 1 − confidence, threshold = ⌈(n+1)(1−α)⌉/n quantile. Expose `policies.Conformal(alpha)`.
- `policy.py`: full outcome set — PROCEED / RETRY_WITH_CONTEXT (bounded, feeds tool results + support evidence back) / CLARIFY_USER (request underspecified → interrupt with a question, not an approval) / ESCALATE_HUMAN.
- Diagnostics: reliability diagram, risk-coverage curve, ECE, empirical-coverage-vs-α plot (`uqguard.diagnostics`).
- Docs honesty: conformal assumes exchangeability; agent steps within a trajectory aren't. Calibrate per step-type/tool, report empirical coverage, say so in README.
**Exit test:** α=0.1 → empirical wrong-action rate among PROCEED steps ≤ ~10% on held-out tasks; fused AUROC > best single signal (assert both in `tests/test_calibration.py`).

### Phase 5 — Real eval (2–3 weekends, the resume-defining part)
**Build:**
- `eval/run_tau2.py`: τ²-bench via agentuq harness; UQ-Guard signals vs their logprob baseline → step-level AUROC/AUARC.
- `eval/run_when2call.py`: map gate outcomes to When2Call's call/ask/can't-answer labels → gate-decision accuracy.
- Baselines: mean token logprob, verbalized confidence, semantic entropy alone, static always-ask-on-tool-X policy.
- Headline plots: (1) wrong-action rate vs cost Pareto, (2) human-interruption rate vs wrong-action rate (vs the static policy), (3) ablation table (each signal off).
- Every number from `make eval`. No invented percentages.
**Exit test:** README results table generated by script; fused beats logprob baseline AUROC on τ²-bench; interruption-rate curve dominates static policy.

### Phase 6 — Ship (1–2 weekends)
**Build:** `ui/audit.py` — Streamlit timeline per run: each step, confidence, signal breakdown, gate, human override, drill-down to raw k samples. README: 10-line quickstart, results table, landscape positioning (vs LM-Polygraph/TruthTorchLM/UQLM = answers not actions; vs KnowNo = middleware not paper code; vs LangChain HITL = confidence-gated not name-gated). Publish to PyPI. 3-min demo video.
**Exit test:** fresh venv, `pip install uqguard`, quickstart works copy-paste; `streamlit run ui/audit.py` renders a real run.

---

## 6. Cut List (explicitly v2 — do not touch in v1)

SEP/UHeads probes, non-LangGraph adapters, adaptive/learned conformal thresholds (arXiv 2502.06884), W&B dashboards, multi-benchmark sweeps, hosted anything.
