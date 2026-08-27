# uqguard

Uncertainty-gated tool calls for LangGraph agents.

LLM agents fail silently by taking confident, wrong actions — booking the wrong flight, refunding the wrong customer — and nothing in the stack knew the agent was unsure. Existing UQ libraries ([LM-Polygraph](https://github.com/IINemo/lm_polygraph), [TruthTorchLM](https://github.com/Ybakman/TruthTorchLM), [UQLM](https://github.com/cvs-health/uqlm)) score the uncertainty of *text answers*. uqguard scores and **gates agent actions**: before a tool call executes, it samples k candidate decisions, fuses multiple uncertainty signals into one calibrated confidence, and routes low-confidence actions to retry, user clarification, or human approval — using LangGraph's native interrupt machinery.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from uqguard import Guard

guard = Guard(k=5, threshold=0.8)
agent = create_agent(model, tools=tools,
                     middleware=guard.middleware,
                     checkpointer=InMemorySaver())
```

Escalations surface as LangGraph interrupts carrying full evidence (chosen action, per-signal breakdown, all k sampled candidates); resume with `Command(resume={"action": "approve"})`, `{"action": "reject"}`, or `{"answer": "..."}` for clarifications.

Works with `invoke` and `ainvoke`/`astream` (async samples the k candidates concurrently). Concurrent conversations on one agent are kept apart by the LangGraph `thread_id` in each invocation's config.

## How it works

1. **Capture** (`CaptureMiddleware`): every model call runs k times; all sampled candidate actions are recorded as an `AgentStep` (JSONL trace). The agent acts on sample 1, so unguarded behavior is preserved.
2. **Score**: pluggable scorers, each `(step, history) -> [0, 1]`:

| scorer | failure mode it catches | cost |
|---|---|---|
| `arg_agreement` | underspecification splits: candidates disagree on tool or args (fabricated dates, invented origins) | free |
| `tool_churn` | doom loops: same tool re-called with mutating args after fruitless results | free |
| `semantic_entropy` | text-answer dispersion (RAG / final answers): the k sampled raw texts say different things | free |
| `retrieval_support` | ungrounded actions: claim tokens (args or answer text) absent from the request + tool results | free (lexical; optional `embed=` upgrade) |
| `OptionsSetScorer` | tie-breaks: unanimous but *arbitrary* pick among equally-valid options (KnowNo-style) | 1 judge call |

Scorers can be scoped **per tool**, which is how you make destructive tools stricter without re-specifying the whole policy — the mechanism the audit asked for to keep judge costs off read-only calls:

```python
from uqguard import Guard, ToolConfig

guard = Guard(
    k=5,
    scorers=["arg_agreement", "tool_churn"],
    tool_config={
        "book_flight": ToolConfig(threshold=1.0),      # any doubt escalates
        "refund": ToolConfig(threshold=1.0,
                              scorers=("arg_agreement", "tool_churn", "options_set")),
    },
)
```

Each `ToolConfig` field (`scorers`, `threshold`, `routes`) falls back to the policy default when unset.

3. **Gate** (`GateMiddleware` + `RoutedPolicy`): fused confidence below the threshold routes by the *weakest signal*, because each signal names its own intervention: request ambiguous → **CLARIFY** the user; agent flailing → **ESCALATE** to a reviewer; model unsure → **RETRY** with feedback (bounded). The CLARIFY route needs `OptionsSetScorer` in the scorer list (it requires a judge model, so the default free scorers can only RETRY/ESCALATE).
4. **Calibrate** (`uqguard.conformal`): split-conformal threshold on a labeled calibration run. Stated precisely: under exchangeability, at most an α fraction of *wrong* steps are accepted; the empirical wrong-rate-among-accepted is reported alongside, never assumed. Agent steps are not i.i.d. — see limitations.

## Why consistency alone is not enough

Live runs of the demo agent produced three distinct failure classes ([docs/progress.md](docs/progress.md)):

- **splits** — samples disagree (caught by agreement),
- **tie-breaks** — samples *unanimously* pick one of several equally-valid options (invisible to any consistency method; caught by the options-set judge),
- **doom-loops** — perfect per-step agreement while churning invented arguments against empty results (caught by churn).

The tie-break class is the "confident hallucination" blindspot described in [arXiv 2605.19220](https://arxiv.org/abs/2605.19220), reproduced in a 200-line fixture. One observed trace: `signals={'arg_agreement': 1.0, 'tool_churn': 1.0, 'options_set': 0.0}` — two signals green, one caught the arbitrary booking. That is the argument for fusion.

## Demo

```bash
uv sync
export GOOGLE_API_KEY=...            # or OPENROUTER_API_KEY / OPENAI_API_KEY / OLLAMA_API_KEY
uv run python examples/demo_agent.py --k 5 --gate --judge          # 10 seeded tasks
uv run python examples/demo_agent.py --k 5 --gate --tool-threshold refund:1.0  # per-tool strictness
uv run python examples/integrate_existing_agent.py --k 5 --gate --judge  # bolt the guard onto an existing agent
uv run python examples/demo_rag.py --k 3                           # semantic_entropy + retrieval_support
uv run python eval/calibrate.py --tasks 30 --k 3 --per-tool        # per-step-type conformal calibration
uv run python eval/run_when2call.py --n 60 --k 3                   # When2Call: call/ask/decline benchmark
uv run streamlit run ui/audit.py                                   # audit trail UI
```

The demo travel agent is instructed to never ask questions — half its tasks are deliberately ambiguous, so it books wrong things silently. With the gate on: clear tasks pass untouched; ambiguous ones pause with evidence.

`examples/integrate_existing_agent.py` is the integration walkthrough: it starts with a plain customer-support agent (`build_agent`, unchanged) and shows the four steps to bolt the guard on — `middleware=guard.middleware` + a checkpointer at `create_agent` time, the interrupt/resume loop, `guard.new_thread()` per conversation, plus per-tool `ToolConfig` strictness, the `options_set` judge, and PII redaction for the trace. Run without `--gate` for the unguarded baseline, with `--gate` for the guarded run.

The RAG demo (`examples/demo_rag.py`) is where the two text-era scorers live: it answers questions from a small policy-document store, and one task is deliberately *not in the docs* — the fabricated answer collapses `semantic_entropy` and `retrieval_support` together, which the travel demo's tool-call steps can't show.

## Results

All numbers below come from `eval/calibrate.py` (generated tasks, programmatic ground truth) and are copied from `eval/out/results.json`. Small-scale run — 29 tasks, k=3, gemini-flash-lite, 90 labeled action steps (41 fit / 49 test) — treat as demonstration, not benchmark. τ²-bench evaluation is still the roadmap's next step.

> **Methodology note:** these numbers predate two fixes now in `eval/calibrate.py`: the split is now three-way at task level (fusion-train / conformal-calibration / test — previously the threshold was computed on the fusion's own training split, which biases it optimistic), and headline metrics now carry task-level bootstrap 95% CIs. The table will be regenerated on the next calibration run.

| metric (test split) | value |
|---|---|
| base wrong-action rate | 0.531 |
| AUROC — fused | **0.865** |
| AUROC — arg_agreement alone | 0.692 |
| AUROC — tool_churn alone | 0.846 |
| wrong-action rate among accepted @ α=0.1 threshold | 0.233 (coverage 0.612) |
| P(accept \| wrong) — the bounded quantity | 0.269 (target ≤ 0.10) |
| per-tool thresholds (`--per-tool`) | computed per step-type on the calibration split; per-tool accepted-error/coverage reported in `results.json["per_tool"]` |

Findings, including the negative one: fusion beats each single signal on AUROC (with only two free signals, logistic fusion ties a plain average — expected). Accepting only above the conformal threshold cuts the wrong-action rate among executed actions from 53% to 23% at 61% coverage. **The conformal bound itself did not transfer**: P(accept | wrong) was 0.269 on test against a 0.10 target — the exchangeability violation we documented up front (steps are correlated within tasks; n is small), not a surprise. Per-step-type calibration and larger calibration sets are the v2 mitigation. The `options_set` judge is excluded from this calibration run (free-tier daily quota); it is validated qualitatively in the live gate runs documented in [docs/progress.md](docs/progress.md).

### When2Call

`eval/run_when2call.py` runs the same fuse-and-calibrate machinery against [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) (`test` config, `mcq` split) — a single-turn, real-world benchmark of whether a model should call a tool, ask a follow-up question, or say it can't help. Numbers below: 60 examples, k=3, `ollama_cloud:gpt-oss:120b`, seed 7, copied from `eval/out/when2call/results.json`.

Scope, stated up front: this only scores the binary decision UQ-Guard's architecture actually makes — call a tool or don't (PROCEED vs not) — against the ground truth's `tool_call` vs `{request_for_info, cannot_answer}`. Disambiguating the latter two is a separate text-classification problem nothing here attempts.

| metric (n=60) | value |
|---|---|
| binary decision accuracy (call vs no-call) | 44/60 = **0.733** |
| AUROC — arg_agreement (only signal available; single-turn has no history for tool_churn) | 0.555 |
| wrong-action rate among accepted @ α=0.1 threshold | undefined — 0 accepted (0% coverage) |
| no-tool predictions, ground-truth split | cannot_answer 20, request_for_info 13, tool_call 3 (false negatives) |

Findings, including the negative one: k=3 sample agreement is a much weaker confidence signal here (AUROC 0.555, near chance) than on the multi-step booking task above (0.692–0.865). Plausible reason: When2Call's ambiguity is mostly in whether calling is appropriate at all, not in which arguments to use — a model can unanimously agree to call the wrong tool, or unanimously decline correctly, either way scoring `arg_agreement=1.0` regardless of correctness. At n=60 the conformal threshold (0.72) accepts nothing on the 20-row test split — too little calibration data for this signal at this task to clear α=0.1 confidently. Consistent-only signals evidently need a task where the *action* is what's uncertain, not just whether to act; When2Call is closer to the tie-break blindspot the options-set judge exists for.

## Positioning

| | scores | gates actions | HITL wiring | calibration |
|---|---|---|---|---|
| LM-Polygraph / TruthTorchLM / UQLM | text answers | ✗ | ✗ | metrics only |
| LangChain HITL middleware | ✗ (static tool-name list) | ✓ | ✓ | ✗ |
| KnowNo (CoRL '23) | multiple-choice robot plans | ✓ (paper code) | ask-for-help | conformal |
| **uqguard** | **agent actions** | **✓ middleware** | **✓ interrupt-native** | **conformal + diagnostics** |

Key research: [KnowNo](https://arxiv.org/abs/2307.01928), [UQ in LLM Agents survey (ACL '26)](https://arxiv.org/abs/2602.05073), [Agentic Abstention](https://arxiv.org/abs/2606.28733), [SAGE-Agent](https://arxiv.org/abs/2511.08798), [uncertainty decomposition for clarification](https://arxiv.org/abs/2606.19559). Full landscape: [docs/prd.md](docs/prd.md).

## Limitations (v1)

- Conformal guarantee assumes exchangeability; sequential agent steps violate it. We calibrate on step-level labels, report empirical coverage, and say so.
- The options-set judge is an LLM judging an LLM — it inherits judge errors and costs one call per scored step (scope it with `OptionsSetScorer(model, tools=("book_flight",))`). Untrusted content in the judge prompt is delimited and the verdict parse is strict/fail-closed, but prompt injection against LLM judges can never be fully ruled out.
- One gate *decision* per step even with parallel tool calls (all calls in the batch are part of the agreement signal, and the one decision blocks or permits the whole batch); LangGraph only; no learned probes. See [docs/SPEC.md](docs/SPEC.md) for the cut list.

## Data handling

Traces persist the raw user request, tool args, and tool results to `runs/*.jsonl`. If those can carry PII or secrets, pass a scrubber: `Guard(redact=my_fn)` / `TraceWriter(redact=my_fn)` — it receives each `AgentStep` before writing and returns the step to persist. Gate evidence is flushed early with `partial: true` so a crash or never-resumed interrupt can't lose the step under audit; the completed line supersedes it on read.

## Development

```bash
uv sync && uv run pytest && uv run ruff check .
```

Layout: `uqguard/` (capture, scorers, policy, gate, fusion, conformal, guard facade, plus a sticky provider-fallback middleware) · `examples/` (demo agent + RAG demo) · `eval/` (calibration) · `ui/` (audit) · `docs/` (prd, spec, progress log).
