# uq-guard progress log

last updated: 2026-08-27. companion to spec.md (build plan) and prd.md (research landscape).

## status: phases 0-6 done (v1 complete, see caveats in what remains)

## what was built

### phase 0, scaffold + toy agent
- uv project, hatchling, ruff, pytest, github actions ci (3.11/3.12)
- `uqguard/step.py`: core data model, `AgentStep` / `CandidateAction` / `Gate` (pydantic)
- `examples/demo_agent.py`: travel booking agent (langchain `create_agent`), 3 tools (`search_flights`, `book_flight`, `refund`) over an in-memory fake api with checkable ground truth
- 10 seeded tasks: 1-5 clear (one correct action), 6-10 ambiguous on purpose
- model resolution by env key: UQGUARD_MODEL, then openrouter, gemini, openai, ollama cloud. `openrouter:<slug>` and `ollama_cloud:<model>` prefixes build hand-configured clients; everything else goes through `init_chat_model`. temperature 0.7 everywhere, k-sampling needs diversity
- task-level retry on 429 with wait, since free-tier keys rate limit constantly

### phase 1, capture layer
- `uqguard/capture.py`: `CaptureMiddleware`, hooks langchain's `wrap_model_call`, executes each model call k times, parses every sample into a `CandidateAction` (first tool call, or `__none__` for text answers). agent acts on sample 1, so unguarded behavior is preserved
- tool results backfilled onto the step when the next model call arrives, flush at `after_agent`
- `uqguard/trace.py`: append-only jsonl, one step per line, `read_trace` loads it back
- `uqguard/fallback.py`: `ModelFallbackMiddleware`, sticky swap to a configured fallback model on any model error. opt-in via UQGUARD_FALLBACK env, no default
- per-sample logging at every step: sampled action, distinct-action count, tool result, gate decision. demo `-q` to silence

### phase 2, first scorer + working gate
- `uqguard/scorers/base.py`: registry, `@register_scorer`, scorer = callable `(step, history) -> float` in [0,1], higher = more confident
- `uqguard/scorers/agreement.py`: `arg_agreement`, fraction of the k candidates matching the majority (tool, args), dict key order normalized, text-instead-of-tool counts as disagreement
- `uqguard/policy.py`: `ThresholdPolicy`, min of signals against a threshold, PROCEED / ESCALATE. min() is a placeholder for phase 4 fusion
- `uqguard/gate.py`: `GateMiddleware`, hooks `wrap_tool_call`, scores the pending step once, on escalate calls langgraph `interrupt` with full evidence (chosen action, confidence, per-signal breakdown, all k candidates). approve executes, reject returns an error toolmessage telling the agent not to retry. needs a checkpointer
- demo: `--gate`, `--gate-threshold`, human review on stdin or UQGUARD_HUMAN=approve|reject for non-interactive runs

### phase 3, scorer suite (revised after taxonomy, see below)
- `uqguard/scorers/conflict.py`: `tool_churn`, doom-loop detector. same tool re-called with different args after fruitless results, confidence decays 1/(1+n). pure logic, no deps
- `uqguard/scorers/options.py`: `OptionsSetScorer`, tie-break detector, knowno-flavored. one judge call: does the request uniquely determine this choice among the prior tool results. no means 0.0. opt-in via `--judge`, costs one model call per gated step
- capture keeps per-thread `history` of flushed steps and stashes the user request in `step.retrieval_context`, the context scorers judge against
- 21 tests, all offline, no llm needed. lint clean

## what was learned (drives everything after)

failure taxonomy from live runs (10 tasks x k=5, two models):

1. underspecification splits: candidates disagree within a step, split bookings, fabricated dates, invented origins (jfk/lhr/lax for a request that never named an origin). caught by arg_agreement
2. tie-breaks: equally valid options, model picks positionally, unanimous across samples, invisible to any consistency method. this is the arxiv 2605.19220 blindspot observed live. caught by the options_set judge
3. doom-loops: perfect within-step agreement while churning invented args across steps against empty results. caught by tool_churn

live validations that all passed:
- clear task with all scorers armed and reject forced: zero interrupts, correct booking
- unanimous tie-break: agreement 1.0, judge 0.0, escalated, arbitrary booking blocked
- doom-loop task: escalated, reject converted the loop into clean abstention
- double-booking incident on task 8: gate escalated at exactly the off-rails step

## decisions log

- cloud apis only, user does not run local models. ollama paths remain in code as explicit opt-ins. local qwen3:4b was 3-9 min per sample on cpu, and that model build ignores think=false
- gemini free tier limits found the hard way: flash is 20 requests per day, flash-lite is rpm-limited but has daily headroom. dev loop runs on flash-lite
- semantic entropy and retrieval-support scorers deferred with reasons: tool-call responses have empty text so text entropy adds nothing over arg_agreement, and there is no retrieval corpus in the demo to test support against. revisit with a rag example
- no heavy ml deps so far: no torch, no transformers, no sentence-transformers. scorers are stdlib logic plus at most one judge call

### phase 4, fusion + conformal calibration (done 2026-07-10)
- `uqguard/fusion.py`: LogisticFusion (sklearn) + WeightedSum fallback, both callables signals -> confidence
- `uqguard/conformal.py`: split conformal threshold (numpy quantile with finite-sample correction), accepted_error, risk_coverage, ece. guarantee stated precisely: bounds P(accept | wrong) under exchangeability, not P(wrong | accepted)
- `uqguard/policy.py` RoutedPolicy: full outcome set, weakest signal routes the intervention: options_set low -> CLARIFY the user, tool_churn low -> ESCALATE to reviewer, agreement low -> RETRY with feedback (bounded, budget exhausted escalates)
- `uqguard/gate.py`: handles all four outcomes, clarify interrupts resume with an answer that is injected back as tool feedback
- `uqguard/guard.py`: Guard facade, the ten-line integration from the spec

### phase 5, calibration + eval (done at demo scale, 2026-07-10)
- `eval/calibrate.py`: seeded task generator (5 balanced templates, controlled ambiguity), capture run with recursion-limit bound on doom loops, offline step labeling (state-changing action on an ambiguous task = wrong, abstention = right), task-level split, fusion fit, conformal threshold, diagnostics json + risk-coverage and reliability pngs. resumable: completed task traces reload from disk, transient network/429 errors retry
- results (29 tasks, k=3, 90 steps, flash-lite, eval/out/results.json): fused auroc 0.865 vs 0.692 agreement-alone and 0.846 churn-alone. accepted-error 23% at 61% coverage vs 53% base wrong rate. honest negative: P(accept | wrong) = 0.269 on test vs alpha 0.1 target, the exchangeability violation we documented, small n plus within-task correlation
- vague tasks doom-looped 4 of 4 times captured, richest label source

### phase 6, ship (done except items below)
- `ui/audit.py`: streamlit trace browser, per-step signals, candidates, gate badges, human overrides. verified serving http 200
- readme rewritten: quickstart, signal table, taxonomy, real results table with the negative finding, positioning table, limitations
- `uv build` produces dist/uqguard-0.0.1 wheel + sdist
- Guard facade shipped, `=4.2.7` junk file deleted

## what remains (post-v1)

full codebase audit 2026-07-10: see audit.md, 18 findings, all tracked as github issues #11-#24 on raptor7197/uq-guard. the broken-wheel packaging bug found during audit was fixed immediately (numpy added to core deps, wheel re-verified in a clean venv).

- pypi publish: build artifacts ready in dist/, publish is a user action (needs pypi account/token)
- judge-inclusive calibration rerun: flash-lite daily quota (500/day) was exhausted 2026-07-10; rerun `eval/calibrate.py --tasks 30 --k 3 --judge` after reset, saved traces reload for free and only judge calls cost quota
- tau2-bench via agentuq harness: real blockers found round 4 (below), not just budget — scoped as a follow-up
- when2call: done round 4 (below)
- demo video
- rotate the gemini api key, it appeared in a chat transcript

## round 2 (2026-08-01): knocked off three tracked items

- **per-tool risk config**: `uqguard/policy.py` gains `ToolConfig` (scorers / threshold / routes per tool); `ThresholdPolicy` and `RoutedPolicy` take `tool_config`, `Guard(tool_config=)` threads it through. this is the mechanism audit findings #7/#17 asked for (judge only on destructive tools, stricter thresholds where blast radius is larger). demo exposes it as `--tool-threshold TOOL:FLOAT` (repeatable).
- **per-step-type conformal calibration**: `eval/calibrate.py` gains `--per-tool` — conformal threshold computed per tool on the calibration split (unseen tools fall back to the global threshold), applied on test, and reported in `results.json["per_tool"]` per tool (threshold, n_cal, n_test, accepted error, coverage) plus `thresholds_per_tool`. this is the exchangeability mitigation the phase-5 data demanded.
- **semantic_entropy + retrieval_support scorers** (deferred since phase 3, now with the rag example): `scorers/entropy.py` — entropy over equivalence-clustered k sampled raw texts, neutral on text-free tool-call steps, pluggable `equivalence` (nli upgrade path); `scorers/support.py` — lexical grounding of claim tokens (arg values or answer text) against request + tool results, pluggable `embed=` (sentence-transformers upgrade path), fail-closed on no evidence. `examples/demo_rag.py` exercises both on a policy-doc store incl. a question that is not in the corpus (both signals collapse together).
- validation: 24 new/updated tests (90 total, all offline), ruff clean, imports verified. the judge-inclusive calibration rerun still needs the user's api key/quota; `--per-tool` and the two new scorers are exercised by unit tests until then.

## round 3 (2026-08-03): priority bugfixes from the audit (issues #35-#40)

- **#35 conformal lockout (high)**: `conformal_threshold`'s `nextafter(t, inf)` nudge is now capped at 1.0. before, a single wrong cal step at confidence 1.0 produced a 1.0000000000000002 threshold and the gate rejected EVERY action forever (including perfect confident-correct ones); the worst case is now a 1.0 threshold whose failure is honestly reported by `accepted_error` / `wrong_accept_rate_test` instead of silently locking the agent out.
- **#36 weighted-sum missing signals**: `WeightedSum` now takes `missing=0.5` + `names=` so an absent signal (judge skipped, scorer raised) imputes neutral instead of re-normalizing away (a missing judge used to read as approval when the other signals agreed). `LogisticFusion` already learned names from fit data; `names=` gives the label-free fallback the same fail-neutral semantics.
- **#37 tool_config dict crash**: `_cfg()` coerces a plain dict `tool_config` entry into `ToolConfig(**cfg)` and raises a clear `TypeError` for anything else, instead of `AttributeError` deep inside `decide()`.
- **#38 k=0 crash**: `Guard` and `CaptureMiddleware` now raise `ValueError` for `k < 1` (was an `IndexError` on `candidates[0]` at the first model call).
- **#39 tiny-dataset eval crash**: `calibrate.evaluate()` raises actionable errors for < 3 tasks (three-way split impossible) and for a one-class fit split (sklearn `ValueError`), instead of a cryptic solver error.
- **#40 per-tool threshold swings**: `per_tool_thresholds(cal, alpha, min_wrong=10)` skips a tool with fewer than 10 wrong calibration steps (below the floor the (n+1) correction needs at alpha=0.1) and it falls back to the global threshold, so tiny splits can't set accept-all (0.0) or reject-all thresholds.
- validation: 9 new/updated tests (99 total, all offline), ruff clean.

## round 3b (2026-08-06): closed the rest of the audit + coverage to 100%

Picked up from a stalled prior session (its transcript was investigated, cross-checked against the actual repo state, then discarded — the 5 commits below matched it exactly, nothing was lost).

- **#34 license**: MIT `LICENSE`, (c) 2026 raptor7197.
- **#41 formatting/CI**: `ruff format` across the codebase, `ruff format --check` added to CI so drift fails the build.
- **#42 unbounded warning registry**: the warned-thread registry is now a bounded `OrderedDict` (cap 1024, evict oldest) so a long-lived deployment with unbounded distinct threads can't grow it forever.
- **#43 run-id collision**: a model call with no `thread_id` in its config now keys state by the per-invocation `run_id` instead of a shared fallback slot, so concurrent no-thread-id invocations stop colliding.
- **#44 single-turn context**: `retrieval_context` now carries every `HumanMessage` in the conversation (bounded by `_join_context`), not just the last one, so multi-turn requests ("book the cheap one" after "I want to go to Paris") are judged against the full dialog.
- `tests/test_edge_cases.py` (21 tests) + `tests/test_properties.py` (13 hypothesis invariants: conformal threshold bounds/monotonicity/finite-sample wrong-acceptance bound, `normalize_value` idempotence, unit-interval scorer outputs, `to_candidate` robustness) — statement coverage 637/637 (was 96%), 137 tests total.
- `examples/integrate_existing_agent.py`: walkthrough for bolting the guard onto an existing customer-support agent (per-tool `ToolConfig`, judge scorer, PII redaction); README updated to match.
- validation: 137 tests, ruff check + format clean. all 5 commits pushed to `origin/main` (`97d66aa..c8aebbd`).

## round 4 (2026-08-27): P5 eval harness, When2Call track

Roadmap phase P5 (github issue #1) calls for real-benchmark eval against tau2-bench and when2call — the one item "what remains" (above) still listed as open going into this round.

- researched tau2-bench + agentuq + when2call before committing to scope. tau2-bench/agentuq turned out to have real blockers, not just extra work: tau2-bench needs python <3.14 (project floor is >=3.11 with no ceiling; this repo's `.venv` is 3.14 — would need an isolated `uv venv --python 3.12` just for it), agentuq is a fixed cli pipeline (`tau2 run` / `extract-uq-from-trajs` / `evaluate-uq`) wrapping *its own* agent loop that doesn't compose with uqguard's langgraph `Guard` middleware, and the roadmap's mean-token-logprob baseline needs real work first — `CandidateAction.logprob` exists in the schema but `capture.py` never populates it today. scoped as a follow-up rather than force it in half-verified against an external repo's api.
- when2call was tractable now: single-turn, 3-way ground truth (`tool_call` / `request_for_info` / `cannot_answer`), no external harness needed beyond `datasets`.
- `eval/run_when2call.py`: samples n rows from `nvidia/When2Call` (test config, mcq split, 3652 rows), converts its bfcl-style tool schemas (`"type": "dict"`, recursively) to `bind_tools()`-compatible specs, k-samples the model directly (no agent graph needed for a single turn — reuses the same tool_calls-vs-text parsing branch as `uqguard.capture.to_candidate`), scores the binary decision uqguard's architecture actually makes (call a tool or not) via `arg_agreement` fed through the exact fit/cal/test split + conformal calibration `eval/calibrate.py` already has. deliberately does not attempt to disambiguate `request_for_info` vs `cannot_answer` within the no-call bucket (a separate text-classification problem nothing in this codebase attempts) — reported only as an unscored breakdown, not folded into the accuracy number.
- `eval/calibrate.py`: `plots()` takes `out_dir=` so `run_when2call.py` reuses it instead of duplicating plotting code.
- results (n=60, k=3, `ollama_cloud:gpt-oss:120b`, seed 7, `eval/out/when2call/results.json`): binary decision accuracy 44/60 = **0.733**. auroc of arg_agreement = 0.555, near chance — honest negative, much weaker than the multi-step booking task's 0.692-0.865. plausible reason: when2call's ambiguity is mostly *whether* to call at all, not *which args* to use, so a model can unanimously agree to call the wrong tool or unanimously decline correctly either way, scoring `arg_agreement=1.0` regardless of correctness. conformal threshold at alpha=0.1 accepted 0/20 test rows — too little calibration data for this signal at this task to clear the bound confidently.
- validation: 10 new tests (147 total, all offline), ruff check + format clean.
- deleted the stray `session-ses_0c23.md` transcript file from the repo root (untracked, not part of the codebase, no longer needed once its contents were cross-checked into round 3b above).
- found in passing (not when2call-specific): `policy._score`'s `getattr(entry, "name", entry.__name__)` evaluates the default arg eagerly, so any scorer *instance* (e.g. `OptionsSetScorer`, which has `.name` but no `__name__`) crashes the moment it's routed through `ThresholdPolicy`/`RoutedPolicy` — every `--judge` example does exactly that. `eval/calibrate.py` never hit it because it calls the judge directly, bypassing policy. Fixed with a short-circuit `or`; regression test added.
- committed (`e2ff7ec` When2Call track + one more for the getattr fix), **not pushed** — pending user go-ahead.
