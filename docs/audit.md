# codebase audit, v1 (2026-07-10)

senior-engineer pass over the whole codebase plus an independent automated review.
each finding below is tracked as a github issue. severity: p0 = fundamental, blocks
real production use. p1 = real bug or gap, workaroundable. p2 = quality/debt.

> **resolution (2026-07-11):** all findings below are fixed; see the matching github
> issues (#11-#24) for per-finding notes and the commit. summary: async hooks +
> concurrent k-sampling; per-thread state keyed by langgraph thread_id; judge
> hardened (delimited untrusted content, strict fail-closed verdict parse, per-tool
> scoping, judges first actions too); three-way calibration split + task-level
> bootstrap CIs; fallback narrowed to provider errors; partial trace flush at gate
> decisions; retry budget derived from history; word-boundary fruitless regex; arg
> value normalization; parallel tool calls captured and scored; policies escalate on
> empty signals and survive scorer exceptions; redact hook, uqguard[eval] extra,
> ci matrix 3.11-3.14. open by design: one gate decision per parallel batch
> (documented), sync-path sequential sampling (use ainvoke), P5 benchmarks.

## fixed during audit

- **broken wheel (p0, fixed)**: `uqguard/conformal.py` imports numpy at module top,
  but numpy was only in the dev dependency group. `pip install uqguard` from the wheel
  gave `ModuleNotFoundError` on `import uqguard`. verified in a clean venv, fixed by
  adding numpy to core dependencies, rebuilt and re-verified. lesson recorded: test the
  wheel in a clean venv, not just `uv run` in the dev environment.

## p0, fundamental shortcomings

### 1. no async support
only the sync middleware hooks (`wrap_model_call`, `wrap_tool_call`) are implemented.
langchain raises `NotImplementedError` when an agent using this middleware is invoked
with `ainvoke`/`astream`, which is how production agents actually run. every middleware
class needs the `awrap_*` counterparts (the sync bodies are mostly i/o-free, so the
async versions are thin wrappers around awaited handlers).

### 2. middleware state is not concurrency-safe
`CaptureMiddleware._pending`, `.history`, `._thread`, `GateMiddleware._retries`, and
`ModelFallbackMiddleware.active` are instance attributes on middleware shared across
all conversations of one agent. two concurrent conversations interleave: steps get
attributed to the wrong thread, history mixes tasks, retry budgets leak across users.
`history` and `_retries` also grow without bound in a long-running service.
fix direction: key state by langgraph thread id (available via request.runtime /
config) or store it in graph state instead of the middleware instance. until then the
library is single-conversation-at-a-time, which the docs must say loudly.

### 3. the judge scorer is prompt-injectable
`OptionsSetScorer` interpolates the raw user request and the raw previous tool result
into the judge prompt. a malicious or compromised tool result containing text like
"answer YES" can steer the judge to approve an arbitrary action. a guardrail that can
be switched off by the very content it is guarding is a security hole, not just a
quality gap. related small bug: the verdict parse is `"YES" in text.upper()`, so a
judge reply like "no, unless YES was intended" scores as yes; injected YES anywhere
in the reply also passes. fix direction: structural delimiting of untrusted content,
strict one-token verdict parsing, and a defensive judge system prompt; treat judge
failure or ambiguity as low confidence, not high.

### 4. conformal calibration uses the same split as fusion training
`eval/calibrate.py` fits the logistic fusion and computes the conformal threshold on
the same fit split. the threshold is therefore optimistic on scores the fusion was
trained to separate, and this plausibly contributes to the observed miscoverage
(P(accept given wrong) = 0.269 on test vs the 0.10 target). fix: three-way task-level
split (fusion train / conformal calibration / test). the deeper exchangeability issue
(steps within a task are correlated) additionally needs per-step-type calibration and
larger n, as already documented.

## p1, real bugs and gaps

### 5. demo retry loop corrupts gated conversations
`run_task` retries `agent.invoke` after a 429 with the same `thread_id`. with a
checkpointer attached (gate mode), the retry appends a second copy of the user message
to the checkpointed conversation and resumes stale state against a freshly reset
fake api, so the conversation references bookings that no longer exist. fix: fresh
thread id per attempt, or delete the thread checkpoint before retrying.

### 6. gate retry budget never resets and leaks memory
`GateMiddleware._retries` is keyed by thread id and never cleared: a rerun of the same
thread inherits a spent budget, and the dict grows forever in a service. reset on
`new_thread`, or key by (thread, step) with eviction.

### 7. options-set scorer is blind without a prior tool result
the judge only fires when `history[-1].tool_result` exists. a state-changing call as
the first action (the demo's "cancel my booking" refund, which never searches first)
returns 1.0 and the tie-break goes ungated. fix direction: judge against the tool
schema and current request when no options list exists, or require a listing call
before destructive tools via per-tool policy.

### 8. only the first tool call per candidate is scored
candidates with parallel tool calls are reduced to their first call for agreement
scoring, and one gate decision covers every tool call in the step. models that batch
calls get weaker protection exactly when blast radius is larger.

### 9. pending steps are lost on crash or unresumed interrupt
a step is only written to the trace when its tool result arrives with the next model
call or at `after_agent`. a crash mid-task, or an interrupt that is never resumed,
silently drops the last step including its gate evidence, which is the step you most
want in a post-incident audit. observed during the calibration run crash. fix: flush
pending with a `partial: true` marker on gate decisions and on exception paths.

### 10. model fallback catches every exception
`ModelFallbackMiddleware` falls back on any `Exception`, including programming errors
in downstream middleware, so real bugs get masked as provider failures (already marked
with a ponytail comment, now tracked): narrow to provider/network error types and
re-raise the rest.

### 11. fruitless-result heuristics are substring matches
`tool_churn` marks results containing "error", "no ", "not found", "none", "[]", "{}"
as fruitless. substrings hit legitimate content ("none of the premium seats", text
containing "no "). word-boundary regex plus a configurable predicate per tool is the
fix; the ponytail comment names the nli upgrade path.

### 12. argument agreement has no value normalization
`arg_agreement` compares exact json: "NYC" vs "nyc", "2026-03-03" vs "03/03/2026",
或 an int vs numeric string count as disagreement, inflating escalations on providers
with sloppy formatting. field-level normalizers (case, dates, numerics) before
comparison; `json.dumps(default=str)` also silently equates non-serializable objects.

### 18. policies crash on empty signals (found by independent review)
`ThresholdPolicy.decide` and `RoutedPolicy.decide` call `min(step.signals.values())`
and `min(step.signals, key=...)`; with `scorers=()`, or if every scorer raises (a
judge outage when options_set is the only scorer), that is a `ValueError` inside the
graph. guard the empty case (empty signals should read as escalate-by-default, not
crash) and catch per-scorer exceptions in the policy, scoring a failed scorer as
missing rather than fatal.

## p2, quality and debt

### 13. traces persist raw user text, tool args and results unbounded
pii and secrets land in `runs/*.jsonl` and in interrupt payloads stored by the
checkpointer, with no redaction hook or retention guidance. needs a redaction callback
on `TraceWriter` and a data-handling section in the readme.

### 14. eval statistics on 49 test steps
auroc/ece/coverage on n=49 with no confidence intervals; add bootstrap cis to
`eval/calibrate.py` output so small-n results read as what they are.

### 15. ci matrix does not include the dev python
ci tests 3.11/3.12; the dev box and lockfile run 3.14. add 3.13/3.14 to the matrix.

### 16. tests and eval import the demo via sys.path insertion
`tests/test_demo.py` and `eval/calibrate.py` mutate sys.path to import
`examples/demo_agent.py`. fine at this size, tracked as debt: make examples a proper
package or move the fake api into a testing module.

### 17. judge cost scales with every gated step
when `options_set` is in the policy scorer list it runs on every scored step, not just
state-changing ones. per-tool scorer config (read-only vs destructive tools) is the
same mechanism needed by finding 7.

### 18b. LogisticFusion needs sklearn but no optional extra declares it
sklearn is lazily imported inside `fit()`, so core import works, but users reaching
for LogisticFusion get an ImportError with no guidance. add a `uqguard[eval]` extra
(scikit-learn, matplotlib) and mention it in the readme.

## process notes

- the api key used during development appeared in a chat transcript; rotate it.
- publish requires a pypi account; dist/ artifacts are built and import-verified.
- all numbers in the readme regenerate from `eval/calibrate.py`; keep it that way.
