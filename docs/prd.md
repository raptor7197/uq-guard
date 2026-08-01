# UQ-Guard — Deep Dive + PRD

## Uncertainty Quantification Middleware for LLM Agents

---

## The One-Paragraph Version

Every LLM agent in production has the same silent failure mode: it takes a confident wrong action. It calls the wrong API, fills the wrong form field, books the wrong date — and nothing in the stack knew it was unsure. There's a large research literature on *measuring* LLM uncertainty (semantic entropy, kernel language entropy, conformal methods), and there are research libraries that implement the scoring math. But almost nobody has packaged this as **agent-level middleware**: a thing that sits inside a LangGraph/tool-calling loop, scores the uncertainty of each *decision* (not just each text generation), and *gates the action* — proceed, retry with more context, or escalate to a human. UQ-Guard is that middleware. The research gives you the scoring primitives; the engineering contribution is turning per-token/per-answer uncertainty into per-action decision control with calibration guarantees.

---

## Why This Is a Real Gap (Not Already Solved)

The honest landscape, so you can articulate the gap precisely in interviews:

**What exists (scoring the uncertainty of a single text answer):**
- **LM-Polygraph** — the closest existing thing. A Python framework implementing a battery of uncertainty-estimation methods with unified interfaces, plus a benchmark and a chat demo that shows confidence scores. It is explicitly framed as an *engineering* contribution to a field that was mostly theoretical. BUT: it's built around scoring text-generation answers in QA settings, not around gating actions inside an agent loop.
- **TruthTorchLM (TTLM)** — a newer library with 30+ "truth methods" for predicting truthfulness of LLM outputs. Broader than LM-Polygraph (adds supervised, document-checking, and tool-based strategies). Again: output-truthfulness focused, not agent-action-gating.
- **UQLM** (CVS Health, arXiv 2507.06196) — `pip install uqlm`; black-box (consistency), white-box (token-probability), LLM-as-judge, and ensemble scorers producing response-level confidence in [0,1]. A third scoring library to wrap/compare. Still response-level, not action-gating — but it is actively maintained and pitched at working engineers, so the README must position against it explicitly.
- **Semantic Entropy Probes (OATML)** — approximates semantic entropy from hidden states of a *single* generation, dropping the 5–10× sampling cost. Great primitive to wrap.
- Various method-specific repos: Kernel Language Entropy (NeurIPS'24), SNNE (ACL'25), Evidential Semantic Entropy (EACL'26), conformal semantic entropy methods.

**What exists at the ACTION level (research prototypes — closer than this PRD's first draft admitted; position against these honestly):**
- **KnowNo** (Ren et al., CoRL 2023, arXiv 2307.01928) — the closest research prior art to the whole idea. Conformal prediction over an LLM planner's candidate next actions; the robot acts alone when the conformal prediction set is a singleton and *asks a human for help* otherwise, with statistical guarantees on task completion. BUT: it scores multiple-choice candidate sets in robot planning, not open-ended tool calls with free-form arguments, and it is paper code, not reusable middleware. Expect "isn't this KnowNo?" in any serious conversation.
- **Agentic UQ / AUQ** (Zhang et al., arXiv 2601.15703, Salesforce) — dual-process framework that turns verbalized uncertainty into active control signals (uncertainty-aware memory + uncertainty-triggered reflection), training-free, with trajectory-level calibration results. Conceptually a competitor to the gating idea; practically a research prototype, not framework middleware.
- **SAGE-Agent** (Suri et al., arXiv 2511.08798) — structured uncertainty over *tool-call parameters* (POMDP + expected-value-of-perfect-information question selection), plus the ClarifyBench benchmark. This overlaps the "argument uncertainty" scorer — cite it rather than claiming that idea as novel; the novel part is fusing it with other signals under conformal control.
- **The 2026 clarification-seeking line** — uncertainty decomposition separating action confidence from request underspecification (arXiv 2606.19559), information-gain-driven clarification (arXiv 2606.03135), clarification in coding agents (arXiv 2603.26233). Validates the RETRY/ESCALATE design. Note their deployment argument: multi-sampling is often ruled out by latency budgets, so prompt-based estimation must be a first-class scorer, not an afterthought.
- **Agentic Abstention** (Luo et al., arXiv 2606.28733) — formalizes "when should an agent stop acting" as a *sequential* decision problem (act / gather-info / abstain per turn), evaluated on 28k+ tasks; finds agents abstain too late or never. Strong motivation citation and eval-design material.
- **agentuq** (`deeplearning-wisc/agentuq`) — the anchor survey's official codebase: a runtime UQ pipeline on τ²-bench capturing token-level logprobs across multi-turn agent-tool interactions, with AUROC/AUARC evaluation CLI. Not a gating layer, but it overlaps the AgentStep capture layer — reuse it as the eval harness and baseline instead of building capture from zero.

**What does NOT exist (the gap that survives the honest landscape review):**
- A middleware layer that operates on *agent steps* — where a "decision" is "call tool X with args Y", not "emit answer Z" — shipped as an installable SDK rather than paper code.
- Action-level uncertainty that fuses multiple signals: generation uncertainty + tool-argument uncertainty + tool-result-conflict + retrieval-support. The 2026 survey on UQ in LLM *agents* explicitly names action-conditional uncertainty as an open problem; every prototype above uses a single signal family.
- Uncertainty-*driven* gating in a production agent framework. LangChain/LangGraph now ship a native human-in-the-loop middleware (`humanInTheLoopMiddleware` + `interrupt`), but it gates on a **static tool-name policy** ("always interrupt on `execute_sql`"). Nothing decides *when* to interrupt from calibrated confidence. UQ-Guard's one-line pitch: **replace name-based interrupts with calibrated confidence-based interrupts** — fewer human interruptions at the same (or bounded) error rate.
- A gating policy layer with conformal calibration guarantees, wired into that existing interrupt machinery.
- A clean SDK a working engineer can `pip install` and wrap around their existing agent in ten lines.

So the pitch is: **the scoring math is solved and available; the agent-integration and decision-control layer is not.** You are the systems engineer who bridges research primitives into a usable reliability tool. That's exactly the AI-engineering (not AI-research) value proposition you want.

---

## What You Can Reuse vs. What You Build

### Reuse (don't reinvent)
| Component | Use this | Notes |
|---|---|---|
| Core UQ scoring methods | **LM-Polygraph** (`pip install lm-polygraph`), **TruthTorchLM**, and/or **UQLM** | Batteries of implemented methods with unified interfaces. Wrap these rather than reimplementing semantic entropy from scratch. UQLM is the most engineer-friendly of the three (LangChain-compatible generation API). |
| Agent-step UQ capture + eval harness | **agentuq** (`deeplearning-wisc/agentuq`) | τ²-bench pipeline that already captures per-turn logprobs and computes AUROC/AUARC. Reuse for the eval and as the published baseline to beat. |
| HITL interrupt plumbing | LangChain `humanInTheLoopMiddleware` / LangGraph `interrupt` | Don't rebuild approve/edit/reject flows. UQ-Guard supplies the *decision* of when to interrupt; the middleware supplies the mechanics. |
| Cheap single-pass semantic entropy | **OATML/semantic-entropy-probes** | If you want low-latency scoring without k-sampling. |
| Semantic clustering (NLI-based) | HF NLI models (e.g., DeBERTa-MNLI) | Standard for grouping semantically-equivalent samples before entropy. |
| Embeddings for similarity signals | `BAAI/bge-large-en-v1.5` or similar | For inter/intra-cluster similarity and tool-result conflict scoring. |
| Conformal calibration | **MAPIE** or **crepes** (Python conformal libs) | For distribution-free abstention guarantees. |
| Agent framework + HITL | **LangGraph** | Its `interrupt` + checkpointer machinery is literally built for "pause and ask a human". |
| Calibration metrics/plots | `netcal` (calibration lib), scikit-learn, matplotlib | ECE, reliability diagrams, risk-coverage curves. |
| Experiment tracking | **W&B** | Track calibration across method configs. |

### Build (your original contribution)
1. **The AgentStep abstraction** — a wrapper that captures, for each agent decision: the sampled candidate actions, their arguments, the retrieval context, and any tool results, then routes them to scorers.
2. **The action-level signal fusion** — combine generation uncertainty (from LM-Polygraph) + argument-level uncertainty + tool-result conflict + retrieval support into one calibrated action-confidence score. This fusion is the novel bit.
3. **The gating policy engine** — thresholds (conformally calibrated) mapping confidence → {proceed, retry-with-more-context, escalate-to-human}, with configurable risk tolerance.
4. **The LangGraph integration** — decorators/nodes that drop into an existing graph with minimal code.
5. **The audit UI** — a per-run view showing every decision, its confidence, which signals drove it, and what the gate did. This is the demo centerpiece.

---

## Full Reference List (papers + repos that touch this)

### Foundational uncertainty methods
1. Kuhn et al. (2023), *Semantic Uncertainty / Semantic Entropy* — the seminal cluster-then-entropy method. (ICLR 2023)
2. Farquhar et al. (2024), *Detecting hallucinations in LLMs using semantic entropy* — Nature paper, the high-profile SE result.
3. Nikitin et al. (2024), *Kernel Language Entropy* (NeurIPS'24) — von Neumann entropy over semantic-similarity kernels; generalizes SE. Repo: `AlexanderVNikitin/kernel-language-entropy`.
4. Nguyen et al. (2025), *Beyond Semantic Entropy (SNNE)* (ACL Findings'25) — pairwise semantic similarity, better for longer responses. Repo: `BigML-CS-UCLA/SNNE`.
5. OATML, *Semantic Entropy Probes* — single-pass SE approximation from hidden states. Repo: `OATML/semantic-entropy-probes`.
6. Kunitomo-Jacquin et al. (2026), *Evidential Semantic Entropy (EVSE)* (EACL'26) — evidence theory for unobserved-answer uncertainty. Repo: `lucieK-J/EvidentialSemanticEntropy`.
7. *LLMs UQ via Adaptive Conformal Semantic Entropy* (2026, arXiv 2605.04295) — conformal calibration + semantic entropy with finite-sample abstention guarantees. Directly relevant to your gating layer.
8. *Position: UQ in LLMs is Just Unsupervised Clustering* (2026, arXiv 2605.19220) — useful framing paper; good for your README's "how these methods relate" section.

### Agent-level uncertainty (the actual gap)
9. Oh et al. (2026), *Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities* (ACL 2026, arXiv 2602.05073) — **your anchor paper**. Names action-conditional uncertainty for interactive agents as open/unimplemented. NOTE: the arXiv v1 of this same paper was titled *Towards Reducible Uncertainty Modeling for Reliable LLM Agents* — it is ONE paper, retitled between versions, not two. Official codebase: `deeplearning-wisc/agentuq` (τ²-bench UQ pipeline).
10. Ren et al. (2023), *Robots That Ask For Help: Uncertainty Alignment for LLM Planners* (KnowNo, CoRL 2023, arXiv 2307.01928) — conformal gating of LLM planner actions with human escalation and task-completion guarantees. The closest research ancestor of the gating policy engine; cite it and differentiate (multiple-choice robot plans vs. open-ended tool calls; paper code vs. middleware).
11. Zhang et al. (2026), *Agentic Uncertainty Quantification* (AUQ, arXiv 2601.15703) — verbalized uncertainty as bi-directional control signals (uncertainty-aware memory + reflection); training-free, trajectory-level calibration.
12. Luo et al. (2026), *Agentic Abstention: Do Agents Know When to Stop Instead of Act?* (arXiv 2606.28733) — sequential act/gather/abstain formulation, 28k-task evaluation; agents abstain too late or never.
13. Suri et al. (2025), *Structured Uncertainty guided Clarification for LLM Agents* (SAGE-Agent, arXiv 2511.08798) — POMDP/EVPI uncertainty over tool-call *parameters*; ClarifyBench benchmark. Prior art for the argument-uncertainty scorer.
14. Matsnev (2026), *Uncertainty Decomposition for Clarification Seeking in LLM Agents* (arXiv 2606.19559) — separates action confidence from request underspecification; argues prompt-based estimation is the latency-viable family at deployment.
15. *Uncertainty-Aware Clarification in LLM Agents with Information Gain* (2026, arXiv 2606.03135) — information-gain objective for when to ask.
16. Edwards & Schuster (2026), *Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents* (arXiv 2603.26233) — clarification on underspecified SWE-bench; decouples underspecification detection from execution.
17. Feng et al. (2024), *DiverseAgentEntropy* (arXiv 2412.09572) — black-box uncertainty via agreement across query perturbations; useful scorer for API-only models.
18. Ni, Fadeeva et al. (2025), *Reasoning with Confidence: … Uncertainty Heads* (UHeads, arXiv 2511.06209) — lightweight (<10M param) heads over frozen-LLM internals for step-level verification; the natural v2 alternative to SEP probes.

### Gating, abstention & conformal decision layer
19. Tayebati et al. (2025), *Learning Conformal Abstention Policies for Adaptive Risk Management in LLMs/VLMs* (arXiv 2502.06884) — RL-learned adaptive conformal thresholds; the v2 upgrade path for static gating thresholds.
20. Gui et al. (2024), *Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees* (arXiv 2405.10301) — certifying outputs against a trust criterion with FDR-style guarantees.
21. *LLMs Uncertainty Quantification via Adaptive Conformal Semantic Entropy* (2026, arXiv 2605.04295) — conformal calibration + semantic entropy with finite-sample abstention guarantees. Directly relevant to your gating layer.
22. Kirichenko et al. (2025), *AbstentionBench* (arXiv 2506.09038) — abstention is unsolved and reasoning fine-tuning *degrades* it; motivation for external gating rather than trusting the model to abstain.
23. Wang et al. (2026), *Are LLM Decisions Faithful to Verbal Confidence?* (RiskEval, arXiv 2601.07767) — models do not convert stated confidence into risk-sensitive decisions, even under extreme penalties; the single best motivation citation for an external gate.
24. *Entropy Alone is Insufficient for Safe Selective Prediction in LLMs* (2026, arXiv 2603.21172) — supports the multi-signal fusion thesis against single-score gating.
25. *Position: UQ in LLMs is Just Unsupervised Clustering* (2026, arXiv 2605.19220) — consistency-based UQ is blind to "confident hallucinations"; your tool-conflict and retrieval-support signals are the answer to this critique — say so in the README.

### Engineering libraries (what you wrap/compare against)
26. Fadeeva et al. (2023), *LM-Polygraph* (EMNLP'23 demo) — battery of UE methods, unified Python API, benchmark. PyPI: `lm-polygraph`.
27. Yaldiz et al. (2025), *TruthTorchLM* — 30+ truthfulness methods. Repo: `Ybakman/TruthTorchLM`.
28. Bouchard & Chauhan (2025), *UQLM: A Python Package for UQ in LLMs* (arXiv 2507.06196) — response-level scorers (black-box, white-box, judge, ensemble). Repo: `cvs-health/uqlm`.
29. *GuardrailsAI* — document-grounded verification; complementary, useful to position against. Likewise NVIDIA *NeMo Guardrails* (rule-programmable rails, EMNLP'23 demo) — rails are hand-written rules; UQ-Guard gates on measured uncertainty.
30. LangChain `humanInTheLoopMiddleware` — static tool-name-policy interrupts; the plumbing UQ-Guard drives with calibrated confidence instead of a name list.

### Benchmarks for the eval
31. Barres et al. (2025), *τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment* (arXiv 2506.07982) — verifiable tool-calling tasks; agentuq already provides a UQ pipeline on it.
32. Ross et al. (2025), *When2Call: When (not) to Call Tools* (NAACL 2025, arXiv 2504.18851) — evaluates call/clarify/abstain decisions, exactly the gate's output space. Dataset: `nvidia/When2Call`.
33. ClarifyBench (from SAGE-Agent, #13) — multi-turn tool-augmented disambiguation.

### Calibration & conformal tooling
34. **MAPIE** / **crepes** — conformal prediction in Python.
35. **netcal** — calibration metrics and reliability diagrams.

---

# Product Requirements Document — UQ-Guard v1.0

## 1. Problem Statement
LLM agents fail silently by taking confident, wrong actions. There is no lightweight, framework-native way to measure how uncertain an agent is *about a specific action* and to gate that action accordingly. Existing UQ libraries score text answers, not agent decisions.

## 2. Goal & Non-Goals

**Goals**
- Ship a `pip install`-able Python SDK that wraps any LangGraph agent and gates actions by calibrated confidence.
- Fuse multiple uncertainty signals into a single per-action confidence score.
- Provide conformal calibration so users can set a target error rate on accepted actions.
- Provide an audit UI and a reproducible eval showing reduced wrong-action rate at fixed cost/latency budget.

**Non-Goals (v1)**
- Not a general observability platform (LangSmith already exists; integrate, don't compete).
- Not reimplementing UQ math — wrap LM-Polygraph/TTLM.
- Not training custom probes in v1 (SEP-style probes are a v2 stretch).
- Not multi-framework day one — LangGraph first; adapters later.

## 3. Target Users & JD Mapping
Primary user: an AI engineer shipping a tool-calling agent who needs reliability guarantees before production. Maps to 2026 JD phrases: "agent reliability", "LLM evaluation and calibration", "guardrails", "human-in-the-loop systems", "production LLM safety".

## 4. Core Requirements

### 4.1 The `AgentStep` capture layer
- MUST intercept each agent decision point and capture: candidate action(s), sampled variants (k samples), tool name + arguments, retrieval context used, and (post-hoc) tool result.
- MUST support both black-box (API models, no logits) and white-box (local models, logits/hidden states) modes.

### 4.2 Signal scorers (pluggable)
- MUST implement/wrap at least four signal sources:
  1. **Generation uncertainty** — semantic entropy via LM-Polygraph over k sampled actions.
  2. **Argument uncertainty** — variance/disagreement across sampled tool arguments (e.g., do 5 samples agree on the date/ID/amount?).
  3. **Tool-result conflict** — after execution, does the result contradict the agent's stated expectation? (embedding/NLI check).
  4. **Retrieval support** — is the action grounded in retrieved context, or unsupported? (faithfulness-style check).
- MUST allow users to register custom scorers.

### 4.3 Fusion & calibration
- MUST fuse signals into one confidence score (start with a simple weighted/learned combination; document the choice).
- Rationale to state in docs: consistency-only UQ is provably blind to "confident hallucinations" (arXiv 2605.19220), and entropy alone is insufficient for safe selective prediction (arXiv 2603.21172). Tool-result-conflict and retrieval-support anchor the score to *external* evidence — that is why fusion, not a bigger single score, is the design.
- MUST support conformal calibration (via MAPIE/crepes) so accepted-action error rate is bounded by a user-set tolerance α.
- MUST expose calibration diagnostics (ECE, reliability diagram, risk-coverage curve).

### 4.4 Gating policy engine
- MUST map confidence → {PROCEED, RETRY_WITH_CONTEXT, CLARIFY_USER, ESCALATE_HUMAN} via configurable thresholds.
- CLARIFY vs ESCALATE are different failure modes, per the uncertainty-decomposition literature (arXiv 2606.19559): CLARIFY = the *request* is underspecified → ask the end user a cheap question; ESCALATE = the *action* is uncertain/high-stakes → route to a reviewer for approval. Conflating them either spams reviewers or asks users to approve SQL.
- RETRY MUST feed additional context/tools back into the agent and re-score (bounded retries).
- ESCALATE MUST reuse LangChain's HITL middleware / LangGraph `interrupt` (approve/edit/reject semantics) rather than reimplementing pause-resume; UQ-Guard only decides *when* to trigger it.

### 4.5 Audit UI
- MUST render a per-run timeline: each decision, its fused confidence, per-signal breakdown, the gate outcome, and (for escalations) the human's override.
- SHOULD link each score to the raw samples that produced it.

### 4.6 SDK ergonomics
- MUST wrap an existing LangGraph agent in ≤10 lines.
- Example target API:
  ```python
  from uqguard import Guard, policies

  guard = Guard(
      scorers=["semantic_entropy", "arg_agreement", "tool_conflict", "retrieval_support"],
      policy=policies.Conformal(alpha=0.1),   # ≤10% error on accepted actions
      on_escalate=my_human_review_handler,
  )
  app = guard.wrap(my_langgraph_app)   # returns an instrumented graph
  ```

## 5. Evaluation Plan (the resume-defining part)
- **Tasks (concrete, all with verifiable ground truth):**
  - **τ²-bench** (arXiv 2506.07982) — primary. The anchor survey's `agentuq` pipeline already runs UQ on it: reuse the harness, and its logprob-only AUROC/AUARC numbers become your published baseline to beat with fused signals.
  - **When2Call** (NAACL'25, arXiv 2504.18851, `nvidia/When2Call`) — its label space (call / ask follow-up / can't-answer) is literally the gate's output space {PROCEED, CLARIFY, ESCALATE}; report gate-decision accuracy on it.
  - Stretch: **ClarifyBench** (arXiv 2511.08798) for the CLARIFY path.
- **Primary metric**: wrong-action rate at fixed compute budget, with vs. without UQ-Guard. Show a Pareto curve (accuracy vs. cost/latency).
- **Second headline curve (the KnowNo framing)**: human-interruption rate vs. wrong-action rate. The comparison that sells the tool: UQ-Guard vs. LangChain's static name-based HITL policy — fewer interruptions at equal error, or lower error at equal interruptions.
- **Baselines (each maps to a known approach):** raw mean token logprob (agentuq baseline); verbalized confidence (shown decision-unfaithful by RiskEval, arXiv 2601.07767); single-signal semantic entropy (LM-Polygraph); static always-ask-on-tool-X policy (LangChain HITL default).
- **Calibration metrics**: ECE, AUROC of confidence vs. correctness, risk-coverage curve (plus AUARC, to match agentuq's reporting), empirical coverage vs. conformal target α.
- **Ablations**: each signal on/off, to show fusion beats any single signal.
- **Honesty rule**: every number reproducible from a repo script. No invented percentages.

## 6. Milestones (fits in ~10–12 weeks solo)
- **M1 (wk 1–2):** Wrap LM-Polygraph; score a single agent's text actions; get semantic entropy working on k samples.
- **M2 (wk 3–4):** `AgentStep` capture layer + LangGraph integration (proceed/escalate only).
- **M3 (wk 5–6):** Add argument-agreement, tool-conflict, retrieval-support scorers; implement fusion.
- **M4 (wk 7–8):** Conformal calibration + gating policy engine (add retry path); calibration diagnostics.
- **M5 (wk 9–10):** Eval on τ²-bench via the agentuq harness + When2Call gate-decision eval; Pareto + interruption-rate + calibration results in W&B.
- **M6 (wk 11–12):** Audit UI, README with results table + landscape comparison (LM-Polygraph/TTLM/Guardrails), 3-min demo, blog post.

## 7. Success Criteria
- Wrapping a demo agent in ≤10 lines works end to end.
- Measured reduction in wrong-action rate at equal or lower cost, shown on a Pareto curve.
- Fewer human interruptions than a static name-based HITL policy at the same wrong-action rate (the KnowNo-style autonomy curve).
- Beats the agentuq logprob-only baseline on AUROC/AUARC at the step level.
- Conformal coverage empirically matches the target α within tolerance (per step-type, with the exchangeability caveat documented).
- README clearly articulates the gap vs. existing libraries (LM-Polygraph, TruthTorchLM, UQLM, LangChain HITL, KnowNo, AUQ) and cites the anchor agent-UQ survey.

## 8. Risks & Mitigations
- **Latency from k-sampling** → offer SEP-style single-pass mode; make k configurable; cache. Also ship a prompt-based scorer as a first-class citizen — the 2026 clarification-seeking work (arXiv 2606.19559) argues multi-sampling is ruled out by interactive latency budgets in exactly the deployments UQ-Guard targets.
- **Black-box models lack logits** → lean on sampling-based (black-box) methods (incl. DiverseAgentEntropy-style query perturbation, arXiv 2412.09572); document white-box-only features.
- **Conformal validity on sequential agent steps** → split-conformal guarantees assume exchangeable calibration/test data; steps within a trajectory are neither i.i.d. nor exchangeable, and the anchor survey flags uncertainty *dynamics* as an open challenge. Mitigate: calibrate per step-type/tool on step-level correctness labels, state the assumption explicitly in docs, report empirical coverage rather than only the nominal α, and keep adaptive/learned thresholds (arXiv 2502.06884) as the v2 upgrade path. Do NOT market a guarantee the math doesn't give.
- **Interview traps** (have each answer crisp):
  - *"Isn't this just LM-Polygraph / UQLM?"* → those score *answers*; UQ-Guard scores and gates *actions*, fuses agent-specific signals, and wires into HITL.
  - *"Isn't this KnowNo?"* → KnowNo conformally gates multiple-choice robot plans and asks for help; UQ-Guard gates open-ended tool calls with free-form arguments, fuses external-evidence signals, and ships as framework middleware. KnowNo is the intellectual ancestor — say so, then differentiate.
  - *"LangGraph already has human-in-the-loop middleware"* → yes, gated by a static tool-name list. UQ-Guard decides *when* to interrupt from calibrated confidence; it drives that middleware, not replaces it.
- **Gap erosion** → agent-UQ is moving fast (KnowNo '23 → AUQ, Agentic Abstention, clarification-seeking, all early '26). The defensible moat is the *engineering*: fusion + calibration + framework integration + audit UI in one installable package. Re-scan the landscape at each milestone; if someone ships the middleware first, pivot the pitch to benchmark rigor and multi-signal fusion.
- **Scope creep** → v1 is LangGraph-only, no custom probe training, single primary benchmark. Everything else is v2.

## 9. Portfolio Value Summary
This is a *library-shaped* project: it demonstrates API/SDK design, systems thinking, calibration rigor, and reads directly onto the most under-supplied 2026 skill — agent reliability. It's grounded in a named research gap, reuses credible research libraries (so you look like someone who knows the literature, not someone reinventing it), and produces honest, benchmark-backed metrics. A clean repo here plausibly earns organic GitHub stars, which is itself an interview signal.
