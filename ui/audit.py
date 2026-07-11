"""Audit UI: per-run timeline of every gated decision.

Run: uv run streamlit run ui/audit.py
Reads the JSONL traces in runs/ (or a directory passed with --). Each step
shows the fused confidence, the per-signal breakdown, all k sampled
candidates, the tool result, the gate outcome, and any human override.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from uqguard import read_trace  # noqa: E402

GATE_BADGE = {
    "PROCEED": "🟢 PROCEED",
    "RETRY": "🟡 RETRY",
    "CLARIFY": "🟠 CLARIFY",
    "ESCALATE": "🔴 ESCALATE",
    None: "⚪ ungated",
}

st.set_page_config(page_title="uqguard audit", layout="wide")
st.title("uqguard audit trail")

trace_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
traces = sorted(trace_dir.glob("**/*.jsonl"), reverse=True)
if not traces:
    st.warning(f"no traces found under {trace_dir}/ — run the demo with --k first")
    st.stop()

path = st.selectbox("run", traces, format_func=lambda p: str(p.relative_to(trace_dir)))
steps = read_trace(path)
st.caption(f"{len(steps)} steps · {path}")

threads = sorted({s.thread_id for s in steps})
picked = st.multiselect("threads", threads, default=threads)

for step in steps:
    if step.thread_id not in picked:
        continue
    badge = GATE_BADGE.get(step.gate, step.gate)
    conf = f" · confidence {step.confidence:.2f}" if step.confidence is not None else ""
    partial = " · ⚠ partial (step never completed)" if step.partial else ""
    title = f"{badge} · {step.step_id} · {step.chosen.tool_name} {step.chosen.args}{conf}{partial}"
    with st.expander(title):
        left, right = st.columns(2)
        with left:
            st.markdown("**signals**")
            if step.signals:
                for name, value in sorted(step.signals.items()):
                    st.progress(min(max(value, 0.0), 1.0), text=f"{name}: {value:.2f}")
            else:
                st.caption("ungated step (no signals computed)")
            if step.human_action:
                st.markdown(f"**human action:** `{step.human_action}`")
        with right:
            st.markdown(f"**candidates (k={len(step.candidates)})**")
            for i, c in enumerate(step.candidates, 1):
                mark = "→" if i == 1 else " "
                st.code(f"{mark} {c.tool_name} {c.args}", language=None)
        if step.tool_result:
            st.markdown("**tool result**")
            st.code(step.tool_result[:2000], language=None)
