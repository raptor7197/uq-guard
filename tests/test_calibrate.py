"""Per-tool (step-type) conformal calibration -- the exchangeability
mitigation in eval/calibrate.py. These run offline on synthetic rows; no
model calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from calibrate import evaluate, per_tool_thresholds  # noqa: E402


def _rows(n_per_tool=40):
    """Two tools with different confidence->correctness behavior: book_flight
    wrong steps cluster at high confidence (the tie-break class), search wrong
    steps at low confidence. A single global threshold mis-calibrates one of
    them; per-tool thresholds should fix it."""
    rows = []
    for i in range(n_per_tool):
        # book_flight: wrong steps score 0.7-0.85 (unanimous-but-wrong)
        wrong = i % 2 == 1
        conf = 0.75 + 0.1 * (i % 4) if wrong else 0.9 + 0.05 * (i % 4)
        rows.append({"task": i // 5, "kind": "tie", "step": f"b/{i}", "tool": "book_flight",
                     "signals": {"arg_agreement": conf, "tool_churn": conf},
                     "correct": not wrong})
        # search_flights: wrong steps score 0.2-0.35 (fabricated origin)
        wrong = i % 3 == 0
        conf = 0.2 + 0.05 * (i % 4) if wrong else 0.8 + 0.1 * (i % 4)
        rows.append({"task": i // 5 + 100, "kind": "vague", "step": f"s/{i}", "tool": "search_flights",
                     "signals": {"arg_agreement": conf, "tool_churn": conf},
                     "correct": not wrong})
    return rows


def test_per_tool_thresholds_separate_tools():
    from calibrate import LogisticFusion

    rows = _rows()
    # fuse on all rows for a quick threshold-only check (split logic is in evaluate)
    fusion = LogisticFusion().fit([r["signals"] for r in rows], [r["correct"] for r in rows])
    for r in rows:
        r["_conf"] = fusion(r["signals"])
    thrs = per_tool_thresholds(rows, alpha=0.1)
    assert set(thrs) == {"book_flight", "search_flights"}
    # the tie-break tool needs a high threshold; the split tool a low one
    assert thrs["book_flight"] > thrs["search_flights"]


def test_evaluate_per_tool_reports_and_applies():
    rows = _rows()
    results, (conf, y) = evaluate(rows, alpha=0.1, seed=7, per_tool=True)
    assert results["thresholds_per_tool"]
    assert results["per_tool"]["book_flight"]["n_test"] > 0
    assert results["per_tool"]["search_flights"]["n_test"] > 0
    # wrong-accept rate on test must respect the alpha bound per tool
    for tool, info in results["per_tool"].items():
        if info["n_accepted_test"]:
            assert info["accepted_error_test"] <= 0.15  # alpha=0.1 + small-n slack


def test_evaluate_global_matches_legacy_keys():
    rows = _rows()
    results, _ = evaluate(rows, alpha=0.1, seed=7, per_tool=False)
    # headline keys that existed before per-tool work must still be present
    for key in ("accepted_error_test", "n_accepted_test", "coverage_test",
                "wrong_accept_rate_test", "ece_fused_test", "auroc"):
        assert key in results
    assert results["thresholds_per_tool"] == {}
    assert results["per_tool"] == {}
