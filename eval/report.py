"""Report and baseline generator for Phase 6 evaluation (eval/results.md and eval/baseline.json).

Reads raw per-item evaluation JSON files from eval/results/raw/ (plus optional retrieval
ablation JSON from eval/results/retrieval_ablation.json) and produces:
  - Per-system and per-category task success rates (both strict and excluding infra failures)
  - SQL execution accuracy per system
  - Combined-question set precision, recall, and F1 per system
  - Refusal/safety accuracy per system
  - Retrieval ablation table (vector-only, BM25-only, hybrid RRF)
  - Latency p50/p95, average cost per request, average LLM calls and tokens per system
  - Spot-check log of LLM-judged items with the judge's reason
  - Every failed question with the reason and the system's answer
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.metrics_summary import percentile
from eval.retrieval_eval import ABLATION_OUT, format_ablation_table
from eval.systems import SYSTEM_DESCRIPTIONS

CATEGORIES = ("documents", "database", "combined", "refusal")


def load_raw_results(raw_dir: Path) -> list[dict[str, Any]]:
    """Load all saved (system, question, repeat) JSON files from eval/results/raw/."""
    if not raw_dir.exists():
        return []
    results = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den > 0 else None


def _fmt_pct(rate: float | None) -> str:
    return f"{rate * 100:.1f}%" if rate is not None else "-"


def _avg(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize_system(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all Phase 6 aggregate metrics for one system's raw result items."""
    total = len(items)
    passed = sum(1 for x in items if x["score"]["passed"])
    infra_failures = sum(1 for x in items if x["score"].get("infra_error"))
    non_infra = [x for x in items if not x["score"].get("infra_error")]

    by_category: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        cat_items = [x for x in items if x["category"] == cat]
        cat_non_infra = [x for x in cat_items if not x["score"].get("infra_error")]
        cat_passed = sum(1 for x in cat_items if x["score"]["passed"])
        by_category[cat] = {
            "total": len(cat_items),
            "passed": cat_passed,
            "infra_failures": len(cat_items) - len(cat_non_infra),
            "strict_success": _rate(cat_passed, len(cat_items)),
            "excl_infra_success": _rate(cat_passed, len(cat_non_infra)),
        }

    # SQL execution accuracy across database (and kpi_sql combined) questions
    db_items = [x for x in non_infra if x["category"] == "database"]
    sql_ok_count = sum(1 for x in db_items if x["score"].get("sql_execution_accurate"))
    sql_accuracy = _rate(sql_ok_count, len(db_items))

    # Combined-question set precision & recall
    set_items = [
        x["score"]["set_metrics"]
        for x in non_infra
        if x["category"] == "combined" and x["score"].get("set_metrics") is not None
    ]
    combined_precision = _avg([m["precision"] for m in set_items])
    combined_recall = _avg([m["recall"] for m in set_items])
    combined_f1 = _avg([m["f1"] for m in set_items])

    # Refusal accuracy
    refusal_accuracy = by_category["refusal"]["excl_infra_success"]

    # Latency, tokens, LLM calls, cost from Phase 5 metrics records
    records = [x["metrics_record"] for x in items if x.get("metrics_record")]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    llm_calls = [r["llm_calls"] for r in records if r.get("llm_calls") is not None]
    input_tokens = [r["input_tokens"] for r in records if r.get("input_tokens") is not None]
    output_tokens = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    thinking_tokens = [r["thinking_tokens"] for r in records if r.get("thinking_tokens") is not None]

    return {
        "total": total,
        "passed": passed,
        "infra_failures": infra_failures,
        "strict_success": _rate(passed, total),
        "excl_infra_success": _rate(passed, len(non_infra)),
        "by_category": by_category,
        "sql_execution_accuracy": sql_accuracy,
        "combined_set_count": len(set_items),
        "combined_precision": combined_precision,
        "combined_recall": combined_recall,
        "combined_f1": combined_f1,
        "refusal_accuracy": refusal_accuracy,
        "latency_p50_ms": round(p50, 1) if p50 is not None else None,
        "latency_p95_ms": round(p95, 1) if p95 is not None else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "avg_llm_calls": round(sum(llm_calls) / len(llm_calls), 2) if llm_calls else None,
        "avg_input_tokens": round(sum(input_tokens) / len(input_tokens), 1) if input_tokens else None,
        "avg_output_tokens": round(sum(output_tokens) / len(output_tokens), 1) if output_tokens else None,
        "avg_thinking_tokens": round(sum(thinking_tokens) / len(thinking_tokens), 1) if thinking_tokens else None,
    }


def render_results_markdown(
    raw_results: list[dict[str, Any]],
    retrieval_ablation: dict[str, Any] | None = None,
) -> str:
    """Build the full markdown report for eval/results.md."""
    systems = sorted({r["system"] for r in raw_results})
    summaries = {sys_id: summarize_system([r for r in raw_results if r["system"] == sys_id]) for sys_id in systems}

    lines: list[str] = [
        "# Phase 6 Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 1. Systems Compared",
        "",
    ]
    for sys_id in systems:
        desc = SYSTEM_DESCRIPTIONS.get(sys_id, sys_id)
        lines.append(f"- **System {sys_id}**: {desc}")
    lines.append("")

    # Table 1: Overall & Per-Category Task Success
    lines.extend(
        [
            "## 2. Task Success by System and Category",
            "",
            "| System | Overall (Strict) | Overall (Excl. Infra) | Infra Failures | Documents | Database | Combined | Refusal / Safety |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for sys_id in systems:
        s = summaries[sys_id]
        bc = s["by_category"]
        lines.append(
            f"| **System {sys_id}** | {_fmt_pct(s['strict_success'])} ({s['passed']}/{s['total']}) | "
            f"{_fmt_pct(s['excl_infra_success'])} | {s['infra_failures']} | "
            f"{_fmt_pct(bc['documents']['excl_infra_success'])} ({bc['documents']['passed']}/{bc['documents']['total']}) | "
            f"{_fmt_pct(bc['database']['excl_infra_success'])} ({bc['database']['passed']}/{bc['database']['total']}) | "
            f"{_fmt_pct(bc['combined']['excl_infra_success'])} ({bc['combined']['passed']}/{bc['combined']['total']}) | "
            f"{_fmt_pct(bc['refusal']['excl_infra_success'])} ({bc['refusal']['passed']}/{bc['refusal']['total']}) |"
        )
    lines.append("")

    # Table 2: Specialized Accuracy Metrics (SQL execution, Combined P/R, Refusal)
    lines.extend(
        [
            "## 3. SQL Execution Accuracy, Combined Precision/Recall, and Refusal Accuracy",
            "",
            "| System | SQL Execution Accuracy | Combined Set Precision | Combined Set Recall | Combined Set F1 | Refusal Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for sys_id in systems:
        s = summaries[sys_id]
        lines.append(
            f"| **System {sys_id}** | {_fmt_pct(s['sql_execution_accuracy'])} | "
            f"{_fmt_pct(s['combined_precision'])} | {_fmt_pct(s['combined_recall'])} | "
            f"{_fmt_pct(s['combined_f1'])} | {_fmt_pct(s['refusal_accuracy'])} |"
        )
    lines.append("")

    # Table 3: Latency, Cost, Calls & Tokens
    lines.extend(
        [
            "## 4. Latency, Cost, LLM Calls, and Tokens per Request",
            "",
            "| System | Latency p50 (s) | Latency p95 (s) | Avg Cost / Req (USD) | Avg LLM Calls | Avg Input Tokens | Avg Output Tokens | Avg Thinking Tokens |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for sys_id in systems:
        s = summaries[sys_id]
        p50_s = f"{s['latency_p50_ms'] / 1000:.1f}s" if s["latency_p50_ms"] is not None else "-"
        p95_s = f"{s['latency_p95_ms'] / 1000:.1f}s" if s["latency_p95_ms"] is not None else "-"
        cost_s = f"${s['avg_cost_usd']:.5f}" if s["avg_cost_usd"] is not None else "n/a"
        calls_s = f"{s['avg_llm_calls']:.1f}" if s["avg_llm_calls"] is not None else "-"
        in_tok = f"{s['avg_input_tokens']:.0f}" if s["avg_input_tokens"] is not None else "-"
        out_tok = f"{s['avg_output_tokens']:.0f}" if s["avg_output_tokens"] is not None else "-"
        th_tok = f"{s['avg_thinking_tokens']:.0f}" if s["avg_thinking_tokens"] is not None else "-"
        lines.append(
            f"| **System {sys_id}** | {p50_s} | {p95_s} | {cost_s} | {calls_s} | {in_tok} | {out_tok} | {th_tok} |"
        )
    lines.append("")

    # Table 4: Retrieval Ablation
    lines.extend(["## 5. Retrieval Ablation (Document Questions, No LLM)", ""])
    if retrieval_ablation:
        lines.append(format_ablation_table(retrieval_ablation))
    else:
        lines.append("_Retrieval ablation has not been run yet (`make eval-retrieval`)._")
    lines.append("")

    # Section 6: LLM Judge Spot-Check Log
    judged = [r for r in raw_results if r.get("score", {}).get("judge")]
    lines.extend(["## 6. LLM Judge Verdicts (for Spot-Checking)", ""])
    if not judged:
        lines.append("_No items required an LLM judge call in this run._")
    else:
        lines.extend(
            [
                "| System | ID | Category | Judge Verdict | Judge Reason |",
                "| :--- | :--- | :--- | :---: | :--- |",
            ]
        )
        for r in judged:
            j = r["score"]["judge"]
            verdict_str = "PASS" if j.get("passed") else "FAIL"
            reason_clean = str(j.get("reason", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {r['system']} | `{r['id']}` | {r['category']} | {verdict_str} | {reason_clean} |")
    lines.append("")

    # Section 7: Failed Questions List
    failed = [r for r in raw_results if not r["score"]["passed"]]
    lines.extend(["## 7. Failed Questions", ""])
    if not failed:
        lines.append("All evaluated questions passed.")
    else:
        for r in failed:
            ans = (r.get("response") or {}).get("answer") or "(no answer returned)"
            ans_compact = " ".join(ans.split())[:400]
            kind = "INFRA FAILURE" if r["score"].get("infra_error") else "WRONG ANSWER"
            lines.extend(
                [
                    f"### System {r['system']} — `{r['id']}` ({r['category']}, repeat {r.get('repeat', 1)}) [{kind}]",
                    f"- **Question**: {r['question']}",
                    f"- **Failure Reason**: {r['score']['reason']}",
                    f"- **System Answer**: {ans_compact}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def write_report_and_baseline(
    raw_dir: Path = Path("eval/results/raw"),
    report_path: Path = Path("eval/results.md"),
    baseline_path: Path = Path("eval/baseline.json"),
    ablation_path: Path = ABLATION_OUT,
    update_baseline: bool = True,
) -> dict[str, Any]:
    """Generate eval/results.md from raw results and optionally write eval/baseline.json from System C."""
    raw_results = load_raw_results(raw_dir)
    ablation = None
    if ablation_path.exists():
        try:
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ablation = None

    markdown = render_results_markdown(raw_results, retrieval_ablation=ablation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    c_items = [r for r in raw_results if r["system"] == "C"]
    baseline_data: dict[str, Any] = {}
    if c_items:
        c_summary = summarize_system(c_items)
        smoke_items = [r for r in c_items if r.get("smoke")]
        smoke_summary = summarize_system(smoke_items) if smoke_items else c_summary
        baseline_data = {
            "system": "C",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "full_total": c_summary["total"],
            "full_strict_success": c_summary["strict_success"],
            "full_excl_infra_success": c_summary["excl_infra_success"],
            "smoke_total": smoke_summary["total"],
            "smoke_strict_success": smoke_summary["strict_success"],
            "smoke_excl_infra_success": smoke_summary["excl_infra_success"],
            "by_category": c_summary["by_category"],
            "sql_execution_accuracy": c_summary["sql_execution_accuracy"],
            "combined_precision": c_summary["combined_precision"],
            "combined_recall": c_summary["combined_recall"],
            "refusal_accuracy": c_summary["refusal_accuracy"],
        }
        if update_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(baseline_data, indent=2) + "\n", encoding="utf-8")

    return baseline_data
