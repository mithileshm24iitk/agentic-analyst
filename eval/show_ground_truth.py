"""make eval-ground-truth: print the ground truth of every golden item, for checking by hand.

For each item it prints how the ground truth is obtained, then:
  documents: the expected sources (all required) and, for every must_include pattern,
             the first line in those files that matches it (or NOT FOUND)
  database:  the reference SQL (paste it into the BigQuery console) and the rows it returns
  combined:  the SQL, a table of every category (return rate, threshold, excess, level, owner)
             for return-rate items, and the expected set or facts; kpi_sql items show SQL + rows
  refusal:   the state checks and the judge expectation (or the decline rule)

Usage: python -m eval.show_ground_truth [--id comb_07] [--category combined]
Needs BigQuery access for database and combined items (no LLM calls).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from eval.ground_truth import (
    category_return_rates_sql,
    compute_category_breaches,
    compute_combined_ground_truth,
    run_reference_sql,
)
from eval.run_eval import GOLDEN_PATH, load_golden
from eval.scoring import JUDGED_REFUSAL_TYPES

RETURN_RATE_TYPES = {"breaching_categories", "breaching_owners_set", "worst_breach_owner", "category_escalation"}


def ground_truth_kind(item: dict[str, Any]) -> str:
    """How an item's ground truth is obtained: fixed text, computed live, or judged."""
    spec = item["check"]
    if item["category"] == "documents":
        return "FIXED (document text)"
    if item["category"] == "database":
        return "COMPUTED (live BigQuery)"
    if item["category"] == "combined":
        return "COMPUTED (live BigQuery + rules parsed from documents)"
    if spec.get("refusal_type") in JUDGED_REFUSAL_TYPES:
        return "JUDGED (LLM judge vs fixed expectation) + state checks"
    return "RULE (state checks + decline pattern)"


def find_pattern_lines(pattern: str, sources: list[str], docs_dir: Path) -> list[str]:
    """First matching 'file:line: text' for the pattern in each expected source, or []."""
    hits = []
    for src in sources:
        path = docs_dir / src
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(pattern, line, flags=re.I) or pattern.lower() in line.lower():
                hits.append(f"{src}:{n}: {line.strip()[:140]}")
                break
    return hits


def format_breach_table(rows: list[dict[str, Any]]) -> str:
    header = f"    {'category':<30} {'returned':>8} {'items':>7} {'rate %':>8} {'thresh %':>8} {'excess pp':>9} {'lvl':>3}  owner"
    lines = [header, "    " + "-" * (len(header) - 4)]
    for r in sorted(rows, key=lambda r: r["category"]):
        mark = " *" if r["breaching"] else ""
        lines.append(
            f"    {r['category']:<30} {r['returned_items']:>8} {r['total_items']:>7} {r['return_rate_pct']:>8.4f} "
            f"{r['threshold_pct']:>8.1f} {r['excess_pp']:>+9.4f} {r['escalation_level']:>3}  {r['owner']}{mark}"
        )
    lines.append("    (* = above threshold; level from escalation_process.md: L1 <= 1.0 pp, L2 <= 2.0 pp, L3 > 2.0 pp)")
    return "\n".join(lines)


def describe_item(
    item: dict[str, Any],
    docs_dir: Path,
    dataset: str,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
    breach_cache: dict[Any, list[dict[str, Any]]] | None = None,
) -> str:
    """Return the printable ground truth for one item."""
    spec = item["check"]
    cache = breach_cache if breach_cache is not None else {}
    out = [
        "=" * 100,
        f"{item['id']}  [{item['category']}]  smoke={bool(item.get('smoke'))}"
        + ("  drafted_by=ai" if item.get("drafted_by") == "ai" else ""),
        f"Q: {item['question']}",
        f"Ground truth: {ground_truth_kind(item)}",
    ]

    if item["category"] == "documents":
        sources = spec.get("expected_sources", [])
        out.append(f"  expected_sources (ALL required): {sources}")
        for pat in spec.get("must_include", []):
            hits = find_pattern_lines(pat, sources, docs_dir)
            out.append(f"  must_include {pat!r}:")
            if hits:
                out.extend(f"    {h}" for h in hits)
            else:
                out.append("    NOT FOUND in expected sources")

    elif item["category"] == "database":
        out.append(f"  SQL: {spec['reference_sql']}")
        rows = run_reference_sql(spec["reference_sql"], query_runner=query_runner)
        out.append(f"  rows ({len(rows)}): {json.dumps(rows, default=str)}")

    elif item["category"] == "combined":
        ctype = spec["combined_type"]
        out.append(f"  combined_type: {ctype}   expected_sources (ALL required): {spec.get('expected_sources', [])}")
        if ctype in RETURN_RATE_TYPES:
            year = spec.get("year")
            out.append(f"  SQL: {category_return_rates_sql(dataset, year)}")
            if year not in cache:
                cache[year] = compute_category_breaches(docs_dir, dataset, year=year, query_runner=query_runner)
            out.append(format_breach_table(cache[year]))
        gt = compute_combined_ground_truth(spec, docs_dir, dataset, query_runner=query_runner)
        if gt["type"] == "set":
            filters = {k: spec[k] for k in ("year", "owner", "team") if k in spec}
            out.append(f"  expected set {filters or ''}: {gt['expected_set']}")
        elif gt["type"] == "fact":
            out.append(f"  must_include: {gt['must_include']}")
            d = gt["details"]
            out.append(
                f"  details: {d['category']} rate {d['return_rate_pct']:.4f}% vs {d['threshold_pct']:.1f}% "
                f"-> excess {d['excess_pp']:+.4f} pp -> Level {d['escalation_level']} "
                f"(contact: {d['escalation_contact']}; owner: {d['owner']})"
            )
        else:  # kpi_sql
            out.append(f"  SQL: {spec['reference_sql']}")
            out.append(f"  rows: {json.dumps(gt['reference_rows'], default=str)}")

    else:  # refusal
        rtype = spec["refusal_type"]
        out.append(f"  refusal_type: {rtype}   state checks: no write SQL"
                   + (", no saved report" if spec.get("forbid_saved_report") else "")
                   + (", no SQL rows" if rtype == "out_of_scope" else ""))
        if rtype in JUDGED_REFUSAL_TYPES:
            out.append(f"  judge expectation: {spec['judge_expectation']}")
        else:
            out.append("  text check: the answer must clearly decline (DECLINE_PATTERNS in eval/scoring.py)")

    out.append(f"  notes: {spec.get('notes', '')}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the ground truth for every golden item.")
    parser.add_argument("--id", action="append", help="Only this item id (repeatable).")
    parser.add_argument("--category", help="Only this category.")
    args = parser.parse_args(argv)

    settings = get_settings()
    items = load_golden(GOLDEN_PATH)
    if args.id:
        items = [it for it in items if it["id"] in set(args.id)]
    if args.category:
        items = [it for it in items if it["category"] == args.category]

    cache: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        print(describe_item(item, settings.docs_dir, settings.bq_dataset, breach_cache=cache))
    print(f"\n{len(items)} item(s). Dataset: {settings.bq_dataset}. Thresholds/owners/levels from {settings.docs_dir}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
