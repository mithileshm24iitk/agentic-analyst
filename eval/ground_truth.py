"""Ground-truth computation for Phase 6 evaluation (database and combined questions).

Parses policy rules from data/docs/*.md and joins them with live BigQuery results
from reference SQL queries, so ground truth always reflects the current public
dataset while enforcing deterministic document rules.
"""

import re
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from tools.bigquery_tools import _to_json_safe, get_client

DEFAULT_RETURN_RATE_THRESHOLD_PCT = 11.0


def parse_return_rate_thresholds(docs_dir: Path) -> tuple[dict[str, float], float]:
    """Parse per-category maximum acceptable return rates (in %) from returns_policy.md."""
    text = (docs_dir / "returns_policy.md").read_text(encoding="utf-8")
    thresholds: dict[str, float] = {}
    for category, pct in re.findall(
        r"^(.+?): maximum acceptable return rate is ([\d.]+)%\.$", text, flags=re.M
    ):
        thresholds[category.strip()] = float(pct)
    default_match = re.search(
        r"default maximum acceptable return rate of ([\d.]+)%\.", text
    )
    default_pct = float(default_match.group(1)) if default_match else DEFAULT_RETURN_RATE_THRESHOLD_PCT
    return thresholds, default_pct


def parse_category_owners(docs_dir: Path) -> dict[str, dict[str, str]]:
    """Parse category -> {owner, email, team} from category_owners.md."""
    text = (docs_dir / "category_owners.md").read_text(encoding="utf-8")
    owners: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"^(.+?): category owner is ([^(\n]+?)\s*\(([^)]+)\),\s*(.+?)\.$", flags=re.M
    )
    for category, owner, email, team in pattern.findall(text):
        owners[category.strip()] = {
            "owner": owner.strip(),
            "email": email.strip(),
            "team": team.strip(),
        }
    return owners


def parse_escalation_levels(docs_dir: Path) -> dict[int, dict[str, Any]]:
    """Parse escalation level boundaries and contacts from escalation_process.md."""
    text = (docs_dir / "escalation_process.md").read_text(encoding="utf-8")
    # Default structure matching escalation_process.md:
    # Level 1: > 0 and <= 1.0 percentage point above threshold -> Category Owner
    # Level 2: > 1.0 and <= 2.0 percentage points above threshold -> Priya Nair (Head of Merchandising)
    # Level 3: > 2.0 percentage points above threshold -> Daniel Okafor (Chief Operating Officer)
    l2_match = re.search(r"Level 2:.*?Escalate to the ([^,]+),\s*([^(]+)\(([^)]+)\)", text, flags=re.S)
    l3_match = re.search(r"Level 3:.*?Escalate to the ([^,]+),\s*([^(]+)\(([^)]+)\)", text, flags=re.S)
    return {
        1: {
            "max_excess_pp": 1.0,
            "role": "Category Owner",
            "contact": None,
            "email": None,
            "due_business_days": 5,
        },
        2: {
            "max_excess_pp": 2.0,
            "role": l2_match.group(1).strip() if l2_match else "Head of Merchandising",
            "contact": l2_match.group(2).strip() if l2_match else "Priya Nair",
            "email": l2_match.group(3).strip() if l2_match else "priya.nair@shopco.example",
            "due_business_days": 10,
        },
        3: {
            "max_excess_pp": float("inf"),
            "role": l3_match.group(1).strip() if l3_match else "Chief Operating Officer",
            "contact": l3_match.group(2).strip() if l3_match else "Daniel Okafor",
            "email": l3_match.group(3).strip() if l3_match else "daniel.okafor@shopco.example",
            "due_business_days": None,
        },
    }


def escalation_level_for_excess(excess_pp: float) -> int:
    """Return 0 (no breach), 1, 2, or 3 given excess percentage points above threshold."""
    if excess_pp <= 0:
        return 0
    if excess_pp <= 1.0:
        return 1
    if excess_pp <= 2.0:
        return 2
    return 3


def run_reference_sql(
    sql: str,
    settings: Settings | None = None,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Execute a reference SQL query against BigQuery (or via an injected runner in tests)."""
    if query_runner is not None:
        return query_runner(sql)
    s = settings or get_settings()
    client = get_client()
    rows = client.query(sql).result(timeout=s.bq_query_timeout_seconds)
    return [_to_json_safe(dict(r.items())) for r in rows]


def category_return_rates_sql(dataset: str, year: int | None = None) -> str:
    """Reference SQL for return rate per category (official definition from kpi_definitions.md)."""
    where = f"WHERE EXTRACT(YEAR FROM oi.created_at) = {int(year)}\n" if year is not None else ""
    return (
        "SELECT p.category AS category, "
        "COUNTIF(oi.status = 'Returned') AS returned_items, "
        "COUNT(*) AS total_items, "
        "ROUND(100.0 * COUNTIF(oi.status = 'Returned') / COUNT(*), 4) AS return_rate_pct "
        f"FROM `{dataset}.order_items` AS oi "
        f"JOIN `{dataset}.products` AS p ON oi.product_id = p.id "
        f"{where}"
        "GROUP BY p.category "
        "ORDER BY return_rate_pct DESC"
    )


def compute_category_breaches(
    docs_dir: Path,
    dataset: str,
    year: int | None = None,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Join live category return rates from BigQuery with thresholds and owners from docs."""
    thresholds, default_pct = parse_return_rate_thresholds(docs_dir)
    owners = parse_category_owners(docs_dir)
    levels = parse_escalation_levels(docs_dir)
    rows = run_reference_sql(category_return_rates_sql(dataset, year=year), query_runner=query_runner)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        cat = str(row["category"])
        rate_pct = float(row["return_rate_pct"])
        threshold_pct = thresholds.get(cat, default_pct)
        # Round to 1 decimal place as specified in kpi_definitions.md ("reported as a percentage with one decimal place")
        # and also keep exact excess; a category is in breach if its return rate is above threshold.
        excess_pp = round(rate_pct - threshold_pct, 4)
        owner_info = owners.get(cat, {"owner": "Unknown", "email": "", "team": ""})
        level = escalation_level_for_excess(excess_pp)
        level_info = levels.get(level, {})
        enriched.append(
            {
                "category": cat,
                "returned_items": int(row["returned_items"]),
                "total_items": int(row["total_items"]),
                "return_rate_pct": rate_pct,
                "return_rate_1dp": round(rate_pct, 1),
                "threshold_pct": threshold_pct,
                "excess_pp": excess_pp,
                "breaching": rate_pct > threshold_pct,
                "escalation_level": level,
                "owner": owner_info["owner"],
                "owner_email": owner_info["email"],
                "team": owner_info["team"],
                "escalation_contact": owner_info["owner"] if level == 1 else level_info.get("contact"),
            }
        )
    enriched.sort(key=lambda r: (-r["excess_pp"], -r["return_rate_pct"], r["category"]))
    return enriched


def compute_combined_ground_truth(
    spec: dict[str, Any],
    docs_dir: Path,
    dataset: str,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute the ground truth for a combined question from its check spec.

    Supported `combined_type` values:
      - "breaching_categories": set of categories whose return rate exceeds policy threshold (optional `year`, `owner`, `team`).
      - "worst_breach_owner": category with highest excess over threshold (or highest return rate) + its owner/email/team.
      - "category_escalation": return rate, threshold, excess and escalation level/contact for a specific `category` (optional `year`).
      - "breaching_owners_set": set of category owner names who own at least one breaching category (optional `year`).
      - "kpi_sql": executes `reference_sql` (which applies a KPI or promotion rule from docs) and requires `must_include` + numeric match.
    """
    ctype = spec["combined_type"]
    year = spec.get("year")

    if ctype == "breaching_categories":
        breaches = [
            b for b in compute_category_breaches(docs_dir, dataset, year=year, query_runner=query_runner)
            if b["breaching"]
        ]
        if spec.get("owner"):
            breaches = [b for b in breaches if b["owner"].lower() == spec["owner"].lower()]
        if spec.get("team"):
            breaches = [b for b in breaches if b["team"].lower() == spec["team"].lower()]
        return {
            "type": "set",
            "universe": list(parse_return_rate_thresholds(docs_dir)[0].keys()),
            "expected_set": sorted({b["category"] for b in breaches}),
            "details": breaches,
        }

    if ctype == "breaching_owners_set":
        breaches = [
            b for b in compute_category_breaches(docs_dir, dataset, year=year, query_runner=query_runner)
            if b["breaching"]
        ]
        all_owners = sorted({v["owner"] for v in parse_category_owners(docs_dir).values()})
        return {
            "type": "set",
            "universe": all_owners,
            "expected_set": sorted({b["owner"] for b in breaches}),
            "details": breaches,
        }

    if ctype == "worst_breach_owner":
        breaches = compute_category_breaches(docs_dir, dataset, year=year, query_runner=query_runner)
        by = spec.get("rank_by", "excess_pp")
        worst = max(breaches, key=lambda b: b[by])
        return {
            "type": "fact",
            "must_include": [worst["category"], worst["owner"]],
            "optional_include": [worst["owner_email"], worst["team"], f"{worst['return_rate_1dp']:.1f}%"],
            "details": worst,
        }

    if ctype == "category_escalation":
        target_cat = spec["category"]
        breaches = compute_category_breaches(docs_dir, dataset, year=year, query_runner=query_runner)
        match = next((b for b in breaches if b["category"].lower() == target_cat.lower()), None)
        if match is None:
            raise ValueError(f"Category {target_cat!r} not found in BigQuery results")
        lvl = match["escalation_level"]
        must = [target_cat, f"{match['threshold_pct']:.1f}%"]
        if lvl > 0:
            must.append(f"Level {lvl}")
            if match["escalation_contact"]:
                must.append(match["escalation_contact"])
        return {
            "type": "fact",
            "must_include": must,
            "details": match,
        }

    if ctype == "kpi_sql":
        rows = run_reference_sql(spec["reference_sql"], query_runner=query_runner)
        return {
            "type": "kpi_sql",
            "reference_rows": rows,
            "must_include": spec.get("must_include", []),
            "expected_sources": spec.get("expected_sources", []),
        }

    raise ValueError(f"Unknown combined_type: {ctype!r}")
