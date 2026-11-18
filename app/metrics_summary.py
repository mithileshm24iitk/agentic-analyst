"""GET /metrics/summary: aggregate the JSONL metrics file, overall and per route."""

import json
from pathlib import Path
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    """p-th percentile (0-100) with linear interpolation between ranks (same as numpy's default)."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def read_records(path: Path) -> list[dict[str, Any]]:
    """All valid JSON lines; a missing file is an empty list, broken lines are skipped."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in records]
    known_costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    return {
        "requests": len(records),
        "error_rate": _avg([0.0 if r["status"] == "ok" else 1.0 for r in records]),
        "latency_ms": {
            "avg": _avg(latencies),
            "p50": round(p50, 1) if p50 is not None else None,
            "p95": round(p95, 1) if p95 is not None else None,
        },
        "avg_input_tokens": _avg([r["input_tokens"] for r in records]),
        "avg_output_tokens": _avg([r["output_tokens"] for r in records]),
        "avg_thinking_tokens": _avg([r["thinking_tokens"] for r in records]),
        "avg_cost_usd": _avg(known_costs),
        "requests_with_unknown_cost": len(records) - len(known_costs),
        "avg_llm_calls": _avg([r["llm_calls"] for r in records]),
        "avg_sql_attempts": _avg([r.get("sql_attempts", 0) for r in records]),
        "revision_rate": _avg([1.0 if r.get("revisions", 0) > 0 else 0.0 for r in records]),
    }


def build_summary(path: Path) -> dict[str, Any]:
    records = read_records(path)
    routes = sorted({r.get("route") or "unknown" for r in records})
    return {
        "overall": summarize(records),
        "by_route": {
            route: summarize([r for r in records if (r.get("route") or "unknown") == route]) for route in routes
        },
    }
