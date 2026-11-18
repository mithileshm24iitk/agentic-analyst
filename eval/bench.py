"""Latency benchmark: `make bench` (BENCH_RUNS runs of 5 fixed questions through /ask).

Runs the real app in-process (real Gemini, BigQuery, embeddings), then reads each
request's metrics record (by X-Request-ID) from METRICS_PATH and prints where the
time went. Each answer also gets a simple correctness check, so a faster setting
can be judged on quality too. Results are saved to data/metrics/bench-*.json.

Cost: about 5 x BENCH_RUNS orchestrator requests (each a handful of Gemini calls).
"""

import json
import logging
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.metrics_summary import read_records

# Categories above their limit in the returns policy vs. the live data (checked in Phase 3).
BREACHING = ["Suits", "Underwear", "Active", "Jumpsuits & Rompers", "Plus"]


def _numbers_in(rows: list[dict]) -> set[str]:
    return {str(v) for row in rows for v in row.values() if isinstance(v, (int, float))}


def _digits(text: str) -> str:
    return text.replace(",", "")


def check_docs(body: dict) -> str | None:
    if body["route"] != "docs":
        return f"route {body['route']}"
    if "11" not in body["answer"] or "returns_policy.md" not in body["cited_sources"]:
        return "missing 11% / citation"
    return None


def check_database(body: dict) -> str | None:
    if body["route"] != "sql" or not body["rows"]:
        return f"route {body['route']} rows {len(body['rows'])}"
    if not any(n in _digits(body["answer"]) for n in _numbers_in(body["rows"])):
        return "answer does not contain the queried number"
    return None


def check_breaches(body: dict) -> str | None:
    if body["route"] != "both":
        return f"route {body['route']}"
    missing = [c for c in BREACHING if c not in body["answer"]]
    return f"missing {missing}" if missing else None


def check_out_of_scope(body: dict) -> str | None:
    return None if body["route"] == "none" and not body["agents_called"] else f"route {body['route']}"


def check_revenue(body: dict) -> str | None:
    if body["route"] != "both" or not body["rows"]:
        return f"route {body['route']} rows {len(body['rows'])}"
    sql = body["sql"] or ""
    if "Cancelled" not in sql or "Returned" not in sql:
        return "SQL does not apply the official revenue definition"
    if not re.search(r"\d{3},?\d{3}", body["answer"]):
        return "no revenue figure in answer"
    return None


QUESTIONS: list[tuple[str, str, Callable[[dict], str | None]]] = [
    ("docs", "What is the maximum acceptable return rate for Jeans?", check_docs),
    ("database", "How many order items have the status 'Returned'?", check_database),
    ("combined", "Which product categories are currently above their maximum acceptable return rate?", check_breaches),
    ("out_of_scope", "What will the weather be in Paris tomorrow?", check_out_of_scope),
    ("combined_kpi", "What was ShopCo's total revenue in 2025, using the official revenue definition?", check_revenue),
]


# A call is "stalled" when it is slow but produced little: the time is spent waiting
# on the server, not generating (thinking + output) tokens.
STALL_MIN_MS = 10_000
STALL_MAX_TOKENS_PER_S = 5.0


def is_stalled(call: dict) -> bool:
    generated = call["output_tokens"] + call["thinking_tokens"]
    return call["latency_ms"] >= STALL_MIN_MS and generated / (call["latency_ms"] / 1000) < STALL_MAX_TOKENS_PER_S


class RetryCounter(logging.Handler):
    """Counts google-genai's own retry log lines, to rule out hidden retries as the cause of slow calls."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "retry" in record.getMessage().lower():
            self.count += 1


def table(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def run() -> None:
    settings = get_settings()
    label = settings.thinking_level or "model-default"
    print(f"Benchmark: {settings.bench_runs} runs x {len(QUESTIONS)} questions, "
          f"model={settings.worker_model}, thinking_level={label}\n", flush=True)

    retries = RetryCounter()
    genai_logger = logging.getLogger("google_genai")
    genai_logger.addHandler(retries)
    genai_logger.setLevel(logging.INFO)

    results: list[dict[str, Any]] = []
    with TestClient(app) as client:  # `with` runs the startup hook: real agents
        for key, question, check in QUESTIONS:
            for n in range(1, settings.bench_runs + 1):
                started = time.perf_counter()
                response = client.post("/ask", json={"question": question})
                wall = (time.perf_counter() - started) * 1000
                request_id = response.headers["X-Request-ID"]
                record = next(r for r in read_records(settings.metrics_path) if r["request_id"] == request_id)
                body = response.json()
                # An error (auth, quota, timeout) is not a wrong answer: keep the two apart.
                error = None if response.status_code == 200 else body.get("error_type", str(response.status_code))
                problem = check(body) if error is None else None
                results.append({"key": key, "run": n, "wall_ms": wall, "record": record,
                                "error": error, "correct": error is None and problem is None,
                                "problem": problem, "answer": body.get("answer", "")})
                verdict = f"ERROR {error}" if error else ("correct" if problem is None else f"WRONG ({problem})")
                print(f"  {key} run {n}: {wall / 1000:.1f}s {verdict}", flush=True)

    print()
    for key, question, _ in QUESTIONS:
        rows = []
        for r in (x for x in results if x["key"] == key):
            rec = r["record"]
            slowest = max(rec["llm_detail"], key=lambda c: c["latency_ms"], default=None)
            rows.append([
                f"run {r['run']}",
                f"{rec['latency_ms'] / 1000:.1f}s",
                f"{rec['llm_time_ms'] / 1000:.1f}s",
                f"{rec['tool_time_ms'] / 1000:.1f}s",
                rec["llm_calls"],
                f"{slowest['agent']} {slowest['latency_ms'] / 1000:.1f}s" if slowest else "-",
                rec["thinking_tokens"],
                rec["output_tokens"],
                sum(is_stalled(c) for c in rec["llm_detail"]),
                f"ERROR: {r['error']}" if r["error"] else ("yes" if r["correct"] else f"NO: {r['problem']}"),
            ])
        print(f"[{key}] {question}")
        print(table(["", "total", "LLM", "tools", "calls", "slowest LLM call", "thinking", "output", "stalled", "correct"], rows))
        print()

    # Where does the time go, per agent, across all runs?
    per_agent: dict[str, list[dict]] = {}
    for r in results:
        for c in r["record"]["llm_detail"]:
            per_agent.setdefault(c["agent"], []).append(c)
    rows = []
    for agent, calls in sorted(per_agent.items()):
        lat = [c["latency_ms"] for c in calls]
        rows.append([
            agent, len(calls),
            f"{sum(lat) / 1000:.1f}s",
            f"{statistics.mean(lat) / 1000:.1f}s",
            f"{max(lat) / 1000:.1f}s",
            round(statistics.mean(c["thinking_tokens"] for c in calls)),
            round(statistics.mean(c["output_tokens"] for c in calls)),
        ])
    print("LLM time by agent (all runs)")
    print(table(["agent", "calls", "total", "mean", "max", "mean thinking", "mean output"], rows))

    totals = [r["record"]["latency_ms"] for r in results]
    llm = sum(r["record"]["llm_time_ms"] for r in results)
    tools = sum(r["record"]["tool_time_ms"] for r in results)
    correct = sum(r["correct"] for r in results)
    errors = sum(r["error"] is not None for r in results)
    all_calls = [c for r in results for c in r["record"]["llm_detail"]]
    stalled = [c for c in all_calls if is_stalled(c)]
    stalled_ms = sum(c["latency_ms"] for c in stalled)
    print(f"\nStalled calls (>= {STALL_MIN_MS / 1000:.0f}s and < {STALL_MAX_TOKENS_PER_S:.0f} generated tokens/s): "
          f"{len(stalled)} of {len(all_calls)}, {stalled_ms / 1000:.1f}s = {stalled_ms / sum(totals):.0%} of all time")
    for c in sorted(stalled, key=lambda c: -c["latency_ms"]):
        print(f"  {c['agent']}: {c['latency_ms'] / 1000:.1f}s for {c['output_tokens']} output + "
              f"{c['thinking_tokens']} thinking tokens (traffic={c.get('traffic_type')})")
    print(f"google-genai retries logged: {retries.count}; traffic types: "
          f"{sorted({str(c.get('traffic_type')) for c in all_calls})}")
    print(f"\nOverall: mean {statistics.mean(totals) / 1000:.1f}s, median {statistics.median(totals) / 1000:.1f}s, "
          f"max {max(totals) / 1000:.1f}s; LLM {llm / sum(totals):.0%} of time, tools {tools / sum(totals):.0%}; "
          f"correct {correct}/{len(results)}, errors {errors}")

    out = settings.metrics_path.parent / f"bench-{label}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"thinking_level": label, "runs": settings.bench_runs, "retries_logged": retries.count,
                               "results": results}, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    run()
