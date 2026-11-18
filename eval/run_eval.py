"""Phase 6 Evaluation Runner (`eval/run_eval.py`).

Features:
  - Parallelism: runs up to `--concurrency` (default 4, from EVAL_CONCURRENCY) tasks in parallel.
  - Caching & resume: saves every (system, question, repeat) result to
    `eval/results/raw/<system>__<id>__r<repeat>.json` as soon as it finishes.
    Re-running skips completed items (items that failed due to infrastructure can be
    retried on resume).
  - Infrastructure retry: distinguishes infra failures (timeout, 503, quota, auth) from
    wrong answers and retries an infra failure once before recording it.
  - Cost & call estimation: before a full run, prints estimated LLM calls and estimated
    USD cost from the Phase 5 benchmark averages, and waits for confirmation.
  - Flags: `--smoke`, `--system`, `--category`, `--repeats N`, `--concurrency N`, `--yes`,
    `--check-baseline`, `--max-drop-points`.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from eval.judge import JudgeFn, make_llm_judge
from eval.report import summarize_system, write_report_and_baseline
from eval.retrieval_eval import ABLATION_OUT, evaluate_retrieval, format_ablation_table
from eval.scoring import JUDGED_REFUSAL_TYPES, is_infra_error, score_item
from eval.systems import SYSTEM_DESCRIPTIONS, ask_in_process, build_system

GOLDEN_PATH = Path("eval/golden.jsonl")
RAW_DIR = Path("eval/results/raw")
REPORT_PATH = Path("eval/results.md")
BASELINE_PATH = Path("eval/baseline.json")

# Phase 5 measured averages per question (from data/metrics/bench-LOW-20261118-213834.json):
# System C (full orchestrator + sub-agents + critic):
#   docs: ~5 calls ($0.0045), database: ~6 calls ($0.0047), combined: ~9 calls ($0.0256), refusal: ~2 calls ($0.0014)
# Weighted across 12 docs + 12 db + 10 combined + 6 refusal = 40 questions:
PHASE5_CALLS_PER_Q: dict[tuple[str, str], float] = {
    ("A", "documents"): 2.2,
    ("A", "database"): 3.0,
    ("A", "combined"): 4.0,
    ("A", "refusal"): 1.0,
    ("B", "documents"): 4.0,
    ("B", "database"): 5.0,
    ("B", "combined"): 8.0,
    ("B", "refusal"): 1.0,
    ("C", "documents"): 5.0,
    ("C", "database"): 6.0,
    ("C", "combined"): 9.0,
    ("C", "refusal"): 2.0,
    ("D", "documents"): 5.0,
    ("D", "database"): 6.0,
    ("D", "combined"): 9.0,
    ("D", "refusal"): 2.0,
}

PHASE5_COST_PER_Q_USD: dict[tuple[str, str], float] = {
    ("A", "documents"): 0.0020,
    ("A", "database"): 0.0024,
    ("A", "combined"): 0.0110,
    ("A", "refusal"): 0.0007,
    ("B", "documents"): 0.0033,
    ("B", "database"): 0.0042,
    ("B", "combined"): 0.0225,
    ("B", "refusal"): 0.0009,
    ("C", "documents"): 0.0045,
    ("C", "database"): 0.0047,
    ("C", "combined"): 0.0256,
    ("C", "refusal"): 0.0014,
    ("D", "documents"): 0.0065,
    ("D", "database"): 0.0068,
    ("D", "combined"): 0.0320,
    ("D", "refusal"): 0.0020,
}

JUDGE_COST_PER_CALL_USD = 0.0009


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_questions(
    items: list[dict[str, Any]],
    *,
    smoke: bool = False,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = items
    if smoke:
        selected = [it for it in selected if it.get("smoke")]
    if categories:
        wanted = {c.strip().lower() for c in categories}
        selected = [it for it in selected if it["category"].lower() in wanted]
    return selected


def cache_file_for(raw_dir: Path, system_id: str, question_id: str, repeat: int) -> Path:
    return raw_dir / f"{system_id}__{question_id}__r{repeat}.json"


def load_cached_result(path: Path, retry_cached_infra: bool = True) -> dict[str, Any] | None:
    """Return the cached result dict if it exists and is complete, else None."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if retry_cached_infra and data.get("score", {}).get("infra_error"):
        return None
    return data


def estimate_run(
    tasks: list[tuple[str, dict[str, Any], int]],
    include_judge: bool = True,
) -> dict[str, float]:
    """Estimate total model calls and USD cost for the pending tasks using Phase 5 numbers."""
    agent_calls = 0.0
    judge_calls = 0.0
    cost_usd = 0.0
    for sys_id, item, _ in tasks:
        cat = item["category"]
        agent_calls += PHASE5_CALLS_PER_Q.get((sys_id, cat), 6.0)
        cost_usd += PHASE5_COST_PER_Q_USD.get((sys_id, cat), 0.008)
        if include_judge and cat == "documents":
            judge_calls += 1.0
            cost_usd += JUDGE_COST_PER_CALL_USD
        elif include_judge and cat == "combined" and item["check"].get("combined_type") in (
            "worst_breach_owner",
            "category_escalation",
        ):
            judge_calls += 1.0
            cost_usd += JUDGE_COST_PER_CALL_USD
        elif include_judge and cat == "refusal" and item["check"].get("refusal_type") in JUDGED_REFUSAL_TYPES:
            judge_calls += 1.0
            cost_usd += JUDGE_COST_PER_CALL_USD
    return {
        "tasks": len(tasks),
        "estimated_agent_calls": round(agent_calls),
        "estimated_judge_calls": round(judge_calls),
        "estimated_total_calls": round(agent_calls + judge_calls),
        "estimated_cost_usd": round(cost_usd, 4),
    }


async def evaluate_one_task(
    system_id: str,
    service: Any,
    item: dict[str, Any],
    repeat: int,
    *,
    raw_dir: Path,
    settings: Settings,
    ask_fn: Callable[..., Any] = ask_in_process,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
    judge: JudgeFn | None = None,
    retry_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """Run one (system, question, repeat) task, retrying once on infrastructure failure, and cache to disk."""
    out_path = cache_file_for(raw_dir, system_id, item["id"], repeat)
    cached = load_cached_result(out_path)
    if cached is not None:
        return {**cached, "cached": True}

    attempts = 0
    body: dict[str, Any] | None = None
    record: dict[str, Any] = {}
    error_type: str | None = None

    while True:
        attempts += 1
        body, record, error_type = await ask_fn(
            service, item["question"], settings.request_timeout_seconds
        )
        if error_type is not None and is_infra_error(error_type) and attempts < 2:
            if retry_delay_seconds > 0:
                await asyncio.sleep(retry_delay_seconds)
            continue
        break

    score = await asyncio.to_thread(
        score_item,
        item,
        body,
        error_type,
        docs_dir=settings.docs_dir,
        dataset=settings.bq_dataset,
        query_runner=query_runner,
        judge=judge,
    )

    payload = {
        "system": system_id,
        "id": item["id"],
        "category": item["category"],
        "smoke": bool(item.get("smoke", False)),
        "repeat": repeat,
        "question": item["question"],
        "attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response": body,
        "metrics_record": record,
        "error_type": error_type,
        "score": score,
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {**payload, "cached": False}


async def run_evaluation(
    *,
    systems: list[str],
    items: list[dict[str, Any]],
    repeats: int = 1,
    concurrency: int = 4,
    raw_dir: Path = RAW_DIR,
    settings: Settings | None = None,
    service_Builder: Callable[[str, Settings], Any] = build_system,
    ask_fn: Callable[..., Any] = ask_in_process,
    query_runner: Callable[[str], list[dict[str, Any]]] | None = None,
    judge: JudgeFn | None = None,
    retry_delay_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Run all (system, question, repeat) tasks with a concurrency semaphore and per-item disk caching."""
    s = settings or get_settings()
    services = {sys_id: service_Builder(sys_id, s) for sys_id in systems}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(sys_id: str, item: dict[str, Any], rep: int) -> dict[str, Any]:
        async with sem:
            res = await evaluate_one_task(
                sys_id,
                services[sys_id],
                item,
                rep,
                raw_dir=raw_dir,
                settings=s,
                ask_fn=ask_fn,
                query_runner=query_runner,
                judge=judge,
                retry_delay_seconds=retry_delay_seconds,
            )
            status = (
                "CACHED"
                if res.get("cached")
                else (
                    "PASS"
                    if res["score"]["passed"]
                    else ("INFRA_ERR" if res["score"].get("infra_error") else "FAIL")
                )
            )
            lat = (res.get("metrics_record") or {}).get("latency_ms", 0.0) / 1000.0
            print(
                f"  [{sys_id}] {item['id']} (r{rep}) [{item['category']}]: {status} ({lat:.1f}s) - {res['score']['reason']}",
                flush=True,
            )
            return res

    coros = [
        _worker(sys_id, item, rep)
        for sys_id in systems
        for rep in range(1, repeats + 1)
        for item in items
    ]
    return list(await asyncio.gather(*coros))


def check_against_baseline(
    results: list[dict[str, Any]],
    baseline_path: Path = BASELINE_PATH,
    max_drop_points: float = 10.0,
) -> tuple[bool, str]:
    """Compare current System C success rate against eval/baseline.json.

    `max_drop_points` is in percentage points (0-100), e.g. 10.0 means a drop of up to 10.0% is allowed.
    """
    if not baseline_path.exists():
        return (False, f"Baseline file {baseline_path} does not exist. Run the evaluation first to create it.")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    c_results = [r for r in results if r["system"] == "C"]
    if not c_results:
        return (False, "No System C results found to compare against baseline.")

    summary = summarize_system(c_results)
    current_rate = summary["excl_infra_success"]
    if current_rate is None:
        return (False, "All System C requests failed with infrastructure errors; cannot verify accuracy.")

    is_smoke_only = all(r.get("smoke") for r in c_results) and len(c_results) <= 16
    base_key = "smoke_excl_infra_success" if is_smoke_only and baseline.get("smoke_excl_infra_success") is not None else "full_excl_infra_success"
    baseline_rate = baseline.get(base_key)
    if baseline_rate is None:
        baseline_rate = baseline.get("full_strict_success", 0.0)

    current_pct = current_rate * 100.0
    baseline_pct = float(baseline_rate) * 100.0
    drop_points = round(baseline_pct - current_pct, 2)

    msg = (
        f"Baseline check: current={current_pct:.1f}% vs baseline={baseline_pct:.1f}% "
        f"(drop={drop_points:+.1f} percentage points, max allowed drop={max_drop_points:.1f} pp)"
    )
    return (drop_points <= max_drop_points, msg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 Evaluation Harness")
    parser.add_argument("--smoke", action="store_true", help="Run the 8-question smoke set on System C only.")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Comma-separated systems to run (e.g. 'C' or 'A,B,C' or 'A,B,C,D'). Default: C for --smoke, A,B,C for full.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Comma-separated categories to run (documents,database,combined,refusal). Default: all.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per (system, question). Default: 1.")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests (default: EVAL_CONCURRENCY=4).")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip interactive confirmation prompt.")
    parser.add_argument("--no-judge", action="store_true", help="Run deterministic checks only (skip LLM judge calls).")
    parser.add_argument("--check-baseline", action="store_true", help="Check System C accuracy against eval/baseline.json and exit non-zero on regression.")
    parser.add_argument("--max-drop-points", type=float, default=None, help="Allowed percentage-point drop below baseline (default: EVAL_MAX_DROP_POINTS).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    settings.export_to_environ()

    if args.system:
        systems = [s.strip().upper() for s in args.system.split(",") if s.strip()]
    elif args.smoke:
        systems = ["C"]
    else:
        systems = ["A", "B", "C"]

    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None
    all_items = load_golden(GOLDEN_PATH)
    items = select_questions(all_items, smoke=args.smoke, categories=categories)
    concurrency = args.concurrency or settings.eval_concurrency

    # Identify which tasks are not cached yet
    pending = [
        (sys_id, item, rep)
        for sys_id in systems
        for rep in range(1, args.repeats + 1)
        for item in items
        if load_cached_result(cache_file_for(RAW_DIR, sys_id, item["id"], rep)) is None
    ]
    total_tasks = len(systems) * len(items) * args.repeats
    cached_count = total_tasks - len(pending)

    est = estimate_run(pending, include_judge=not args.no_judge)
    print(
        f"Evaluation Plan:\n"
        f"  Systems: {', '.join(f'{s} ({SYSTEM_DESCRIPTIONS.get(s, s)})' for s in systems)}\n"
        f"  Questions: {len(items)} ({'smoke set' if args.smoke else 'golden set'}) x {args.repeats} repeat(s) = {total_tasks} total tasks\n"
        f"  Cached (will skip): {cached_count} | Pending to run: {len(pending)}\n"
        f"  Concurrency: {concurrency}\n"
        f"  Estimated LLM calls (Phase 5 numbers): ~{est['estimated_total_calls']} "
        f"({est['estimated_agent_calls']} agent + {est['estimated_judge_calls']} judge)\n"
        f"  Estimated cost (Phase 5 numbers): ~${est['estimated_cost_usd']:.4f} USD\n",
        flush=True,
    )

    if not args.smoke and pending and not args.yes:
        if not sys.stdin.isatty():
            print("Non-interactive stdin detected without --yes; stopping before full run for confirmation.")
            return 0
        reply = input("Proceed with evaluation run? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted by user.")
            return 0

    # Ensure retrieval ablation exists for the report
    if not ABLATION_OUT.exists():
        try:
            ablation = evaluate_retrieval(settings=settings)
            ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
            ABLATION_OUT.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
            print("Computed retrieval ablation table:\n" + format_ablation_table(ablation) + "\n", flush=True)
        except Exception as exc:
            print(f"Note: could not compute retrieval ablation ({exc})", flush=True)

    judge = None if args.no_judge else make_llm_judge(settings)
    results = asyncio.run(
        run_evaluation(
            systems=systems,
            items=items,
            repeats=args.repeats,
            concurrency=concurrency,
            raw_dir=RAW_DIR,
            settings=settings,
            judge=judge,
        )
    )

    update_baseline = not args.check_baseline
    write_report_and_baseline(
        raw_dir=RAW_DIR,
        report_path=REPORT_PATH,
        baseline_path=BASELINE_PATH,
        ablation_path=ABLATION_OUT,
        update_baseline=update_baseline,
    )
    print(f"\nWrote report to {REPORT_PATH}", flush=True)
    if update_baseline and "C" in systems:
        print(f"Updated baseline in {BASELINE_PATH}", flush=True)

    if args.check_baseline:
        max_drop = args.max_drop_points if args.max_drop_points is not None else settings.eval_max_drop_points
        ok, msg = check_against_baseline(results, BASELINE_PATH, max_drop_points=max_drop)
        print(msg, flush=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
