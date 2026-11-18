# 2-minute demo script

Five questions, about 20 seconds of talking each. Model latency varies from run to run. If time is tight, run each question once before the demo and walk through the saved JSON responses, then run one question live.

## Setup (before the demo)

Against the deployed service (see [DEPLOY.md](DEPLOY.md) for `SERVICE_URL` and `API_KEY`):

```bash
ask() {
  curl -s -X POST "$SERVICE_URL/ask" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$1")" | python3 -m json.tool
}
```

Locally, use `make run` in one terminal and `make ask Q="..."` in another.

Optional 10-second opener: `make smoke-prod` shows `/health` is open and that a missing or wrong key gets **401**.

## 1. Documents: "What is the maximum acceptable return rate for Jeans?"

- `route` is `docs` and `agents_called` is `["rag_agent"]`.
- The answer cites `[returns_policy.md]`. `cited_sources` is checked against the chunks that were actually retrieved, so a made-up file name would not appear there.
- Mention hybrid retrieval: vector search + BM25, merged with reciprocal rank fusion.
- `metrics` shows the LLM calls, tokens and estimated cost for this one request.

## 2. Database: "How many orders were created in the year 2024?"

- `route` is `sql`. The response includes the exact `sql` that ran and the `rows` it returned, so the number can be audited.
- `metrics.sql_attempts`: the SQL agent retries on errors, up to `SQL_MAX_ATTEMPTS`.
- The guardrails are in code ([tools/bigquery_tools.py](../tools/bigquery_tools.py)): read-only, one dataset, dry-run byte cap, LIMIT clamp.

## 3. Combined: "Which product categories are currently above their maximum acceptable return rate across all recorded order items?"

- `route` is `both`, and `agents_called` shows `rag_agent` **before** `sql_agent`. The orchestrator fetches the policy thresholds first, then gives the SQL agent the exact category names and definition.
- The answer compares each category's actual rate with its policy limit. The orchestrator's instructions require document citations and ask for database numbers to be labelled as such.
- `critic.supported` and `critic.revisions`: the critic checked every claim against the evidence. Any revision is capped in Python code, not by the prompt.

## 4. Combined + action: "Which product categories are above their maximum acceptable return rate? Save it as a report titled Return rate breaches."

- `saved_report` is set. On Cloud Run the report is a Cloud Storage object (only the object name is returned). Locally it is a file in `reports/`.
- `save_report` runs only when the user explicitly asks. Questions 1–3 did not save anything.
- If the critic forced a revision, the same report is overwritten with the corrected answer, so no unreviewed draft is left behind.

## 5. Out of scope: "What will the weather be in Tokyo tomorrow, and should I pack an umbrella?"

- `route` is `none` and `agents_called` is empty. The orchestrator declines without calling a specialist or guessing.
- Compare its `metrics` (LLM calls, cost) with questions 1–4: no specialist was called.
- If there is time, mention that the evaluation set also includes destructive SQL and prompt-injection questions ([eval/golden.jsonl](../eval/golden.jsonl)).

## Closing line

"Every answer comes with its evidence (SQL, rows, sources), a critic verdict, and a cost. It's evaluated against 48 golden questions, compared with a single-agent baseline, and the safety limits are in code, not just prompts."
