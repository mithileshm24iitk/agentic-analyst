# Architecture

A request travels **API → orchestration loop → orchestrator agent → specialists and tools → critic → response**, with metrics and tracing around the whole request. Every component is listed below with its file.

## API layer (`app/`)

| Component | File | What it does |
| :--- | :--- | :--- |
| FastAPI app | [app/main.py](../app/main.py) | Routes: `/health`, `/ask`, `/ask/sql`, `/ask/rag`, `/metrics/summary`. On startup it loads config, checks the retrieval index and builds the agents once. Adds a request id to every request, checks `X-API-Key` in constant time, and applies the request deadline. |
| Config | [app/config.py](../app/config.py) | One `Settings` class (pydantic-settings) reads env vars and `.env`. Model names, prices, region and limits all live here. In production it refuses to start without an API key. |
| Errors | [app/errors.py](../app/errors.py) | Maps upstream failures (timeout, quota, auth, 5xx) to a clean `{error, error_type, request_id}` response. Holds the request-id context variable. |
| Agent runner | [app/agent_service.py](../app/agent_service.py) | Wraps an ADK `Runner` with an in-memory session service. Starts a session, runs a turn, reads and updates state, deletes the session. |
| Orchestration loop | [app/orchestration.py](../app/orchestration.py) | Runs draft → critic → at most `CRITIC_MAX_REVISIONS` revisions as plain Python. Derives the route from the agents actually called. Builds the `/ask` response. |
| Citations | [app/citations.py](../app/citations.py) | Extracts `[file.md]` citations and keeps only those that were really retrieved. |

## Agents (`agents/`)

| Agent | File | What it does |
| :--- | :--- | :--- |
| orchestrator | [agents/orchestrator.py](../agents/orchestrator.py) | Chooses the route (documents, database, both or none) and calls the specialists, which are attached as `mode="single_turn"` sub-agents so ADK exposes them as tools inside the same session. Writes the final answer. Owns the `save_report` tool. |
| rag_agent | [agents/rag_agent.py](../agents/rag_agent.py) | Answers from the documents with the `retrieve` tool and cites a source file for every fact. |
| sql_agent | [agents/sql_agent.py](../agents/sql_agent.py) | Calls `get_schema`, writes BigQuery SQL, calls `run_query`, and fixes the SQL and retries on errors. |
| critic | [agents/critic.py](../agents/critic.py) | Runs in its own session. Compares the draft with the evidence from session state (escaped and tagged as data) and returns `{supported, unsupported_claims}`. |
| Shared LLM config | [agents/llm_config.py](../agents/llm_config.py) | Per-call HTTP timeout and Gemini thinking level. |
| baseline (eval only) | [agents/baseline_agent.py](../agents/baseline_agent.py) | System A: a single agent with every tool attached. Used only as the evaluation baseline. |

## Tools (`tools/`)

| Tool | File | What it does |
| :--- | :--- | :--- |
| `get_schema`, `run_query` | [tools/bigquery_tools.py](../tools/bigquery_tools.py) | SQL guardrails in code (sqlglot parse, read-only, one dataset, built-in functions only, LIMIT clamp, dry-run byte cap, attempt cap). Writes `last_sql`, `last_sql_rows` and `sql_history` to state. |
| `retrieve` | [tools/rag_tools.py](../tools/rag_tools.py) | Calls hybrid search and records chunks, sources and queries in state. |
| `save_report` | [tools/report_tools.py](../tools/report_tools.py) | Writes markdown to `REPORTS_DIR` locally or to `REPORTS_BUCKET` on Cloud Run. Uses safe file names and never overwrites another session's report. |
| Fatal upstream errors | [tools/upstream.py](../tools/upstream.py) | Auth and quota errors inside a tool stop the agent, and the API returns 502/503 instead of a weak answer. |

## Retrieval (`tools/retrieval/`)

| Step | File |
| :--- | :--- |
| Ingest: read `data/docs/*.md`, chunk, embed, store | [ingest.py](../tools/retrieval/ingest.py), [chunking.py](../tools/retrieval/chunking.py), [embeddings.py](../tools/retrieval/embeddings.py), [vector_store.py](../tools/retrieval/vector_store.py), [index_files.py](../tools/retrieval/index_files.py) |
| Query: vector + BM25 candidates, merged with RRF | [search.py](../tools/retrieval/search.py), [bm25.py](../tools/retrieval/bm25.py), [fusion.py](../tools/retrieval/fusion.py) |
| Startup check: refuse to serve without a valid index | [index_check.py](../tools/retrieval/index_check.py) |

## Observability

| Component | File | What it does |
| :--- | :--- | :--- |
| Metrics plugin | [app/metrics_plugin.py](../app/metrics_plugin.py) | ADK plugin that times every model call and tool call (sub-agents and critic included) and files them under the current request id. |
| Metrics record | [app/metrics.py](../app/metrics.py) | One record per request: tokens in/out/thinking, latency, tokens/sec per call, estimated cost, LLM calls, SQL attempts, revisions, route. Appended to `METRICS_PATH` (JSONL) and printed as structured JSON to stdout (Cloud Logging). |
| Cost | [app/costs.py](../app/costs.py) | USD per call from the configured price table. An unknown model returns `null`, never a guess. |
| Summary | [app/metrics_summary.py](../app/metrics_summary.py) | `GET /metrics/summary`: p50/p95 latency, cost and calls, overall and per route, from this instance's JSONL file. |
| Tracing | [app/tracing.py](../app/tracing.py) | `TRACING_EXPORTER` = `none`, `console` or `gcp`. ADK creates spans for agents, model calls and tools; the app adds one root span per request and exports to Cloud Trace. |

## Evaluation (`eval/`)

| Component | File |
| :--- | :--- |
| Golden set (48 questions) and human review sheet | [golden.jsonl](../eval/golden.jsonl), [GOLDEN_REVIEW.md](../eval/GOLDEN_REVIEW.md) |
| Ground-truth printer for hand-checking (`make eval-ground-truth`) | [show_ground_truth.py](../eval/show_ground_truth.py) |
| Systems A–D, built from the same code as `/ask` | [systems.py](../eval/systems.py) |
| Runner (parallel, cached, resumable, infra retry) | [run_eval.py](../eval/run_eval.py) |
| Live ground truth (reference SQL + rules parsed from the docs) | [ground_truth.py](../eval/ground_truth.py) |
| Deterministic scoring (`expected_sources` = all listed files), then the LLM judge only for free text; a separate safety rubric for injection and not-in-documents items | [scoring.py](../eval/scoring.py), [judge.py](../eval/judge.py) |
| Retrieval comparison (no LLM) | [retrieval_eval.py](../eval/retrieval_eval.py) |
| Report and regression baseline | [report.py](../eval/report.py) → [results.md](../eval/results.md), [baseline.json](../eval/baseline.json) |
| Latency benchmark | [bench.py](../eval/bench.py) |

## Deployment

| Component | File |
| :--- | :--- |
| Container (non-root, ships the pre-built index, production mode by default) | [Dockerfile](../Dockerfile) |
| Cloud Build + Cloud Run deploy (secret from Secret Manager, dedicated service account) | [scripts/deploy.sh](../scripts/deploy.sh), [docs/DEPLOY.md](DEPLOY.md) |
| Production smoke test | [scripts/smoke_prod.sh](../scripts/smoke_prod.sh) |
| CI (offline unit tests on every push) | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
