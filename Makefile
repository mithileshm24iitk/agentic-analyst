PY := .venv/bin/python
PORT ?= 8000
IMAGE ?= agentic-analyst:local
# Local API key header for the curl targets (empty = open local dev).
AUTH = $(if $(API_KEY),-H "X-API-Key: $(API_KEY)")
Q ?= Which product categories are above their maximum acceptable return rate?

.DEFAULT_GOAL := help

.PHONY: help install run test test-integration ingest ask ask-sql ask-rag bench summary eval-smoke eval eval-retrieval eval-check eval-ground-truth eval-rescore \
	docker-build deploy smoke-prod

help: ## List every target (the default when you run plain `make`)
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime + test dependencies into .venv
	$(PY) -m pip install -r requirements-dev.txt

run: ## Run the API locally with auto-reload (PORT, default 8000)
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

test: ## Run the offline unit tests (no GCP needed)
	$(PY) -m pytest -q

# Orchestrator (documents + database + critic): make ask Q="your question"
ask: ## Ask the orchestrator: make ask Q="..."
	$(PY) -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$(Q)" | \
		curl -s -X POST localhost:$(PORT)/ask -H "Content-Type: application/json" $(AUTH) -d @-; echo

# SQL agent smoke test: make ask-sql Q="How many orders were placed in 2024?"
ask-sql: ## Ask the SQL agent only: make ask-sql Q="..."
	$(PY) -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$(Q)" | \
		curl -s -X POST localhost:$(PORT)/ask/sql -H "Content-Type: application/json" $(AUTH) -d @-; echo

# Real GCP tests: BigQuery, RAG and the full orchestrator (needs make ingest; a few Gemini calls)
test-integration: ## Run integration tests against real BigQuery, RAG and Gemini
	RUN_BQ_INTEGRATION=1 RUN_RAG_INTEGRATION=1 RUN_ORCHESTRATOR_INTEGRATION=1 $(PY) -m pytest -q \
		tests/test_bigquery_integration.py tests/test_rag_integration.py tests/test_orchestrator_integration.py

# Build the document index (ChromaDB + chunks.json). Safe to re-run.
ingest: ## Build the document index in data/chroma (safe to re-run)
	$(PY) -m tools.retrieval.ingest

# RAG agent smoke test: make ask-rag Q="What is the maximum return rate for Jeans?"
ask-rag: ## Ask the RAG agent only: make ask-rag Q="..."
	$(PY) -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$(Q)" | \
		curl -s -X POST localhost:$(PORT)/ask/rag -H "Content-Type: application/json" $(AUTH) -d @-; echo

# Latency benchmark: 5 fixed questions x BENCH_RUNS through /ask (real GCP calls).
# Try another thinking level with: make bench THINKING_LEVEL=LOW
bench: ## Latency benchmark through /ask (THINKING_LEVEL=LOW to compare)
	$(if $(THINKING_LEVEL),THINKING_LEVEL=$(THINKING_LEVEL)) $(PY) -m eval.bench

# Aggregated metrics from METRICS_PATH (needs make run)
summary: ## Show aggregated metrics from the running API
	curl -s $(AUTH) localhost:$(PORT)/metrics/summary; echo

# Phase 6 evaluation harness targets
eval-retrieval: ## Retrieval comparison: vector vs BM25 vs hybrid (no LLM)
	$(PY) -m eval.retrieval_eval

eval-smoke: ## Evaluation on the 8-question smoke set
	$(PY) -m eval.run_eval --smoke

eval: ## Full evaluation run; cached items are skipped, infra failures rerun (ARGS="--system B --ids x,y --rerun")
	$(PY) -m eval.run_eval $(ARGS)

eval-ground-truth: ## Print every golden item's ground truth for hand-checking (BigQuery, no LLM)
	$(PY) -m eval.show_ground_truth $(ARGS)

eval-rescore: ## Re-score cached raw results with the current scorer (no system calls); rewrites results.md
	$(PY) -m eval.run_eval --rescore $(ARGS)

eval-check: ## Smoke evaluation + regression check against eval/baseline.json
	$(PY) -m eval.run_eval --smoke --check-baseline $(if $(MAX_DROP),--max-drop-points $(MAX_DROP))

# Phase 7: container and Cloud Run (see docs/DEPLOY.md)
# Local image build (needs Docker). linux/amd64 = what Cloud Run runs.
docker-build: ## Build the Docker image locally (linux/amd64)
	docker build --platform linux/amd64 -t $(IMAGE) .

# Build on Cloud Build and deploy to Cloud Run. Reads PROJECT_ID, REGION, ... from the environment.
deploy: ## Build on Cloud Build and deploy to Cloud Run (see docs/DEPLOY.md)
	./scripts/deploy.sh

# /health, 401 without a key, then one real question. Needs SERVICE_URL and API_KEY.
smoke-prod: ## Smoke-test the deployed service (needs SERVICE_URL, API_KEY)
	./scripts/smoke_prod.sh
