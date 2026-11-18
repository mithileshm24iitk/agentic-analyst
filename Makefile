PY := .venv/bin/python
PORT ?= 8000
IMAGE ?= agentic-analyst:local
# Local API key header for the curl targets (empty = open local dev).
AUTH = $(if $(API_KEY),-H "X-API-Key: $(API_KEY)")
Q ?= Which product categories are above their maximum acceptable return rate?

.PHONY: install run test test-integration ingest ask ask-sql ask-rag bench summary eval-smoke eval eval-retrieval eval-check \
	docker-build deploy smoke-prod

install:
	$(PY) -m pip install -r requirements-dev.txt

run:
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

test:
	$(PY) -m pytest -q

# Orchestrator (documents + database + critic): make ask Q="your question"
ask:
	curl -s -X POST localhost:$(PORT)/ask -H "Content-Type: application/json" $(AUTH) \
		-d '{"question": "$(Q)"}'; echo

# SQL agent smoke test: make ask-sql Q="How many orders were placed in 2024?"
ask-sql:
	curl -s -X POST localhost:$(PORT)/ask/sql -H "Content-Type: application/json" $(AUTH) \
		-d '{"question": "$(Q)"}'; echo

# Real GCP tests: BigQuery, RAG and the full orchestrator (needs make ingest; a few Gemini calls)
test-integration:
	RUN_BQ_INTEGRATION=1 RUN_RAG_INTEGRATION=1 RUN_ORCHESTRATOR_INTEGRATION=1 $(PY) -m pytest -q \
		tests/test_bigquery_integration.py tests/test_rag_integration.py tests/test_orchestrator_integration.py

# Build the document index (ChromaDB + chunks.json). Safe to re-run.
ingest:
	$(PY) -m tools.retrieval.ingest

# RAG agent smoke test: make ask-rag Q="What is the maximum return rate for Jeans?"
ask-rag:
	curl -s -X POST localhost:$(PORT)/ask/rag -H "Content-Type: application/json" $(AUTH) \
		-d '{"question": "$(Q)"}'; echo

# Latency benchmark: 5 fixed questions x BENCH_RUNS through /ask (real GCP calls).
# Try another thinking level with: make bench THINKING_LEVEL=LOW
bench:
	$(if $(THINKING_LEVEL),THINKING_LEVEL=$(THINKING_LEVEL)) $(PY) -m eval.bench

# Aggregated metrics from METRICS_PATH (needs make run)
summary:
	curl -s $(AUTH) localhost:$(PORT)/metrics/summary; echo

# Phase 6 evaluation harness targets
eval-retrieval:
	$(PY) -m eval.retrieval_eval

eval-smoke:
	$(PY) -m eval.run_eval --smoke

eval:
	$(PY) -m eval.run_eval $(ARGS)

eval-check:
	$(PY) -m eval.run_eval --smoke --check-baseline $(if $(MAX_DROP),--max-drop-points $(MAX_DROP))

# Phase 7: container and Cloud Run (see docs/DEPLOY.md)
# Local image build (needs Docker). linux/amd64 = what Cloud Run runs.
docker-build:
	docker build --platform linux/amd64 -t $(IMAGE) .

# Build on Cloud Build and deploy to Cloud Run. Reads PROJECT_ID, REGION, ... from the environment.
deploy:
	./scripts/deploy.sh

# /health, 401 without a key, then one real question. Needs SERVICE_URL and API_KEY.
smoke-prod:
	./scripts/smoke_prod.sh
