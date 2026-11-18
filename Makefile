PY := .venv/bin/python
PORT ?= 8000
Q ?= In one sentence, what is a KPI?

.PHONY: install run test test-integration ingest ask ask-sql ask-rag

install:
	$(PY) -m pip install -r requirements.txt

run:
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

test:
	$(PY) -m pytest -q

# Smoke test against a running server: make ask Q="your question"
ask:
	curl -s -X POST localhost:$(PORT)/ask -H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}'; echo

# SQL agent smoke test: make ask-sql Q="How many orders were placed in 2024?"
ask-sql:
	curl -s -X POST localhost:$(PORT)/ask/sql -H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}'; echo

# Real GCP tests: BigQuery (a few MB of scan) and RAG (needs make ingest; one embedding call)
test-integration:
	RUN_BQ_INTEGRATION=1 RUN_RAG_INTEGRATION=1 $(PY) -m pytest -q tests/test_bigquery_integration.py tests/test_rag_integration.py

# Build the document index (ChromaDB + chunks.json). Safe to re-run.
ingest:
	$(PY) -m tools.retrieval.ingest

# RAG agent smoke test: make ask-rag Q="What is the maximum return rate for Jeans?"
ask-rag:
	curl -s -X POST localhost:$(PORT)/ask/rag -H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}'; echo
