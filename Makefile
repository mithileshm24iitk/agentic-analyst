PY := .venv/bin/python
PORT ?= 8000
Q ?= In one sentence, what is a KPI?

.PHONY: install run test test-integration ask ask-sql

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

# Real BigQuery test (costs a few MB of scan in your project)
test-integration:
	RUN_BQ_INTEGRATION=1 $(PY) -m pytest -q tests/test_bigquery_integration.py
