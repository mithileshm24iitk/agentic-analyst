PY := .venv/bin/python
PORT ?= 8000
Q ?= In one sentence, what is a KPI?

.PHONY: install run test ask

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
