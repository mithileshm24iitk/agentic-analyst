"""Real end-to-end /ask flow (Gemini + BigQuery + embeddings). Skipped unless RUN_ORCHESTRATOR_INTEGRATION=1.

Needs .env, gcloud ADC and `make ingest`. Cost: a handful of Gemini calls per question
and a few MB of BigQuery scan. LLM routing is not fully deterministic; the assertions
check the route and the presence of evidence, not exact wording.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ORCHESTRATOR_INTEGRATION") != "1",
    reason="set RUN_ORCHESTRATOR_INTEGRATION=1 to run the real orchestrator",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the startup hook, which builds the real agents
        yield c


def ask(client, question: str) -> dict:
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_docs_question(client):
    body = ask(client, "What is the maximum acceptable return rate for Jeans?")
    assert body["route"] == "docs", body
    assert "returns_policy.md" in body["cited_sources"]
    assert "11" in body["answer"]


def test_database_question(client):
    body = ask(client, "How many order items have the status 'Returned'?")
    assert body["route"] == "sql", body
    assert body["sql"] and body["rows"]


def test_combined_question(client):
    body = ask(
        client,
        "Using the returns policy thresholds and the sales database, which product categories "
        "are currently above their maximum acceptable return rate?",
    )
    assert body["route"] == "both", body
    assert body["sql"] and body["rows"]
    assert "returns_policy.md" in body["sources"]
    assert body["critic"]["supported"] is not None
