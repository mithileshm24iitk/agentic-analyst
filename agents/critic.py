"""Critic agent: checks a draft answer against the evidence and returns a structured verdict."""

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field

from tools.bigquery_tools import STATE_HISTORY
from tools.rag_tools import STATE_CHUNKS

MAX_ROWS_PER_QUERY = 100


class CriticVerdict(BaseModel):
    supported: bool = Field(description="True if every factual claim in the draft is supported by the evidence.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Each claim in the draft that the evidence does not support, quoted or paraphrased.",
    )


INSTRUCTION = """You are a strict fact checker. You receive a question, a draft answer, and the
evidence the answer was built from (document chunks and SQL query results).

Check every factual claim in the draft: numbers, percentages, names, emails, dates,
category names, thresholds, policies and comparisons.
- A claim is supported if the evidence states it, or if it follows from the evidence by
  simple arithmetic or comparison (e.g. 10.9% is above a 10.0% threshold).
- Rounding to fewer decimals is fine.
- Statements that information is missing or unavailable are supported.
- Generic connecting phrases are not claims.
- A claim is unsupported if the evidence does not contain it or contradicts it.

Return supported=true only if there are no unsupported claims. List each unsupported claim
briefly. Do not judge style or completeness, only factual support."""


def build_critic(model: str | BaseLlm) -> Agent:
    return Agent(
        name="critic",
        model=model,
        description="Checks whether a draft answer is supported by the evidence.",
        instruction=INSTRUCTION,
        output_schema=CriticVerdict,
    )


def format_evidence(state: dict[str, Any]) -> str:
    """Turn the session state into a readable evidence block for the critic."""
    parts: list[str] = []

    chunks = state.get(STATE_CHUNKS, [])
    if chunks:
        parts.append("## Document chunks")
        for c in chunks:
            parts.append(f"[{c['source']}] ({c['chunk_id']})\n{c['text']}")

    queries = [q for q in state.get(STATE_HISTORY, []) if q.get("ok")]
    if queries:
        parts.append("## SQL query results (sales database)")
        for i, q in enumerate(queries, start=1):
            rows = q["rows"][:MAX_ROWS_PER_QUERY]
            parts.append(f"Query {i}:\n{q['sql']}\nRows ({len(rows)}):\n{json.dumps(rows, default=str)}")

    return "\n\n".join(parts) if parts else "(no evidence was gathered)"


def build_critic_prompt(question: str, draft: str, state: dict[str, Any]) -> str:
    return (
        f"# Question\n{question}\n\n"
        f"# Draft answer\n{draft}\n\n"
        f"# Evidence\n{format_evidence(state)}"
    )


def parse_verdict(text: str) -> CriticVerdict:
    """Parse the critic's JSON output (tolerates a ```json fence)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{") :]
    return CriticVerdict.model_validate_json(cleaned)
