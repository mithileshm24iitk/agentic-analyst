"""Retrieval evaluation (Phase 6 item 2): no LLM needed.

For every document question in eval/golden.jsonl, evaluates three retrievers:
  1. vector_only (ChromaDB cosine distance)
  2. bm25_only (BM25Okapi keyword search)
  3. hybrid (Reciprocal Rank Fusion of vector + BM25)

Computes hit@3, hit@5 (whether ALL expected source files are in the top k results)
and MRR (Mean Reciprocal Rank of the first chunk from an expected source file),
prints an ablation table, and saves the results to eval/results/retrieval_ablation.json.
"""

import json
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from tools.retrieval.embeddings import Embedder, get_embedder
from tools.retrieval.fusion import fuse
from tools.retrieval.search import CANDIDATE_MULTIPLIER, _load_index
from tools.retrieval.vector_store import open_collection, vector_search

GOLDEN_PATH = Path("eval/golden.jsonl")
ABLATION_OUT = Path("eval/results/retrieval_ablation.json")


def hit_at_k(ranked_sources: list[str], expected_sources: list[str], k: int) -> float:
    """1.0 if ALL expected sources appear in ranked_sources[:k], else 0.0.

    Same meaning of expected_sources as in eval/scoring.py: every listed file is needed.
    """
    top = set(ranked_sources[:k])
    return 1.0 if expected_sources and all(src in top for src in expected_sources) else 0.0


def reciprocal_rank(ranked_sources: list[str], expected_sources: list[str]) -> float:
    """1 / rank of the first result whose source is in expected_sources (1-indexed), or 0.0."""
    wanted = set(expected_sources)
    for rank, src in enumerate(ranked_sources, start=1):
        if src in wanted:
            return 1.0 / rank
    return 0.0


def score_rankings(
    queries_and_expected: list[tuple[str, list[str], list[str]]],
) -> dict[str, Any]:
    """Compute mean hit@3, hit@5 and MRR over a list of (id, ranked_sources, expected_sources)."""
    if not queries_and_expected:
        return {"count": 0, "hit_at_3": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "per_question": []}

    per_q = []
    for qid, ranked, expected in queries_and_expected:
        h3 = hit_at_k(ranked, expected, 3)
        h5 = hit_at_k(ranked, expected, 5)
        rr = reciprocal_rank(ranked, expected)
        per_q.append(
            {
                "id": qid,
                "expected_sources": expected,
                "top_5_sources": ranked[:5],
                "hit_at_3": h3,
                "hit_at_5": h5,
                "rr": round(rr, 4),
            }
        )

    n = len(per_q)
    return {
        "count": n,
        "hit_at_3": round(sum(x["hit_at_3"] for x in per_q) / n, 4),
        "hit_at_5": round(sum(x["hit_at_5"] for x in per_q) / n, 4),
        "mrr": round(sum(x["rr"] for x in per_q) / n, 4),
        "per_question": per_q,
    }


def run_three_retrievers(
    question: str,
    k: int = 5,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> dict[str, list[str]]:
    """Return top-k source lists for 'vector_only', 'bm25_only', and 'hybrid'."""
    s = settings or get_settings()
    emb = embedder or get_embedder()
    n = max(k * CANDIDATE_MULTIPLIER, 15)

    index = _load_index(s)
    collection = open_collection(s.chroma_dir, s.rag_collection, s.embedding_model, s.embedding_dim)

    q_vec = emb.embed_query(question)
    vector_hits = [cid for cid, _ in vector_search(collection, q_vec, n) if cid in index.chunks]
    bm25_hits = [cid for cid, _ in index.bm25.search(question, n) if cid in index.chunks]
    fused_hits = [cid for cid, _ in fuse([vector_hits, bm25_hits], s.rrf_k)]

    return {
        "vector_only": [index.chunks[cid].source for cid in vector_hits[:k]],
        "bm25_only": [index.chunks[cid].source for cid in bm25_hits[:k]],
        "hybrid": [index.chunks[cid].source for cid in fused_hits[:k]],
    }


def load_document_questions(golden_path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    items = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [item for item in items if item["category"] == "documents"]


def evaluate_retrieval(
    doc_questions: list[dict[str, Any]] | None = None,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run vector_only, bm25_only and hybrid on all document questions and score each."""
    questions = doc_questions if doc_questions is not None else load_document_questions()
    by_mode: dict[str, list[tuple[str, list[str], list[str]]]] = {
        "vector_only": [],
        "bm25_only": [],
        "hybrid": [],
    }
    for item in questions:
        qid = item["id"]
        expected = item["check"]["expected_sources"]
        rankings = run_three_retrievers(item["question"], k=k, embedder=embedder, settings=settings)
        for mode, sources in rankings.items():
            by_mode[mode].append((qid, sources, expected))

    return {mode: score_rankings(entries) for mode, entries in by_mode.items()}


def format_ablation_table(summary: dict[str, dict[str, Any]]) -> str:
    """Render a markdown table comparing vector_only, bm25_only, and hybrid (RRF)."""
    labels = {
        "vector_only": "Vector only (ChromaDB)",
        "bm25_only": "BM25 only (rank_bm25)",
        "hybrid": "Hybrid (RRF)",
    }
    lines = [
        "| Retriever | Questions | Hit@3 | Hit@5 | MRR |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for mode in ("vector_only", "bm25_only", "hybrid"):
        if mode not in summary:
            continue
        m = summary[mode]
        lines.append(
            f"| {labels.get(mode, mode)} | {m['count']} | {m['hit_at_3']:.1%} | {m['hit_at_5']:.1%} | {m['mrr']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    summary = evaluate_retrieval()
    print("Retrieval Ablation (Document Questions, no LLM):\n")
    print(format_ablation_table(summary))
    ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved retrieval ablation to {ABLATION_OUT}")


if __name__ == "__main__":
    main()
