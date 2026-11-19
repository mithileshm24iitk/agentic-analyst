"""retrieve(): the RAG agent's only tool. Wraps hybrid_search and records results in session state."""

import logging

from google.adk.tools import ToolContext

from app.errors import current_request_id
from tools.retrieval.search import hybrid_search
from tools.upstream import fatal_error_type, fatal_message, record_fatal_error, sanitize_tool_error

logger = logging.getLogger(__name__)

# Session state keys (read by /ask/rag, the orchestrator, the critic and Phase 5 metrics).
# Scoped to the session; each API request uses a fresh session.
STATE_CHUNKS = "rag_chunks"  # every chunk retrieved in this session, deduplicated
STATE_SOURCES = "rag_sources"  # sorted unique source file names
STATE_QUERIES = "rag_queries"  # the queries the agent searched with


def retrieve(query: str, tool_context: ToolContext) -> dict:
    """Searches ShopCo's company documents (policies, KPI definitions, data dictionary).

    Returns the most relevant text chunks, each with its source file name.
    Cite the `source` of every chunk you use. Call again with different words
    if the first results do not answer the question.
    """
    state = tool_context.state
    # Assign new lists (not .append) so ADK records the change in the state delta.
    state[STATE_QUERIES] = state.get(STATE_QUERIES, []) + [query]

    try:
        hits = hybrid_search(query)
    except Exception as e:  # never raise to the agent
        fatal = fatal_error_type(e)  # auth / quota on the embeddings API
        if fatal:
            record_fatal_error(state, fatal, "retrieve")
            return {"ok": False, "chunks": [], "sources": [], "error": fatal_message(fatal)}
        logger.warning("request_id=%s retrieve failed", current_request_id(), exc_info=e)
        return {"ok": False, "chunks": [], "sources": [], "error": sanitize_tool_error(e)}

    chunks = [
        {
            "chunk_id": h["chunk_id"],
            "source": h["source"],
            "heading": h["heading"],
            "text": h["text"],
            "score": h["score"],
        }
        for h in hits
    ]
    previous = state.get(STATE_CHUNKS, [])
    seen = {c["chunk_id"] for c in previous}
    state[STATE_CHUNKS] = previous + [c for c in chunks if c["chunk_id"] not in seen]
    state[STATE_SOURCES] = sorted({c["source"] for c in state[STATE_CHUNKS]})

    return {
        "ok": True,
        "chunks": chunks,
        "sources": sorted({c["source"] for c in chunks}),
        "error": None,
    }
