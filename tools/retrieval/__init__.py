"""Hybrid retrieval over the company documents in data/docs.

chunking -> embeddings -> vector_store (ChromaDB) + bm25 (keyword) -> fusion (RRF) -> search
ingest.py builds the stores; search.hybrid_search queries them.
"""
