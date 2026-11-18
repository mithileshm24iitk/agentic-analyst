# Project: Agentic Enterprise Knowledge & Analytics Assistant

Goal: a multi-agent system that answers business questions from documents (RAG) and BigQuery (SQL), with evaluation and tracing. Portfolio project for a Google Cloud Forward Deployed Engineer internship, so it must be simple enough for me to explain every file.

## Final design
- Orchestrator agent delegates to rag_agent and sql_agent, plus a critic step. One small action tool: save_report.
- Shared session state for intermediate results.
- Every request produces a metrics record: tokens in/out, latency, tokens/sec, estimated cost, SQL retries.
- Stack: Python 3.11, Google ADK, Gemini on Vertex AI, BigQuery (bigquery-public-data.thelook_ecommerce), ChromaDB + rank_bm25, FastAPI, OpenTelemetry -> Cloud Trace, Docker, Cloud Run.
- Out of scope: frontend, other databases, extra tools, fine-tuning.

## Roadmap (one phase at a time, I will tell you which)
1 Skeleton + one agent | 2 SQL agent | 3 RAG agent | 4 Orchestrator + critic + save_report | 5 Metrics + tracing | 6 Evaluation harness | 7 Cloud Run deploy | 8 README

## Rules for you
1. Before coding a phase, read the official ADK and Vertex AI docs for the APIs you will use. Do not rely on memory. Pin versions in requirements.txt.
2. Show a short plan first and wait for my OK. Stay inside the current phase.
3. No secrets or project IDs in code. Use env vars and a .env.example.
4. Model names, prices and region are config values, never hard-coded.
5. Write tests for anything with logic and run them.
6. Keep code simple and readable.
7. End each phase with: (a) how to run it, (b) a 5-line explanation of how it works, (c) what could break. Then stop.