# Agentic Analyst API image for Cloud Run.
# Build: make docker-build (local) or scripts/deploy.sh (Cloud Build). See docs/DEPLOY.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is reused until requirements.txt changes.
# Every package is a prebuilt wheel, so no compiler is needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged user. It can write only to data/ (Chroma's SQLite file, the metrics
# JSONL) and reports/; the code itself is owned by root and read-only to it.
RUN useradd --create-home --uid 10001 app

COPY app/ app/
COPY agents/ agents/
COPY tools/ tools/
# The pre-built retrieval index (make ingest, run locally). The container never ingests
# and never calls the embedding API at build time; queries are embedded at request time.
COPY --chown=app:app data/chroma/ data/chroma/
RUN test -s data/chroma/chunks.json && test -s data/chroma/chroma.sqlite3 \
      || { echo "ERROR: data/chroma has no index. Run 'make ingest' before building." >&2; exit 1; } \
    && mkdir -p data/metrics reports \
    && chown app:app data data/metrics reports

USER app

# Secure default: production mode refuses to start without API_KEY.
# Cloud Run sets PORT (8080); uvicorn listens on it. One process: requests run concurrently
# in its event loop, and Cloud Run's --concurrency caps how many reach one instance.
ENV APP_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT}\" --no-access-log"]
