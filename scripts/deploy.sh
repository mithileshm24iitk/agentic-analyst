#!/usr/bin/env bash
# Build the image on Cloud Build and deploy it to Cloud Run.
# One-time setup (bucket, secret, service account, roles) is in docs/DEPLOY.md.
#
# Every project-specific value comes from the environment; nothing is hard-coded here.
# Required: PROJECT_ID REGION API_KEY_SECRET_VERSION
#           GOOGLE_CLOUD_LOCATION WORKER_MODEL EMBEDDING_MODEL EMBEDDING_LOCATION
# Optional (defaults in brackets): SERVICE [agentic-analyst], AR_REPO [agentic-analyst],
#           REPORTS_BUCKET [PROJECT_ID-agentic-analyst-reports],
#           API_KEY_SECRET [agentic-analyst-api-key],
#           RUNTIME_SA [agentic-analyst-run@PROJECT_ID.iam.gserviceaccount.com],
#           DEPLOY_TRACING_EXPORTER [gcp], REQUEST_TIMEOUT_SECONDS [270]
# Safe after `source .env`: an empty REPORTS_BUCKET or a local TRACING_EXPORTER=none
# does not leak into the deployment.
# Any app setting in PASS_THROUGH below is forwarded when set and non-empty.
set -euo pipefail

die() { echo "deploy: $*" >&2; exit 1; }

for name in PROJECT_ID REGION API_KEY_SECRET_VERSION \
            GOOGLE_CLOUD_LOCATION WORKER_MODEL EMBEDDING_MODEL EMBEDDING_LOCATION; do
  [[ -n "${!name:-}" ]] || die "$name is not set (see docs/DEPLOY.md)"
done
[[ "$API_KEY_SECRET_VERSION" =~ ^[0-9]+$ ]] \
  || die "API_KEY_SECRET_VERSION must be a version number like 1 (pinned, not 'latest')"

SERVICE="${SERVICE:-agentic-analyst}"
AR_REPO="${AR_REPO:-agentic-analyst}"
REPORTS_BUCKET="${REPORTS_BUCKET:-${PROJECT_ID}-agentic-analyst-reports}"
API_KEY_SECRET="${API_KEY_SECRET:-agentic-analyst-api-key}"
RUNTIME_SA="${RUNTIME_SA:-agentic-analyst-run@${PROJECT_ID}.iam.gserviceaccount.com}"
TRACING_EXPORTER="${DEPLOY_TRACING_EXPORTER:-gcp}"  # not TRACING_EXPORTER: .env sets that for local runs
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-270}"
CLOUD_RUN_TIMEOUT=300  # seconds; the app answers with clean JSON before this

(( REQUEST_TIMEOUT_SECONDS < CLOUD_RUN_TIMEOUT )) \
  || die "REQUEST_TIMEOUT_SECONDS ($REQUEST_TIMEOUT_SECONDS) must be below the Cloud Run timeout ($CLOUD_RUN_TIMEOUT)"

# The image ships the pre-built index; fail here rather than after a 5-minute build.
[[ -s data/chroma/chunks.json && -s data/chroma/chroma.sqlite3 ]] \
  || die "data/chroma has no index. Run 'make ingest' first."

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
[[ -z "$(git status --porcelain 2>/dev/null)" ]] || GIT_SHA="${GIT_SHA}-dirty"
TAG="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE}:${TAG}"

# Runtime settings go in a YAML file: --set-env-vars cannot carry values with commas
# (e.g. MODEL_PRICES JSON). API_KEY is never here: it comes from Secret Manager.
ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT

yaml_line() {  # KEY: 'value' (single quotes; a ' inside the value is doubled)
  local q="'"
  local value="${2//$q/$q$q}"
  printf "%s: '%s'\n" "$1" "$value" >> "$ENV_FILE"
}

yaml_line APP_ENV production
yaml_line GOOGLE_CLOUD_PROJECT "$PROJECT_ID"
yaml_line GOOGLE_CLOUD_LOCATION "$GOOGLE_CLOUD_LOCATION"
yaml_line GOOGLE_GENAI_USE_ENTERPRISE true
yaml_line WORKER_MODEL "$WORKER_MODEL"
yaml_line EMBEDDING_MODEL "$EMBEDDING_MODEL"
yaml_line EMBEDDING_LOCATION "$EMBEDDING_LOCATION"
yaml_line REPORTS_BUCKET "$REPORTS_BUCKET"
yaml_line TRACING_EXPORTER "$TRACING_EXPORTER"
yaml_line REQUEST_TIMEOUT_SECONDS "$REQUEST_TIMEOUT_SECONDS"

PASS_THROUGH=(
  ORCHESTRATOR_MODEL CRITIC_MODEL THINKING_LEVEL MODEL_PRICES LLM_TIMEOUT_SECONDS
  BQ_DATASET BQ_LOCATION BQ_MAX_BYTES_BILLED BQ_MAX_ROWS BQ_QUERY_TIMEOUT_SECONDS SQL_MAX_ATTEMPTS
  EMBEDDING_DIM RAG_COLLECTION RAG_TOP_K RRF_K CRITIC_MAX_REVISIONS MAX_QUESTION_CHARS
)
for name in "${PASS_THROUGH[@]}"; do
  if [[ -n "${!name:-}" ]]; then yaml_line "$name" "${!name}"; fi
done

echo "==> Building $IMAGE on Cloud Build"
gcloud builds submit . --project="$PROJECT_ID" --region="$REGION" --tag="$IMAGE"

echo "==> Deploying $SERVICE to Cloud Run ($REGION)"
gcloud run deploy "$SERVICE" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --image="$IMAGE" \
  --service-account="$RUNTIME_SA" \
  --cpu=1 \
  --memory=2Gi \
  --concurrency=4 \
  --timeout="$CLOUD_RUN_TIMEOUT" \
  --min-instances=0 \
  --max-instances=2 \
  --cpu-boost \
  --env-vars-file="$ENV_FILE" \
  --set-secrets="API_KEY=${API_KEY_SECRET}:${API_KEY_SECRET_VERSION}" \
  --allow-unauthenticated

URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')"
echo
echo "Deployed: $URL"
echo "Smoke test:  export SERVICE_URL=$URL  (and API_KEY, see docs/DEPLOY.md), then: make smoke-prod"
