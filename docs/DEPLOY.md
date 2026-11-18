# Deploying to Cloud Run

The API runs as one Cloud Run service. The image is built by Cloud Build from the
`Dockerfile`, and it ships the retrieval index already built (`data/chroma`), so the
container never ingests documents.

```
client --X-API-Key--> Cloud Run (agentic-analyst, runtime SA agentic-analyst-run)
                        |-- Vertex AI (Gemini + query embeddings)   roles/aiplatform.user
                        |-- BigQuery (public thelook dataset)       roles/bigquery.jobUser
                        |-- Cloud Trace (telemetry.googleapis.com)  roles/cloudtrace.agent
                        |-- Cloud Storage reports bucket            roles/storage.objectCreator (bucket only)
                        |-- Secret Manager API key secret           roles/secretmanager.secretAccessor (secret only)
                        '-- stdout JSON -> Cloud Logging (per-request metrics)
```

Steps 1–6 are **one-time setup**. Steps 7–8 are **every deploy**.
All commands use the shell variables from step 0, so set those first in each new terminal.

## 0. Variables (every terminal session)

```bash
export PROJECT_ID=your-project-id          # your project; never committed
export REGION=us-central1
export SERVICE=agentic-analyst
export AR_REPO=agentic-analyst
export REPORTS_BUCKET="${PROJECT_ID}-agentic-analyst-reports"   # bucket names are global
export API_KEY_SECRET=agentic-analyst-api-key
export RUNTIME_SA="agentic-analyst-run@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud config set project "$PROJECT_ID"
```

`scripts/deploy.sh` uses the same defaults for `SERVICE`, `AR_REPO`, `API_KEY_SECRET` and `RUNTIME_SA`.

## One-time setup (run once per project)

### 1. Enable the APIs the service calls
The Cloud Build and Artifact Registry APIs are **already enabled** in this project, so they are not listed. Enabling an API that is already on does nothing.
```bash
gcloud services enable run.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com \
  bigquery.googleapis.com storage.googleapis.com telemetry.googleapis.com cloudtrace.googleapis.com
```

### 2. Create the Docker repository the images are pushed to
The Artifact Registry API is enabled, but the repository itself does not exist yet.
```bash
gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION" \
  --description="Agentic Analyst images"
```

### 3. Create the private reports bucket
This uses uniform bucket-level access (IAM only, no per-object ACLs), with public access blocked.
```bash
gcloud storage buckets create "gs://${REPORTS_BUCKET}" --location="$REGION" \
  --uniform-bucket-level-access --public-access-prevention
```

### 4. Create the API key secret (version 1)
This generates a random 64-character key and pipes it straight into Secret Manager, so it is never written to a file, a script or your shell history. `tr` removes the trailing newline.
```bash
openssl rand -hex 32 | tr -d '\n' | gcloud secrets create "$API_KEY_SECRET" \
  --replication-policy=automatic --data-file=-
```

### 5. Create the runtime service account
The service runs as this identity instead of the Compute Engine default account (which often has Editor).
```bash
gcloud iam service-accounts create agentic-analyst-run --display-name="Agentic Analyst Cloud Run runtime"
```

### 6. Grant the service account only what it needs
Call Gemini and the embedding model on Vertex AI:
```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/aiplatform.user --condition=None
```
Run BigQuery query jobs, billed to this project. The public dataset is readable by anyone, so no dataset role is needed:
```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/bigquery.jobUser --condition=None
```
Write trace spans. This role includes `telemetry.traces.write`, which the exporter to `telemetry.googleapis.com` needs:
```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/cloudtrace.agent --condition=None
```
Create report objects in this one bucket. The service cannot read, overwrite or delete them:
```bash
gcloud storage buckets add-iam-policy-binding "gs://${REPORTS_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/storage.objectCreator
```
Read this one secret, and no other:
```bash
gcloud secrets add-iam-policy-binding "$API_KEY_SECRET" \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/secretmanager.secretAccessor
```

## Every deploy

### 7. Build and deploy
`deploy.sh` reads the model settings from the environment. The easiest way to provide them is to load your local `.env` (it is never uploaded: `.gcloudignore` and `.dockerignore` are allow-lists). The script does not forward local paths or `API_KEY`.
```bash
make ingest                      # only if data/docs changed: the image ships this index
set -a; source .env; set +a      # GOOGLE_CLOUD_LOCATION, WORKER_MODEL, EMBEDDING_MODEL, ...
export API_KEY_SECRET_VERSION=1  # pinned secret version (Google recommends pinning for env vars)
make deploy
```
The script does three things:
1. Uploads the allow-listed files.
2. Builds the image on Cloud Build and pushes it to `REGION-docker.pkg.dev/PROJECT_ID/AR_REPO/SERVICE:<time>-<git sha>`.
3. Runs `gcloud run deploy` with these settings:

| Setting | Value | Why |
|---|---|---|
| CPU / memory | 1 / 2 GiB | ChromaDB + ADK fit with room for 4 concurrent requests |
| Concurrency | 4 | Requests per instance; most time is spent waiting on Gemini |
| Request timeout | 300 s | The app stops at `REQUEST_TIMEOUT_SECONDS=270` and returns clean JSON first |
| Min / max instances | 0 / 2 | Scales to zero when idle; at most 8 requests in flight (cost cap) |
| CPU boost | on | Faster cold starts (imports + index check) |
| Ingress auth | `--allow-unauthenticated` | Cloud Run IAM is open; the **app** requires `X-API-Key` on every route except `/health` |
| `APP_ENV` | `production` | The app refuses to start if `API_KEY` is missing |
| `API_KEY` | `--set-secrets API_KEY=agentic-analyst-api-key:<version>` | Injected from Secret Manager at instance start |

### 8. Smoke test
This fetches the service URL and your key into the current shell only, then runs three checks: `/health`, a 401 without a key, and one real question.
```bash
export SERVICE_URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
export API_KEY="$(gcloud secrets versions access "$API_KEY_SECRET_VERSION" --secret="$API_KEY_SECRET")"
make smoke-prod
```
Calling the API yourself:
```bash
curl -s -X POST "$SERVICE_URL/ask" -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" -d '{"question": "Which categories are above their maximum return rate?"}'
```

## Operating it

**Metrics.**
- Every request writes one JSON line to stdout. Cloud Logging stores it as a structured entry.
- To see all instances, open Logs Explorer and query:
  ```
  resource.type="cloud_run_revision" AND resource.labels.service_name="agentic-analyst"
  jsonPayload.log_type="request_metrics"
  ```
  The same from the terminal (read-only):
  `gcloud logging read 'jsonPayload.log_type="request_metrics"' --freshness=1d --limit=20 --format=json`
- `GET /metrics/summary` only reads the local JSONL file of **the instance that answers**. That file lives in the instance's memory and is lost when the instance stops, so the summary covers only the current instance (the response has a `scope` field saying so). Use Cloud Logging for real totals.

**Reports.**
- With `REPORTS_BUCKET` set, `save_report` creates `gs://$REPORTS_BUCKET/<title>-<timestamp>.md`. The API response contains only the object name.
- The service can only create objects. So a corrected version saved in the same session (after a critic revision) becomes `<name>-r2.md`, and the response returns the newest name. The earlier draft stays in the bucket.
- To list reports: `gcloud storage ls "gs://${REPORTS_BUCKET}"`.

**Rotating the key.**
1. Add a new secret version: `openssl rand -hex 32 | tr -d '\n' | gcloud secrets versions add "$API_KEY_SECRET" --data-file=-`
2. Set `API_KEY_SECRET_VERSION` to the new number and run `make deploy` again.
3. When the new revision is serving, disable the old version with `gcloud secrets versions disable`.

**Running the image locally** (needs Docker; optional): `make docker-build`, then
```bash
docker run --rm -p 8080:8080 --env-file .env -e APP_ENV=production -e API_KEY="$(openssl rand -hex 32)" \
  -v "$HOME/.config/gcloud:/home/app/.config/gcloud:ro" agentic-analyst:local
```
The mounted folder provides your Application Default Credentials. Without `API_KEY` the container refuses to start, which is intended.

## If something fails

| Symptom | Cause / fix |
|---|---|
| Deploy fails: "revision is not ready", logs show `IndexMissing` | The image has no index. Run `make ingest`, check that `data/chroma/chunks.json` exists, and deploy again. |
| Logs show `APP_ENV=production requires API_KEY` | The `--set-secrets` binding failed, or the version is disabled. Check step 6 (secret accessor) and `API_KEY_SECRET_VERSION`. |
| Build fails pushing the image | Cloud Build's service account (on newer projects, the Compute Engine default SA) needs `roles/artifactregistry.writer` on the repo and `roles/logging.logWriter`. |
| `--allow-unauthenticated` fails with an organization-policy error | Domain-restricted sharing blocks `allUsers`. Use `--no-invoker-iam-check` instead in `scripts/deploy.sh`; the app's API key still protects every route. |
| `/ask` returns 502 `auth_error` | A role from step 6 is missing, or an API from step 1 is disabled. |
| `save_report` says the report could not be saved | Check the `objectCreator` binding on the bucket and the `REPORTS_BUCKET` value. |
| No traces in Cloud Trace | The deploy always sets `TRACING_EXPORTER=gcp` (override with `DEPLOY_TRACING_EXPORTER`). Spans are exported in the background; Cloud Run throttles CPU between requests, so they can arrive late. |
