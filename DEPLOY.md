# Deploy — Vertex AI Agent Engine (GCP)

Complete guide to deploy the F1 agent on **Vertex AI Agent Engine** with infrastructure via **Terraform** and CI/CD via **GitHub Actions**.

## Architecture

```
GitHub Actions (CI/CD)
  ├── PR → lint + tests
  └── merge main → deploy production (with manual approval)

GCP (f1-regulations-agent-chat)
  ├── Vertex AI Agent Engine  ← ADK agent (f1_agent)
  ├── Secret Manager          ← GEMINI_API_KEY
  ├── Cloud Storage           ← staging bucket + artifacts
  └── IAM                     ← dedicated service account
```

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- GitHub repository with Actions enabled
- Pre-generated data artifacts (`vector_store/` and `f1_data/`) — see [Section 2](#2-generate-data-artifacts)

---

## 1) Initial GCP setup

### 1.1) Environment variables

```fish
set -x PROJECT_ID "f1-regulations-agent-chat"
set -x LOCATION "us-central1"
set -x STAGING_BUCKET "gs://f1-agent-staging"
set -x SA_NAME "f1-agent-engine"
set -x SA_EMAIL "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
```

### 1.2) Authentication and APIs

```fish
gcloud auth login
gcloud auth application-default login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com
```

### 1.3) Create Service Account

```fish
gcloud iam service-accounts create $SA_NAME \
  --display-name "F1 Agent Engine SA" \
  --project $PROJECT_ID

# Required roles
for role in roles/aiplatform.user roles/storage.admin roles/secretmanager.secretAccessor roles/logging.logWriter
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member "serviceAccount:$SA_EMAIL" \
    --role $role
end
```

### 1.4) Create Staging Bucket

```fish
gcloud storage buckets create $STAGING_BUCKET \
  --project $PROJECT_ID \
  --location $LOCATION \
  --uniform-bucket-level-access
```

### 1.5) Configure AI Platform Service Agent

```fish
gcloud beta services identity create \
  --service aiplatform.googleapis.com \
  --project $PROJECT_ID

# Grant bucket access to the service agent
set -l PROJECT_NUMBER (gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
set -l SERVICE_AGENT "service-$PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding $STAGING_BUCKET \
  --member "serviceAccount:$SERVICE_AGENT" \
  --role roles/storage.objectAdmin
```

### 1.6) Store secrets in Secret Manager

```fish
# Gemini API key (required)
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create google-api-key \
  --project $PROJECT_ID \
  --replication-policy automatic \
  --data-file=-

# Fine-tuned model endpoint (optional — falls back to gemini-2.5-flash if not set)
# Only needed after running a fine-tuning job (see DEVELOPMENT.md)
echo -n "projects/PROJECT_NUMBER/locations/us-central1/endpoints/ENDPOINT_ID" | \
  gcloud secrets create f1-tuned-model \
  --project $PROJECT_ID \
  --replication-policy automatic \
  --data-file=-

# Grant access to SA
for secret in google-api-key f1-tuned-model
  gcloud secrets add-iam-policy-binding $secret \
    --project $PROJECT_ID \
    --member "serviceAccount:$SA_EMAIL" \
    --role roles/secretmanager.secretAccessor
end
```

> If any key was ever committed or exposed in logs/docs, rotate it immediately in the provider console and update Secret Manager with the new value.

---

## 2) Generate data artifacts

The artifacts (`vector_store/` and `f1_data/`) are generated **once** locally and uploaded as `extra_packages` during deploy.

```fish
# Install dependencies
uv sync

# Ensure source data is in docs/
# - FIA 2026 regulation PDFs (Sections A-F)
# - "Formula 1 World Championship (1950 - 2024)" folder with Kaggle CSVs

# Generate indices
uv run build_index.py

# Verify artifacts were created
ls vector_store/   # index.faiss, index.pkl
ls f1_data/        # f1_history.db
```

> **Important**: these directories must be present at deploy time. `.gitignore` excludes them from the repository, so they must be generated locally or downloaded from a bucket before deploying in CI.

---

## 3) Terraform — Infrastructure

### 3.1) Structure

```
deployment/
└── terraform/
    ├── main.tf          # provider, backend
    ├── variables.tf     # input variables
    ├── outputs.tf       # outputs (resource_name, etc.)
    ├── iam.tf           # service account + roles
    ├── storage.tf       # staging bucket
    ├── secrets.tf       # Secret Manager
    └── environments/
        └── production.tfvars
```

### 3.2) Initialize Terraform

```fish
# Create state bucket (once)
gcloud storage buckets create gs://f1-agent-terraform-state \
  --project $PROJECT_ID \
  --location $LOCATION \
  --uniform-bucket-level-access

# Initialize and apply
cd deployment/terraform
terraform init
terraform plan -var-file=environments/production.tfvars
terraform apply -var-file=environments/production.tfvars
```

---

## 4) Deploy script

The agent is deployed to Agent Engine via the Python SDK using `deployment/deploy.py`.

### 4.1) Generate requirements-deploy.txt

Agent Engine requires a flat `requirements.txt` (no `pyproject.toml` extras). Generate with:

```fish
uv export --frozen --no-hashes --no-dev --no-editable --no-annotate --no-header --no-emit-project -o requirements-deploy.txt
```

## 5) Prepare Vertex RAG corpus (optional but recommended)

Use this once (or when regulation PDFs change) to externalize regulations retrieval.

> For new projects, RAG Engine in `us-central1`, `us-east1`, and `us-east4` may require allowlist access. If you hit that error, run corpus ingestion in `europe-west4` or `europe-west3`.

```fish
uv run python deployment/rag_engine_ingest.py \
  --project-id $PROJECT_ID \
  --location "europe-west4" \
  --display-name "f1-regulations-rag" \
  --paths "gs://<BUCKET>/regulations/" \
  --chunk-size 1024 \
  --chunk-overlap 200
```

Copy the returned corpus resource name:

```text
projects/<PROJECT_NUMBER>/locations/europe-west4/ragCorpora/<RAG_CORPUS_ID>
```

## 5.1) Prepare Example Store for dynamic few-shot (A5, optional)

Use manual curation JSONL (v1) and sync with:

```fish
uv run python deployment/example_store_sync.py \
  --project-id $PROJECT_ID \
  --location us-central1 \
  --dataset data/example_store/manual_examples.v1.jsonl \
  --example-store-name "projects/<PROJECT_NUMBER>/locations/us-central1/exampleStores/<EXAMPLE_STORE_ID>"
```

If you need to create a store first, omit `--example-store-name` and pass a display name:

```fish
uv run python deployment/example_store_sync.py \
  --project-id $PROJECT_ID \
  --location us-central1 \
  --dataset data/example_store/manual_examples.v1.jsonl \
  --display-name "f1-real-errors"
```

---

## 6) Manual deploy (local)

For direct deploy without CI/CD:

```fish
# 1. Ensure artifacts exist
ls vector_store/ f1_data/

# 2. Sync environment (installs f1_agent as editable package)
uv sync

# 3. Generate requirements
uv export --frozen --no-hashes --no-dev --no-editable --no-annotate --no-header --no-emit-project -o requirements-deploy.txt

# 4. Deploy
uv run python deployment/deploy.py \
  --project-id $PROJECT_ID \
  --location $LOCATION \
  --staging-bucket $STAGING_BUCKET \
  --display-name "f1-agent" \
  --service-account $SA_EMAIL \
  --rag-backend auto \
  --rag-corpus "projects/<PROJECT_NUMBER>/locations/europe-west4/ragCorpora/<RAG_CORPUS_ID>" \
  --rag-location "europe-west4" \
  --example-store-enabled \
  --example-store-name "projects/<PROJECT_NUMBER>/locations/us-central1/exampleStores/<EXAMPLE_STORE_ID>" \
  --example-store-top-k 3 \
  --example-store-min-score 0.65 \
  --code-execution-enabled \
  --code-execution-location "us-central1" \
  --code-execution-agent-engine-name "projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>" \
  --code-execution-sandbox-ttl-seconds 3600 \
  --code-execution-max-rows 500
```

### 6.1) Smoke test

```fish
uv run python deployment/smoke_agent_engine.py \
  --project-id $PROJECT_ID \
  --location $LOCATION \
  --resource-name $RESOURCE_NAME \
  --display-name "f1-agent" \
  --user-id "smoke-user" \
  --session-ttl-seconds 864000
```

This smoke test now validates:
- Agent Engine list/get
- Sessions create/get/list/delete (managed)
- TTL payload applied on session creation

---

## 7) CI/CD — GitHub Actions

### 7.1) GitHub Secrets

Configure under **Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | `f1-regulations-agent-chat` |
| `GCP_SA_KEY` | Service Account JSON key (or use Workload Identity Federation) |
| `GCP_REGION` | `us-central1` |
| `GCP_STAGING_BUCKET` | `gs://f1-agent-staging` |
| `GCP_SA_EMAIL` | `f1-agent-engine@f1-regulations-agent-chat.iam.gserviceaccount.com` |
| `GCP_RAG_CORPUS` | `projects/<PROJECT_NUMBER>/locations/europe-west4/ragCorpora/<RAG_CORPUS_ID>` (optional) |
| `GCP_RAG_REGION` | `europe-west4` (optional; defaults to `GCP_REGION`) |
| `GCP_CODE_EXECUTION_ENABLED` | `true` or `false` (optional; enables A6 rollout in deploy workflow) |
| `GCP_CODE_EXECUTION_AGENT_ENGINE_NAME` | `projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>` (optional, recommended when enabling A6) |

> **Recommended**: Use [Workload Identity Federation](https://github.com/google-github-actions/auth#workload-identity-federation) instead of SA key for keyless authentication.

### 7.2) Workflows

- **`.github/workflows/ci.yml`** — Lint and format check on PRs
- **`.github/workflows/deploy.yml`** — Deploy to production on merge to `main` (requires environment approval). Pass `--rag-backend auto`, `--rag-corpus`, and optional `--rag-location`.

### 7.3) Configure GitHub Environment

Under **Settings > Environments**, create:

- **production** — with **Required reviewers** (manual approval before deploy)

---

## 8) Upload artifacts to bucket

Since `vector_store/` and `f1_data/` are generated once and excluded from git, upload them to the artifacts bucket:

```fish
# Upload (run once, or when data changes)
gcloud storage cp -r vector_store/* "gs://f1-regulations-agent-chat-artifacts/vector_store/"
gcloud storage cp -r f1_data/* "gs://f1-regulations-agent-chat-artifacts/f1_data/"
```

CI automatically downloads these artifacts before deploy (see "Download data artifacts" step in the workflow).

---

## 9) Update the agent

To update an already deployed agent without creating a new instance:

```fish
# deploy.py automatically detects if the agent exists (by display_name)
# and runs update instead of create.
uv run python deployment/deploy.py \
  --project-id $PROJECT_ID \
  --location $LOCATION \
  --staging-bucket $STAGING_BUCKET \
  --display-name "f1-agent" \
  --service-account $SA_EMAIL \
  --rag-backend auto \
  --rag-corpus "projects/<PROJECT_NUMBER>/locations/europe-west4/ragCorpora/<RAG_CORPUS_ID>" \
  --rag-location "europe-west4"
```

---

## 10) Delete agent

```fish
set -x RESOURCE_NAME "projects/.../locations/us-central1/reasoningEngines/..."

python -c '
import os
import vertexai

client = vertexai.Client(
    project=os.environ["PROJECT_ID"],
    location=os.environ["LOCATION"],
)
client.agent_engines.delete(name=os.environ["RESOURCE_NAME"])
print("Deleted:", os.environ["RESOURCE_NAME"])
'
```

---

## Notes

- **Reserved variables**: Never set `GOOGLE_CLOUD_PROJECT` or `GOOGLE_APPLICATION_CREDENTIALS` as env vars in the deploy. Use `vertexai.Client(...)` with explicit `project` and `location`.
- **Region**: Agent Engine must use the same region as the staging bucket.
- **LLM model version**: Production uses `gemini-2.5-pro` for complex queries and the fine-tuned Flash endpoint (`F1_TUNED_MODEL`) for simple queries. Model routing is automatic via callbacks.
- **RAG backend**: `F1_RAG_BACKEND=auto` is the recommended default. It prefers Vertex RAG when configured (`F1_RAG_CORPUS`) and falls back to local FAISS+BM25 for resilience.
- **RAG location**: `F1_RAG_LOCATION` can be different from Agent Engine region; use `europe-west4` (or `europe-west3`) when `us-central1` is allowlist-restricted.
- **Example Store**: only enable dynamic few-shot when `F1_EXAMPLE_STORE_NAME` points to a valid store. Recommended region for Example Store is `us-central1`.
- **Code Execution (A6)**: keep disabled by default and enable only with a clear safety policy. Current implementation is restricted to allowlisted analytical templates and `us-central1`.
- **Fine-tuned model**: The `f1-tuned-model` secret is optional. If not set, simple queries fall back to `gemini-2.5-flash`. See [DEVELOPMENT.md](./DEVELOPMENT.md#fine-tuning-production-only) for details.
- **Scaling**: Adjust `min_instances` and `max_instances` according to demand.
- **Data artifacts**: If PDFs or CSVs are updated, re-run `build_index.py` locally and upload the new artifacts to the bucket.
- **Telemetry**: The deploy already enables traces and logs via OpenTelemetry. Access them in the Vertex AI console under **Dashboard** and **Traces**.
- **Session contract**: client should propagate `user_id` + `session_id` on every request. If the app has no login, use a stable browser `client_id` and derive `user_id=anon-<hash(client_id)>`.
