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

### 1.6) Store API Key in Secret Manager

```fish
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create google-api-key \
  --project $PROJECT_ID \
  --replication-policy automatic \
  --data-file=-

# Grant access to SA
gcloud secrets add-iam-policy-binding google-api-key \
  --project $PROJECT_ID \
  --member "serviceAccount:$SA_EMAIL" \
  --role roles/secretmanager.secretAccessor
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

## 5) Manual deploy (local)

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
  --service-account $SA_EMAIL
```

### 5.1) Smoke test

```fish
python -c '
import os, asyncio, vertexai
from vertexai import agent_engines

vertexai.init(
    project=os.environ["PROJECT_ID"],
    location=os.environ["LOCATION"],
)

remote = agent_engines.get(os.environ["RESOURCE_NAME"])

async def main():
    async for event in remote.async_stream_query(
        user_id="smoke-test",
        message="Who has the most wins in F1 history?",
    ):
        print(event)

asyncio.run(main())
'
```

---

## 6) CI/CD — GitHub Actions

### 6.1) GitHub Secrets

Configure under **Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | `f1-regulations-agent-chat` |
| `GCP_SA_KEY` | Service Account JSON key (or use Workload Identity Federation) |
| `GCP_REGION` | `us-central1` |
| `GCP_STAGING_BUCKET` | `gs://f1-agent-staging` |
| `GCP_SA_EMAIL` | `f1-agent-engine@f1-regulations-agent-chat.iam.gserviceaccount.com` |

> **Recommended**: Use [Workload Identity Federation](https://github.com/google-github-actions/auth#workload-identity-federation) instead of SA key for keyless authentication.

### 6.2) Workflows

- **`.github/workflows/ci.yml`** — Lint and format check on PRs
- **`.github/workflows/deploy.yml`** — Deploy to production on merge to `main` (requires environment approval)

### 6.3) Configure GitHub Environment

Under **Settings > Environments**, create:

- **production** — with **Required reviewers** (manual approval before deploy)

---

## 7) Upload artifacts to bucket

Since `vector_store/` and `f1_data/` are generated once and excluded from git, upload them to the artifacts bucket:

```fish
# Upload (run once, or when data changes)
gcloud storage cp -r vector_store/ "gs://f1-regulations-agent-chat-artifacts/vector_store/"
gcloud storage cp -r f1_data/ "gs://f1-regulations-agent-chat-artifacts/f1_data/"
```

CI automatically downloads these artifacts before deploy (see "Download data artifacts" step in the workflow).

---

## 8) Update the agent

To update an already deployed agent without creating a new instance:

```fish
# deploy.py automatically detects if the agent exists (by display_name)
# and runs update instead of create.
uv run python deployment/deploy.py \
  --project-id $PROJECT_ID \
  --location $LOCATION \
  --staging-bucket $STAGING_BUCKET \
  --display-name "f1-agent" \
  --service-account $SA_EMAIL
```

---

## 9) Delete agent

```fish
set -x RESOURCE_NAME "projects/.../locations/us-central1/reasoningEngines/..."

python -c '
import os, vertexai
from vertexai import agent_engines

vertexai.init(project=os.environ["PROJECT_ID"], location=os.environ["LOCATION"])
agent_engines.delete(os.environ["RESOURCE_NAME"])
print("Deleted:", os.environ["RESOURCE_NAME"])
'
```

---

## Notes

- **Reserved variables**: Never set `GOOGLE_CLOUD_PROJECT` or `GOOGLE_APPLICATION_CREDENTIALS` as env vars in the deploy. Use `vertexai.init()`.
- **Region**: Agent Engine must use the same region as the staging bucket.
- **LLM model version**: Production uses the model configured in `f1_agent/agent.py` (`root_agent.model`, currently `gemini-2.5-pro`). Any model change requires a new deploy.
- **Scaling**: Adjust `min_instances` and `max_instances` according to demand.
- **Data artifacts**: If PDFs or CSVs are updated, re-run `build_index.py` locally and upload the new artifacts to the bucket.
- **Telemetry**: The deploy already enables traces and logs via OpenTelemetry. Access them in the Vertex AI console under **Dashboard** and **Traces**.
