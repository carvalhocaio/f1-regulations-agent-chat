"""Deploy the F1 agent to Vertex AI Agent Engine."""

import argparse
import json
from datetime import datetime

import vertexai
from google.cloud import secretmanager
from google.genai import errors as genai_errors
from vertexai import types

from f1_agent.agent import root_agent


def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Fetch a secret value from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def _resource_name(agent_engine: object) -> str | None:
    """Extract resource name across SDK object variations."""
    resource_name = getattr(agent_engine, "resource_name", None)
    if resource_name:
        return str(resource_name)

    resource_name = getattr(agent_engine, "resourceName", None)
    if resource_name:
        return str(resource_name)

    name = getattr(agent_engine, "name", None)
    if name:
        return str(name)

    api_resource = getattr(agent_engine, "api_resource", None)
    if api_resource is not None:
        api_name = getattr(api_resource, "name", None)
        if api_name:
            return str(api_name)

        if isinstance(api_resource, dict):
            for key in ("name", "resource_name", "resourceName"):
                value = api_resource.get(key)
                if value:
                    return str(value)

    return None


def _display_name(agent_engine: object) -> str | None:
    """Extract display name across SDK object variations."""
    display_name = getattr(agent_engine, "display_name", None)
    if display_name:
        return str(display_name)

    display_name = getattr(agent_engine, "displayName", None)
    if display_name:
        return str(display_name)

    api_resource = getattr(agent_engine, "api_resource", None)
    if api_resource is not None:
        api_display_name = getattr(api_resource, "display_name", None)
        if api_display_name:
            return str(api_display_name)

        api_display_name = getattr(api_resource, "displayName", None)
        if api_display_name:
            return str(api_display_name)

        if isinstance(api_resource, dict):
            for key in ("display_name", "displayName"):
                value = api_resource.get(key)
                if value:
                    return str(value)

    return None


def _created_at(agent_engine: object) -> datetime | None:
    """Extract creation time across SDK object variations."""
    candidates: list[object] = []

    for attr in ("create_time", "createTime", "created_at", "createdAt"):
        value = getattr(agent_engine, attr, None)
        if value is not None:
            candidates.append(value)

    api_resource = getattr(agent_engine, "api_resource", None)
    if api_resource is not None:
        for attr in ("create_time", "createTime"):
            value = getattr(api_resource, attr, None)
            if value is not None:
                candidates.append(value)
        if isinstance(api_resource, dict):
            for key in ("create_time", "createTime"):
                value = api_resource.get(key)
                if value is not None:
                    candidates.append(value)

    for value in candidates:
        if isinstance(value, datetime):
            return value
        text = str(value)
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue

    return None


def find_existing_agent(client: vertexai.Client, display_name: str) -> str | None:
    """Find an existing agent by display name, return resource_name or None."""
    matches: list[tuple[datetime, str]] = []

    for engine in client.agent_engines.list():
        candidate_name = _resource_name(engine)
        if not candidate_name:
            continue

        listed_display_name = _display_name(engine)

        # Some SDK list responses may omit display_name fields.
        if listed_display_name is None:
            listed_display_name = _display_name(
                client.agent_engines.get(name=candidate_name)
            )

        if listed_display_name != display_name:
            continue

        created_at = _created_at(engine) or datetime.min
        matches.append((created_at, candidate_name))

    if matches:
        # If there are duplicates, prefer the most recently created one.
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    return None


def build_agent_engine_config(
    args: argparse.Namespace, env_vars: dict[str, str]
) -> types.AgentEngineConfig:
    """Build a shared Agent Engine config for create/update."""
    return types.AgentEngineConfig(
        requirements="requirements-deploy.txt",
        extra_packages=["f1_agent", "vector_store", "f1_data"],
        display_name=args.display_name,
        description=args.description,
        service_account=args.service_account,
        env_vars=env_vars,
        staging_bucket=args.staging_bucket,
    )


def _without_service_account(
    config: types.AgentEngineConfig,
) -> types.AgentEngineConfig:
    """Clone config without service account for fallback deploy."""
    payload = config.model_dump(exclude_none=True)
    payload.pop("service_account", None)
    return types.AgentEngineConfig.model_validate(payload)


def _is_service_account_actas_error(exc: Exception) -> bool:
    """Return true when error indicates missing iam.serviceAccountUser."""
    if not isinstance(exc, genai_errors.ClientError):
        return False
    message = str(exc).lower()
    return "permission to act as service_account" in message


def main():
    parser = argparse.ArgumentParser(description="Deploy F1 agent to Agent Engine")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--staging-bucket", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--description", default="F1 Regulations & History Agent")
    parser.add_argument("--service-account", default=None)
    parser.add_argument(
        "--rag-backend",
        default="auto",
        choices=["auto", "local", "vertex"],
        help="RAG backend routing mode for regulations retrieval",
    )
    parser.add_argument(
        "--rag-corpus",
        default="",
        help="Vertex RAG corpus resource name (required for vertex mode)",
    )
    parser.add_argument(
        "--rag-location",
        default="",
        help="Optional Vertex RAG location override (defaults to --location)",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=5,
        help="Top-k retrieval docs for Vertex RAG",
    )
    parser.add_argument(
        "--rag-vector-distance-threshold",
        type=float,
        default=None,
        help="Optional Vertex RAG vector distance threshold",
    )
    parser.add_argument(
        "--example-store-enabled",
        action="store_true",
        help="Enable dynamic few-shot retrieval from Example Store",
    )
    parser.add_argument(
        "--example-store-name",
        default="",
        help="Example Store resource name (projects/.../locations/.../exampleStores/...)",
    )
    parser.add_argument(
        "--example-store-top-k",
        type=int,
        default=3,
        help="Top-k dynamic examples retrieved per request",
    )
    parser.add_argument(
        "--example-store-min-score",
        type=float,
        default=0.65,
        help="Minimum similarity score required for dynamic examples",
    )
    args = parser.parse_args()

    if args.example_store_enabled and not args.example_store_name:
        raise ValueError(
            "--example-store-enabled requires --example-store-name "
            "(projects/.../locations/.../exampleStores/...)"
        )

    client = vertexai.Client(
        project=args.project_id,
        location=args.location,
    )

    # F1_TUNED_MODEL is optional — falls back to gemini-2.5-flash if not set
    try:
        tuned_model = get_secret(args.project_id, "f1-tuned-model")
    except Exception:
        tuned_model = ""
        print("Warning: f1-tuned-model secret not found, using base Flash model")

    env_vars = {
        "GEMINI_API_KEY": get_secret(args.project_id, "google-api-key"),
        "F1_TUNED_MODEL": tuned_model,
        "F1_RAG_BACKEND": args.rag_backend,
        "F1_RAG_PROJECT_ID": args.project_id,
        "F1_RAG_LOCATION": args.rag_location or args.location,
        "F1_RAG_TOP_K": str(max(1, args.rag_top_k)),
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
        "F1_EXAMPLE_STORE_ENABLED": "true" if args.example_store_enabled else "false",
        "F1_EXAMPLE_STORE_NAME": args.example_store_name,
        "F1_EXAMPLE_STORE_TOP_K": str(max(1, args.example_store_top_k)),
        "F1_EXAMPLE_STORE_MIN_SCORE": str(args.example_store_min_score),
    }

    if args.rag_corpus:
        env_vars["F1_RAG_CORPUS"] = args.rag_corpus

    if args.rag_vector_distance_threshold is not None:
        env_vars["F1_RAG_VECTOR_DISTANCE_THRESHOLD"] = str(
            args.rag_vector_distance_threshold
        )

    config = build_agent_engine_config(args, env_vars)
    existing = find_existing_agent(client, args.display_name)

    if existing:
        print(f"Updating existing agent: {existing}")
        try:
            client.agent_engines.update(
                name=existing,
                agent=root_agent,
                config=config,
            )
        except Exception as exc:
            if args.service_account and _is_service_account_actas_error(exc):
                print(
                    "Warning: missing iam.serviceAccountUser on configured service "
                    "account; retrying update without explicit service_account"
                )
                client.agent_engines.update(
                    name=existing,
                    agent=root_agent,
                    config=_without_service_account(config),
                )
            else:
                raise
        resource_name = existing
    else:
        print("Creating new agent...")
        try:
            remote = client.agent_engines.create(
                agent=root_agent,
                config=config,
            )
        except Exception as exc:
            if args.service_account and _is_service_account_actas_error(exc):
                print(
                    "Warning: missing iam.serviceAccountUser on configured service "
                    "account; retrying create without explicit service_account"
                )
                remote = client.agent_engines.create(
                    agent=root_agent,
                    config=_without_service_account(config),
                )
            else:
                raise
        resource_name = _resource_name(remote)

    if not resource_name:
        raise RuntimeError(
            "Failed to resolve Agent Engine resource name from SDK response"
        )

    print(f"Resource name: {resource_name}")

    metadata = {"resource_name": resource_name, "display_name": args.display_name}
    with open("deployment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
