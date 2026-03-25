"""Deploy the F1 agent to Vertex AI Agent Engine."""

import argparse
import json

import vertexai
from google.cloud import secretmanager
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

    return None


def _display_name(agent_engine: object) -> str | None:
    """Extract display name across SDK object variations."""
    display_name = getattr(agent_engine, "display_name", None)
    if display_name:
        return str(display_name)

    display_name = getattr(agent_engine, "displayName", None)
    if display_name:
        return str(display_name)

    return None


def find_existing_agent(client: vertexai.Client, display_name: str) -> str | None:
    """Find an existing agent by display name, return resource_name or None."""
    for engine in client.agent_engines.list():
        candidate_name = _resource_name(engine)
        listed_display_name = _display_name(engine)

        # Some SDK list responses may omit display_name fields.
        if listed_display_name is None and candidate_name:
            listed_display_name = _display_name(
                client.agent_engines.get(name=candidate_name)
            )

        if listed_display_name == display_name:
            return candidate_name
    return None


def build_agent_engine_config(
    args: argparse.Namespace, env_vars: dict[str, str]
) -> types.AgentEngineConfig:
    """Build a shared Agent Engine config for create/update."""
    return types.AgentEngineConfig(
        requirements_file="requirements-deploy.txt",
        extra_packages=["f1_agent", "vector_store", "f1_data"],
        display_name=args.display_name,
        description=args.description,
        service_account=args.service_account,
        env_vars=env_vars,
        staging_bucket=args.staging_bucket,
    )


def main():
    parser = argparse.ArgumentParser(description="Deploy F1 agent to Agent Engine")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--staging-bucket", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--description", default="F1 Regulations & History Agent")
    parser.add_argument("--service-account", default=None)
    args = parser.parse_args()

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
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }

    config = build_agent_engine_config(args, env_vars)
    existing = find_existing_agent(client, args.display_name)

    if existing:
        print(f"Updating existing agent: {existing}")
        client.agent_engines.update(
            name=existing,
            agent_engine=root_agent,
            config=config,
        )
        resource_name = existing
    else:
        print("Creating new agent...")
        remote = client.agent_engines.create(
            agent_engine=root_agent,
            config=config,
        )
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
