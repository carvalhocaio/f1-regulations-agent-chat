"""Deploy the F1 agent to Vertex AI Agent Engine."""

import argparse
import json

import vertexai
from google.cloud import secretmanager
from vertexai import agent_engines

from f1_agent.agent import root_agent


def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Fetch a secret value from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def find_existing_agent(display_name: str) -> str | None:
    """Find an existing agent by display name, return resource_name or None."""
    for engine in agent_engines.list():
        if engine.display_name == display_name:
            return engine.resource_name
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy F1 agent to Agent Engine")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--staging-bucket", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--description", default="F1 Regulations & History Agent")
    parser.add_argument("--service-account", default=None)
    args = parser.parse_args()

    vertexai.init(
        project=args.project_id,
        location=args.location,
        staging_bucket=args.staging_bucket,
    )

    env_vars = {
        "GEMINI_API_KEY": get_secret(args.project_id, "google-api-key"),
        "F1_TUNED_MODEL": get_secret(args.project_id, "f1-tuned-model"),
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }

    extra_packages = ["f1_agent", "vector_store", "f1_data"]

    existing = find_existing_agent(args.display_name)

    if existing:
        print(f"Updating existing agent: {existing}")
        remote = agent_engines.get(existing)
        remote.update(
            agent_engine=root_agent,
            requirements="requirements-deploy.txt",
            extra_packages=extra_packages,
            display_name=args.display_name,
            description=args.description,
            env_vars=env_vars,
        )
        resource_name = existing
    else:
        print("Creating new agent...")
        remote = agent_engines.create(
            agent_engine=root_agent,
            requirements="requirements-deploy.txt",
            extra_packages=extra_packages,
            display_name=args.display_name,
            description=args.description,
            service_account=args.service_account,
            env_vars=env_vars,
        )
        resource_name = remote.resource_name

    print(f"Resource name: {resource_name}")

    metadata = {"resource_name": resource_name, "display_name": args.display_name}
    with open("deployment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
