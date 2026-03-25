"""Smoke tests for Agent Engine client-based SDK flows.

Validates list/get and optionally delete against a deployed resource.
"""

from __future__ import annotations

import argparse

import vertexai


def _resource_name(agent_engine: object) -> str | None:
    resource_name = getattr(agent_engine, "resource_name", None)
    if resource_name:
        return str(resource_name)

    name = getattr(agent_engine, "name", None)
    if name:
        return str(name)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test Agent Engine list/get/delete using vertexai.Client"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--resource-name", required=True)
    parser.add_argument(
        "--display-name",
        default=None,
        help="Optional expected display name for extra validation",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete resource at the end (destructive)",
    )
    args = parser.parse_args()

    client = vertexai.Client(project=args.project_id, location=args.location)

    print("[1/3] list")
    listed = list(client.agent_engines.list())
    listed_names = {_resource_name(engine) for engine in listed}
    if args.resource_name not in listed_names:
        raise RuntimeError(
            f"Target resource not found in list() response: {args.resource_name}"
        )
    print(f"Found {len(listed)} resource(s); target is listed")

    print("[2/3] get")
    engine = client.agent_engines.get(name=args.resource_name)
    got_name = _resource_name(engine)
    if got_name != args.resource_name:
        raise RuntimeError(f"get() returned unexpected resource name: {got_name}")

    if args.display_name and engine.display_name != args.display_name:
        raise RuntimeError(
            "get() returned unexpected display_name: "
            f"{engine.display_name} (expected {args.display_name})"
        )
    print(f"Fetched: {got_name}")

    if args.delete:
        print("[3/3] delete")
        client.agent_engines.delete(name=args.resource_name)
        print(f"Deleted: {args.resource_name}")
    else:
        print("[3/3] delete skipped (pass --delete to enable)")

    print("Smoke test passed")


if __name__ == "__main__":
    main()
