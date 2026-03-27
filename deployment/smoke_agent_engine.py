"""Smoke tests for Agent Engine resource lifecycle checks.

Validates list/get for Agent Engine and optional resource deletion.
"""

from __future__ import annotations

import argparse

import vertexai


def _resource_name(agent_engine: object) -> str | None:
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


def _session_name(session: object) -> str | None:
    name = getattr(session, "name", None)
    if name:
        return str(name)
    if isinstance(session, dict):
        value = session.get("name")
        if value:
            return str(value)
    return None


def _is_session_resource_name(name: str | None) -> bool:
    return bool(name and "/sessions/" in name)


def _session_user_id(session: object) -> str | None:
    value = getattr(session, "user_id", None)
    if value:
        return str(value)
    value = getattr(session, "userId", None)
    if value:
        return str(value)
    if isinstance(session, dict):
        dict_value = session.get("user_id")
        if dict_value:
            return str(dict_value)
        dict_value = session.get("userId")
        if dict_value:
            return str(dict_value)
    return None


def _extract_session_name(create_response: object) -> str | None:
    nested_response = getattr(create_response, "response", None)
    if nested_response is not None:
        nested_name = _session_name(nested_response)
        if _is_session_resource_name(nested_name):
            return nested_name

    direct = _session_name(create_response)
    if _is_session_resource_name(direct):
        return direct

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

    print("[1/5] list")
    listed = list(client.agent_engines.list())
    listed_names = {_resource_name(engine) for engine in listed}
    if args.resource_name not in listed_names:
        raise RuntimeError(
            f"Target resource not found in list() response: {args.resource_name}"
        )
    print(f"Found {len(listed)} resource(s); target is listed")

    print("[2/5] get")
    engine = client.agent_engines.get(name=args.resource_name)
    got_name = _resource_name(engine)
    if got_name != args.resource_name:
        raise RuntimeError(f"get() returned unexpected resource name: {got_name}")

    fetched_display_name = _display_name(engine)
    if args.display_name and fetched_display_name != args.display_name:
        raise RuntimeError(
            "get() returned unexpected display_name: "
            f"{fetched_display_name} (expected {args.display_name})"
        )
    print(f"Fetched: {got_name}")

    print("[3/3] reasoning engine delete")
    if args.delete:
        client.agent_engines.delete(name=args.resource_name)
        print(f"Deleted: {args.resource_name}")
    else:
        print("Delete skipped (pass --delete to enable)")

    print("Smoke test passed")


if __name__ == "__main__":
    main()
