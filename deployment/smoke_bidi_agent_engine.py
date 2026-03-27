"""Smoke test for Agent Engine bidirectional streaming.

Connects to a deployed Agent Engine via live bidi channel and prints protocol
v1 events (`turn_start`, `delta`, `turn_end`, `error`) for a single turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import vertexai
from google import genai

from f1_agent.bidi import iter_bidi_turn_events
from f1_agent.response_contract import validate_contract_payload
from f1_agent.websocket_bridge import build_agent_connection_factory


async def _run(args: argparse.Namespace) -> None:
    genai_client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )
    remote_agent = vertexai.Client(
        project=args.project_id,
        location=args.location,
    ).agent_engines.get(name=args.resource_name)

    connection_factory = build_agent_connection_factory(
        client=genai_client,
        agent_engine=args.resource_name,
        remote_agent=remote_agent,
        class_method="bidi_stream_query",
    )

    request_payload = {"input": args.message, "user_id": args.user_id}
    if args.response_contract_id:
        request_payload["response_contract_id"] = args.response_contract_id

    response_text_chunks: list[str] = []

    async with connection_factory() as connection:
        request_id = str(uuid.uuid4())
        async for event in iter_bidi_turn_events(
            connection=connection,
            request=request_payload,
            request_id=request_id,
            user_id=args.user_id,
            session_id=args.session_id,
            receive_timeout_seconds=args.receive_timeout_seconds,
        ):
            print(json.dumps(event, ensure_ascii=True))
            if event.get("event_type") == "delta":
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    text = payload.get("text")
                    if isinstance(text, str):
                        response_text_chunks.append(text)

    if args.response_contract_id:
        _validate_structured_output(
            contract_id=args.response_contract_id,
            response_text="".join(response_text_chunks),
        )


def _validate_structured_output(*, contract_id: str, response_text: str) -> None:
    if not response_text.strip():
        raise RuntimeError(
            "Structured response validation failed: empty response text from stream"
        )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Structured response validation failed: invalid JSON ({exc})"
        ) from exc

    is_valid, error = validate_contract_payload(contract_id, payload)
    if not is_valid:
        raise RuntimeError(
            "Structured response validation failed: "
            f"contract={contract_id} error={error}"
        )

    print(
        json.dumps(
            {
                "structured_validation": "ok",
                "contract_id": contract_id,
            },
            ensure_ascii=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test bidirectional streaming for Agent Engine"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--resource-name", required=True)
    parser.add_argument(
        "--message",
        default="Give me a one-line summary of Formula 1.",
        help="Input text sent via bidi stream",
    )
    parser.add_argument(
        "--user-id",
        default="smoke-bidi-user",
        help="Logical user id for protocol envelope",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id for protocol envelope",
    )
    parser.add_argument(
        "--receive-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-receive timeout while waiting for stream events",
    )
    parser.add_argument(
        "--response-contract-id",
        default="",
        help=(
            "Optional structured response contract id. If provided, smoke test "
            "validates streamed JSON payload against the contract."
        ),
    )
    args = parser.parse_args()

    if args.receive_timeout_seconds <= 0:
        raise ValueError("--receive-timeout-seconds must be > 0")
    if args.response_contract_id:
        # Only validate known contracts in strict mode.
        if args.response_contract_id not in {
            "sources_block_v1",
            "comparison_table_v1",
        }:
            raise ValueError(
                "--response-contract-id must be one of: "
                "sources_block_v1, comparison_table_v1"
            )

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
