"""Benchmark request/streaming modes for Agent Engine UX latency.

Compares:
- `query` (request/response)
- `async_stream_query` (unidirectional stream)
- `bidi_stream_query` (bidirectional live stream)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

import vertexai
from google import genai

from f1_agent.resilience import classify_transient_error
from f1_agent.streaming_protocol import extract_bidi_text, is_bidi_end_of_turn
from f1_agent.websocket_bridge import build_agent_connection_factory


@dataclass
class ModeResult:
    ok: bool
    ttft_seconds: float
    turn_seconds: float
    error: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    is_transient: bool = False


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(index, len(ordered) - 1))]


async def _run_query_mode(
    *, remote_agent: Any, user_id: str, message: str, timeout_seconds: float
) -> ModeResult:
    start = perf_counter()
    try:
        if hasattr(remote_agent, "query"):
            await asyncio.wait_for(
                asyncio.to_thread(remote_agent.query, user_id=user_id, message=message),
                timeout=timeout_seconds,
            )
        elif hasattr(remote_agent, "stream_query"):

            def _consume_sync_stream() -> None:
                for _ in remote_agent.stream_query(user_id=user_id, message=message):
                    pass

            await asyncio.wait_for(
                asyncio.to_thread(_consume_sync_stream),
                timeout=timeout_seconds,
            )
        else:
            raise RuntimeError("Remote agent does not support query/stream_query")

        elapsed = perf_counter() - start
        return ModeResult(ok=True, ttft_seconds=elapsed, turn_seconds=elapsed)
    except Exception as exc:
        is_transient, status_code, error_type = classify_transient_error(exc)
        return ModeResult(
            ok=False,
            ttft_seconds=0.0,
            turn_seconds=perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
            error_type=error_type,
            status_code=status_code,
            is_transient=is_transient,
        )


async def _run_async_stream_mode(
    *, remote_agent: Any, user_id: str, message: str, timeout_seconds: float
) -> ModeResult:
    start = perf_counter()
    first_event_at: float | None = None
    try:

        async def _consume() -> None:
            nonlocal first_event_at
            async for _event in remote_agent.async_stream_query(
                user_id=user_id, message=message
            ):
                if first_event_at is None:
                    first_event_at = perf_counter()

        await asyncio.wait_for(_consume(), timeout=timeout_seconds)
        end = perf_counter()
        ttft = (first_event_at or end) - start
        return ModeResult(ok=True, ttft_seconds=ttft, turn_seconds=end - start)
    except Exception as exc:
        is_transient, status_code, error_type = classify_transient_error(exc)
        return ModeResult(
            ok=False,
            ttft_seconds=0.0,
            turn_seconds=perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
            error_type=error_type,
            status_code=status_code,
            is_transient=is_transient,
        )


async def _run_bidi_mode(
    *,
    connection_factory: Any,
    user_id: str,
    message: str,
    timeout_seconds: float,
) -> ModeResult:
    start = perf_counter()
    first_event_at: float | None = None
    try:
        async with connection_factory() as connection:
            await asyncio.wait_for(
                connection.send({"input": message, "user_id": user_id}),
                timeout=timeout_seconds,
            )
            while True:
                raw = await asyncio.wait_for(
                    connection.receive(), timeout=timeout_seconds
                )
                if first_event_at is None:
                    first_event_at = perf_counter()

                if isinstance(raw, dict):
                    if extract_bidi_text(raw) is not None:
                        pass
                    if is_bidi_end_of_turn(raw):
                        break
        end = perf_counter()
        ttft = (first_event_at or end) - start
        return ModeResult(ok=True, ttft_seconds=ttft, turn_seconds=end - start)
    except Exception as exc:
        is_transient, status_code, error_type = classify_transient_error(exc)
        return ModeResult(
            ok=False,
            ttft_seconds=0.0,
            turn_seconds=perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
            error_type=error_type,
            status_code=status_code,
            is_transient=is_transient,
        )


def _summarize(mode: str, results: list[ModeResult]) -> dict[str, object]:
    oks = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]

    report: dict[str, object] = {
        "mode": mode,
        "total_requests": len(results),
        "success_count": len(oks),
        "error_count": len(errors),
    }

    if oks:
        ttfts = [r.ttft_seconds for r in oks]
        turns = [r.turn_seconds for r in oks]
        report.update(
            {
                "ttft_avg_seconds": round(mean(ttfts), 4),
                "ttft_p50_seconds": round(_percentile(ttfts, 0.50), 4),
                "ttft_p95_seconds": round(_percentile(ttfts, 0.95), 4),
                "turn_avg_seconds": round(mean(turns), 4),
                "turn_p50_seconds": round(_percentile(turns, 0.50), 4),
                "turn_p95_seconds": round(_percentile(turns, 0.95), 4),
            }
        )

    if errors:
        report["errors"] = [r.error for r in errors[:5]]

    return report


async def _run_mode(
    *,
    mode: str,
    total_requests: int,
    concurrency: int,
    user_pool_size: int,
    message: str,
    timeout_seconds: float,
    remote_agent: Any,
    genai_client: genai.Client,
    resource_name: str,
    user_id_prefix: str,
) -> list[ModeResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(idx: int) -> ModeResult:
        async with semaphore:
            user_id = f"{user_id_prefix}-{idx % max(1, user_pool_size)}"
            if mode == "query":
                return await _run_query_mode(
                    remote_agent=remote_agent,
                    user_id=user_id,
                    message=message,
                    timeout_seconds=timeout_seconds,
                )
            if mode == "async_stream":
                return await _run_async_stream_mode(
                    remote_agent=remote_agent,
                    user_id=user_id,
                    message=message,
                    timeout_seconds=timeout_seconds,
                )
            if mode == "bidi":
                return await _run_bidi_mode(
                    connection_factory=build_agent_connection_factory(
                        client=genai_client,
                        agent_engine=resource_name,
                        remote_agent=remote_agent,
                        class_method="bidi_stream_query",
                    ),
                    user_id=user_id,
                    message=message,
                    timeout_seconds=timeout_seconds,
                )
            raise ValueError(f"Unsupported mode: {mode}")

    tasks = [_one(i) for i in range(total_requests)]
    return await asyncio.gather(*tasks)


async def _run(args: argparse.Namespace) -> dict[str, object]:
    client = vertexai.Client(project=args.project_id, location=args.location)
    remote_agent = client.agent_engines.get(name=args.resource_name)
    genai_client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )

    selected_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in selected_modes:
        if mode not in {"query", "async_stream", "bidi"}:
            raise ValueError(f"Unsupported mode in --modes: {mode}")

    reports = []
    for mode in selected_modes:
        print(
            f"Running mode={mode} requests={args.total_requests} concurrency={args.concurrency}"
        )
        results = await _run_mode(
            mode=mode,
            total_requests=args.total_requests,
            concurrency=args.concurrency,
            user_pool_size=args.user_pool_size,
            message=args.message,
            timeout_seconds=args.timeout_seconds,
            remote_agent=remote_agent,
            genai_client=genai_client,
            resource_name=args.resource_name,
            user_id_prefix=args.user_id_prefix,
        )
        reports.append(_summarize(mode, results))

    return {
        "project_id": args.project_id,
        "location": args.location,
        "resource_name": args.resource_name,
        "total_requests_per_mode": args.total_requests,
        "concurrency": args.concurrency,
        "modes": selected_modes,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark query vs async_stream vs bidi stream modes"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--resource-name", required=True)
    parser.add_argument(
        "--modes",
        default="query,async_stream,bidi",
        help="Comma-separated: query,async_stream,bidi",
    )
    parser.add_argument(
        "--message",
        default="Give me a one-line summary of Formula 1.",
    )
    parser.add_argument("--total-requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--user-pool-size", type=int, default=10)
    parser.add_argument("--user-id-prefix", default="stream-bench-user")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.total_requests < 1:
        raise ValueError("--total-requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.user_pool_size < 1:
        raise ValueError("--user-pool-size must be >= 1")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")

    report = asyncio.run(_run(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
