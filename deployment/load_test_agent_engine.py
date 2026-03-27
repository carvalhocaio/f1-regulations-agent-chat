"""Basic load test for a deployed Vertex AI Agent Engine resource.

Runs concurrent async_stream_query calls and reports latency percentiles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from time import perf_counter

import vertexai

from f1_agent.resilience import classify_transient_error


@dataclass
class RequestResult:
    ok: bool
    latency_seconds: float
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


async def _consume_stream(remote_agent: object, user_id: str, message: str) -> None:
    async for _event in remote_agent.async_stream_query(
        user_id=user_id, message=message
    ):
        pass


async def _run_one(
    *,
    semaphore: asyncio.Semaphore,
    remote_agent: object,
    user_id: str,
    message: str,
    timeout_seconds: float,
) -> RequestResult:
    async with semaphore:
        start = perf_counter()
        try:
            await asyncio.wait_for(
                _consume_stream(
                    remote_agent=remote_agent, user_id=user_id, message=message
                ),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            is_transient, status_code, error_type = classify_transient_error(exc)
            return RequestResult(
                ok=False,
                latency_seconds=perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
                error_type=error_type,
                status_code=status_code,
                is_transient=is_transient,
            )
        return RequestResult(ok=True, latency_seconds=perf_counter() - start)


async def _run_load(args: argparse.Namespace) -> dict[str, object]:
    client = vertexai.Client(project=args.project_id, location=args.location)
    remote_agent = client.agent_engines.get(name=args.resource_name)

    if args.warmup_requests > 0:
        print(f"Warmup: running {args.warmup_requests} request(s)")
        warmup_tasks = [
            _run_one(
                semaphore=asyncio.Semaphore(max(1, args.concurrency)),
                remote_agent=remote_agent,
                user_id=f"{args.user_id_prefix}-warmup-{idx}",
                message=args.message,
                timeout_seconds=args.timeout_seconds,
            )
            for idx in range(args.warmup_requests)
        ]
        await asyncio.gather(*warmup_tasks)

    total_requests = args.total_requests
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        _run_one(
            semaphore=semaphore,
            remote_agent=remote_agent,
            user_id=f"{args.user_id_prefix}-{idx % max(1, args.user_pool_size)}",
            message=args.message,
            timeout_seconds=args.timeout_seconds,
        )
        for idx in range(total_requests)
    ]

    started = perf_counter()
    results = await asyncio.gather(*tasks)
    wall_time = perf_counter() - started

    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    latencies = [r.latency_seconds for r in ok]

    report: dict[str, object] = {
        "project_id": args.project_id,
        "location": args.location,
        "resource_name": args.resource_name,
        "total_requests": total_requests,
        "concurrency": args.concurrency,
        "success_count": len(ok),
        "error_count": len(errors),
        "wall_time_seconds": round(wall_time, 4),
        "throughput_rps": round(total_requests / wall_time, 4)
        if wall_time > 0
        else 0.0,
    }

    if latencies:
        report.update(
            {
                "latency_avg_seconds": round(mean(latencies), 4),
                "latency_p50_seconds": round(_percentile(latencies, 0.50), 4),
                "latency_p95_seconds": round(_percentile(latencies, 0.95), 4),
                "latency_p99_seconds": round(_percentile(latencies, 0.99), 4),
                "latency_max_seconds": round(max(latencies), 4),
            }
        )

    if errors:
        report["errors"] = [r.error for r in errors[:10]]
        report.update(
            _summarize_error_metrics(errors=errors, total_requests=total_requests)
        )

    return report


def _summarize_error_metrics(
    *, errors: list[RequestResult], total_requests: int
) -> dict[str, object]:
    error_buckets: Counter[str] = Counter()
    error_type_buckets: Counter[str] = Counter()
    transient_error_count = 0
    quota_transient_error_count = 0

    for item in errors:
        bucket = _error_bucket_key(item)
        error_buckets[bucket] += 1
        error_type_buckets[item.error_type or "UnknownError"] += 1

        if item.is_transient:
            transient_error_count += 1
            if _is_quota_like_error(item):
                quota_transient_error_count += 1

    return {
        "error_buckets": dict(error_buckets.most_common()),
        "error_type_buckets": dict(error_type_buckets.most_common()),
        "transient_error_count": transient_error_count,
        "transient_error_rate": round(
            transient_error_count / max(1, total_requests), 4
        ),
        "quota_transient_error_count": quota_transient_error_count,
        "quota_transient_error_rate": round(
            quota_transient_error_count / max(1, total_requests), 4
        ),
    }


def _error_bucket_key(item: RequestResult) -> str:
    status = str(item.status_code) if item.status_code is not None else "unknown"
    error_type = item.error_type or "UnknownError"
    return f"{status}:{error_type}"


def _is_quota_like_error(item: RequestResult) -> bool:
    if item.status_code in {429, 503}:
        return True
    text = (item.error or "").lower()
    return any(
        token in text
        for token in ("resourceexhausted", "too many requests", "rate limit")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test a deployed Agent Engine")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--resource-name", required=True)
    parser.add_argument(
        "--message",
        default="Give me a one-line summary of Formula 1.",
        help="Prompt sent in each request",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=100,
        help="Total number of requests in this run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of in-flight requests",
    )
    parser.add_argument(
        "--user-pool-size",
        type=int,
        default=20,
        help="Distinct user_id cardinality rotated during the run",
    )
    parser.add_argument(
        "--user-id-prefix",
        default="load-test-user",
        help="Prefix used to generate user IDs",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Per-request timeout",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=10,
        help="Requests sent before measurement to reduce cold-start noise",
    )
    args = parser.parse_args()

    if args.total_requests < 1:
        raise ValueError("--total-requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.user_pool_size < 1:
        raise ValueError("--user-pool-size must be >= 1")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")

    report = asyncio.run(_run_load(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
