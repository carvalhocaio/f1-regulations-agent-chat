"""Benchmark retrieval backends: local, vertex, vector_search.

Measures latency and basic quality metrics (`recall@k`) when ground truth IDs are
provided in the query dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from langchain_core.documents import Document

from f1_agent.tools import (
    _search_regulations_local,
    _search_regulations_vector_search,
    _search_regulations_vertex,
)


@dataclass
class QueryCase:
    query: str
    expected_ids: list[str]


@dataclass
class RetrievalResult:
    ok: bool
    latency_ms: float
    retrieved_ids: list[str]
    error: str | None = None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _doc_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown")).strip()
    page = str(doc.metadata.get("page", "unknown")).strip()
    article = str(doc.metadata.get("article", "")).strip()
    prefix = doc.page_content[:80].strip().replace("\n", " ")
    return f"{source}|{page}|{article}|{prefix}"


def _load_queries(path: Path) -> list[QueryCase]:
    entries: list[QueryCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            query = str(obj.get("query", "")).strip()
            if not query:
                continue
            expected_ids = [str(v) for v in list(obj.get("expected_ids", []) or [])]
            entries.append(QueryCase(query=query, expected_ids=expected_ids))
    if not entries:
        raise ValueError(f"No benchmark queries found in {path}")
    return entries


def _backend_search(backend: str, query: str, k: int) -> list[Document]:
    if backend == "local":
        return _search_regulations_local(query, k=k)
    if backend == "vertex":
        return _search_regulations_vertex(query, k=k)
    if backend == "vector_search":
        return _search_regulations_vector_search(query, k=k)
    raise ValueError(f"Unsupported backend: {backend}")


async def _run_one(backend: str, case: QueryCase, k: int, timeout_s: float):
    started = perf_counter()
    try:
        docs = await asyncio.wait_for(
            asyncio.to_thread(_backend_search, backend, case.query, k),
            timeout=timeout_s,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            ok=True,
            latency_ms=latency_ms,
            retrieved_ids=[_doc_id(doc) for doc in docs],
        )
    except Exception as exc:
        return RetrievalResult(
            ok=False,
            latency_ms=(perf_counter() - started) * 1000.0,
            retrieved_ids=[],
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_backend(
    *,
    backend: str,
    cases: list[QueryCase],
    k: int,
    concurrency: int,
    timeout_s: float,
) -> list[RetrievalResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _wrapped(case: QueryCase) -> RetrievalResult:
        async with semaphore:
            return await _run_one(backend, case, k, timeout_s)

    tasks = [_wrapped(case) for case in cases]
    return await asyncio.gather(*tasks)


def _recall_at_k(case: QueryCase, result: RetrievalResult, k: int) -> float | None:
    if not case.expected_ids:
        return None
    expected = set(case.expected_ids)
    if not expected:
        return None
    got = set(result.retrieved_ids[:k])
    return len(expected.intersection(got)) / len(expected)


def _summarize(
    backend: str,
    cases: list[QueryCase],
    results: list[RetrievalResult],
    k: int,
) -> dict[str, object]:
    oks = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    latencies = [r.latency_ms for r in oks]

    report: dict[str, object] = {
        "backend": backend,
        "total_queries": len(results),
        "success_count": len(oks),
        "error_count": len(errors),
        "timeout_rate": round(len(errors) / max(1, len(results)), 4),
    }
    non_empty = [r for r in oks if r.retrieved_ids]
    report["non_empty_count"] = len(non_empty)
    report["empty_result_rate"] = round(
        (len(oks) - len(non_empty)) / max(1, len(oks)), 4
    )

    if latencies:
        report.update(
            {
                "retrieval_p50_ms": round(_percentile(latencies, 0.50), 2),
                "retrieval_p95_ms": round(_percentile(latencies, 0.95), 2),
                "retrieval_p99_ms": round(_percentile(latencies, 0.99), 2),
                "retrieval_avg_ms": round(mean(latencies), 2),
            }
        )

    recalls: list[float] = []
    for case, result in zip(cases, results):
        recall = _recall_at_k(case, result, k=k)
        if recall is not None:
            recalls.append(recall)
    if recalls:
        report[f"recall@{k}"] = round(mean(recalls), 4)

    if errors:
        report["errors"] = [r.error for r in errors[:10]]

    return report


async def _run(args: argparse.Namespace) -> dict[str, object]:
    cases = _load_queries(Path(args.queries_file))
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    reports = []
    started = perf_counter()
    for backend in backends:
        if backend not in {"local", "vertex", "vector_search"}:
            raise ValueError(f"Unsupported backend in --backends: {backend}")
        print(
            f"Running backend={backend} queries={len(cases)} concurrency={args.concurrency}"
        )
        results = await _run_backend(
            backend=backend,
            cases=cases,
            k=args.top_k,
            concurrency=args.concurrency,
            timeout_s=args.timeout_seconds,
        )
        reports.append(_summarize(backend, cases, results, k=args.top_k))

    wall_s = perf_counter() - started
    return {
        "queries_file": args.queries_file,
        "total_queries": len(cases),
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "backends": backends,
        "wall_time_seconds": round(wall_s, 3),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval backends (local, vertex, vector_search)"
    )
    parser.add_argument(
        "--queries-file", required=True, help="JSONL query dataset file"
    )
    parser.add_argument(
        "--backends",
        default="local,vertex,vector_search",
        help="Comma-separated backends: local,vertex,vector_search",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")

    report = asyncio.run(_run(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
