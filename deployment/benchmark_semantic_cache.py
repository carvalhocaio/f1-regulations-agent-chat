"""Benchmark semantic cache lookup scalability.

Measures lookup latency curves for the ANN-backed semantic cache and compares
against a simulated brute-force O(N) cosine scan baseline.

By default this script uses deterministic fake embeddings (no API calls).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np

from f1_agent.cache import SemanticCache


@dataclass
class RunMetrics:
    size: int
    ann_avg_ms: float
    ann_p50_ms: float
    ann_p95_ms: float
    ann_p99_ms: float
    ann_hits: int
    ann_hit_rate: float
    vector_scan_avg_ms: float
    vector_scan_p50_ms: float
    vector_scan_p95_ms: float
    vector_scan_p99_ms: float


class FakeEmbeddings:
    """Deterministic hash-based embedding provider for local benchmarks."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _build_questions(size: int) -> list[str]:
    return [
        f"Who won the F1 championship in season {1950 + (i % 75)}? #{i}"
        for i in range(size)
    ]


def _run_ann_lookups(
    cache: SemanticCache, queries: list[str]
) -> tuple[list[float], int]:
    latencies_ms: list[float] = []
    hits = 0
    for question in queries:
        result = cache.lookup(question)
        latencies_ms.append(result.lookup_ms)
        if result.answer is not None:
            hits += 1
    return latencies_ms, hits


def _run_bruteforce_lookups(
    vectors_matrix: np.ndarray,
    embedding_client: FakeEmbeddings,
    queries: list[str],
) -> list[float]:
    latencies_ms: list[float] = []
    for question in queries:
        q = np.asarray(embedding_client.embed_query(question), dtype=np.float32)
        start = perf_counter()
        _ = np.dot(vectors_matrix, q)
        latencies_ms.append((perf_counter() - start) * 1000)
    return latencies_ms


def _benchmark_size(
    *,
    size: int,
    lookups: int,
    warmup: int,
    rng: Random,
) -> RunMetrics:
    embedding_client = FakeEmbeddings()
    questions = _build_questions(size)

    with tempfile.TemporaryDirectory(prefix="semantic-cache-bench-") as tmpdir:
        cache = SemanticCache(
            cache_dir=Path(tmpdir),
            embeddings=embedding_client,
            sweep_every_ops=1000000,
            sweep_interval_seconds=1000000,
            max_entries=max(2 * size, 1000),
            top_k=8,
            hnsw_m=32,
            hnsw_ef_search=64,
        )

        for i, question in enumerate(questions):
            cache.put(question, f"Cached answer {i}")

        sample_queries = [
            questions[rng.randrange(0, len(questions))] for _ in range(lookups)
        ]
        warmup_queries = [
            questions[rng.randrange(0, len(questions))] for _ in range(warmup)
        ]

        _run_ann_lookups(cache, warmup_queries)

        ann_latencies, ann_hits = _run_ann_lookups(cache, sample_queries)

        vectors_matrix = np.stack(
            [
                np.asarray(embedding_client.embed_query(q), dtype=np.float32)
                for q in questions
            ]
        )
        vector_scan_latencies = _run_bruteforce_lookups(
            vectors_matrix=vectors_matrix,
            embedding_client=embedding_client,
            queries=sample_queries,
        )

    return RunMetrics(
        size=size,
        ann_avg_ms=round(mean(ann_latencies), 4),
        ann_p50_ms=round(_percentile(ann_latencies, 0.50), 4),
        ann_p95_ms=round(_percentile(ann_latencies, 0.95), 4),
        ann_p99_ms=round(_percentile(ann_latencies, 0.99), 4),
        ann_hits=ann_hits,
        ann_hit_rate=round(ann_hits / max(1, lookups), 4),
        vector_scan_avg_ms=round(mean(vector_scan_latencies), 4),
        vector_scan_p50_ms=round(_percentile(vector_scan_latencies, 0.50), 4),
        vector_scan_p95_ms=round(_percentile(vector_scan_latencies, 0.95), 4),
        vector_scan_p99_ms=round(_percentile(vector_scan_latencies, 0.99), 4),
    )


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [size for size in sizes if size > 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark semantic cache lookup latency"
    )
    parser.add_argument(
        "--sizes",
        default="500,2000,5000,10000",
        help="Comma-separated cache sizes to benchmark",
    )
    parser.add_argument(
        "--lookups",
        type=int,
        default=500,
        help="Number of measured hit lookups per size",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup lookups per size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling",
    )
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    if not sizes:
        raise SystemExit("No valid sizes provided")

    rng = Random(args.seed)
    metrics = [
        _benchmark_size(size=size, lookups=args.lookups, warmup=args.warmup, rng=rng)
        for size in sizes
    ]

    report = {
        "sizes": sizes,
        "lookups": args.lookups,
        "warmup": args.warmup,
        "seed": args.seed,
        "notes": [
            "ANN metrics include full cache lookup path (ANN search + SQLite answer fetch).",
            "vector_scan metrics are a synthetic O(N) cosine scan baseline over vectors only.",
        ],
        "curve": [metric.__dict__ for metric in metrics],
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
