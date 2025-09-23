#!/usr/bin/env python3
"""
Async load test for /api/predict.

- Spawns N concurrent clients, each sending M requests
- Measures per-request latency and success rate
- Checks that p95 and max latency stay <= 1s (configurable)

Usage:
  python3 scripts/load_test_predict.py \
    --base_url http://localhost:8000 \
    --input examples/submission2.csv \
    --concurrency 100 \
    --requests-per-client 40 \
    --timeout 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import List, Tuple
import csv
import os
from dataclasses import dataclass

import aiohttp


DEFAULT_CSV = os.environ.get(
    "LOADTEST_INPUT",
    os.path.join(os.path.dirname(__file__), "..", "examples", "submission2.csv"),
)


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    http_status: int
    query: str


async def one_request(session: aiohttp.ClientSession, url: str, query: str, timeout_s: float) -> RequestResult:
    t0 = time.perf_counter()
    try:
        async with session.post(url, json={"input": query}, timeout=timeout_s) as resp:
            _ = await resp.json()
            dt = time.perf_counter() - t0
            return RequestResult(ok=(resp.status == 200), latency_s=dt, http_status=resp.status, query=query)
    except Exception:
        dt = time.perf_counter() - t0
        return RequestResult(ok=False, latency_s=dt, http_status=0, query=query)


async def client_worker(
    base_url: str,
    requests_per_client: int,
    timeout_s: float,
) -> List[RequestResult]:
    url = base_url.rstrip("/") + "/api/predict"
    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        results: List[RequestResult] = []
        for _ in range(requests_per_client):
            q = random.choice(GLOBAL_QUERIES)
            results.append(await one_request(session, url, q, timeout_s))
        return results


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def load_queries_from_csv(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            q = (r.get("search_query") or "").strip()
            if q:
                queries.append(q)
    return queries


def write_log(path: str, results: List[RequestResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "query", "ok", "http_status", "latency_s"])
        for i, r in enumerate(results):
            w.writerow([i, r.query, int(r.ok), r.http_status, f"{r.latency_s:.6f}"])


async def run_load(
    base_url: str,
    concurrency: int,
    requests_per_client: int,
    timeout_s: float,
    log_path: str | None,
):
    tasks = [
        asyncio.create_task(client_worker(base_url, requests_per_client, timeout_s))
        for _ in range(concurrency)
    ]
    all_results: List[RequestResult] = []
    for t in asyncio.as_completed(tasks):
        all_results.extend(await t)

    successes = [r.latency_s for r in all_results if r.ok]
    failures = [r.latency_s for r in all_results if not r.ok]
    total = len(all_results)
    ok_count = len(successes)
    fail_count = total - ok_count

    successes.sort()
    p50 = percentile(successes, 0.50)
    p90 = percentile(successes, 0.90)
    p95 = percentile(successes, 0.95)
    p99 = percentile(successes, 0.99)
    mx = max(successes) if successes else 0.0
    avg = statistics.mean(successes) if successes else 0.0

    print("=== Load Test Results ===")
    print(f"Total: {total}  Success: {ok_count}  Fail: {fail_count}")
    print(f"Latency (s) -> avg: {avg:.3f}  p50: {p50:.3f}  p90: {p90:.3f}  p95: {p95:.3f}  p99: {p99:.3f}  max: {mx:.3f}")
    sla_ok = (p95 <= timeout_s) and (mx <= timeout_s)
    print(f"SLA (<= {timeout_s:.3f}s) p95 & max: {'PASS' if sla_ok else 'FAIL'}")
    if log_path:
        write_log(log_path, all_results)
        print(f"Requests log: {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Async load test for /api/predict")
    ap.add_argument("--base_url", default="http://localhost:8000")
    ap.add_argument("--input", default=DEFAULT_CSV, help="CSV with column 'search_query'")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--requests-per-client", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=1.0, help="SLA threshold in seconds")
    ap.add_argument("--log_requests", default="", help="Optional CSV path to save all sent requests")
    args = ap.parse_args()

    global GLOBAL_QUERIES
    GLOBAL_QUERIES = load_queries_from_csv(args.input)
    if not GLOBAL_QUERIES:
        print(f"Warning: no queries loaded from {args.input}. Nothing to test.")
        return

    asyncio.run(
        run_load(
            base_url=args.base_url,
            concurrency=args.concurrency,
            requests_per_client=args.requests_per_client,
            timeout_s=args.timeout,
            log_path=(args.log_requests or None),
        )
    )


if __name__ == "__main__":
    main()
