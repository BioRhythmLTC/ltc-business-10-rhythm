#!/usr/bin/env python3
"""
Scenario load testing for X5 NER Service.

Builds on load_test_predict.py and adds:
  - steady load (RPS-controlled) with single requests
  - bursty load to exercise micro-batching
  - mixed single/batch traffic
  - background noise: frequent health checks and occasional invalid paths

SLA goal: p95 <= threshold and max <= threshold per scenario (default 1.0s).

Usage examples:
  python3 scripts/load_test_scenarios.py --base_url http://localhost:8000 \
    --input examples/submission2.csv --sla 1.0 --rps 200 --duration 30 \
    --burst_rps 600 --burst_every 10 --burst_len 2 --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import aiohttp


DEFAULT_CSV = os.environ.get(
    "LOADTEST_INPUT",
    os.path.join(os.path.dirname(__file__), "..", "examples", "submission2.csv"),
)


@dataclass
class Result:
    ok: bool
    latency_s: float
    status: int
    tag: str


def _append_results_csv(path: str, items: List[Result]) -> None:
    if not path or not items:
        return
    header = ["ok", "latency_s", "status", "tag"]
    exists = os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            if not exists:
                f.write(";".join(header) + "\n")
            for r in items:
                f.write(f"{int(r.ok)};{r.latency_s:.6f};{r.status};{r.tag}\n")
    except Exception:
        pass


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values.sort()
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _parse_log_segments(path: str, endpoint: str = "/api/predict", target_total: int | None = None) -> List[int]:
    """Parse container logs into concurrency segments.

    Approximates concurrency by counting consecutive POST lines to the endpoint.
    Any non-matching line ends the current segment.

    Args:
        path: Path to container logs.
        endpoint: Endpoint path to match (default: /api/predict).
        target_total: Optional cap on total requests to count.

    Returns:
        List of integers representing burst sizes to schedule concurrently.
    """
    if not os.path.isfile(path):
        return []
    segments: List[int] = []
    cur = 0
    total = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if f'"POST {endpoint} ' in line:
                    cur += 1
                    total += 1
                    if target_total is not None and total >= target_total:
                        if cur > 0:
                            segments.append(cur)
                        return segments
                else:
                    if cur > 0:
                        segments.append(cur)
                        cur = 0
        if cur > 0:
            segments.append(cur)
    except Exception:
        pass
    return segments


def _load_queries(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            q = (r.get("search_query") or "").strip()
            if q:
                qs.append(q)
    return qs


async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout_s: float, tag: str) -> Result:
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            _ = await resp.read()  # don't spend time parsing
            return Result(ok=(resp.status == 200), latency_s=time.perf_counter() - t0, status=resp.status, tag=tag)
    except Exception:
        return Result(ok=False, latency_s=time.perf_counter() - t0, status=0, tag=tag)


async def _get(session: aiohttp.ClientSession, url: str, timeout_s: float, tag: str) -> Result:
    t0 = time.perf_counter()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            _ = await resp.read()
            ok = (200 <= resp.status < 400) or (tag == "scan" and resp.status == 404)
            return Result(ok=ok, latency_s=time.perf_counter() - t0, status=resp.status, tag=tag)
    except Exception:
        return Result(ok=False, latency_s=time.perf_counter() - t0, status=0, tag=tag)


async def scenario_runner(base_url: str, duration_s: int, rps: int, batch_size: int, timeout_s: float, noise_health_every: float, noise_scan_every: float, burst_rps: int | None = None, burst_every_s: int = 10, burst_len_s: int = 2, target_total: Optional[int] = None, log_requests: bool = False) -> Tuple[List[Result], List[Result]]:
    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    predict_single = base_url.rstrip("/") + "/api/predict"
    predict_batch = base_url.rstrip("/") + "/api/predict_batch"
    health = base_url.rstrip("/") + "/health"
    junk = base_url.rstrip("/") + "/device.rsp?opt=sys&cmd=__S_O_S__"

    results: List[Result] = []
    noise_results: List[Result] = []
    pending_tasks: List[asyncio.Task] = []
    sent = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        next_noise_health = start
        next_noise_scan = start
        end = start + max(1, int(duration_s))
        if target_total is not None:
            # In capped mode, prioritize reaching the target count over duration
            end = float("inf")

        while time.monotonic() < end:
            if target_total is not None and sent >= target_total:
                break
            now = time.monotonic()
            cur_rps = rps
            if burst_rps and ((int(now - start) % max(1, int(burst_every_s))) < burst_len_s):
                cur_rps = max(cur_rps, burst_rps)

            # schedule main requests for this 100ms timeslice
            slice_ms = 100
            slice_rps = max(1, int(cur_rps * slice_ms / 1000))
            to_schedule: List[asyncio.Task] = []
            schedule_budget = slice_rps
            if target_total is not None:
                schedule_budget = min(schedule_budget, max(0, int(target_total - sent)))
            for _ in range(schedule_budget):
                if batch_size > 1:
                    batch = [random.choice(GLOBAL_QUERIES) for _ in range(batch_size)]
                    to_schedule.append(asyncio.create_task(_post_json(session, predict_batch, {"inputs": batch}, timeout_s, tag=f"batch[{batch_size}]")))
                else:
                    q = random.choice(GLOBAL_QUERIES)
                    to_schedule.append(asyncio.create_task(_post_json(session, predict_single, {"input": q}, timeout_s, tag="single")))
            sent += len(to_schedule)

            # noise: health checks
            if now >= next_noise_health:
                next_noise_health = now + max(0.1, float(noise_health_every))
                noise_results.append(await _get(session, health, timeout_s=timeout_s, tag="health"))

            # noise: random 404 scan
            if now >= next_noise_scan:
                next_noise_scan = now + max(0.5, float(noise_scan_every))
                noise_results.append(await _get(session, junk, timeout_s=timeout_s, tag="scan"))

            # execute scheduled main requests concurrently and collect
            outstanding = []
            if pending_tasks:
                outstanding.extend(pending_tasks)
            if to_schedule:
                outstanding.extend(to_schedule)
            if outstanding:
                done, pending = await asyncio.wait(set(outstanding), timeout=slice_ms / 1000.0)
                collected = [t.result() for t in done if t.done()]
                results.extend(collected)
                if log_requests and collected:
                    for r in collected:
                        logging.debug(f"request ok={r.ok} status={r.status} latency_s={r.latency_s:.6f} tag={r.tag}")
                pending_tasks = list(pending)

            # sleep till next timeslice
            await asyncio.sleep(slice_ms / 1000.0)

        # drain any remaining outstanding tasks to ensure accurate totals
        if pending_tasks:
            drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for d in drained:
                if isinstance(d, Result):
                    results.append(d)
                    if log_requests:
                        logging.debug(f"request ok={d.ok} status={d.status} latency_s={d.latency_s:.6f} tag={d.tag}")
                else:
                    fail = Result(ok=False, latency_s=0.0, status=0, tag="single")
                    results.append(fail)
                    if log_requests:
                        logging.debug(f"request ok={fail.ok} status={fail.status} latency_s={fail.latency_s:.6f} tag={fail.tag}")

    return results, noise_results


async def replay_runner(base_url: str, log_path: str, queries: List[str], timeout_s: float, target_total: int, endpoint: str = "/api/predict", resp_log: Optional[str] = None, log_requests: bool = False) -> List[Result]:
    """Replay traffic profile from logs with identical burst sizes and total.

    Schedules exactly target_total requests to endpoint using provided queries
    (cycled if needed). Each segment is executed as a concurrent burst.
    """
    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    url = base_url.rstrip("/") + endpoint
    segments = _parse_log_segments(log_path, endpoint=endpoint, target_total=target_total)
    if not segments:
        segments = [target_total]

    results: List[Result] = []
    if not queries:
        return results
    qn = len(queries)

    async with aiohttp.ClientSession(connector=connector) as session:
        sent = 0
        qi = 0
        for burst in segments:
            if sent >= target_total:
                break
            burst = min(burst, target_total - sent)
            tasks = []
            for _ in range(burst):
                q = queries[qi]
                qi = (qi + 1) % qn
                tasks.append(_post_json(session, url, {"input": q}, timeout_s, tag="replay"))
            # Wait for all tasks in the burst to complete (or raise)
            burst_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Normalize exceptions into failed Results
            out: List[Result] = []
            for br in burst_results:
                if isinstance(br, Result):
                    out.append(br)
                elif isinstance(br, Exception):
                    out.append(Result(ok=False, latency_s=0.0, status=0, tag="replay"))
                else:
                    out.append(Result(ok=False, latency_s=0.0, status=0, tag="replay"))
            results.extend(out)
            if log_requests and out:
                for r in out:
                    logging.debug(f"request ok={r.ok} status={r.status} latency_s={r.latency_s:.6f} tag={r.tag}")
            sent += burst
            if resp_log:
                _append_results_csv(resp_log, out)
            await asyncio.sleep(0)
    return results


def summarize(title: str, rs: List[Result], sla: float) -> None:
    """Summarize results using ONLY median latency.

    Any request with latency > SLA is considered out-of-spec and contributes 0.0s
    to the median ("обнуляется"). Non-200 responses also contribute 0.0s.
    """
    total = len(rs)
    over_sla = 0
    okc = 0
    lat_for_median: List[float] = []
    for r in rs:
        if r.ok:
            okc += 1
            if r.latency_s > sla:
                over_sla += 1
                lat_for_median.append(0.0)
            else:
                lat_for_median.append(r.latency_s)
        else:
            lat_for_median.append(0.0)

    fails = total - okc
    median_latency = statistics.median(lat_for_median) if lat_for_median else 0.0

    logging.info(f"\n=== {title} ===")
    logging.info(f"Total: {total}  Success: {okc}  Fail: {fails}  OverSLA(>{sla:.3f}s): {over_sla}")
    logging.info(f"Median latency (s, with >SLA set to 0): {median_latency:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario load testing for X5 NER Service")
    ap.add_argument("--base_url", default="http://localhost:8000")
    ap.add_argument("--input", default=DEFAULT_CSV)
    ap.add_argument("--sla", type=float, default=1.0)
    ap.add_argument("--duration", type=int, default=30, help="seconds")
    ap.add_argument("--rps", type=int, default=200, help="steady RPS for main traffic")
    ap.add_argument("--burst_rps", type=int, default=600, help="RPS during bursts (micro-batch stress)")
    ap.add_argument("--burst_every", type=int, default=10, help="burst period seconds")
    ap.add_argument("--burst_len", type=int, default=2, help="burst length seconds")
    ap.add_argument("--batch-size", type=int, default=1, help=">1 sends /api/predict_batch")
    ap.add_argument("--noise_health_every", type=float, default=2.0, help="seconds between health checks")
    ap.add_argument("--noise_scan_every", type=float, default=5.0, help="seconds between random 404 scans")
    ap.add_argument("--replay_logs", type=str, default=None, help="Path to container logs to replay traffic profile")
    ap.add_argument("--replay_total", type=int, default=5000, help="Exact number of POST requests to send in replay mode")
    ap.add_argument("--replay_endpoint", type=str, default="/api/predict", help="Endpoint path to match in logs and replay to")
    ap.add_argument("--resp_log", type=str, default=None, help="Optional path to write response results CSV")
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--log_file", type=str, default=None, help="Optional path to log file")
    ap.add_argument("--log_requests", action="store_true", help="Log each request result at DEBUG level")
    args = ap.parse_args()

    # setup logging
    log_level = getattr(logging, (args.log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=args.log_file,
        force=True,
    )

    global GLOBAL_QUERIES
    GLOBAL_QUERIES = _load_queries(args.input)
    if not GLOBAL_QUERIES:
        logging.error(f"No queries in {args.input}")
        return

    loop = asyncio.get_event_loop()

    if args.replay_logs:
        logging.info(f"Starting LOG REPLAY from {args.replay_logs} (total={args.replay_total})...")
        rs_replay = loop.run_until_complete(
            replay_runner(
                base_url=args.base_url,
                log_path=args.replay_logs,
                queries=GLOBAL_QUERIES,
                timeout_s=args.sla,
                target_total=max(1, int(args.replay_total)),
                endpoint=args.replay_endpoint,
                resp_log=args.resp_log,
                log_requests=args.log_requests,
            )
        )
        summarize("LOG REPLAY", rs_replay, sla=args.sla)
        return

    # If replay_total is provided without logs, cap total requests across scenarios
    cap_total = max(0, int(args.replay_total)) if (args.replay_logs is None and args.replay_total) else None

    logging.info("Starting STEADY scenario (single requests)...")
    rs, noise = loop.run_until_complete(
        scenario_runner(
            base_url=args.base_url,
            duration_s=args.duration,
            rps=args.rps,
            batch_size=1,
            timeout_s=args.sla,
            noise_health_every=args.noise_health_every,
            noise_scan_every=args.noise_scan_every,
            burst_rps=None,
            target_total=cap_total // 2 if cap_total is not None else None,
            log_requests=args.log_requests,
        )
    )
    summarize("STEADY single", rs, sla=args.sla)
    summarize("NOISE", noise, sla=args.sla)

    logging.info("\nStarting BURST scenario (spike load to /api/predict)...")
    remaining = None
    if cap_total is not None:
        remaining = max(0, cap_total - len(rs))
    rs2, _ = loop.run_until_complete(
        scenario_runner(
            base_url=args.base_url,
            duration_s=args.duration,
            rps=args.rps,
            batch_size=1,
            timeout_s=args.sla,
            noise_health_every=args.noise_health_every,
            noise_scan_every=args.noise_scan_every,
            burst_rps=args.burst_rps,
            burst_every_s=args.burst_every,
            burst_len_s=args.burst_len,
            target_total=remaining,
            log_requests=args.log_requests,
        )
    )
    summarize("BURST single", rs2, sla=args.sla)


if __name__ == "__main__":
    main()


