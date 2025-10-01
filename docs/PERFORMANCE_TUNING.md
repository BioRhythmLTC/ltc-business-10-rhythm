## Performance Tuning

This service includes several controls for throughput and latency.

### Inference concurrency
- `PREDICT_MAX_CONCURRENCY` limits concurrent CPU-bound predictions per worker process (default `2`).
- Used by a semaphore around `run_in_executor` offloading.

### Micro-batching
Environment variables:
- `MICRO_BATCH_ENABLED` (default `true`)
- `MICRO_BATCH_MAX_SIZE` (default `32`)
- `MICRO_BATCH_MAX_WAIT_MS` (default `3`) — soft wait to gather a micro-batch
- `MICRO_BATCH_HARD_TIMEOUT_MS` (default `500`) — per-item timeout when queued
- `MICRO_BATCH_QUEUE_MAXSIZE` (default `0` unlimited)

When enabled, concurrent single requests are grouped to reduce tokenizer/model overhead and maximize cache hit opportunities.

### Caching
The in-memory TTL cache avoids recomputation for repeated inputs:
- `CACHE_ENABLED` (default `true`)
- `CACHE_MAX_SIZE` (default `1000`)
- `CACHE_TTL_SECONDS` (default `3600`)

Endpoints:
- `GET /cache/stats`, `GET /cache/info`, `DELETE /cache/clear`.

### Threading and CPU settings
Tune per environment to avoid oversubscription:
- `TOKENIZERS_PARALLELISM=false`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- `TORCH_NUM_THREADS=1`, `TORCH_NUM_INTEROP_THREADS=1`

### Input handling
- `MAX_INPUT_CHARS` truncates inputs to cap tokenization cost (default `512`).

### Fail-safe behavior
- `PREDICT_FAIL_SAFE=true` returns empty results on unexpected errors (HTTP 200), keeping latency predictable during transient issues.

### Device selection
- `X5_FORCE_DEVICE` or `FORCE_DEVICE` can force `cpu|cuda|mps` (auto-detect otherwise).

### Gunicorn workers
- Start with `workers=(cpu*2)+1` and measure. Increase only if CPU bound and latency is acceptable.


