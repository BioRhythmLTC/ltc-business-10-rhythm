## API Reference

Base URL: by default `http://localhost:8000`

### Health
- Method: GET
- Path: `/health`
- Response: `HealthResponse`

Example:
```bash
curl -s http://localhost:8000/health
```

Response shape:
```json
{
  "status": "ok",
  "device": "cpu|cuda|mps",
  "artifacts": "/abs/path/to/artifacts",
  "model_loaded": true,
  "cache_stats": {"enabled": true, "max_size": 1000, "current_size": 10, "ttl_seconds": 3600, "hit_count": 5, "miss_count": 2, "hit_rate_percent": 71.43, "total_requests": 7, "memory_usage_mb": 0.03}
}
```

### Predict (single)
- Method: POST
- Path: `/api/predict`
- Request body: `PredictRequest`
- Response: `List[EntitySpan]`

Schema:
```json
{
  "input": "строка для анализа"
}
```

Response item schema:
```json
{
  "start_index": 0,
  "end_index": 5,
  "entity": "B-BRAND|I-BRAND|B-TYPE|I-TYPE|B-VOLUME|I-VOLUME|B-PERCENT|I-PERCENT"
}
```

Notes:
- Empty or whitespace-only input returns `[]`.
- Inputs longer than `MAX_INPUT_CHARS` are truncated.
- On internal error, if `PREDICT_FAIL_SAFE=true` the response is `[]` with status 200; otherwise 500.

Example:
```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"кока кола 0.5л"}'
```

### Predict (batch)
- Method: POST
- Path: `/api/predict_batch`
- Request body: `PredictBatchRequest`
- Response: `List[List[EntitySpan]]`

Schema:
```json
{
  "inputs": ["строка 1", "строка 2"]
}
```

Notes:
- Preserves order; identical strings are de-duplicated internally.
- Uses cache per unique normalized string.
- On internal error with `PREDICT_FAIL_SAFE=true`, returns list of empty lists of the same length.

### Warmup
- Method: POST
- Path: `/warmup`
- Response: `{ "status": "warmed_up", "device": "cpu|cuda|mps" }`

### Cache endpoints
- GET `/cache/stats` → `CacheStatsResponse`
- GET `/cache/info` → detailed cache config and env
- DELETE `/cache/clear` → `{ "status": "cache_cleared" }`

### Root and favicon
- GET `/` → 404 by default (production hardening). If `ROOT_PUBLIC=true` returns basic service info.
- GET `/favicon.ico` → 204.

### Models (Pydantic)

`PredictRequest`
```json
{"input": "text"}
```

`PredictBatchRequest`
```json
{"inputs": ["text1", "text2"]}
```

`EntitySpan`
```json
{"start_index": 0, "end_index": 5, "entity": "B-TYPE"}
```

`HealthResponse`
```json
{"status":"ok","device":"cpu","artifacts":"/path","model_loaded":true,"cache_stats":{}}
```

`CacheStatsResponse`
```json
{"enabled":true,"max_size":1000,"current_size":10,"ttl_seconds":3600,"hit_count":5,"miss_count":2,"hit_rate_percent":71.43,"total_requests":7,"memory_usage_mb":0.02}
```

### Status codes
- 200: Success
- 204: No content (favicon)
- 404: Not Found (root in production)
- 422: Validation error (pydantic schema validation)
- 500: Internal error (when `PREDICT_FAIL_SAFE=false`)


