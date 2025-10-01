## Deployment Guide

This service is a FastAPI app packaged with Gunicorn for production and Uvicorn for local runs.

### Prerequisites: download model artifacts
- Download the model archive and checksum from the artifacts drive (`https://drive.google.com/drive/folders/13WxzEEXwLt8el3-_sm_XkO_0YqUde5EA`), then unzip into the repository root so that the model files appear under `./artifacts/<alias>/<run_id>`.

```bash
# Example (preferred): ruBERT base cased
# Files: rubert_base_cased_20250930_165530.zip and rubert_base_cased_20250930_165530.zip.sha256

# (optional) verify checksum
shasum -a 256 -c rubert_base_cased_20250930_165530.zip.sha256

# unzip at repo root so the following path exists:
# ./artifacts/rubert-base-cased/20250930-165530/{config.json, tokenizer.json, model.safetensors, ...}
unzip -o rubert_base_cased_20250930_165530.zip -d .

# Alternative (smaller model): rubert-tiny
shasum -a 256 -c rubert-tiny-latest.zip.sha256 || true
unzip -o rubert-tiny-latest.zip -d .
```

If you choose a different alias or run id, set `ARTIFACTS_DIR` accordingly or update the compose environment to point to the actual folder containing `config.json`, `tokenizer.json`, and `model.safetensors`.

### Local (Uvicorn)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt
ARTIFACTS_DIR=./artifacts/rubert-base-cased/20250930-165530 \
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

### Production (Gunicorn)
Use the provided `gunicorn_conf.py` and `scripts/run_gunicorn.sh`.

Environment knobs (see `docs/ENV_VARS.md`):
- `GUNICORN_WORKERS` (default `(cpu*2)+1`)
- `GUNICORN_TIMEOUT`, `GUNICORN_GRACEFUL_TIMEOUT`, `GUNICORN_KEEPALIVE`
- `GUNICORN_MAX_REQUESTS`, `GUNICORN_MAX_REQUESTS_JITTER`
- `GUNICORN_LOGLEVEL`
- `GUNICORN_PRELOAD_APP` (defaults false)

Run:
```bash
HOST=0.0.0.0 PORT=8000 \
GUNICORN_WORKERS=8 \
./scripts/run_gunicorn.sh
```

### Docker
Build and run:
```bash
docker build -t x5-ner:local .
docker run --rm -p 8000:8000 \
  -e ARTIFACTS_DIR=/app/artifacts/rubert-base-cased/20250930-165530 \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD/artifacts:/app/artifacts:ro" x5-ner:local
```

### Docker Compose
`compose.yaml` maps `./artifacts` and sets `ARTIFACTS_DIR=/app/artifacts/rubert-base-cased/20250930-165530` along with recommended env defaults.

```bash
docker compose up --build
```

Healthcheck uses `/health`.

### Ports and health
- Port: `8000` (set via `PORT`)
- Health: `GET /health` should return `{ "status": "ok", ... }`

### Observability
- Standard Gunicorn access/error logs to stdout/stderr
- Add Sentry or similar in the future via `SENTRY_DSN`


