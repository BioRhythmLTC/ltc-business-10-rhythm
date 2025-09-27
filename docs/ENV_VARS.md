# Environment Variables

Copy these into a local `.env` (do not commit real `.env`). Values below are safe defaults for local runs; tune for production.

## Model artifacts
- ARTIFACTS_DIR: `./artifacts`

## Server
- HOST: `0.0.0.0`
- PORT: `8000`

## Gunicorn (see `gunicorn_conf.py`)
- GUNICORN_WORKERS: `` (empty to use default `(cpu*2)+1`)
- GUNICORN_TIMEOUT: `30`
- GUNICORN_GRACEFUL_TIMEOUT: `30`
- GUNICORN_KEEPALIVE: `5`
- GUNICORN_MAX_REQUESTS: `1000`
- GUNICORN_MAX_REQUESTS_JITTER: `100`
- GUNICORN_LOGLEVEL: `info`

## Performance knobs
- TOKENIZERS_PARALLELISM: `false`
- OMP_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`

## Warmup behavior
- DISABLE_WARMUP: `false`
- ALLOW_MPS_WARMUP: `false`

## Cache configuration
- CACHE_ENABLED: `true`
- CACHE_MAX_SIZE: `1000`
- CACHE_TTL_SECONDS: `3600`

## Third-party integrations (configure via CI/CD secrets, not committed)
- SENTRY_DSN: ``
