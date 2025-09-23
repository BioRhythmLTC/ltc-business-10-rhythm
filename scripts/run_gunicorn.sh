#!/usr/bin/env bash
set -euo pipefail

# Sensible defaults; override via env vars
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${GUNICORN_WORKERS:=}"  # if empty, conf uses (cpu*2)+1

# Choose python: prefer provided PY_BIN, else python3 from current env
PY_BIN="${PY_BIN:-python3}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

bind="${HOST}:${PORT}"

if [[ -n "${GUNICORN_WORKERS}" ]]; then
  exec "${PY_BIN}" -m gunicorn service.main:app \
    -k uvicorn.workers.UvicornWorker \
    -c gunicorn_conf.py \
    --bind "${bind}" \
    --workers "${GUNICORN_WORKERS}"
else
  exec "${PY_BIN}" -m gunicorn service.main:app \
    -k uvicorn.workers.UvicornWorker \
    -c gunicorn_conf.py \
    --bind "${bind}"
fi
