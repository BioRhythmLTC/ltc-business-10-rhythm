# X5 NER Service

FastAPI service for Named Entity Recognition over Russian search queries. Loads a HuggingFace-compatible tokenizer/model from ARTIFACTS_DIR and exposes two endpoints: predict (single) and predict_batch.

## API

- GET /health
  - Returns JSON with status, selected device, and artifacts path.
  - Example: `{ "status": "ok", "device": "cpu|cuda|mps", "artifacts": "/abs/path" }`
- POST /api/predict
  - Request: `{ "input": "текст запроса" }`
  - Response: list of spans, each with:
    - `start_index` (int): start char index of a word span
    - `end_index` (int): end char index (exclusive)
    - `entity` (str): BIO-tag like `B-TYPE`, `I-BRAND`, ...
- POST /api/predict_batch
  - Request: `{ "inputs": ["строка 1", "строка 2", ...] }`
  - Response: array of arrays in the same shape as inputs

OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc` once the service runs.

## Quickstart

### Local (CPU)

1. Create and activate a virtualenv (Python 3.10+ recommended)
2. Install dependencies:
   - Production (minimal runtime): `pip install -r requirements-prod.txt`
   - Development (includes notebooks, tests, linters): `pip install -r requirements-dev.txt`
3. Put model artifacts into `artifacts/` (see below) or set `ARTIFACTS_DIR` to your path.
4. Configure environment variables (see `docs/ENV_VARS.md`). Optionally create a local `.env` (do not commit).
5. Run development server:
   - `uvicorn service.main:app --host 0.0.0.0 --port 8000`
6. Or production server (Gunicorn):
   - `bash run_gunicorn.sh` (or `python -m gunicorn service.main:app -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py --bind 0.0.0.0:8000`)

Environment variables (summary):

- ARTIFACTS_DIR: path to model artifacts (default: ./artifacts)
- TOKENIZERS_PARALLELISM=false (recommended)
- OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 (avoid CPU over-subscription)
- GUNICORN_* (see gunicorn_conf.py): workers, timeouts, logging
- DISABLE_WARMUP=1 to skip model warmup on startup
- ALLOW_MPS_WARMUP=1 to allow warmup on macOS MPS

### Health, Liveness, Readiness

- Liveness probe: `GET /health` returns 200 when the process is alive
- Readiness probe: also `GET /health`; for stricter readiness, ensure artifacts are present and consider a warmup (default warmup can be disabled via `DISABLE_WARMUP=1`)
- Kubernetes example:
  - livenessProbe: httpGet path `/health`, port `8000`
  - readinessProbe: httpGet path `/health`, port `8000`

### Docker

A multi-stage `Dockerfile` and `.dockerignore` are included.

- Build:
  - `docker build -t x5-ner:local .`
- Run (mount artifacts):
  - `docker run --rm -p 8000:8000 -e ARTIFACTS_DIR=/app/artifacts -v "$PWD/artifacts:/app/artifacts:ro" x5-ner:local`
- Swagger: `http://localhost:8000/docs`
- Compose (optional):
  - `docker compose up --build`
  - See `compose.yaml` for envs, volume, healthcheck.

## Model artifacts

Place HuggingFace-compatible files in `ARTIFACTS_DIR` (default `./artifacts`):
- config.json, tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
- model.safetensors (or framework-specific weights)
- (optional) label_mapping.json or id2label in config.json

See `docs/ARTIFACTS.md` for packaging guidance and a future GitHub Releases flow (assets only). For now, download or prepare artifacts locally and point `ARTIFACTS_DIR` to their folder.

## Gunicorn guidance

- Workers (`GUNICORN_WORKERS`): start with `(CPU cores * 2) + 1` for CPU-only. For GPU-bound inference, fewer workers may be better (e.g., 1–2 per GPU) to avoid memory contention.
- Timeouts: `GUNICORN_TIMEOUT=30`, `GUNICORN_GRACEFUL_TIMEOUT=30` are sane defaults; adjust for your SLA.
- Max requests: `GUNICORN_MAX_REQUESTS=1000` with jitter helps memory stability for long runs.
- Logging: defaults to stdout/stderr. In containers, collect via the platform’s log driver (e.g., Docker/Pod logs). Set `GUNICORN_LOGLEVEL=info|warning|debug` as needed.

## Load testing

`scripts/load_test_predict.py` — async нагрузочный тест `/api/predict`.

- CSV формат: `;`-разделитель, колонка `search_query` (см. `examples/sample_input.csv`).
- Основные флаги: `--base_url`, `--input`, `--concurrency`, `--requests-per-client`, `--timeout`, `--log_requests`.
- Пример:
  - `python scripts/load_test_predict.py --base_url http://localhost:8000 --input examples/sample_input.csv --concurrency 100 --requests-per-client 50 --timeout 1.0 --log_requests eval_out/load_requests.csv`
- Где лог: если указать `--log_requests PATH`, сохранит CSV с отправленными запросами в `PATH` и выведет путь в консоль.

## Offline evaluation

`scripts/evaluate_service.py` — оффлайн-оценка предсказаний сервиса.

- Вход: CSV `id;search_query;annotation`, где `annotation` — Python-список кортежей `[(start,end,'B-TYPE'), ...]`.
- Запуск:
  - `python scripts/evaluate_service.py --input examples/annotated_sample.csv --output_dir eval_out --base_url http://localhost:8000 --batch_size 32`
- Выходы (в `--output_dir`, по умолчанию используйте папку наподобие `eval_out/`):
  - `eval_results.csv` — покомпонентные результаты
  - `eval_report.html` — интерактивный отчет (открыть в браузере)
  - `eval_stats.json` — агрегированные метрики
- Git: каталоги `eval_out*/` исключены из репозитория. В CI можно сохранять их как артефакты job’ов.

## Development

- Keep notebooks in `notebooks/` and generated outputs out of Git.
- Use pre-commit for consistent style:
  - `pip install pre-commit && pre-commit install`
  - Hooks: black, isort, flake8, trailing-whitespace, end-of-file-fixer

## License

Add MIT or Apache-2.0 license file to the repo root.
