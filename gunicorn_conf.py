import multiprocessing
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


_default_workers = (multiprocessing.cpu_count() * 2) + 1

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
workers = _int_env("GUNICORN_WORKERS", _default_workers)
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")

timeout = _int_env("GUNICORN_TIMEOUT", 30)
graceful_timeout = _int_env("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _int_env("GUNICORN_KEEPALIVE", 5)

# Tune memory/perf stability under load
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 1000)
max_requests_jitter = _int_env("GUNICORN_MAX_REQUESTS_JITTER", 100)

# Logging
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Preload can reduce worker startup latency; keep off by default to avoid GPU/MPS quirks
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "false").lower() in {"1", "true", "yes"}

# Worker env
raw_env = [
    f"TOKENIZERS_PARALLELISM={os.getenv('TOKENIZERS_PARALLELISM', 'false')}",
    f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', '1')}",
    f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS', '1')}",
]
