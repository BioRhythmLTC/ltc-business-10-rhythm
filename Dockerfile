# syntax=docker/dockerfile:1.7-labs

# --- Base builder (deps layer) ---
FROM python:3.10-slim AS deps
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements-prod.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install -r requirements-prod.txt \
    && apt-get purge -y build-essential && rm -rf /var/lib/apt/lists/*

# --- Runtime image ---
FROM python:3.10-slim AS runtime
WORKDIR /app

# Minimal runtime env
ENV TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install runtime OS deps needed by PyTorch/Tokenizers and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      curl \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early so COPY --chown works
RUN useradd -u 10001 -m appuser

# Copy installed site-packages from deps (requires root)
COPY --from=deps /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy project files
COPY --chown=appuser:appuser service/ service/
COPY --chown=appuser:appuser gunicorn_conf.py ./
COPY --chown=appuser:appuser scripts/run_gunicorn.sh ./run_gunicorn.sh
RUN chmod +x ./run_gunicorn.sh

# Switch to non-root user after files are in place
USER appuser

# Artifacts are mounted or copied at runtime to /app/artifacts by default
ENV ARTIFACTS_DIR=/app/artifacts

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python - <<'PY' || exit 1


EXPOSE 8000
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["./run_gunicorn.sh"]
