# Downloading Artifacts from GitHub Releases

This project distributes model artifacts via GitHub Releases. Do not commit model weights to Git.

## What you need
- An artifacts archive (e.g., `artifacts-v1.0.0.zip`)
- A checksum file `artifacts-v1.0.0.zip.sha256` (optional but recommended)
- Optional test CSV `submission2.csv` for load testing

## Where to place artifacts locally
- Unzip the archive into the project root as `./artifacts/` (default used by the service), or
- Set an absolute path via env var `ARTIFACTS_DIR`.

Examples:
```bash
# Option A: place next to the repo and use default path
unzip artifacts-v1.0.0.zip -d artifacts

# Option B: place anywhere and point the service to it
export ARTIFACTS_DIR=/absolute/path/to/artifacts
```

## How to download via CLI
Replace OWNER and REPO with your GitHub org/user and repository.

```bash
TAG=v1.0.0
OWNER=<owner>
REPO=<repo>

# Download artifacts archive and checksum
curl -L -o artifacts-$TAG.zip \
  "https://github.com/$OWNER/$REPO/releases/download/$TAG/artifacts-$TAG.zip"

curl -L -o artifacts-$TAG.zip.sha256 \
  "https://github.com/$OWNER/$REPO/releases/download/$TAG/artifacts-$TAG.zip.sha256"

# (optional) verify checksum
shasum -a 256 -c artifacts-$TAG.zip.sha256

# Unpack
unzip -o artifacts-$TAG.zip -d artifacts
```

## Load test sample (submission2.csv)
The load test script expects a semicolon-delimited CSV with the column `search_query`.

- Download `submission2.csv` (if published as a separate asset) to `examples/submission2.csv`:
```bash
curl -L -o examples/submission2.csv \
  "https://github.com/$OWNER/$REPO/releases/download/$TAG/submission2.csv"
```

- Run the load test (example: 5000 inputs in batches of 10):
```bash
/abs/path/to/x5_ner_env/bin/python scripts/load_test_predict.py \
  --base_url http://127.0.0.1:8000 \
  --input examples/submission2.csv \
  --concurrency 100 \
  --requests-per-client 5 \
  --batch-size 10 \
  --timeout 1.0
```

## Using with Docker Compose
The compose file maps `./artifacts` from your host to `/app/artifacts` in the container and sets `ARTIFACTS_DIR=/app/artifacts/rubert-tiny/latest`.

Unzip so that this path exists on the host:

```bash
# Ensure the repo root contains ./artifacts/rubert-tiny/latest
unzip -o rubert-tiny-latest.zip -d .

# Resulting tree (important part):
# ./artifacts/rubert-tiny/latest/{config.json, tokenizer.json, ..., model.safetensors}

docker compose up -d --build
```

## Service startup with artifacts
```bash
# If artifacts are in ./artifacts
uvicorn service.main:app --host 0.0.0.0 --port 8000

# Or explicitly set the path
ARTIFACTS_DIR=./artifacts uvicorn service.main:app --host 0.0.0.0 --port 8000
```
