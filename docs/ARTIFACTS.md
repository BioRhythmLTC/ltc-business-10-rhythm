# Model Artifacts (Packaging Guide)

This project does not commit model weights. To run the service, you need HuggingFace-compatible artifacts placed in `ARTIFACTS_DIR` (default `./artifacts`).

## Expected contents
- config.json
- tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
- model.safetensors (or framework-specific weights)
- (optional) label_mapping.json
- (optional) versions.json metadata (see template below)

## versions.json (template)
Use this file to record model metadata for traceability. Do not commit real values; attach with release assets.

See `docs/versions.example.json` for a template.

## Preparing a Release (assets only)
- Do not publish now; this is a future workflow.
- Zip the artifacts directory (without large unrelated files):
  - `artifacts-vX.Y.Z.zip` containing only files listed above
- Include `versions.json` inside the zip with:
  - model_name, model_sha, training_data_version, tokenizer_sha, created_at, notes
- Upload the zip as a GitHub Release asset (when ready).

## Using artifacts locally
- Unzip the downloaded archive next to the repo and point the service to it:
  - `ARTIFACTS_DIR=./artifacts uvicorn service.main:app --host 0.0.0.0 --port 8000`
- Or set `ARTIFACTS_DIR` to an absolute path.

## Downloading from GitHub Releases
See `docs/ARTIFACTS_DOWNLOAD.md` for step-by-step instructions to download, verify, and place artifacts (and optional `examples/submission2.csv`).
