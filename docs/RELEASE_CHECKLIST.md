# Release Checklist (Artifacts as Assets)

This checklist prepares artifacts for a future GitHub Release. Do not publish now.

1) Bump version (SemVer) and update CHANGELOG
2) Export artifacts
   - Ensure `config.json`, tokenizer files, `model.safetensors` exist
   - Generate/refresh `versions.json` (use `docs/versions.example.json` as template)
3) Package
   - `zip -r artifacts-vX.Y.Z.zip artifacts/`
   - Verify archive size and contents
4) Smoke test locally
   - Unzip and set `ARTIFACTS_DIR` to the unpacked folder
   - Start service and hit `/health` and `/api/predict`
5) Create Release (when ready)
   - Tag `vX.Y.Z`
   - Upload `artifacts-vX.Y.Z.zip` as asset
   - Attach `versions.json` and brief release notes
6) Post-release
   - Update README with the new version reference if needed
