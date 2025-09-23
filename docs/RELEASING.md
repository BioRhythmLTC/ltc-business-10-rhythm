# Releasing

## Versioning
- Semantic Versioning: `vMAJOR.MINOR.PATCH` (e.g., `v1.2.3`)
- Bump version when making incompatible/public changes, features, or patches

## Tags and Changelog
- Use conventional commits (`feat:`, `fix:`, `chore:`, etc.)
- Update `CHANGELOG.md` for notable changes
- Create an annotated tag: `git tag -a vX.Y.Z -m "release vX.Y.Z"` and push tags

## Artifacts (models handled separately)
- Package artifacts as described in `docs/ARTIFACTS.md`
- Attach `artifacts-vX.Y.Z.zip` and `versions.json` as release assets

## CI
- On `v*.*.*` tags, run lint/type/tests and (optionally later) build/push Docker image
- This repo is prepared but does not build Docker on CI yet per current policy
