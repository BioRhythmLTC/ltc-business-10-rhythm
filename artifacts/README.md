# Artifacts directory

This directory is intentionally kept out of Git history. Place runtime model files here (tokenizer, config, weights). Typical contents:

- config.json, tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
- model.safetensors (or framework-specific weights)
- label_mapping.json or model id2label in config.json

How to provide artifacts:

- Download from a release or a model registry 
- Set env var ARTIFACTS_DIR to this path before starting the service

Do not commit model weights to Git. For reproducibility, document the artifact source and version in the project README or release notes.
