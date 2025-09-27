"""Configuration and constants for X5 NER Service."""

import os

# Paths
ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts")
)
ARTIFACTS_DIR = os.path.abspath(ARTIFACTS_DIR)

# Device configuration
SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}

# Entity types
ENTITY_TYPES = {"TYPE", "BRAND", "VOLUME", "PERCENT"}
BIO_PREFIXES = {"B", "I"}

# Cache configuration
CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))  # 1 hour default
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
