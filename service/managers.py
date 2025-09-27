"""Business logic managers for model and cache operations."""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from cachetools import TTLCache
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .config import (
    ARTIFACTS_DIR,
    CACHE_ENABLED,
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    SUPPORTED_DEVICES,
)
from .utils import (
    _extract_spans_from_bio,
    _spans_to_api_spans,
    _token_tags_to_char_bio,
    preprocess_text_with_mapping,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference with proper error handling."""

    def __init__(self, artifacts_dir: str) -> None:
        self.artifacts_dir = artifacts_dir
        self.model: Optional[AutoModelForTokenClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cpu"
        self.id2label: Dict[int, str] = {}
        self._loaded = False

    def _select_device(self) -> str:
        """Select the best available device for inference.

        Returns:
            Device string (cpu, cuda, or mps).
        """
        # Allow environment override
        forced = os.environ.get("X5_FORCE_DEVICE") or os.environ.get("FORCE_DEVICE")
        if forced:
            device = forced.strip().lower()
            if device in SUPPORTED_DEVICES:
                logger.info(f"Using forced device: {device}")
                return device
            else:
                logger.warning(
                    f"Unsupported forced device: {device}, falling back to auto-detection"
                )

        # Auto-detect best device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Selected device: {device}")
        return device

    def _load_artifacts(self) -> None:
        """Load model artifacts with proper error handling.

        Raises:
            FileNotFoundError: If artifacts directory doesn't exist.
            RuntimeError: If model loading fails.
        """
        if not os.path.exists(self.artifacts_dir):
            raise FileNotFoundError(
                f"Artifacts directory not found: {self.artifacts_dir}"
            )

        try:
            logger.info(f"Loading model from {self.artifacts_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.artifacts_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.artifacts_dir
            )

            self.device = self._select_device()
            if self.model is not None:
                self.model = self.model.to(self.device).eval()

            # Load id2label mapping
            if self.model is not None:
                id2label = getattr(self.model.config, "id2label", None)
                if not isinstance(id2label, dict):
                    num_labels = getattr(self.model.config, "num_labels", 0)
                    id2label = {i: str(i) for i in range(num_labels)}
                    logger.warning("No id2label found in config, using default mapping")
            else:
                id2label = {}

            self.id2label = {int(k): v for k, v in id2label.items()}
            self._loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def ensure_loaded(self) -> None:
        """Ensure model is loaded, loading if necessary."""
        if not self._loaded:
            self._load_artifacts()

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Predict entities in text.

        Args:
            text: Input text to analyze.

        Returns:
            List of entity spans with start_index, end_index, and entity type.

        Raises:
            RuntimeError: If model is not loaded or prediction fails.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess text
            normalized_text, _ = preprocess_text_with_mapping(
                text, do_lower=True, apply_translit_map=True
            )

            with torch.no_grad():
                # Tokenize
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer not loaded")
                encoding = self.tokenizer(
                    normalized_text,
                    return_offsets_mapping=True,
                    truncation=True,
                    return_tensors="pt",
                )
                offsets: List[Tuple[int, int]] = encoding["offset_mapping"][0].tolist()

                # Move inputs to device
                inputs = {
                    k: v.to(self.device)
                    for k, v in encoding.items()
                    if k != "offset_mapping"
                }

                # Get predictions
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred_ids = logits.argmax(-1)[0].tolist()

                # Convert to labels
                token_tags = [self._id2label(token_id) for token_id in pred_ids]

                # Convert to character-level BIO tags
                char_bio = _token_tags_to_char_bio(normalized_text, token_tags, offsets)

                # Extract spans
                max_offset = max([e for (_, e) in offsets if e is not None] or [0])
                spans = _extract_spans_from_bio(
                    normalized_text[:max_offset],
                    char_bio,
                )

                # Convert to API format
                return _spans_to_api_spans(normalized_text, spans, include_O=True)

        except Exception as e:
            logger.error(f"Prediction failed for text '{text[:50]}...': {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def _id2label(self, token_id: int) -> str:
        """Convert token ID to label with fallback.

        Args:
            token_id: Token ID from model.

        Returns:
            Label string.
        """
        # Try config mapping first
        cfg_map = None
        if self.model is not None and hasattr(self.model, 'config'):
            cfg_map = getattr(self.model.config, "id2label", None)
        if isinstance(cfg_map, dict):
            result = cfg_map.get(
                token_id, cfg_map.get(str(token_id), self.id2label.get(token_id, "O"))
            )
            return str(result) if result is not None else "O"
        return self.id2label.get(token_id, "O")


class CacheManager:
    """High-performance in-memory cache for prediction requests.

    Uses TTLCache from cachetools for thread-safe, time-based caching
    with automatic eviction and size limits.
    """

    def __init__(
        self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS
    ) -> None:
        """Initialize cache manager.

        Args:
            max_size: Maximum number of items in cache.
            ttl_seconds: Time-to-live for cache items in seconds.
        """
        self.cache: TTLCache[str, Any] = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.hit_count = 0
        self.miss_count = 0
        self.enabled = CACHE_ENABLED
        logger.info(
            f"In-memory cache initialized: max_size={max_size}, ttl={ttl_seconds}s, enabled={self.enabled}"
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate efficient cache key using SHA-256 hash.

        Args:
            text: Input text to cache.

        Returns:
            SHA-256 hash of the text (64 characters).
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result for text.

        Args:
            text: Input text.

        Returns:
            Cached result or None if not found/expired.
        """
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(text)
        result: Optional[List[Dict[str, Any]]] = self.cache.get(cache_key)

        if result is not None:
            self.hit_count += 1
            logger.debug(f"Cache HIT: {text[:30]}...")
            return result
        else:
            self.miss_count += 1
            logger.debug(f"Cache MISS: {text[:30]}...")
            return None

    def set(self, text: str, result: List[Dict[str, Any]]) -> None:
        """Cache result for text with automatic TTL.

        Args:
            text: Input text.
            result: Prediction result to cache.
        """
        if not self.enabled:
            return

        cache_key = self._get_cache_key(text)
        self.cache[cache_key] = result
        logger.debug(f"Cached: {text[:30]}...")

    def clear(self) -> None:
        """Clear all cached items and reset statistics."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("In-memory cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance metrics.
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = (
            (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0
        )

        return {
            "enabled": self.enabled,
            "max_size": self.cache.maxsize,
            "current_size": len(self.cache),
            "ttl_seconds": self.cache.ttl,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "memory_usage_mb": round(len(str(self.cache)) / 1024 / 1024, 2),
        }


# Global instances
model_manager = ModelManager(ARTIFACTS_DIR)
cache_manager = CacheManager()
