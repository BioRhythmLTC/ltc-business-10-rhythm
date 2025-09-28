"""Business logic managers for model and cache operations."""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from cachetools import TTLCache
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import (
    ARTIFACTS_DIR,
    CACHE_ENABLED,
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    SUPPORTED_DEVICES,
)
from .utils import predict_one_pp_preloaded


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference with proper error handling."""

    def __init__(self, artifacts_dir: str) -> None:
        self.artifacts_dir = artifacts_dir
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.device: str = "cpu"
        self.id2label: Dict[int, str] = {}
        self._loaded = False
        # Default model alias; can be overridden via env X5_MODEL_ALIAS
        self.model_alias: str = os.environ.get("X5_MODEL_ALIAS", "rubert-base-cased")

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
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = "mps"
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
            self.model = AutoModelForTokenClassification.from_pretrained(self.artifacts_dir)

            self.device = self._select_device()
            if self.model is not None:
                # Do not reassign to preserve precise type information for mypy
                m_any = cast(Any, self.model)
                m_any.to(torch.device(self.device))
                m_any.eval()

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
            assert self.model is not None and self.tokenizer is not None
            result = predict_one_pp_preloaded(
                text=text,
                model=cast(PreTrainedModel, self.model),
                tokenizer=cast(PreTrainedTokenizerBase, self.tokenizer),
                device=self.device,
            )
            spans = cast(List[Dict[str, Any]], result.get("api_spans", []))
            return spans

        except Exception as e:
            logger.error(f"Prediction failed for text '{text[:50]}...': {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    


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
