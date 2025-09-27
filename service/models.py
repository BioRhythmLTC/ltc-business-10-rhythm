"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for single prediction."""

    input: str = Field(..., description="Text to analyze for entities", min_length=0)


class PredictBatchRequest(BaseModel):
    """Request model for batch prediction."""

    inputs: List[str] = Field(..., description="List of texts to analyze", min_items=1)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    device: str = Field(..., description="Current device")
    artifacts: str = Field(..., description="Artifacts directory path")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""

    enabled: bool = Field(..., description="Whether caching is enabled")
    max_size: int = Field(..., description="Maximum cache size")
    current_size: int = Field(..., description="Current cache size")
    ttl_seconds: int = Field(..., description="Time-to-live in seconds")
    hit_count: int = Field(..., description="Number of cache hits")
    miss_count: int = Field(..., description="Number of cache misses")
    hit_rate_percent: float = Field(..., description="Cache hit rate percentage")
    total_requests: int = Field(..., description="Total requests processed")
    memory_usage_mb: float = Field(..., description="Estimated memory usage in MB")


class EntitySpan(BaseModel):
    """Entity span response model."""

    start_index: int = Field(..., description="Start character index")
    end_index: int = Field(..., description="End character index")
    entity: str = Field(..., description="Entity type in BIO format")
