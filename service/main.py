"""
X5 NER Service - FastAPI application with modular architecture.

This module provides a FastAPI service for Named Entity Recognition
in Russian product search queries for the Pyaterochka mobile app.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

from .config import ARTIFACTS_DIR, CACHE_ENABLED, CACHE_MAX_SIZE, CACHE_TTL_SECONDS
from .managers import cache_manager, model_manager
from .models import (
    CacheStatsResponse,
    EntitySpan,
    HealthResponse,
    PredictBatchRequest,
    PredictRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan."""
    # Startup
    logger.info("Starting X5 NER Service")
    try:
        model_manager.ensure_loaded()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down X5 NER Service")


app = FastAPI(
    title="X5 NER Service",
    version="1.0.0",
    description="Named Entity Recognition service for Pyaterochka product search",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Service health status and configuration.
    """
    return HealthResponse(
        status="ok",
        device=model_manager.device,
        artifacts=ARTIFACTS_DIR,
        model_loaded=model_manager._loaded,
        cache_stats=cache_manager.get_stats(),
    )


@app.post("/api/predict", response_model=List[EntitySpan])
async def predict(req: PredictRequest) -> List[EntitySpan]:
    """Predict entities in a single text.

    Args:
        req: Prediction request with input text.

    Returns:
        List of detected entity spans.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        # Handle empty input
        if not req.input.strip():
            return []

        # Check cache first
        cached_result = cache_manager.get(req.input)
        if cached_result is not None:
            logger.info(f"Cache HIT for prediction: {req.input[:50]}...")
            return [EntitySpan(**span) for span in cached_result]

        # Run prediction in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.predict, req.input
        )

        # Cache the result
        cache_manager.set(req.input, result)
        logger.info(f"Cache MISS for prediction: {req.input[:50]}...")

        # Convert to response model
        return [EntitySpan(**span) for span in result]

    except RuntimeError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/predict_batch", response_model=List[List[EntitySpan]])
async def predict_batch(req: PredictBatchRequest) -> List[List[EntitySpan]]:
    """Predict entities in multiple texts.

    Args:
        req: Batch prediction request with list of texts.

    Returns:
        List of entity spans for each input text.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        # Create tasks for parallel processing
        tasks = [
            asyncio.get_event_loop().run_in_executor(None, model_manager.predict, text)
            for text in req.inputs
        ]

        # Execute all predictions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results: List[List[EntitySpan]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch prediction failed for item {i}: {result}")
                processed_results.append([])  # Return empty list for failed items
            else:
                # Type check: result should be List[Dict[str, Any]]
                if isinstance(result, list):
                    processed_results.append([EntitySpan(**span) for span in result])
                else:
                    logger.error(f"Unexpected result type for item {i}: {type(result)}")
                    processed_results.append([])

        return processed_results

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# Optional warmup endpoint for testing
@app.post("/warmup")
async def warmup() -> Dict[str, str]:
    """Warmup endpoint to test model loading.

    Returns:
        Warmup status.
    """
    try:
        model_manager.ensure_loaded()
        # Test with empty string
        _ = model_manager.predict("")
        return {"status": "warmed_up", "device": model_manager.device}
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """Get cache statistics.

    Returns:
        Cache statistics including hit rate and usage.
    """
    stats = cache_manager.get_stats()
    return CacheStatsResponse(**stats)


@app.delete("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear all cached predictions.

    Returns:
        Confirmation message.
    """
    cache_manager.clear()
    return {
        "status": "cache_cleared",
        "message": "All cached predictions have been cleared",
    }


@app.get("/cache/info")
async def get_cache_info() -> Dict[str, Any]:
    """Get detailed cache information.

    Returns:
        Detailed cache configuration and statistics.
    """
    stats = cache_manager.get_stats()
    return {
        "cache_config": {
            "enabled": CACHE_ENABLED,
            "max_size": CACHE_MAX_SIZE,
            "ttl_seconds": CACHE_TTL_SECONDS,
        },
        "cache_stats": stats,
        "environment_variables": {
            "CACHE_ENABLED": os.environ.get("CACHE_ENABLED", "true"),
            "CACHE_MAX_SIZE": os.environ.get("CACHE_MAX_SIZE", "1000"),
            "CACHE_TTL_SECONDS": os.environ.get("CACHE_TTL_SECONDS", "3600"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
