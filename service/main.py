"""
X5 NER Service - FastAPI application with modular architecture.

This module provides a FastAPI service for Named Entity Recognition
in Russian product search queries for the Pyaterochka mobile app.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
logger = logging.getLogger(__name__)

# Per-process bounded concurrency for CPU-bound inference
_PREDICT_MAX_CONCURRENCY = int(os.getenv("PREDICT_MAX_CONCURRENCY", "2"))
_predict_semaphore = asyncio.Semaphore(_PREDICT_MAX_CONCURRENCY)


async def _predict_offthread(text: str) -> List[Dict[str, Any]]:
    """Run model prediction in a bounded thread pool slot.

    Binds CPU-bound inference to at most PREDICT_MAX_CONCURRENCY concurrent tasks
    per worker process to avoid thread oversubscription.
    """
    loop = asyncio.get_running_loop()
    async with _predict_semaphore:
        return await loop.run_in_executor(None, model_manager.predict, text)


async def _predict_batch_offthread(texts: List[str]) -> List[List[Dict[str, Any]]]:
    """Run batched model prediction in a bounded thread pool slot.

    Uses the same semaphore to avoid oversubscription across single and batch calls.
    """
    loop = asyncio.get_running_loop()
    async with _predict_semaphore:
        return await loop.run_in_executor(None, model_manager.predict_batch, texts)


class MicroBatcher:
    def __init__(
        self,
        max_batch_size: int,
        max_wait_ms: int,
        hard_timeout_ms: int,
        enabled: bool,
        queue_maxsize: int = 0,
    ) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self.hard_timeout_ms = max(1, int(hard_timeout_ms))
        self.enabled = enabled
        self._queue: "asyncio.Queue[Tuple[str, asyncio.Future[List[EntitySpan]]]]" = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def enqueue(self, text: str) -> List[EntitySpan]:
        if not self.enabled:
            result = await _predict_offthread(text)
            return [EntitySpan(**span) for span in result]
        fut = cast(
            "asyncio.Future[List[EntitySpan]]",
            asyncio.get_running_loop().create_future(),
        )
        await self._queue.put((text, fut))
        try:
            res: List[EntitySpan] = await asyncio.wait_for(
                fut, timeout=self.hard_timeout_ms / 1000.0
            )
            return res
        except asyncio.TimeoutError:
            if not fut.done():
                fut.cancel()
            result = await _predict_offthread(text)
            return [EntitySpan(**span) for span in result]

    async def _run(self) -> None:
        try:
            while True:
                text0, fut0 = await self._queue.get()
                batch: List[Tuple[str, asyncio.Future[List[EntitySpan]]]] = [(text0, fut0)]
                start = asyncio.get_running_loop().time()
                deadline = start + (self.max_wait_ms / 1000.0)
                while len(batch) < self.max_batch_size:
                    timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                    if timeout == 0.0:
                        break
                    try:
                        itm = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        batch.append(itm)
                    except asyncio.TimeoutError:
                        break

                index_map: Dict[str, List[int]] = {}
                for i, (t, _) in enumerate(batch):
                    index_map.setdefault(t, []).append(i)

                cached_entities: Dict[int, List[EntitySpan]] = {}
                misses: List[str] = []
                for i, (t, _) in enumerate(batch):
                    c = cache_manager.get(t)
                    if c is not None:
                        cached_entities[i] = [EntitySpan(**span) for span in c]
                    else:
                        if t not in misses:
                            misses.append(t)

                miss_results: Dict[str, List[EntitySpan]] = {}
                if misses:
                    try:
                        spans_lists = await _predict_batch_offthread(misses)
                        for txt, spans in zip(misses, spans_lists):
                            cache_manager.set(txt, spans)
                            miss_results[txt] = [EntitySpan(**span) for span in spans]
                    except Exception as e:
                        logger.error(
                            f"Micro-batch execution failed for {len(misses)} items: {e}"
                        )
                        fallback = await asyncio.gather(
                            *[_predict_offthread(t) for t in misses],
                            return_exceptions=True,
                        )
                        for t, r in zip(misses, fallback):
                            if isinstance(r, Exception):
                                miss_results[t] = []
                            else:
                                spans_ok = cast(List[Dict[str, Any]], r)
                                cache_manager.set(t, spans_ok)
                                miss_results[t] = [EntitySpan(**span) for span in spans_ok]

                for i, (t, fut) in enumerate(batch):
                    if fut.cancelled():
                        continue
                    try:
                        if i in cached_entities:
                            fut.set_result(cached_entities[i])
                        else:
                            fut.set_result(miss_results.get(t, []))
                    except Exception as e_set:
                        try:
                            fut.set_exception(e_set)
                        except Exception:
                            pass
        except asyncio.CancelledError:
            return


# Global micro-batcher instance (initialized in lifespan)
_micro_batcher: Optional[MicroBatcher] = None


# FastAPI app with lifespan management
async def _background_warmup() -> None:
    """Run a lightweight model warmup in the background.

    Executes a few representative predictions to initialize tokenizers,
    allocate tensors, and populate internal caches without blocking startup.
    """
    try:
        logger.info("Starting background warmup")
        # A few short, typical queries (including empty) to touch common paths
        samples = [
            "",
            "молоко 1 л",
            "кока кола 0.5л",
            "сыр 45% 200г",
        ]
        loop = asyncio.get_running_loop()
        for s in samples:
            try:
                await loop.run_in_executor(None, model_manager.predict, s)
            except Exception as e_item:
                logger.debug(f"Warmup sample failed (ignored): {e_item}")
        logger.info("Background warmup completed")
    except Exception as e:
        logger.warning(f"Background warmup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan."""
    # Startup
    logger.info("Starting X5 NER Service")
    try:
        model_manager.ensure_loaded()
        # Optional auto-warmup
        disable_warm = os.getenv("DISABLE_WARMUP", "false").lower() in {"1", "true", "yes"}
        allow_mps_warm = os.getenv("ALLOW_MPS_WARMUP", "false").lower() in {"1", "true", "yes"}
        if not disable_warm:
            if model_manager.device == "mps" and not allow_mps_warm:
                logger.info("Auto-warmup skipped on MPS (set ALLOW_MPS_WARMUP=true to enable)")
            else:
                try:
                    asyncio.create_task(_background_warmup())
                    logger.info("Auto-warmup task scheduled")
                except Exception as e_task:
                    logger.debug(f"Failed to schedule warmup task: {e_task}")
        # Micro-batch startup
        try:
            mb_enabled = os.getenv("MICRO_BATCH_ENABLED", "true").lower() in {"1", "true", "yes"}
            mb_max_size = int(os.getenv("MICRO_BATCH_MAX_SIZE", "32"))
            mb_max_wait_ms = int(os.getenv("MICRO_BATCH_MAX_WAIT_MS", "3"))
            mb_hard_timeout_ms = int(os.getenv("MICRO_BATCH_HARD_TIMEOUT_MS", "500"))
            mb_queue_max = int(os.getenv("MICRO_BATCH_QUEUE_MAXSIZE", "0"))
        except Exception:
            mb_enabled, mb_max_size, mb_max_wait_ms, mb_hard_timeout_ms, mb_queue_max = (
                True,
                32,
                3,
                500,
                0,
            )

        global _micro_batcher
        _micro_batcher = MicroBatcher(
            max_batch_size=mb_max_size,
            max_wait_ms=mb_max_wait_ms,
            hard_timeout_ms=mb_hard_timeout_ms,
            enabled=mb_enabled,
            queue_maxsize=mb_queue_max,
        )
        try:
            await _micro_batcher.start()
            if _micro_batcher.enabled:
                logger.info(
                    f"Micro-batching enabled: size={mb_max_size} wait_ms={mb_max_wait_ms}"
                )
            else:
                logger.info("Micro-batching disabled")
        except Exception as e_mb:
            logger.warning(f"Failed to start micro-batching: {e_mb}")

        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down X5 NER Service")
    try:
        if _micro_batcher is not None:
            await _micro_batcher.stop()
    except Exception as e:
        logger.debug(f"Error stopping micro-batcher: {e}")


app = FastAPI(
    title="X5 NER Service",
    version="1.0.0",
    description="Named Entity Recognition service for Pyaterochka product search",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Все домены
    allow_credentials=True,
    allow_methods=["*"],  # Все HTTP методы
    allow_headers=["*"],  # Все заголовки
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

        if _micro_batcher is not None and _micro_batcher.enabled:
            return await _micro_batcher.enqueue(req.input)
        cached_result = cache_manager.get(req.input)
        if cached_result is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Cache HIT for prediction: %s...", req.input[:50])
            return [EntitySpan(**span) for span in cached_result]
        result = await _predict_offthread(req.input)
        cache_manager.set(req.input, result)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Cache MISS for prediction: %s...", req.input[:50])
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
        inputs = list(req.inputs)
        if not inputs:
            return []

        # Pre-check cache and deduplicate to minimize work
        index_map: Dict[str, List[int]] = {}
        cached: Dict[int, List[EntitySpan]] = {}
        misses: List[str] = []
        for idx, text in enumerate(inputs):
            if text not in index_map:
                index_map[text] = []
            index_map[text].append(idx)
            c = cache_manager.get(text)
            if c is not None:
                cached[idx] = [EntitySpan(**span) for span in c]
            else:
                if text not in misses:
                    misses.append(text)

        miss_results: Dict[str, List[EntitySpan]] = {}
        if misses:
            try:
                batch_spans = await _predict_batch_offthread(misses)
                for txt, spans in zip(misses, batch_spans):
                    # Store in cache
                    cache_manager.set(txt, spans)
                    miss_results[txt] = [EntitySpan(**span) for span in spans]
            except Exception as e:
                logger.error(f"Batched prediction failed for {len(misses)} items: {e}")
                # Fallback: process misses individually in bounded slots
                fallback_tasks = [
                    _predict_offthread(t) for t in misses
                ]
                fallback_done = await asyncio.gather(*fallback_tasks, return_exceptions=True)
                for t, r in zip(misses, fallback_done):
                    if isinstance(r, Exception):
                        logger.error(f"Fallback single prediction failed for text: {t[:40]}...: {r}")
                        miss_results[t] = []
                    else:
                        r_ok = cast(List[Dict[str, Any]], r)
                        cache_manager.set(t, r_ok)
                        miss_results[t] = [EntitySpan(**span) for span in r_ok]

        # Assemble results in original order
        out: List[List[EntitySpan]] = []
        for idx, text in enumerate(inputs):
            if idx in cached:
                out.append(cached[idx])
            else:
                out.append(miss_results.get(text, []))
        return out

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
        # Test with empty string without blocking the event loop
        _ = await _predict_offthread("")
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
