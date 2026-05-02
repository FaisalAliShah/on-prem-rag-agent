from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.routes import router, warmup_components
from app.config import settings
from app.utils.logging import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logging.getLogger(__name__).info("application_started")
    if settings.warmup_on_startup:
        try:
            warmup_components()
        except Exception:
            logging.getLogger(__name__).exception("startup_warmup_failed")
    yield
    logging.getLogger(__name__).info("application_stopped")


app = FastAPI(title="On-Prem RAG Agent", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Check logs for request_id.",
                "request_id": request_id,
            },
        )

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response
