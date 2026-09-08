import logging
import sys

import uvicorn
from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Chatbot backend powered by a locally hosted Qwen3-1.7B model.",
    version="0.1.0",
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["health"], summary="Health check")
def health_check() -> dict:
    return {"status": "ok"}


def run() -> None:
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_config=None,  # keep our own logging config from setup_logging
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C")
        sys.exit(0)


if __name__ == "__main__":
    run()