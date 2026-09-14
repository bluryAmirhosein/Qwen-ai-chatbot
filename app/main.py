import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.logging import setup_logging
from app.core.paths import STATIC_DIR
from app.web.pages import router as web_router

settings = get_settings()
setup_logging(settings)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Chatbot backend powered by a locally hosted Qwen3-1.7B model.",
    version="0.1.0",
)

# JSON API — unchanged, still the single source of truth for all data.
app.include_router(api_router, prefix=settings.api_v1_prefix)

# HTML page(s) — render Jinja2 templates that call the JSON API above via fetch().
app.include_router(web_router)

# CSS/JS/assets for the templates, served at /static/...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
