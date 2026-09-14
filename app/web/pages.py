"""HTML page routes.

These are the only routes in the project that return rendered templates
instead of JSON — everything the page talks to afterwards (chat, history,
rag, files, summarizer) still goes through the existing JSON API under
`settings.api_v1_prefix`, called asynchronously from the browser via
`app/static/js`.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.core.paths import TEMPLATES_DIR

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/")
async def chat_page(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "api_prefix": settings.api_v1_prefix,
        },
    )
