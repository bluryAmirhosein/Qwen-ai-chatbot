from fastapi import APIRouter

from app.api.v1.endpoints import chat, files, history, rag

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(history.router, prefix="/history", tags=["history"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
