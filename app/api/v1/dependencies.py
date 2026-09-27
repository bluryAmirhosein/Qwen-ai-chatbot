"""Shared FastAPI dependency providers for the v1 API.

Every endpoint module (chat, files, history, rag) pulls its services from
here instead of constructing them locally. Keeping this in one place avoids
duplicated `@lru_cache` singletons and circular imports between endpoint
files that would otherwise need each other's service getters (e.g. both
`chat` and `rag` need a `RagService`).
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.history_service import HistoryService
from app.services.model_service import ModelService
from app.services.rag.embedding_service import EmbeddingService
from app.services.rag.rag_service import RagService
from app.services.web_search import WebSearchService
from app.services.summarizer_service import SummarizerService


@lru_cache
def get_model_service() -> ModelService:
    """Singleton ModelService instance, shared across requests.

    Using lru_cache (instead of app.state) keeps this a plain, easily
    overridable FastAPI dependency for tests via `dependency_overrides`.
    """
    return ModelService(settings=get_settings())


@lru_cache
def get_search_service() -> WebSearchService:
    settings = get_settings()
    return WebSearchService(
        max_results=settings.web_search_max_results,
        timeout=settings.web_search_timeout,
    )


@lru_cache
def get_file_service() -> FileService:
    return FileService()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        cache_dir=settings.model_cache_dir,
    )


def get_history_service(session: AsyncSession = Depends(get_db)) -> HistoryService:
    """Request-scoped: HistoryService now owns an AsyncSession, so it can no
    longer be an lru_cache singleton — each request gets its own session."""
    return HistoryService(session=session)


def get_rag_service(
    session: AsyncSession = Depends(get_db),
    file_service: FileService = Depends(get_file_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RagService:
    return RagService(
        session=session,
        file_service=file_service,
        embedding_service=embedding_service,
    )


def get_chat_service(
    model_service: ModelService = Depends(get_model_service),
    search_service: WebSearchService = Depends(get_search_service),
    file_service: FileService = Depends(get_file_service),
    rag_service: RagService = Depends(get_rag_service),
    history_service: HistoryService = Depends(get_history_service),
) -> ChatService:
    return ChatService(
        model_service=model_service,
        search_service=search_service,
        file_service=file_service,
        rag_service=rag_service,
        history_service=history_service,
    )


def get_summarizer_service(
    model_service: ModelService = Depends(get_model_service),
    settings=Depends(get_settings),
) -> SummarizerService:
    return SummarizerService(model_service=model_service, settings=settings)