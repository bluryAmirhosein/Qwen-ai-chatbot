# app/api/v1/endpoints/summarizer.py

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.dependencies import get_summarizer_service
from app.schemas.summarizer import (
    SummarizeByQueryRequest,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.summarizer_service import SummarizerService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a piece of text",
    description="Returns a general summary of the given text, capped at roughly max_length words.",
)
async def summarize_text(
    payload: SummarizeRequest,
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
) -> SummarizeResponse:
    logger.info("Summarize request (text_length=%d)", len(payload.text))
    try:
        result = summarizer_service.summarize(text=payload.text, max_length=payload.max_length)
    except Exception as exc:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail="Failed to summarize the text") from exc
    return SummarizeResponse(**result)


@router.post(
    "/summarize/by-query",
    response_model=SummarizeResponse,
    summary="Summarize a piece of text focused on a query",
    description="Returns a summary of the given text focused only on parts relevant to `query`.",
)
async def summarize_by_query(
    payload: SummarizeByQueryRequest,
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
) -> SummarizeResponse:
    logger.info(
        "Query-focused summarize request (text_length=%d, query=%r)",
        len(payload.text),
        payload.query,
    )
    try:
        result = summarizer_service.summarize_by_query(
            text=payload.text, query=payload.query, max_length=payload.max_length
        )
    except Exception as exc:
        logger.exception("Query-focused summarization failed")
        raise HTTPException(status_code=500, detail="Failed to summarize the text") from exc
    return SummarizeResponse(**result)