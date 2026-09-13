# app/schemas/summarizer.py

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """Request body for general (non-query) summarization."""

    text: str = Field(..., min_length=1, description="Input text to summarize")
    max_length: int = Field(
        default=150,
        ge=20,
        le=1000,
        description="Approximate maximum summary length, in words",
    )


class SummarizeByQueryRequest(BaseModel):
    """Request body for query-focused summarization."""

    text: str = Field(..., min_length=1, description="Input text")
    query: str = Field(
        ..., min_length=1, description="Key phrase/question the summary should be focused on"
    )
    max_length: int = Field(
        default=150,
        ge=20,
        le=1000,
        description="Approximate maximum summary length, in words",
    )


class SummarizeResponse(BaseModel):
    """Summarizer output returned to the client."""

    summary: str = Field(..., description="Generated summary text")
    original_length: int = Field(..., description="Length of the input text, in words")
    summary_length: int = Field(..., description="Length of the generated summary, in words")