from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Result of ingesting a Word document into the RAG index."""

    document_id: int = Field(..., description="Stored document id")
    filename: str = Field(..., description="Original filename")
    chunks_created: int = Field(..., description="Number of chunks stored for this document")


class RagDocument(BaseModel):
    """A single ingested document, as listed in /rag/documents."""

    id: int = Field(..., description="Document id")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks stored for this document")


class RagDocumentListResponse(BaseModel):
    """List of all ingested documents."""

    documents: list[RagDocument] = Field(..., description="Ingested documents")