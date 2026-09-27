from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.v1.dependencies import get_rag_service
from app.schemas.rag import IngestResponse, RagDocument, RagDocumentListResponse
from app.services.file_service import UnsupportedFileTypeError
from app.services.rag.rag_service import RagService

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document into the RAG index",
    description="Extracts text from the uploaded file, splits it into overlapping "
    "chunks, embeds each chunk and stores it. Use `rag=true` on /chat "
    "afterwards to retrieve from it.",
)
async def ingest_document(
    file: UploadFile = File(...),
    rag_service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    try:
        filename, document_id, chunk_count = await rag_service.ingest_file(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(document_id=document_id, filename=filename, chunks_created=chunk_count)


@router.get(
    "/documents",
    response_model=RagDocumentListResponse,
    summary="List ingested documents",
)
async def list_documents(
    rag_service: RagService = Depends(get_rag_service),
) -> RagDocumentListResponse:
    docs = await rag_service.list_documents()
    return RagDocumentListResponse(documents=[RagDocument(**doc) for doc in docs])


@router.delete(
    "/documents/{document_id}",
    summary="Delete an ingested document and its chunks",
)
async def delete_document(
    document_id: int,
    rag_service: RagService = Depends(get_rag_service),
) -> dict:
    deleted = await rag_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}