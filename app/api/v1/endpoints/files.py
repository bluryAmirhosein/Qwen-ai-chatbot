from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.v1.dependencies import get_file_service
from app.schemas.chat import FileExtractResponse
from app.services.file_service import FileService, UnsupportedFileTypeError

router = APIRouter()


@router.post(
    "/extract",
    response_model=FileExtractResponse,
    summary="Extract text from an uploaded file",
    description="Extracts plain text from .txt, .md, .pdf or .docx files. "
    "The returned text can be passed back in the `context` field of a chat request.",
)
async def extract_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service),
) -> FileExtractResponse:
    try:
        content = file_service.extract_text(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileExtractResponse(filename=file.filename or "unknown", content=content)
