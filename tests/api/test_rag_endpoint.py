from unittest.mock import MagicMock

from app.api.v1.dependencies import get_rag_service
from app.services.file_service import UnsupportedFileTypeError


def test_ingest_document_returns_ingestion_summary(app, client):
    mock_service = MagicMock()
    mock_service.ingest_file.return_value = ("report.docx", 1, 5)
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("report.docx", b"raw bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json() == {"document_id": 1, "filename": "report.docx", "chunks_created": 5}


def test_ingest_document_returns_400_for_unsupported_type(app, client):
    mock_service = MagicMock()
    mock_service.ingest_file.side_effect = UnsupportedFileTypeError("Unsupported file type: .zip")
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("archive.zip", b"raw", "application/zip")},
    )

    assert response.status_code == 400


def test_ingest_document_returns_422_when_no_text_extractable(app, client):
    mock_service = MagicMock()
    mock_service.ingest_file.side_effect = ValueError("No extractable text found in the uploaded file")
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


def test_list_documents_returns_all_ingested_documents(app, client):
    mock_service = MagicMock()
    mock_service.list_documents.return_value = [{"id": 1, "filename": "a.txt", "chunk_count": 3}]
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.get("/api/v1/rag/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["filename"] == "a.txt"


def test_delete_document_returns_404_when_missing(app, client):
    mock_service = MagicMock()
    mock_service.delete_document.return_value = False
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.delete("/api/v1/rag/documents/999")

    assert response.status_code == 404


def test_delete_document_succeeds(app, client):
    mock_service = MagicMock()
    mock_service.delete_document.return_value = True
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.delete("/api/v1/rag/documents/1")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "document_id": 1}
