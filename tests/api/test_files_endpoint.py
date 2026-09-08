from unittest.mock import MagicMock

from app.api.v1.dependencies import get_file_service
from app.services.file_service import UnsupportedFileTypeError


def test_extract_file_returns_extracted_text(app, client):
    mock_service = MagicMock()
    mock_service.extract_text.return_value = "extracted plain text"
    app.dependency_overrides[get_file_service] = lambda: mock_service

    response = client.post(
        "/api/v1/files/extract",
        files={"file": ("notes.txt", b"raw bytes", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {"filename": "notes.txt", "content": "extracted plain text"}


def test_extract_file_returns_400_for_unsupported_type(app, client):
    mock_service = MagicMock()
    mock_service.extract_text.side_effect = UnsupportedFileTypeError("Unsupported file type: .zip")
    app.dependency_overrides[get_file_service] = lambda: mock_service

    response = client.post(
        "/api/v1/files/extract",
        files={"file": ("archive.zip", b"raw bytes", "application/zip")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_extract_file_requires_a_file(app, client):
    mock_service = MagicMock()
    app.dependency_overrides[get_file_service] = lambda: mock_service

    response = client.post("/api/v1/files/extract")

    assert response.status_code == 422
    mock_service.extract_text.assert_not_called()
