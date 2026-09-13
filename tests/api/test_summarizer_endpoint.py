# tests/api/test_summarizer_api.py

from unittest.mock import Mock

import pytest

from app.api.v1.dependencies import get_summarizer_service


@pytest.fixture
def mock_summarizer_service():
    return Mock()


@pytest.fixture
def client(app, client, mock_summarizer_service):
    app.dependency_overrides[get_summarizer_service] = lambda: mock_summarizer_service
    return client


class TestSummarizeEndpoint:
    def test_success_returns_200_with_expected_body(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize.return_value = {
            "summary": "A brief summary.",
            "original_length": 42,
            "summary_length": 3,
        }

        response = client.post(
            "/api/v1/summarize", json={"text": "Some text to summarize.", "max_length": 100}
        )

        assert response.status_code == 200
        assert response.json() == {
            "summary": "A brief summary.",
            "original_length": 42,
            "summary_length": 3,
        }

    def test_calls_service_with_payload_values(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize.return_value = {
            "summary": "x",
            "original_length": 1,
            "summary_length": 1,
        }

        client.post("/api/v1/summarize", json={"text": "Hello world.", "max_length": 30})

        mock_summarizer_service.summarize.assert_called_once_with(text="Hello world.", max_length=30)

    def test_uses_default_max_length_when_omitted(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize.return_value = {
            "summary": "x",
            "original_length": 1,
            "summary_length": 1,
        }

        client.post("/api/v1/summarize", json={"text": "Hello world."})

        _, kwargs = mock_summarizer_service.summarize.call_args
        assert kwargs["max_length"] == 150

    def test_service_exception_returns_500(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize.side_effect = RuntimeError("model exploded")

        response = client.post("/api/v1/summarize", json={"text": "Some text."})

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to summarize the text"

    def test_missing_text_returns_422(self, client, mock_summarizer_service):
        response = client.post("/api/v1/summarize", json={"max_length": 100})

        assert response.status_code == 422
        mock_summarizer_service.summarize.assert_not_called()


class TestSummarizeByQueryEndpoint:
    def test_success_returns_200_with_expected_body(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize_by_query.return_value = {
            "summary": "Query-focused summary.",
            "original_length": 60,
            "summary_length": 4,
        }

        response = client.post(
            "/api/v1/summarize/by-query",
            json={"text": "Some long text.", "query": "pricing", "max_length": 80},
        )

        assert response.status_code == 200
        assert response.json() == {
            "summary": "Query-focused summary.",
            "original_length": 60,
            "summary_length": 4,
        }

    def test_calls_service_with_payload_values(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize_by_query.return_value = {
            "summary": "x",
            "original_length": 1,
            "summary_length": 1,
        }

        client.post(
            "/api/v1/summarize/by-query",
            json={"text": "Some text.", "query": "topic", "max_length": 40},
        )

        mock_summarizer_service.summarize_by_query.assert_called_once_with(
            text="Some text.", query="topic", max_length=40
        )

    def test_missing_query_returns_422(self, client, mock_summarizer_service):
        response = client.post(
            "/api/v1/summarize/by-query", json={"text": "Some text.", "max_length": 40}
        )

        assert response.status_code == 422
        mock_summarizer_service.summarize_by_query.assert_not_called()

    def test_service_exception_returns_500(self, client, mock_summarizer_service):
        mock_summarizer_service.summarize_by_query.side_effect = RuntimeError("boom")

        response = client.post(
            "/api/v1/summarize/by-query", json={"text": "Some text.", "query": "q"}
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to summarize the text"