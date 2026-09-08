from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.api.v1.dependencies import get_history_service

_NOW = datetime.now(timezone.utc).isoformat()


def test_list_history_without_query_returns_conversation_list(app, client):
    mock_service = MagicMock()
    mock_service.list_conversations.return_value = [
        {"id": 1, "title": "hello", "created_at": _NOW, "updated_at": _NOW}
    ]
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.get("/api/v1/history")

    assert response.status_code == 200
    assert response.json()["conversations"][0]["id"] == 1
    mock_service.search.assert_not_called()


def test_list_history_with_query_returns_search_results(app, client):
    mock_service = MagicMock()
    mock_service.search.return_value = [
        {"id": 2, "title": "python", "created_at": _NOW, "updated_at": _NOW, "snippet": "...list..."}
    ]
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.get("/api/v1/history", params={"q": "list"})

    assert response.status_code == 200
    assert response.json()["results"][0]["snippet"] == "...list..."
    mock_service.list_conversations.assert_not_called()


def test_get_conversation_returns_404_when_missing(app, client):
    mock_service = MagicMock()
    mock_service.get_conversation.return_value = None
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.get("/api/v1/history/999")

    assert response.status_code == 404


def test_get_conversation_returns_full_conversation(app, client):
    mock_service = MagicMock()
    mock_service.get_conversation.return_value = {
        "id": 1,
        "title": "hello",
        "created_at": _NOW,
        "updated_at": _NOW,
        "messages": [{"role": "user", "content": "hi", "created_at": _NOW}],
    }
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.get("/api/v1/history/1")

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "hi"


def test_delete_conversation_returns_404_when_missing(app, client):
    mock_service = MagicMock()
    mock_service.delete_conversation.return_value = False
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.delete("/api/v1/history/999")

    assert response.status_code == 404


def test_delete_conversation_succeeds(app, client):
    mock_service = MagicMock()
    mock_service.delete_conversation.return_value = True
    app.dependency_overrides[get_history_service] = lambda: mock_service

    response = client.delete("/api/v1/history/1")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "conversation_id": 1}
