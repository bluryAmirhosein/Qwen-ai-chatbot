from unittest.mock import MagicMock

from app.api.v1.dependencies import get_chat_service
from app.services.chat_service import ChatResult, ConversationNotFoundError


def test_send_message_returns_reply_and_conversation_id(app, client):
    mock_service = MagicMock()
    mock_service.get_response.return_value = ChatResult(response="hello!", conversation_id=3)
    app.dependency_overrides[get_chat_service] = lambda: mock_service

    response = client.post(
        "/api/v1/chat",
        content="hi there",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "hello!", "conversation_id": 3}
    _, kwargs = mock_service.get_response.call_args
    assert kwargs["user_message"] == "hi there"


def test_send_message_passes_query_params_through_to_the_service(app, client):
    mock_service = MagicMock()
    mock_service.get_response.return_value = ChatResult(response="ok", conversation_id=None)
    app.dependency_overrides[get_chat_service] = lambda: mock_service

    client.post(
        "/api/v1/chat",
        params={
            "thinking_mode": "deep",
            "web_search": "true",
            "rag": "true",
            "rag_top_k": 10,
            "personality": "humorous",
            "language": "persian",
            "conversation_id": 4,
        },
        content="translate this",
        headers={"Content-Type": "text/plain"},
    )

    _, kwargs = mock_service.get_response.call_args
    assert kwargs["thinking_mode"].value == "deep"
    assert kwargs["web_search"] is True
    assert kwargs["rag"] is True
    assert kwargs["rag_top_k"] == 10
    assert kwargs["personality"].value == "humorous"
    assert kwargs["language"].value == "persian"
    assert kwargs["conversation_id"] == 4


def test_send_message_returns_404_for_unknown_conversation_id(app, client):
    mock_service = MagicMock()
    mock_service.get_response.side_effect = ConversationNotFoundError("Conversation 999 not found")
    app.dependency_overrides[get_chat_service] = lambda: mock_service

    response = client.post(
        "/api/v1/chat",
        params={"conversation_id": 999},
        content="hi",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation 999 not found"


def test_send_message_rejects_out_of_range_rag_top_k(app, client):
    mock_service = MagicMock()
    app.dependency_overrides[get_chat_service] = lambda: mock_service

    response = client.post(
        "/api/v1/chat",
        params={"rag_top_k": 999},
        content="hi",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 422
    mock_service.get_response.assert_not_called()


def test_send_message_rejects_empty_body(app, client):
    mock_service = MagicMock()
    app.dependency_overrides[get_chat_service] = lambda: mock_service

    response = client.post(
        "/api/v1/chat",
        content="",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 422
    mock_service.get_response.assert_not_called()
