from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.v1.dependencies import get_history_service
from app.schemas.history import (
    ConversationDetailOut,
    ConversationListResponse,
    ConversationOut,
    ConversationSearchResponse,
    ConversationSearchResult,
    MessageOut,
)
from app.services.history_service import HistoryService

router = APIRouter()


@router.get(
    "",
    response_model=ConversationListResponse | ConversationSearchResponse,
    summary="List or search saved conversations",
    description="Without `q`, returns the most recently updated conversations. "
    "With `q`, full-text searches message content and returns matches with a "
    "highlighted snippet, most relevant first.",
)
async def list_or_search_history(
    q: str | None = Query(default=None, description="Search text. Omit to just list conversations."),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, description="Ignored when q is set"),
    history_service: HistoryService = Depends(get_history_service),
):
    if q:
        results = history_service.search(q, limit=limit)
        return ConversationSearchResponse(
            results=[ConversationSearchResult(**r) for r in results]
        )

    conversations = history_service.list_conversations(limit=limit, offset=offset)
    return ConversationListResponse(
        conversations=[ConversationOut(**c) for c in conversations]
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailOut,
    summary="Get a saved conversation's full messages",
    description="Fetch a conversation to display or resume — feed its id back as "
    "`conversation_id` on /chat to continue it.",
)
async def get_conversation(
    conversation_id: int,
    history_service: HistoryService = Depends(get_history_service),
) -> ConversationDetailOut:
    conversation = history_service.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetailOut(
        **{**conversation, "messages": [MessageOut(**m) for m in conversation["messages"]]}
    )


@router.delete(
    "/{conversation_id}",
    summary="Delete a saved conversation and its messages",
)
async def delete_conversation(
    conversation_id: int,
    history_service: HistoryService = Depends(get_history_service),
) -> dict:
    deleted = history_service.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}
