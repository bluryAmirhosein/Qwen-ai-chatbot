from datetime import datetime

from pydantic import BaseModel, Field


class MessageOut(BaseModel):
    """A single stored chat message."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    created_at: datetime


class ConversationOut(BaseModel):
    """Summary of a stored conversation, used in list/search results."""

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetailOut(ConversationOut):
    """Full conversation including all messages. Used to view/resume a chat."""

    messages: list[MessageOut]


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]


class ConversationSearchResult(ConversationOut):
    """A conversation match, with a short snippet showing why it matched."""

    snippet: str


class ConversationSearchResponse(BaseModel):
    results: list[ConversationSearchResult]