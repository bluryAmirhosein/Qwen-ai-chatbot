from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation

_TITLE_MAX_LEN = 60


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, title: str) -> Conversation:
        conversation = Conversation(title=_truncate_title(title))
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get_by_id(self, conversation_id: int) -> Conversation | None:
        return await self._session.get(Conversation, conversation_id)

    async def list(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        stmt = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def touch(self, conversation_id: int) -> None:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(timezone.utc)

    async def delete(self, conversation_id: int) -> bool:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is None:
            return False
        await self._session.delete(conversation)
        return True


def _truncate_title(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _TITLE_MAX_LEN:
        return text or "New conversation"
    return text[: _TITLE_MAX_LEN - 1].rstrip() + "…"