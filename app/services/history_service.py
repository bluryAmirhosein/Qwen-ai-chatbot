import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.conversation_repository import ConversationRepository
from app.repository.message_repository import MessageRepository

logger = logging.getLogger(__name__)


class HistoryService:
    """Same public API as the old sqlite3-based HistoryService, now backed
    by Postgres through ConversationRepository / MessageRepository.

    One HistoryService per request/unit-of-work: it owns the commit boundary,
    the repositories only flush.
    """

    def __init__(self, session: AsyncSession):
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)

    async def create_conversation(self, title: str) -> int:
        conversation = await self._conversations.create(title)
        await self._session.commit()
        logger.info("Created conversation id=%d", conversation.id)
        return conversation.id

    async def add_message(self, conversation_id: int, role: str, content: str) -> None:
        await self._messages.add(conversation_id, role, content)
        await self._conversations.touch(conversation_id)
        await self._session.commit()

    async def get_conversation(self, conversation_id: int) -> dict | None:
        conversation = await self._conversations.get_by_id(conversation_id)
        if conversation is None:
            return None
        messages = await self._messages.list_for_conversation(conversation_id)
        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at}
                for m in messages
            ],
        }

    async def get_messages_for_prompt(self, conversation_id: int) -> list[dict]:
        """Returns messages as {"role", "content"} dicts, ready to feed the model."""
        messages = await self._messages.list_for_conversation(conversation_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict]:
        conversations = await self._conversations.list(limit=limit, offset=offset)
        return [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
            for c in conversations
        ]

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        return await self._messages.search(query, limit=limit)

    async def delete_conversation(self, conversation_id: int) -> bool:
        deleted = await self._conversations.delete(conversation_id)
        await self._session.commit()
        if deleted:
            logger.info("Deleted conversation id=%d", conversation_id)
        return deleted