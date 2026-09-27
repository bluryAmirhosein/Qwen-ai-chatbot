from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message

_HEADLINE_OPTIONS = "StartSel=[, StopSel=], MaxWords=8, MinWords=4"


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, conversation_id: int, role: str, content: str) -> Message:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        self._session.add(message)
        await self._session.flush()
        return message

    async def list_for_conversation(self, conversation_id: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full text search over message content, deduped to the best-ranked
        message per conversation (mirrors the old FTS5-based search())."""
        ts_query = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(Message.content_tsv, ts_query)
        headline = func.ts_headline(
            "english", Message.content, ts_query, text(f"'{_HEADLINE_OPTIONS}'")
        )

        ranked = (
            select(
                Message.conversation_id,
                headline.label("snippet"),
                rank.label("rank"),
                func.row_number()
                .over(partition_by=Message.conversation_id, order_by=rank.desc())
                .label("row_num"),
            )
            .where(Message.content_tsv.op("@@")(ts_query))
            .subquery()
        )

        stmt = (
            select(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
                ranked.c.snippet,
            )
            .join(ranked, ranked.c.conversation_id == Conversation.id)
            .where(ranked.c.row_num == 1)
            .order_by(ranked.c.rank.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "id": row.id,
                "title": row.title,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "snippet": row.snippet,
            }
            for row in result.all()
        ]