from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, filename: str) -> Document:
        document = Document(filename=filename)
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_by_id(self, document_id: int) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list_with_chunk_counts(self) -> list[dict]:
        stmt = (
            select(Document.id, Document.filename, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(Document.id)
            .order_by(Document.id)
        )
        result = await self._session.execute(stmt)
        return [
            {"id": row[0], "filename": row[1], "chunk_count": row[2]}
            for row in result.all()
        ]

    async def delete(self, document_id: int) -> bool:
        document = await self._session.get(Document, document_id)
        if document is None:
            return False
        await self._session.delete(document)
        return True