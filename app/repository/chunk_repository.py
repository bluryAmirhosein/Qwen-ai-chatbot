from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    filename: str
    content: str
    score: float


class ChunkRepository:
    """Stores chunk text + embeddings and does brute-force cosine search.

    Same approach as the old SQLiteVectorStore: all vectors for the corpus
    are loaded into memory and ranked with a single numpy matmul, no vector
    extension required. Fine up to tens of thousands of chunks. If the
    corpus grows much larger, swap this for pgvector (vector column + an
    ivfflat/hnsw index) without touching RagService, since this repository
    is the only place that talks to chunk storage.

    Search can optionally be restricted to a set of document ids, in which
    case only those documents' chunks are loaded and ranked.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def add_many(self, document_id: int, chunks: list[str], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self._session.add_all(
            [
                Chunk(
                    document_id=document_id,
                    content=chunk,
                    embedding=embedding.astype(np.float32).tobytes(),
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
        )
        await self._session.flush()
        return len(chunks)

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        document_ids: list[int] | None = None,
    ) -> list[SearchResult]:
        stmt = select(
            Chunk.id, Chunk.document_id, Document.filename, Chunk.content, Chunk.embedding
        ).join(Document, Document.id == Chunk.document_id)

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        result = await self._session.execute(stmt)
        rows = result.all()

        if not rows:
            return []

        vectors = np.vstack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        # Vectors are pre-normalized at embed time, so a dot product is
        # equivalent to cosine similarity.
        scores = vectors @ query_embedding.astype(np.float32)
        top_indices = np.argsort(-scores)[:top_k]

        return [
            SearchResult(
                chunk_id=rows[i][0],
                document_id=rows[i][1],
                filename=rows[i][2],
                content=rows[i][3],
                score=float(scores[i]),
            )
            for i in top_indices
        ]