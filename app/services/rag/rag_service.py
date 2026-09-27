import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.chunk_repository import ChunkRepository, SearchResult
from app.repository.document_repository import DocumentRepository
from app.services.file_service import FileService
from app.services.rag.chunker import TextChunker
from app.services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RagService:
    """Ingests Word documents and retrieves relevant chunks for a query.

    Sits alongside ChatService/ModelService as its own layer: ingestion
    (extract -> chunk -> embed -> store) and retrieval (embed query ->
    search -> format context) both happen here, so the API layer and
    ChatService stay unaware of chunking/embedding/storage details.
    """

    def __init__(
        self,
        session: AsyncSession,
        file_service: FileService,
        embedding_service: EmbeddingService,
        chunker: TextChunker | None = None,
    ):
        self._session = session
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)
        self._file_service = file_service
        self._embedding_service = embedding_service
        self._chunker = chunker or TextChunker()

    async def ingest_file(self, file) -> tuple[str, int, int]:
        """Extract, chunk, embed and store an uploaded file.

        Returns (filename, document_id, chunk_count).
        """
        filename = file.filename or "unknown"
        text = self._file_service.extract_text(file)
        chunks = self._chunker.split(text)

        if not chunks:
            raise ValueError("No extractable text found in the uploaded file")

        embeddings = self._embedding_service.embed_documents(chunks)
        document = await self._documents.create(filename)
        chunk_count = await self._chunks.add_many(document.id, chunks, embeddings)
        await self._session.commit()

        logger.info("Stored document '%s' (id=%d) with %d chunks", filename, document.id, chunk_count)
        return filename, document.id, chunk_count

    async def retrieve(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_embedding = self._embedding_service.embed_query(query)
        return await self._chunks.search(query_embedding, top_k=top_k)

    async def retrieve_context(self, query: str, top_k: int = 4) -> str | None:
        """Retrieve chunks and format them as a single context block for the prompt."""
        results = await self.retrieve(query, top_k=top_k)
        if not results:
            return None

        blocks = [f"[Source: {r.filename}]\n{r.content}" for r in results]
        return "\n\n---\n\n".join(blocks)

    async def list_documents(self) -> list[dict]:
        return await self._documents.list_with_chunk_counts()

    async def delete_document(self, document_id: int) -> bool:
        deleted = await self._documents.delete(document_id)
        await self._session.commit()
        return deleted