import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.chunk_repository import ChunkRepository, SearchResult
from app.repository.document_repository import DocumentRepository
from app.services.file_service import FileService
from app.services.rag.chunker import TextChunker
from app.services.rag.csv_chunker import CsvChunker
from app.services.rag.embedding_service import EmbeddingService
from app.services.rag.excel_chunker import ExcelChunker
from app.services.tables.table_store import (
    QueryResult,
    TableStore,
    frames_from_csv_text,
    frames_from_excel_bytes,
)

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """Raised when one or more requested document ids don't exist."""

    def __init__(self, missing_ids: list[int]):
        self.missing_ids = missing_ids
        super().__init__(f"Document(s) not found: {', '.join(str(i) for i in missing_ids)}")


class RagService:
    """Ingests documents (text-based, CSV and Excel) and retrieves relevant
    chunks for a query.

    Ingestion branches on file type: plain-text-derived documents (.txt/
    .md/.pdf/.docx) go through TextChunker's paragraph-packing logic, .csv
    files go through CsvChunker and .xlsx/.xlsm workbooks go through
    ExcelChunker (one chunk per row, with sheet name and row number).

    CSV and Excel files are additionally stored as real Postgres tables
    through TableStore, so exact questions (lookups, filters, totals, joins
    across files) can be answered by running a SQL query instead of relying
    on top-k semantic retrieval. The table helpers at the bottom of this
    class expose that capability to TableQueryService.

    Retrieval can be scoped to specific documents via `document_ids`. When
    omitted (or empty), all ingested documents are searched.
    """

    def __init__(
        self,
        session: AsyncSession,
        file_service: FileService,
        embedding_service: EmbeddingService,
        table_store: TableStore,
        chunker: TextChunker | None = None,
        csv_chunker: CsvChunker | None = None,
        excel_chunker: ExcelChunker | None = None,
    ):
        self._session = session
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)
        self._file_service = file_service
        self._embedding_service = embedding_service
        self._table_store = table_store
        self._chunker = chunker or TextChunker()
        self._csv_chunker = csv_chunker or CsvChunker()
        self._excel_chunker = excel_chunker or ExcelChunker()

    async def ingest_file(self, file) -> tuple[str, int, int]:
        """Extract, chunk, embed and store an uploaded file.

        Returns (filename, document_id, chunk_count).
        """
        filename = file.filename or "unknown"
        frames = None

        if self._file_service.is_excel(filename):
            data = self._file_service.read_bytes(file)
            try:
                chunks = await asyncio.to_thread(self._excel_chunker.split, data)
                frames = await asyncio.to_thread(frames_from_excel_bytes, data)
            except Exception as exc:
                logger.exception("Failed to parse Excel file '%s'", filename)
                raise ValueError(f"Could not read the Excel file: {exc}") from exc
        else:
            text = self._file_service.extract_text(file)
            if self._file_service.is_csv(filename):
                chunks = self._csv_chunker.split(text)
                try:
                    frames = await asyncio.to_thread(frames_from_csv_text, text)
                except Exception as exc:
                    logger.exception("Failed to parse CSV file '%s'", filename)
                    raise ValueError(f"Could not read the CSV file: {exc}") from exc
            else:
                chunks = self._chunker.split(text)

        if not chunks:
            raise ValueError("No extractable text found in the uploaded file")

        # Embedding 1000+ chunks on CPU is slow; keep it off the event loop.
        embeddings = await asyncio.to_thread(self._embedding_service.embed_documents, chunks)
        document = await self._documents.create(filename)
        chunk_count = await self._chunks.add_many(document.id, chunks, embeddings)

        if frames is not None:
            try:
                await self._table_store.save_document(self._session, document.id, filename, frames)
            except Exception as exc:
                await self._session.rollback()
                logger.exception("Failed to store tables for '%s'", filename)
                raise ValueError(f"Could not store table data: {exc}") from exc

        await self._session.commit()

        logger.info("Stored document '%s' (id=%d) with %d chunks", filename, document.id, chunk_count)
        return filename, document.id, chunk_count

    async def _validate_document_ids(self, document_ids: list[int]) -> None:
        existing = {doc["id"] for doc in await self._documents.list_with_chunk_counts()}
        missing = sorted(set(document_ids) - existing)
        if missing:
            raise DocumentNotFoundError(missing)

    async def retrieve(
        self,
        query: str,
        top_k: int = 4,
        document_ids: list[int] | None = None,
    ) -> list[SearchResult]:
        if document_ids:
            await self._validate_document_ids(document_ids)

        query_embedding = self._embedding_service.embed_query(query)
        return await self._chunks.search(query_embedding, top_k=top_k, document_ids=document_ids)

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        document_ids: list[int] | None = None,
    ) -> str | None:
        """Retrieve chunks and format them as a single context block for the prompt."""
        results = await self.retrieve(query, top_k=top_k, document_ids=document_ids)
        if not results:
            return None

        blocks = [f"[Source: {r.filename}]\n{r.content}" for r in results]
        return "\n\n---\n\n".join(blocks)

    async def list_documents(self) -> list[dict]:
        return await self._documents.list_with_chunk_counts()

    async def delete_document(self, document_id: int) -> bool:
        deleted = await self._documents.delete(document_id)
        if deleted:
            await self._table_store.delete_document(self._session, document_id)
        await self._session.commit()
        return deleted

    async def table_document_ids(self, document_ids: list[int] | None = None) -> list[int]:
        """Ids of documents that have SQL tables, optionally limited to `document_ids`."""
        available = await self._table_store.document_ids_with_tables(self._session)
        if document_ids:
            return sorted(set(document_ids) & available)
        return sorted(available)

    async def describe_tables(self, document_ids: list[int] | None = None) -> str:
        return await self._table_store.describe(self._session, document_ids)

    async def run_table_query(
        self,
        sql: str,
        document_ids: list[int] | None = None,
        max_rows: int = 200,
    ) -> QueryResult:
        return await self._table_store.query(self._session, sql, document_ids, max_rows)