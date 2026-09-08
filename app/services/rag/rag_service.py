import logging

from app.services.file_service import FileService
from app.services.rag.chunker import TextChunker
from app.services.rag.embedding_service import EmbeddingService
from app.services.rag.vector_store import SearchResult, SQLiteVectorStore

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
        file_service: FileService,
        embedding_service: EmbeddingService,
        vector_store: SQLiteVectorStore,
        chunker: TextChunker | None = None,
    ):
        self._file_service = file_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunker = chunker or TextChunker()

    def ingest_file(self, file) -> tuple[str, int, int]:
        """Extract, chunk, embed and store an uploaded file.

        Returns (filename, document_id, chunk_count).
        """
        filename = file.filename or "unknown"
        text = self._file_service.extract_text(file)
        chunks = self._chunker.split(text)

        if not chunks:
            raise ValueError("No extractable text found in the uploaded file")

        embeddings = self._embedding_service.embed_documents(chunks)
        document_id, chunk_count = self._vector_store.add_document(filename, chunks, embeddings)
        return filename, document_id, chunk_count

    def retrieve(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_embedding = self._embedding_service.embed_query(query)
        return self._vector_store.search(query_embedding, top_k=top_k)

    def retrieve_context(self, query: str, top_k: int = 4) -> str | None:
        """Retrieve chunks and format them as a single context block for the prompt."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return None

        blocks = [f"[Source: {r.filename}]\n{r.content}" for r in results]
        return "\n\n---\n\n".join(blocks)

    def list_documents(self) -> list[dict]:
        return self._vector_store.list_documents()

    def delete_document(self, document_id: int) -> bool:
        return self._vector_store.delete_document(document_id)