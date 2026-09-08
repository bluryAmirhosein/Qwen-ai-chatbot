import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    filename: str
    content: str
    score: float


class SQLiteVectorStore:
    """Stores chunk text + embeddings in SQLite and does brute-force cosine search.

    Embeddings are stored as raw float32 blobs. Search loads all vectors for
    the corpus into memory and ranks them with a single numpy matrix
    multiply — no vector extension required. This is fine up to tens of
    thousands of chunks, which comfortably covers a single-user / small-team
    Word-document knowledge base. If the corpus grows much larger, swap this
    class for one backed by sqlite-vec or FAISS without touching RagService,
    since it's the only place that talks to storage.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def add_document(self, filename: str, chunks: list[str], embeddings: np.ndarray) -> tuple[int, int]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        with self._connect() as conn:
            cur = conn.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
            document_id = cur.lastrowid

            conn.executemany(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (?, ?, ?)",
                [
                    (document_id, chunk, embedding.astype(np.float32).tobytes())
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )
            conn.commit()

        logger.info("Stored document '%s' (id=%d) with %d chunks", filename, document_id, len(chunks))
        return document_id, len(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.filename,
                       chunks.content, chunks.embedding
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                """
            ).fetchall()

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

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT documents.id, documents.filename, COUNT(chunks.id)
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.id
                GROUP BY documents.id
                ORDER BY documents.id
                """
            ).fetchall()
        return [{"id": r[0], "filename": r[1], "chunk_count": r[2]} for r in rows]

    def delete_document(self, document_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return cur.rowcount > 0