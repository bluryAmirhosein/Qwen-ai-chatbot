import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

_TITLE_MAX_LEN = 60


class HistoryService:
    """Persists conversations/messages to SQLite and supports full-text search.

    Kept as its own layer (mirrors RagService / SQLiteVectorStore) so ChatService
    and the router never touch raw SQL directly.

    Deletes go through `messages` explicitly (not FK cascade) so the AFTER DELETE
    trigger fires for every row and messages_fts stays in sync.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("HistoryService initialized (db_path=%s)", db_path)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_conversation(self, title: str) -> int:
        now = _now()
        title = _truncate_title(title)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            conversation_id = cur.lastrowid
        logger.info("Created conversation id=%d", conversation_id)
        return conversation_id

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            messages = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": [dict(m) for m in messages],
        }

    def get_messages_for_prompt(self, conversation_id: int) -> list[dict]:
        """Returns messages as {"role", "content"} dicts, ready to feed the model."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over message content (FTS5), deduped by conversation.

        Returns an empty list for a query with no searchable tokens (e.g. only
        punctuation) instead of raising, so callers don't need to know FTS5 exists.
        """
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       snippet(messages_fts, 0, '[', ']', '...', 8) AS snippet
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN conversations c ON c.id = m.conversation_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        seen: set[int] = set()
        results = []
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            results.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "snippet": row["snippet"],
                }
            )
        return results

    def delete_conversation(self, conversation_id: int) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted conversation id=%d", conversation_id)
        return deleted


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_title(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _TITLE_MAX_LEN:
        return text or "New conversation"
    return text[: _TITLE_MAX_LEN - 1].rstrip() + "…"


def _to_fts_query(query: str) -> str:
    """Turns free text into a safe FTS5 MATCH query (AND of quoted tokens)."""
    tokens = [t for t in query.split() if t.strip()]
    if not tokens:
        return ""
    escaped = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens]
    return " AND ".join(escaped)