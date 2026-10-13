import asyncio
import logging
import re
import threading
from dataclasses import dataclass

from app.services.model_service import ModelService
from app.services.rag.rag_service import RagService
from app.services.tables.table_store import TableQueryError

logger = logging.getLogger(__name__)

_SQL_SYSTEM_PROMPT = (
    "You translate questions into PostgreSQL queries over the tables described by the user. "
    "Reply with exactly one SELECT statement and nothing else: no explanation, no markdown. "
    "If the question cannot be answered from these tables, reply with exactly NO_QUERY.\n\n"
    "Rules:\n"
    "- Use only the listed tables and columns. Wrap every table and column name in double quotes, "
    "exactly as written in the schema.\n"
    "- Tables come from different files; JOIN them when the question spans several files.\n"
    "- Prefer aggregates (COUNT, SUM, AVG, MIN, MAX, GROUP BY) over listing rows when the question "
    "asks for totals, counts or comparisons.\n"
    "- Select only the columns needed. Add ORDER BY for rankings and LIMIT 50 for plain row listings.\n"
    "- Use ILIKE with % for partial text matches. When a column lists its possible values in the "
    "schema, use those exact values.\n"
    "- Cast text columns explicitly if you need to do arithmetic or date comparison on them."
)

_MAX_HISTORY_MESSAGES = 4
_MAX_HISTORY_CHARS = 500


@dataclass
class TableQueryOutcome:
    """What the table query step produced for one question.

    Either `result_text` is set (the query ran), or `error` is set (all
    attempts failed). `table_document_ids` lists the documents whose tables
    were available, so ChatService can skip semantic retrieval for them.
    """

    table_document_ids: list[int]
    sql: str | None
    result_text: str | None
    error: str | None


class TableQueryService:
    """Answers table questions exactly by letting the model write SQL.

    Flow: describe the selected documents' tables, ask the model for one
    SELECT, run it through RagService (validated, read-only, time-limited),
    and on failure feed the error back to the model for a corrected query,
    up to `max_attempts` times. The model never sees raw tables, only the
    schema and the exact query result.
    """

    def __init__(
        self,
        model_service: ModelService,
        rag_service: RagService,
        max_attempts: int = 3,
        max_result_chars: int = 6000,
    ):
        self._model_service = model_service
        self._rag_service = rag_service
        self._max_attempts = max_attempts
        self._max_result_chars = max_result_chars

    async def run(
        self,
        question: str,
        document_ids: list[int] | None = None,
        history: list[dict] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TableQueryOutcome | None:
        """Return None when there are no tables or the question is not a table question."""
        table_ids = await self._rag_service.table_document_ids(document_ids)
        if not table_ids:
            return None

        schema = await self._rag_service.describe_tables(table_ids)
        messages = self._build_messages(question, schema, history or [])

        last_sql: str | None = None
        last_error: str | None = None

        for attempt in range(1, self._max_attempts + 1):
            raw = await asyncio.to_thread(
                self._model_service.generate,
                messages,
                enable_thinking=False,
                max_new_tokens_multiplier=1.0,
                cancel_event=cancel_event,
            )
            sql = self._extract_sql(raw)

            if sql is None:
                if attempt == 1:
                    logger.info("Model declined to write a table query for this question")
                    return None
                break

            last_sql = sql
            logger.info("Table query attempt %d: %s", attempt, sql)

            try:
                result = await self._rag_service.run_table_query(sql, table_ids)
            except TableQueryError as exc:
                last_error = str(exc)
                logger.warning("Table query attempt %d failed: %s", attempt, last_error)
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"That query failed with this error:\n{last_error}\n"
                            "Write a corrected single SELECT statement only."
                        ),
                    },
                ]
                continue

            return TableQueryOutcome(
                table_document_ids=table_ids,
                sql=sql,
                result_text=result.to_text(self._max_result_chars),
                error=None,
            )

        return TableQueryOutcome(
            table_document_ids=table_ids,
            sql=last_sql,
            result_text=None,
            error=last_error or "The model did not produce a valid query",
        )

    @staticmethod
    def _build_messages(question: str, schema: str, history: list[dict]) -> list[dict]:
        recent = [
            {"role": m["role"], "content": str(m["content"])[:_MAX_HISTORY_CHARS]}
            for m in history[-_MAX_HISTORY_MESSAGES:]
            if m.get("role") in ("user", "assistant")
        ]
        return [
            {"role": "system", "content": _SQL_SYSTEM_PROMPT},
            *recent,
            {"role": "user", "content": f"Tables:\n\n{schema}\n\nQuestion: {question}\n\nSQL:"},
        ]

    @staticmethod
    def _extract_sql(raw: str) -> str | None:
        """Pull the SQL out of a model reply. Returns None for NO_QUERY."""
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned.upper().startswith("NO_QUERY"):
            return None

        fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

        start = re.search(r"\b(select|with)\b", cleaned, flags=re.IGNORECASE)
        if start:
            cleaned = cleaned[start.start():]
        return cleaned.strip()