import csv
import io
import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
import sqlglot
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlglot import exp

logger = logging.getLogger(__name__)

TABLES_SCHEMA = "doc_tables"

_TABLE_PREFIX_PATTERN = re.compile(r"^d(\d+)__")
_MAX_IDENTIFIER_BYTES = 63
_DISTINCT_VALUES_LIMIT = 10
_DISTINCT_VALUE_MAX_CHARS = 40
_SAMPLE_VALUE_MAX_CHARS = 60
_TEXT_TYPES = {"text", "character varying"}

# Functions that can read files, run nested SQL strings, change settings or
# touch sequences. Blocked on top of the read-only transaction as a second
# layer of defense.
_FORBIDDEN_FUNCTIONS = re.compile(
    r"^(pg_|lo_|dblink|set_config|nextval|setval|currval|ts_stat|"
    r"query_to_|cursor_to_|table_to_|schema_to_|database_to_|txid_)",
    re.IGNORECASE,
)


class TableQueryError(ValueError):
    """Raised when a SQL query is rejected, fails or times out."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool

    def to_text(self, max_chars: int = 6000) -> str:
        if not self.rows:
            return "(no rows)"

        header = " | ".join(self.columns)
        lines = [header]
        used = len(header)
        shown = 0
        for row in self.rows:
            line = " | ".join(_format_value(value) for value in row)
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
            shown += 1

        if shown < len(self.rows) or self.truncated:
            lines.append(f"[truncated: only the first {shown} rows are shown, more rows exist]")
        return "\n".join(lines)


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(round(value, 4), "f") if value.is_finite() else str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def frames_from_csv_text(csv_text: str) -> dict[str | None, pd.DataFrame]:
    """Parse CSV text into a single dataframe keyed by None (no sheet name)."""
    try:
        delimiter = csv.Sniffer().sniff(csv_text[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return {None: pd.read_csv(io.StringIO(csv_text), sep=delimiter)}


def frames_from_excel_bytes(data: bytes) -> dict[str | None, pd.DataFrame]:
    """Parse every sheet of a workbook into dataframes keyed by sheet name."""
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
    return {str(name): frame for name, frame in sheets.items()}


class TableStore:
    """Stores tabular uploads as real Postgres tables so exact queries work.

    Every ingested CSV/Excel document becomes one table (CSV) or one table
    per non-empty sheet (Excel) inside the `doc_tables` schema. Table names
    look like `d<document_id>__<slug>`, so the owning document can always be
    derived from the name and no extra registry table or migration is
    needed. Tables of different documents live side by side, so a single
    query can join them.

    Writes happen inside the caller's session transaction: if ingestion
    fails afterwards and the session is rolled back, the tables disappear
    together with the document row.

    Queries are validated with sqlglot (single SELECT, only the tables of the
    selected documents, no dangerous functions) and then executed inside a
    savepoint that is always rolled back, with a read-only transaction, a
    statement timeout and a restricted search_path. If `readonly_role` is
    given, the query additionally runs under that database role.
    """

    def __init__(self, readonly_role: str | None = None, query_timeout_ms: int = 10_000):
        if readonly_role is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", readonly_role):
            raise ValueError("readonly_role must be a plain SQL identifier")
        self._readonly_role = readonly_role
        self._timeout_ms = int(query_timeout_ms)

    async def save_document(
        self,
        session: AsyncSession,
        document_id: int,
        source_name: str,
        frames: dict[str | None, pd.DataFrame],
    ) -> list[str]:
        """Write the dataframes of one document. Returns the created table names."""
        stem = os.path.splitext(source_name)[0]
        await self._run_raw(session, f"CREATE SCHEMA IF NOT EXISTS {TABLES_SCHEMA}")
        connection = await session.connection()

        created: list[str] = []
        for sheet, frame in frames.items():
            cleaned = self._clean_frame(frame)
            if cleaned.empty:
                continue
            name = self._table_name(stem, sheet, document_id, created)
            await connection.run_sync(self._write_frame, name, cleaned)
            created.append(name)

        if not created:
            raise ValueError("No table data found in the uploaded file")

        logger.info("Saved tables %s for document id=%d", created, document_id)
        return created

    async def delete_document(self, session: AsyncSession, document_id: int) -> None:
        by_document = await self._tables_by_document(session)
        for table in by_document.get(document_id, []):
            await self._run_raw(session, f"DROP TABLE IF EXISTS {TABLES_SCHEMA}.{_quote(table)} CASCADE")

    async def document_ids_with_tables(self, session: AsyncSession) -> set[int]:
        return set(await self._tables_by_document(session))

    async def describe(
        self,
        session: AsyncSession,
        document_ids: list[int] | None = None,
        sample_rows: int = 3,
    ) -> str:
        """Build a schema description (tables, columns, types, samples) for a prompt."""
        tables = await self._select_tables(session, document_ids)
        blocks = [await self._describe_table(session, table, sample_rows) for table in tables]
        return "\n\n".join(blocks)

    async def query(
        self,
        session: AsyncSession,
        sql: str,
        document_ids: list[int] | None = None,
        max_rows: int = 200,
    ) -> QueryResult:
        """Run a single read-only SELECT across the tables of the given documents."""
        tables = await self._select_tables(session, document_ids)
        statement = self._validate_sql(sql, set(tables))

        nested = await session.begin_nested()
        try:
            await self._apply_limits(session)
            try:
                result = await self._run_raw(session, statement)
                columns = list(result.keys())
                rows = result.fetchmany(max_rows + 1)
            except DBAPIError as exc:
                raise TableQueryError(self._short_error(exc)) from exc
        finally:
            # Always roll back: it discards the SET LOCAL changes and recovers
            # the outer transaction if the statement failed.
            await nested.rollback()

        truncated = len(rows) > max_rows
        return QueryResult(columns=columns, rows=[tuple(r) for r in rows[:max_rows]], truncated=truncated)

    async def _apply_limits(self, session: AsyncSession) -> None:
        await self._run_raw(
            session,
            "SELECT "
            f"set_config('statement_timeout', '{self._timeout_ms}', true), "
            f"set_config('search_path', '{TABLES_SCHEMA}', true), "
            "set_config('transaction_read_only', 'on', true)",
        )
        if self._readonly_role:
            await self._run_raw(session, f'SET LOCAL ROLE "{self._readonly_role}"')

    @staticmethod
    async def _run_raw(session: AsyncSession, sql: str):
        # exec_driver_sql avoids SQLAlchemy's bind-parameter parsing, so
        # colons or percent signs inside the SQL are passed through untouched.
        connection = await session.connection()
        return await connection.exec_driver_sql(sql)

    @staticmethod
    def _short_error(exc: DBAPIError) -> str:
        message = str(getattr(exc, "orig", exc)).strip()
        return message.splitlines()[0] if message else "Query failed"

    async def _tables_by_document(self, session: AsyncSession) -> dict[int, list[str]]:
        result = await session.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            ),
            {"schema": TABLES_SCHEMA},
        )
        by_document: dict[int, list[str]] = {}
        for (name,) in result.all():
            match = _TABLE_PREFIX_PATTERN.match(name)
            if match:
                by_document.setdefault(int(match.group(1)), []).append(name)
        return by_document

    async def _select_tables(self, session: AsyncSession, document_ids: list[int] | None) -> list[str]:
        by_document = await self._tables_by_document(session)
        wanted = sorted(set(document_ids)) if document_ids else sorted(by_document)
        tables = [table for doc_id in wanted for table in by_document.get(doc_id, [])]
        if not tables:
            raise TableQueryError("No table data found for the selected documents")
        return tables

    async def _describe_table(self, session: AsyncSession, table: str, sample_rows: int) -> str:
        qualified = f"{TABLES_SCHEMA}.{_quote(table)}"

        column_result = await session.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = :schema AND table_name = :table "
                "ORDER BY ordinal_position"
            ),
            {"schema": TABLES_SCHEMA, "table": table},
        )
        columns = column_result.all()

        count_result = await self._run_raw(session, f"SELECT COUNT(*) FROM {qualified}")
        row_count = count_result.scalar_one()

        distinct_parts: list[str] = []
        for index, (name, data_type) in enumerate(columns):
            if data_type in _TEXT_TYPES:
                column = _quote(name)
                distinct_parts.append(
                    f"(SELECT array_agg(v) FROM (SELECT DISTINCT {column} AS v FROM {qualified} "
                    f"WHERE {column} IS NOT NULL LIMIT {_DISTINCT_VALUES_LIMIT + 1}) s{index}) AS c{index}"
                )
        distinct_values: dict[int, list[str]] = {}
        if distinct_parts:
            distinct_result = await self._run_raw(session, "SELECT " + ", ".join(distinct_parts))
            row = distinct_result.one()
            keys = list(distinct_result.keys())
            for key, values in zip(keys, row):
                if values and len(values) <= _DISTINCT_VALUES_LIMIT:
                    distinct_values[int(key[1:])] = sorted(
                        str(v)[:_DISTINCT_VALUE_MAX_CHARS] for v in values
                    )

        lines = [f'Table "{table}" ({row_count} rows)', "Columns:"]
        for index, (name, data_type) in enumerate(columns):
            line = f'- "{name}" ({data_type})'
            if index in distinct_values:
                line += " values: " + ", ".join(distinct_values[index])
            lines.append(line)

        sample_result = await self._run_raw(session, f"SELECT * FROM {qualified} LIMIT {int(sample_rows)}")
        keys = list(sample_result.keys())
        for sample in sample_result.fetchall():
            rendered = ", ".join(
                f"{key}={str(value)[:_SAMPLE_VALUE_MAX_CHARS]}"
                for key, value in zip(keys, sample)
                if value is not None
            )
            lines.append(f"Sample: {rendered}")
        return "\n".join(lines)

    @staticmethod
    def _validate_sql(sql: str, allowed_tables: set[str]) -> str:
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise TableQueryError("Empty query")

        try:
            parsed = [tree for tree in sqlglot.parse(statement, read="postgres") if tree is not None]
        except sqlglot.errors.SqlglotError as exc:
            raise TableQueryError(f"SQL syntax error: {exc}") from exc

        if len(parsed) != 1:
            raise TableQueryError("Exactly one SELECT statement is allowed")
        tree = parsed[0]

        if not isinstance(tree, (exp.Select, exp.Union)):
            raise TableQueryError("Only SELECT queries are allowed")
        if tree.find(exp.Into):
            raise TableQueryError("SELECT ... INTO is not allowed")
        if tree.find(exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Command):
            raise TableQueryError("Only SELECT queries are allowed")

        lowered_allowed = {name.lower() for name in allowed_tables}
        cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
        for table in tree.find_all(exp.Table):
            schema = (table.db or "").lower()
            name = (table.name or "").lower()
            if table.catalog:
                raise TableQueryError("Cross-database references are not allowed")
            if not schema and name in cte_names:
                continue
            if schema not in ("", TABLES_SCHEMA) or name not in lowered_allowed:
                available = ", ".join(sorted(allowed_tables))
                raise TableQueryError(
                    f'Table "{table.name}" is not available. Available tables: {available}'
                )

        for function in tree.find_all(exp.Func):
            name = function.name if isinstance(function, exp.Anonymous) else function.sql_name()
            if name and _FORBIDDEN_FUNCTIONS.match(name):
                raise TableQueryError(f"Function {name} is not allowed")

        return statement

    @staticmethod
    def _write_frame(sync_connection, name: str, frame: pd.DataFrame) -> None:
        # asyncpg accepts at most 32767 bind parameters per statement.
        chunksize = max(1, 30_000 // max(1, len(frame.columns)))
        frame.to_sql(
            name,
            sync_connection,
            schema=TABLES_SCHEMA,
            index=False,
            if_exists="fail",
            method="multi",
            chunksize=chunksize,
        )

    @classmethod
    def _clean_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.dropna(how="all").dropna(axis=1, how="all").copy()
        if cleaned.empty:
            return cleaned

        cleaned.columns = cls._unique_column_names(list(cleaned.columns))
        cleaned = cleaned.convert_dtypes()

        # Columns with mixed Python types (numbers and text in one column)
        # cannot be sent to a TEXT column as-is; store them as strings.
        for column in cleaned.columns:
            if cleaned[column].dtype == object:
                cleaned[column] = (
                    cleaned[column]
                    .map(lambda v: None if pd.isna(v) else str(v).replace("\x00", ""))
                    .astype(object)
                )
        return cleaned

    @classmethod
    def _unique_column_names(cls, columns: list) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for index, column in enumerate(columns, start=1):
            base = " ".join(str(column).split()) or f"column_{index}"
            candidate = cls._fit_identifier(base)
            counter = 1
            while candidate in seen:
                counter += 1
                suffix = f"_{counter}"
                candidate = cls._fit_identifier(base, reserve=len(suffix)) + suffix
            seen.add(candidate)
            names.append(candidate)
        return names

    @classmethod
    def _table_name(cls, stem: str, sheet: str | None, document_id: int, existing: list[str]) -> str:
        def slug(value: str) -> str:
            return re.sub(r"\W+", "_", value.lower(), flags=re.UNICODE).strip("_")

        parts = [slug(stem) or "table"]
        if sheet is not None:
            parts.append(slug(sheet) or "sheet")
        base = f"d{document_id}__" + "_".join(parts)

        candidate = cls._fit_identifier(base)
        counter = 1
        while candidate in existing:
            counter += 1
            suffix = f"_{counter}"
            candidate = cls._fit_identifier(base, reserve=len(suffix)) + suffix
        return candidate

    @staticmethod
    def _fit_identifier(value: str, reserve: int = 0) -> str:
        # Postgres limits identifiers to 63 bytes, and Persian letters take
        # two bytes each in UTF-8, so trim by encoded length.
        limit = _MAX_IDENTIFIER_BYTES - reserve
        while len(value.encode("utf-8")) > limit:
            value = value[:-1]
        return value