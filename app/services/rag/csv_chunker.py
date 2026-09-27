import csv
import io


class CsvChunker:
    """Formats CSV rows into chunk-ready text for row-level RAG retrieval.

    Unlike TextChunker, which packs paragraphs into size-bound blocks, CSV
    data is chunked one row per chunk by default. Each row becomes a small,
    self-contained "document" (column: value pairs), which keeps embeddings
    precise for lookup-style queries ("what is X for row/order 42?").

    Aggregation-style queries (sums, counts, averages across the whole
    table) are NOT well served by this approach, since retrieval only
    returns the top-k most similar rows rather than the full table.
    """

    def __init__(self, rows_per_chunk: int = 1):
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be at least 1")
        self._rows_per_chunk = rows_per_chunk

    def split(self, csv_text: str) -> list[str]:
        reader = csv.DictReader(io.StringIO(csv_text))
        if reader.fieldnames is None:
            return []

        formatted_rows = [
            self._format_row(index, row) for index, row in enumerate(reader, start=1)
        ]

        if self._rows_per_chunk == 1:
            return formatted_rows

        chunks = []
        for i in range(0, len(formatted_rows), self._rows_per_chunk):
            batch = formatted_rows[i : i + self._rows_per_chunk]
            chunks.append("\n\n".join(batch))
        return chunks

    @staticmethod
    def _format_row(row_number: int, row: dict) -> str:
        fields = " | ".join(f"{key}: {value}" for key, value in row.items() if key)
        return f"Row {row_number}: {fields}"