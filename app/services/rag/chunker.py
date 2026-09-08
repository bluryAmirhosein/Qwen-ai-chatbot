import re


class TextChunker:
    """Splits extracted document text into overlapping chunks for embedding.

    Chunking is done on paragraph boundaries first, then packed greedily up
    to `chunk_size` characters, with `chunk_overlap` characters repeated at
    the start of the next chunk to preserve context across chunk edges.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + 2 + len(para) <= self._chunk_size:
                current = f"{current}\n\n{para}"
            else:
                chunks.append(current.strip())
                overlap_tail = current[-self._chunk_overlap :]
                current = f"{overlap_tail}\n\n{para}".strip()

            # A single paragraph longer than chunk_size: hard-split it.
            while len(current) > self._chunk_size:
                chunks.append(current[: self._chunk_size].strip())
                current = current[self._chunk_size - self._chunk_overlap :]

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in raw if p.strip()]