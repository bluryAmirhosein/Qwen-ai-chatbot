import logging

from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_SUPPORTED_TYPES = {".txt", ".md", ".pdf", ".docx", ".csv"}


class UnsupportedFileTypeError(ValueError):
    pass


class FileService:
    """Extracts plain text from uploaded files so it can be fed to the
    model as context. Kept separate from ChatService so new file types
    can be added without touching conversation logic.
    """

    def extract_text(self, file: UploadFile) -> str:
        suffix = self._get_suffix(file.filename)

        if suffix not in _SUPPORTED_TYPES:
            raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

        raw = file.file.read()

        if suffix in (".txt", ".md"):
            return raw.decode("utf-8", errors="ignore")

        if suffix == ".pdf":
            return self._extract_pdf(raw)

        if suffix == ".docx":
            return self._extract_docx(raw)

        if suffix == ".csv":
            return self._extract_csv(raw)

        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

    @staticmethod
    def is_csv(filename: str | None) -> bool:
        return FileService._get_suffix(filename) == ".csv"

    @staticmethod
    def _get_suffix(filename: str | None) -> str:
        if not filename or "." not in filename:
            return ""
        return "." + filename.rsplit(".", 1)[-1].lower()

    @staticmethod
    def _extract_pdf(raw: bytes) -> str:
        import io

        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _extract_docx(raw: bytes) -> str:
        import io

        document = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in document.paragraphs)

    @staticmethod
    def _extract_csv(raw: bytes) -> str:
        # utf-8-sig strips a leading BOM, which is common in CSVs exported
        # from Excel. The result is handed off as-is to CsvChunker, which
        # does the actual row parsing.
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            logger.warning("CSV file was not valid UTF-8; falling back to latin-1")
            return raw.decode("latin-1", errors="replace")