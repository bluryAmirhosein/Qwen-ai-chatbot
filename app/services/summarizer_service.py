# app/services/summarizer_service.py

import logging

from app.config import Settings
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

_GENERAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a text summarization assistant. Summarize the user's text accurately "
    "and coherently in at most approximately {max_length} words, without adding any "
    "information that isn't in the original text. Reply in the same language as the "
    "input text. Return only the summary, with no preamble or extra commentary."
)

_QUERY_FOCUSED_SYSTEM_PROMPT_TEMPLATE = (
    "You are a text summarization assistant. The user provides a text and a key "
    "query/phrase. Identify only the parts of the text relevant to that query, and "
    "based on them, produce a focused summary of at most approximately {max_length} "
    "words. If the text has no relevant content for the query, say so explicitly "
    "instead of summarizing unrelated content. Reply in the same language as the "
    "input text. Return only the final output, with no preamble."
)


class SummarizerService:
    """Business logic layer for text summarization.

    Mirrors ChatService's separation from ModelService: prompt construction
    and length bookkeeping live here, while model loading/inference stay in
    ModelService — reused as-is, no duplicated generation logic.
    """

    MAX_INPUT_CHARS = 12000  # Prevent context overflow for very long texts

    def __init__(self, model_service: ModelService, settings: Settings):
        self._model_service = model_service
        self._settings = settings

    def summarize(self, text: str, max_length: int = 150) -> dict:
        text = self._truncate(text.strip())
        messages = self._build_general_messages(text, max_length)
        summary = self._generate(messages, max_length)
        return self._build_result(text, summary)

    def summarize_by_query(self, text: str, query: str, max_length: int = 150) -> dict:
        text = self._truncate(text.strip())
        messages = self._build_query_messages(text, query, max_length)
        summary = self._generate(messages, max_length)
        return self._build_result(text, summary)

    def _generate(self, messages: list[dict], max_length: int) -> str:
        return self._model_service.generate(
            messages,
            enable_thinking=False,
            max_new_tokens_multiplier=self._tokens_multiplier_for(max_length),
        )

    def _tokens_multiplier_for(self, max_length_words: int) -> float:
        """Converts max_length (in words) to an appropriate multiplier for ModelService.generate."""
        approx_tokens = max(int(max_length_words * 1.8), 64)
        return approx_tokens / self._settings.max_new_tokens

    def _truncate(self, text: str) -> str:
        if len(text) > self.MAX_INPUT_CHARS:
            logger.warning(
                "Summarizer input truncated from %d to %d chars", len(text), self.MAX_INPUT_CHARS
            )
            return text[: self.MAX_INPUT_CHARS]
        return text

    def _build_general_messages(self, text: str, max_length: int) -> list[dict]:
        return [
            {
                "role": "system",
                "content": _GENERAL_SYSTEM_PROMPT_TEMPLATE.format(max_length=max_length),
            },
            {"role": "user", "content": text},
        ]

    def _build_query_messages(self, text: str, query: str, max_length: int) -> list[dict]:
        return [
            {
                "role": "system",
                "content": _QUERY_FOCUSED_SYSTEM_PROMPT_TEMPLATE.format(max_length=max_length),
            },
            {"role": "user", "content": f"Key query: {query}\n\nText:\n{text}"},
        ]

    def _build_result(self, original_text: str, summary: str) -> dict:
        return {
            "summary": summary,
            "original_length": len(original_text.split()),
            "summary_length": len(summary.split()),
        }