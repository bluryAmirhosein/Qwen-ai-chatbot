import logging
from dataclasses import dataclass

from app.schemas.chat import (
    _PERSONALITY_INSTRUCTIONS,
    Language,
    Personality,
    ThinkingMode,
)
from app.services.file_service import FileService
from app.services.history_service import HistoryService
from app.services.model_service import ModelService
from app.services.rag.rag_service import RagService
from app.services.web_search import WebSearchService

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# (enable_thinking, max_new_tokens_multiplier, extra_instruction)
_THINKING_MODE_CONFIG: dict[ThinkingMode, tuple[bool, float, str | None]] = {
    ThinkingMode.FAST: (False, 0.5, None),
    ThinkingMode.BALANCED: (True, 1.0, None),
    ThinkingMode.DEEP: (
        True,
        2.0,
        "Think carefully and thoroughly, consider multiple angles and edge cases "
        "before giving your final answer.",
    ),
}

_LANGUAGE_NAMES: dict[Language, str] = {
    Language.PERSIAN: "Persian (Farsi)",
    Language.ENGLISH: "English",
}


class ConversationNotFoundError(Exception):
    """Raised when the caller passes a conversation_id that doesn't exist (or was deleted)."""


@dataclass
class ChatResult:
    response: str
    conversation_id: int | None


class ChatService:
    """Business logic layer sitting between the API and the model layer.

    Keeping this separate from ModelService means conversation logic
    (message building, thinking mode, personality, web search / file /
    RAG context injection, history persistence) can be unit tested with a
    fake or mocked ModelService, without ever touching real model weights.

    get_response is async because HistoryService and RagService are backed
    by an AsyncSession; model_service.generate and search_service.search
    stay sync/blocking calls (they don't touch the DB) and FastAPI runs
    them in a worker thread as usual.
    """

    def __init__(
        self,
        model_service: ModelService,
        search_service: WebSearchService | None = None,
        file_service: FileService | None = None,
        rag_service: RagService | None = None,
        history_service: HistoryService | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._model_service = model_service
        self._search_service = search_service or WebSearchService()
        self._file_service = file_service or FileService()
        self._rag_service = rag_service
        self._history_service = history_service
        self._system_prompt = system_prompt

    async def get_response(
        self,
        user_message: str,
        thinking_mode: ThinkingMode = ThinkingMode.FAST,
        web_search: bool = False,
        rag: bool = False,
        rag_top_k: int = 4,
        personality: Personality = Personality.NEUTRAL,
        language: Language = Language.ENGLISH,
        context: str | None = None,
        conversation_id: int | None = None,
    ) -> ChatResult:
        logger.info(
            "Processing user message (length=%d, thinking_mode=%s, web_search=%s, "
            "rag=%s, personality=%s, language=%s, conversation_id=%s)",
            len(user_message),
            thinking_mode.value,
            web_search,
            rag,
            personality.value,
            language.value,
            conversation_id,
        )

        history: list[dict] = []
        if conversation_id is not None:
            if self._history_service is None:
                logger.warning("conversation_id was given but no HistoryService is configured; ignoring it")
                conversation_id = None
            else:
                conversation = await self._history_service.get_conversation(conversation_id)
                if conversation is None:
                    raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
                history = await self._history_service.get_messages_for_prompt(conversation_id)

        enable_thinking, tokens_multiplier, extra_instruction = _THINKING_MODE_CONFIG[thinking_mode]

        extra_context_parts = []

        if context:
            extra_context_parts.append(context)

        if web_search:
            results = self._search_service.search(user_message)
            formatted = self._search_service.format_for_prompt(results)
            if formatted:
                extra_context_parts.append(formatted)

        if rag:
            if self._rag_service is None:
                logger.warning("rag=True was requested but no RagService is configured; skipping retrieval")
            else:
                rag_context = await self._rag_service.retrieve_context(user_message, top_k=rag_top_k)
                if rag_context:
                    extra_context_parts.append(rag_context)

        messages = self._build_messages(
            user_message=user_message,
            history=history,
            personality=personality,
            language=language,
            context="\n\n".join(extra_context_parts) or None,
            extra_instruction=extra_instruction,
        )
        response = self._model_service.generate(
            messages,
            enable_thinking=enable_thinking,
            max_new_tokens_multiplier=tokens_multiplier,
        )

        logger.info("Generated response (length=%d)", len(response))

        if self._history_service is not None:
            if conversation_id is None:
                conversation_id = await self._history_service.create_conversation(title=user_message)
            await self._history_service.add_message(conversation_id, "user", user_message)
            await self._history_service.add_message(conversation_id, "assistant", response)

        return ChatResult(response=response, conversation_id=conversation_id)

    def extract_file_text(self, file) -> str:
        return self._file_service.extract_text(file)

    def _build_messages(
        self,
        user_message: str,
        history: list[dict],
        personality: Personality,
        language: Language,
        context: str | None,
        extra_instruction: str | None,
    ) -> list[dict]:
        system_prompt = self._build_system_prompt(personality, language, context, extra_instruction)
        return [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

    def _build_system_prompt(
        self,
        personality: Personality,
        language: Language,
        context: str | None,
        extra_instruction: str | None,
    ) -> str:
        parts = [self._system_prompt, _PERSONALITY_INSTRUCTIONS[personality]]

        parts.append(
            f"Always reply in {_LANGUAGE_NAMES[language]}, regardless of the language of the input."
        )

        if extra_instruction:
            parts.append(extra_instruction)

        if context:
            parts.append(
                "Use the following context to answer if relevant. "
                "If it doesn't help, ignore it:\n\n" + context
            )

        return "\n\n".join(parts)