import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.v1.dependencies import get_chat_service
from app.schemas.chat import ChatResponse, Language, Personality, ThinkingMode
from app.services.chat_service import ChatService, ConversationNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to the chatbot",
    description="Sends a single user message to the model and returns its reply. "
    "The model is lazily loaded on first call. Only `message` goes in the request "
    "body; every other option is a query parameter with a dropdown in Swagger. "
    "Pass `conversation_id` to continue an earlier conversation, or omit it to "
    "start a new one — the response always includes the conversation_id to use "
    "on your next call.",
)
async def send_message(
    message: str = Body(
        ...,
        min_length=1,
        media_type="text/plain",
        description="User message to send to the chatbot",
    ),
    thinking_mode: ThinkingMode = Query(
        default=ThinkingMode.FAST,
        description="Reasoning depth: fast (no thinking), balanced (thinking, normal length), "
        "deep (thinking, longer/more thorough)",
    ),
    web_search: bool = Query(
        default=False,
        description="If true, run a web search on the message and inject results as context",
    ),
    rag: bool = Query(
        default=False,
        description="If true, retrieve relevant chunks from documents ingested via "
        "/rag/ingest and inject them as context",
    ),
    rag_top_k: int = Query(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve when rag=true",
    ),
    personality: Personality = Query(
        default=Personality.NEUTRAL,
        description="Tone/style preset for the reply",
    ),
    language: Language = Query(
        default=Language.ENGLISH,
        description="Language the reply should be written in",
    ),
    context: str | None = Query(
        default=None,
        description="Optional extra reference text to ground the reply on "
        "(e.g. text extracted from an uploaded file via /files/extract)",
    ),
    conversation_id: int | None = Query(
        default=None,
        description="Existing conversation to continue. Omit to start a new one.",
    ),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    logger.info(
        "Received chat request (thinking_mode=%s, web_search=%s, rag=%s, personality=%s, "
        "language=%s, conversation_id=%s)",
        thinking_mode.value,
        web_search,
        rag,
        personality.value,
        language.value,
        conversation_id,
    )
    try:
        result = await chat_service.get_response(
            user_message=message,
            thinking_mode=thinking_mode,
            web_search=web_search,
            rag=rag,
            rag_top_k=rag_top_k,
            personality=personality,
            language=language,
            context=context,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(response=result.response, conversation_id=result.conversation_id)