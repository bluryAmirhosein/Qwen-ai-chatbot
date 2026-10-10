import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.v1.dependencies import get_cancellation_registry, get_chat_service
from app.schemas.chat import ChatResponse, Language, Personality, ThinkingMode
from app.services.cancellation import CancellationRegistry, GenerationCancelledError
from app.services.chat_service import ChatService, ConversationNotFoundError
from app.services.rag.rag_service import DocumentNotFoundError

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
    "on your next call. With `rag=true`, pass one or more `document_ids` (from "
    "GET /rag/documents) to answer only from those documents. Pass a unique "
    "`request_id` to be able to cancel the generation via POST /chat/stop; a "
    "cancelled request returns 499 and nothing is saved to the history.",
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
    document_ids: list[int] | None = Query(
        default=None,
        description="Only used when rag=true. Restrict retrieval to these document ids "
        "(see GET /rag/documents). Leave empty to search all ingested documents.",
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
    request_id: str | None = Query(
        default=None,
        min_length=8,
        max_length=64,
        description="Client-generated unique id (e.g. a UUID) for this request. "
        "Send the same id to POST /chat/stop to cancel the generation.",
    ),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    if document_ids and not rag:
        raise HTTPException(
            status_code=400,
            detail="document_ids can only be used together with rag=true",
        )

    logger.info(
        "Received chat request (thinking_mode=%s, web_search=%s, rag=%s, document_ids=%s, "
        "personality=%s, language=%s, conversation_id=%s, request_id=%s)",
        thinking_mode.value,
        web_search,
        rag,
        document_ids,
        personality.value,
        language.value,
        conversation_id,
        request_id,
    )
    try:
        result = await chat_service.get_response(
            user_message=message,
            thinking_mode=thinking_mode,
            web_search=web_search,
            rag=rag,
            rag_top_k=rag_top_k,
            document_ids=document_ids,
            personality=personality,
            language=language,
            context=context,
            conversation_id=conversation_id,
            request_id=request_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GenerationCancelledError as exc:
        # 499 is the de-facto "client closed request" status.
        raise HTTPException(status_code=499, detail="Generation cancelled by user") from exc

    return ChatResponse(response=result.response, conversation_id=result.conversation_id)


@router.post(
    "/stop",
    summary="Stop an in-progress generation",
    description="Cancels the generation started by POST /chat with the same `request_id`. "
    "The model stops at the next token, so on a long prompt it can take a moment "
    "to take effect. `status` is `stopping` if the request was found, or "
    "`not_running` if it already finished (or has not started yet).",
)
async def stop_generation(
    request_id: str = Query(
        ...,
        min_length=8,
        max_length=64,
        description="The request_id that was sent to POST /chat",
    ),
    registry: CancellationRegistry = Depends(get_cancellation_registry),
) -> dict:
    was_running = registry.cancel(request_id)
    return {"request_id": request_id, "status": "stopping" if was_running else "not_running"}