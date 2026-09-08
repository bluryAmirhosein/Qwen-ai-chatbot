from enum import Enum

from pydantic import BaseModel, Field


class ThinkingMode(str, Enum):
    """Three-tier reasoning depth exposed to API clients."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class Personality(str, Enum):
    """Predefined tone/style presets for the assistant's replies."""

    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CONCISE = "concise"
    TECHNICAL = "technical"
    HUMOROUS = "humorous"


class Language(str, Enum):
    """Reply language. Limited to the two languages the model handles well."""

    PERSIAN = "persian"
    ENGLISH = "english"


_PERSONALITY_INSTRUCTIONS: dict[Personality, str] = {
    Personality.NEUTRAL: "Respond in a neutral, straightforward tone.",
    Personality.FRIENDLY: "Respond in a warm, friendly, approachable tone.",
    Personality.FORMAL: "Respond in a formal, professional tone.",
    Personality.CONCISE: "Respond as concisely as possible, no filler.",
    Personality.TECHNICAL: "Respond with precise technical language, assume an expert audience.",
    Personality.HUMOROUS: "Respond with a light, humorous tone where appropriate.",
}


class ChatRequest(BaseModel):
    """Incoming chat message from the client."""

    message: str = Field(..., min_length=1, description="User message to send to the chatbot")
    thinking_mode: ThinkingMode = Field(
        default=ThinkingMode.FAST,
        description="Reasoning depth: fast (no thinking), balanced (thinking, normal length), "
        "deep (thinking, longer/more thorough)",
    )
    web_search: bool = Field(
        default=False,
        description="If true, run a web search on the message and inject results as context",
    )
    personality: Personality = Field(
        default=Personality.NEUTRAL,
        description="Tone/style preset for the reply",
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Language the reply should be written in",
    )
    context: str | None = Field(
        default=None,
        description="Optional extra reference text to ground the reply on "
        "(e.g. text extracted from an uploaded file via /files/extract). "
        "Leave empty if not needed.",
    )
    conversation_id: int | None = Field(
        default=None,
        description="Existing conversation to continue. Omit/leave empty to start a new one; "
        "the new conversation's id is returned in the response.",
    )


class ChatResponse(BaseModel):
    """Chatbot reply returned to the client."""

    response: str = Field(..., description="Chatbot generated reply")
    conversation_id: int | None = Field(
        default=None,
        description="Id of the conversation this exchange was saved under "
        "(null only if history storage isn't configured). Pass it back on the "
        "next request to continue this conversation.",
    )


class FileExtractResponse(BaseModel):
    """Extracted text from an uploaded file."""

    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Extracted plain text content")