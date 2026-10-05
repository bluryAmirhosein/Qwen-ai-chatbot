from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # start app settings
    host: str = "127.0.0.1"
    port: int = 8000

    # General app settings
    app_name: str = "Qwen Chatbot API"
    api_v1_prefix: str = "/api/v1"

    # Database settings
    # Docker Compose sets DATABASE_URL directly (built from POSTGRES_* vars);
    # this default only applies for local, non-docker runs.
    database_url: str = "postgresql+asyncpg://qwen:qwen@localhost:5432/qwen_chatbot"

    # Model settings
    model_name: str = "Qwen/Qwen3-1.7B"
    # Where the (large) model weights are cached on disk.
    # Point this to a drive with enough free space, e.g. D:/ai-models/qwen3-1.7b
    model_cache_dir: str = "./model_cache"
    device: str = "auto"  # "auto", "cpu" or "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Inference backend: "transformers" (HF weights) or "llama_cpp" (quantized GGUF).
    model_backend: Literal["transformers", "llama_cpp"] = "transformers"
    # Absolute path to the .gguf file (only used when model_backend == "llama_cpp").
    gguf_model_path: str = ""
    # Context window in tokens: prompt (history + RAG + system) + generated tokens must fit.
    llama_n_ctx: int = 4096
    # None lets llama.cpp pick a default. Set to the number of physical CPU cores for best speed.
    llama_n_threads: int | None = None
    # Prompt-processing batch size.
    llama_n_batch: int = 512
    llama_top_p: float = 0.95
    llama_top_k: int = 20

    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # web search settings
    web_search_max_results: int = 5
    web_search_timeout: int = 10

    # RAG settings
    rag_db_path: str = "./data/rag.db"
    # Multilingual (handles Persian + English) sentence-transformers model.
    # Reuses model_cache_dir so weights live next to the LLM cache.
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_top_k: int = 4

    history_db_path: str = "./data/history.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()