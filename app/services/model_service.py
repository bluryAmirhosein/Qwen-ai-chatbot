import logging
import threading
import time

from app.config import Settings
from app.services.cancellation import GenerationCancelledError
from app.services.model_backends import LLMBackend, create_backend

logger = logging.getLogger(__name__)


class ModelService:
    """Facade over the configured inference backend.

    Kept as its own layer (separate from ChatService) so that:
      - Model loading/inference details never leak into business logic.
      - It can be swapped for a fake/mock implementation in unit tests
        without ever loading real model weights.

    The actual inference engine (transformers or llama.cpp) is selected by
    settings.model_backend; this class only adds locking, token budgeting,
    cancellation checks and throughput logging on top of it.
    """

    def __init__(self, settings: Settings, backend: LLMBackend | None = None):
        self._settings = settings
        self._backend = backend or create_backend(settings)
        # Reentrant because generate() may call load() while already holding the lock.
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    def load(self) -> None:
        """Download/open (if needed) and load the model into memory."""
        with self._lock:
            if self.is_loaded:
                return
            self._backend.load()

    def unload(self) -> None:
        with self._lock:
            self._backend.unload()

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool = False,
        max_new_tokens_multiplier: float = 1.0,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Generate a reply for a list of chat messages.

        messages follows the standard chat format:
            [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

        enable_thinking controls whether the model's extended thinking /
        reasoning mode is enabled. max_new_tokens_multiplier scales the
        configured max_new_tokens, used to approximate different
        "thinking depth" tiers (fast / balanced / deep).

        cancel_event, when set (from another thread), aborts the generation
        and makes this method raise GenerationCancelledError. It is also
        checked after the model lock is acquired, so a request that was
        cancelled while queued behind another generation never starts.
        """
        max_new_tokens = max(1, int(self._settings.max_new_tokens * max_new_tokens_multiplier))

        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelledError("Generation cancelled before it started")

            if not self.is_loaded:
                self.load()

            started_at = time.perf_counter()
            result = self._backend.generate(
                messages,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
                cancel_event=cancel_event,
            )
            elapsed = time.perf_counter() - started_at

        tokens_per_second = result.completion_tokens / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Generation finished (backend=%s, tokens=%d, seconds=%.2f, tokens_per_second=%.2f)",
            self._settings.model_backend,
            result.completion_tokens,
            elapsed,
            tokens_per_second,
        )

        response = result.text.strip()
        logger.debug("Generated response of length %d", len(response))
        return response