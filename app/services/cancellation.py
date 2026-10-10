import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Entries older than this are dropped, so stray stop requests (for ids that
# never started or already finished) can't accumulate forever.
_ENTRY_TTL_SECONDS = 15 * 60


class GenerationCancelledError(Exception):
    """Raised when a generation is stopped by the user before it finishes."""


@dataclass
class _Entry:
    event: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.monotonic)


class CancellationRegistry:
    """Maps a client-supplied request_id to a threading.Event.

    The chat request registers its id before generating; a separate
    /chat/stop request sets the event; the inference loop (running in a
    worker thread) polls the event and aborts. threading.Event is used
    (not asyncio.Event) because the loop that checks it runs in a thread.

    If a stop arrives before the request registers, the event is created
    already set, so the request is cancelled as soon as it registers.

    The registry lives in process memory, so stop only works when the API
    runs as a single process (one uvicorn worker).
    """

    def __init__(self, ttl_seconds: float = _ENTRY_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def register(self, request_id: str) -> threading.Event:
        with self._lock:
            self._purge_expired()
            entry = self._entries.get(request_id)
            if entry is None:
                entry = _Entry()
                self._entries[request_id] = entry
            return entry.event

    def cancel(self, request_id: str) -> bool:
        """Signal cancellation. Returns True if the request was already registered."""
        with self._lock:
            self._purge_expired()
            entry = self._entries.get(request_id)
            was_registered = entry is not None
            if entry is None:
                entry = _Entry()
                self._entries[request_id] = entry
            entry.event.set()

        logger.info("Cancellation requested (request_id=%s, registered=%s)", request_id, was_registered)
        return was_registered

    def release(self, request_id: str) -> None:
        with self._lock:
            self._entries.pop(request_id, None)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [rid for rid, entry in self._entries.items() if now - entry.created_at > self._ttl]
        for rid in expired:
            del self._entries[rid]