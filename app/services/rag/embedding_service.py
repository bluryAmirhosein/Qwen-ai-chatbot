import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a sentence-transformers embedding model.

    Kept as its own layer, mirroring ModelService, so embedding logic and
    model loading never leak into RagService and can be mocked in tests.

    The default model (intfloat/multilingual-e5-small) handles Persian and
    English well and is small enough to run comfortably on CPU alongside
    Qwen3-1.7B. e5 models expect a "query: " / "passage: " prefix on the
    text being embedded, which is applied here.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        logger.info("Loading embedding model '%s'", self._model_name)
        self._model = SentenceTransformer(self._model_name, cache_folder=self._cache_dir)
        logger.info("Embedding model loaded (dim=%d)", self.dimension)

    @property
    def dimension(self) -> int:
        if not self.is_loaded:
            self.load()
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of chunk texts for storage."""
        return self._embed([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string for retrieval."""
        return self._embed([f"query: {text}"])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        with self._lock:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.astype(np.float32)