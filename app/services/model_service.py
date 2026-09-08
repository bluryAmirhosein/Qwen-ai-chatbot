import logging
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelService:
    """Wraps loading and running the language model.

    Kept as its own layer (separate from ChatService) so that:
      - Model loading/inference details never leak into business logic.
      - It can be swapped for a fake/mock implementation in unit tests
        without ever loading real model weights.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (if needed) and load the tokenizer/model into memory."""
        if self.is_loaded:
            return

        logger.info(
            "Loading model '%s' (cache_dir=%s)",
            self._settings.model_name,
            self._settings.model_cache_dir,
        )

        device = self._resolve_device()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._settings.model_name,
            cache_dir=self._settings.model_cache_dir,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._settings.model_name,
            cache_dir=self._settings.model_cache_dir,
            torch_dtype="auto",
        ).to(device)
        self._device = device

        logger.info("Model loaded successfully on device '%s'", device)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        logger.info("Model unloaded")

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool = False,
        max_new_tokens_multiplier: float = 1.0,
    ) -> str:
        """Generate a reply for a list of chat messages.

        messages follows the standard chat format:
            [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

        enable_thinking controls whether the chat template enables the
        model's extended thinking / reasoning mode. max_new_tokens_multiplier
        scales the configured max_new_tokens, used to approximate different
        "thinking depth" tiers (fast / balanced / deep).
        """
        if not self.is_loaded:
            self.load()

        max_new_tokens = max(1, int(self._settings.max_new_tokens * max_new_tokens_multiplier))

        with self._lock:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self._device)

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self._settings.temperature,
            )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        response = response.strip()
        logger.debug("Generated response of length %d", len(response))
        return response

    def _resolve_device(self) -> str:
        if self._settings.device != "auto":
            return self._settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"
