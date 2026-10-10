import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.services.cancellation import GenerationCancelledError

logger = logging.getLogger(__name__)

_MIN_COMPLETION_TOKENS = 32


@dataclass
class GenerationResult:
    text: str
    completion_tokens: int


class LLMBackend(Protocol):
    """Contract every inference backend must satisfy."""

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool,
        max_new_tokens: int,
        cancel_event: threading.Event | None = None,
    ) -> GenerationResult:
        """Generate a reply.

        If cancel_event is given and gets set while generating, the backend
        must stop as soon as possible and raise GenerationCancelledError.
        """
        ...


class TransformersBackend:
    """Runs the original HuggingFace weights through transformers + torch."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._tokenizer = None
        self._model = None
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading transformers model '%s' (cache_dir=%s)",
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

        logger.info("Transformers model loaded on device '%s'", device)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        logger.info("Transformers model unloaded")

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool,
        max_new_tokens: int,
        cancel_event: threading.Event | None = None,
    ) -> GenerationResult:
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self._device)

        extra_kwargs = {}
        if cancel_event is not None:
            extra_kwargs["stopping_criteria"] = self._build_cancel_criteria(cancel_event)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=self._settings.temperature,
            **extra_kwargs,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled by user")

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(text=text, completion_tokens=int(generated_ids.shape[0]))

    @staticmethod
    def _build_cancel_criteria(cancel_event: threading.Event):
        """Stopping criteria that ends generation once cancel_event is set.

        transformers calls it after every generated token. Imports are local
        so the llama.cpp backend never needs torch/transformers installed.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],),
                    cancel_event.is_set(),
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        return StoppingCriteriaList([_CancelCriteria()])

    def _resolve_device(self) -> str:
        if self._settings.device != "auto":
            return self._settings.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


class LlamaCppBackend:
    """Runs a quantized GGUF model through llama.cpp (llama-cpp-python).

    The prompt is built manually in ChatML format, mirroring the Qwen3 chat
    template: when thinking is disabled, an empty <think></think> block is
    prefilled so the model skips the reasoning phase (same as the HF template
    does with enable_thinking=False).

    Generation runs in streaming mode so the loop can check for cancellation
    after every token.
    """

    _STOP_STRINGS = ["<|im_end|>"]

    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        model_path = Path(self._settings.gguf_model_path)
        if not self._settings.gguf_model_path or not model_path.is_file():
            raise FileNotFoundError(
                f"GGUF model file not found at '{self._settings.gguf_model_path}'. "
                "Check GGUF_MODEL_PATH in your .env file."
            )

        from llama_cpp import Llama

        logger.info(
            "Loading GGUF model '%s' (n_ctx=%d, n_threads=%s, n_batch=%d)",
            model_path,
            self._settings.llama_n_ctx,
            self._settings.llama_n_threads,
            self._settings.llama_n_batch,
        )

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self._settings.llama_n_ctx,
            n_threads=self._settings.llama_n_threads,
            n_batch=self._settings.llama_n_batch,
            verbose=False,
        )

        logger.info("GGUF model loaded successfully")

    def unload(self) -> None:
        if self._llm is not None and hasattr(self._llm, "close"):
            self._llm.close()
        self._llm = None
        logger.info("GGUF model unloaded")

    def generate(
        self,
        messages: list[dict],
        enable_thinking: bool,
        max_new_tokens: int,
        cancel_event: threading.Event | None = None,
    ) -> GenerationResult:
        prompt = self._build_prompt(messages, enable_thinking)
        prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

        available = self._llm.n_ctx() - len(prompt_tokens)
        if available < _MIN_COMPLETION_TOKENS:
            raise ValueError(
                f"Prompt is too long for the context window "
                f"(prompt_tokens={len(prompt_tokens)}, n_ctx={self._llm.n_ctx()}). "
                "Reduce the history/RAG context or increase LLAMA_N_CTX."
            )

        stream = self._llm.create_completion(
            prompt=prompt_tokens,
            max_tokens=min(max_new_tokens, available),
            temperature=self._settings.temperature,
            top_p=self._settings.llama_top_p,
            top_k=self._settings.llama_top_k,
            stop=self._STOP_STRINGS,
            stream=True,
        )

        text_parts: list[str] = []
        # In streaming mode there is no usage block, so each chunk is counted
        # as one token. This is only used for the throughput log line.
        completion_tokens = 0

        try:
            for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelledError("Generation cancelled by user")
                text_parts.append(chunk["choices"][0]["text"])
                completion_tokens += 1
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        return GenerationResult(text="".join(text_parts), completion_tokens=completion_tokens)

    @staticmethod
    def _build_prompt(messages: list[dict], enable_thinking: bool) -> str:
        parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ]
        parts.append("<|im_start|>assistant\n")
        if not enable_thinking:
            parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)


def create_backend(settings: Settings) -> LLMBackend:
    if settings.model_backend == "llama_cpp":
        return LlamaCppBackend(settings)
    return TransformersBackend(settings)