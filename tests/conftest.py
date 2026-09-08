"""Shared pytest fixtures and test-session setup.

`ModelService` and `EmbeddingService` wrap `torch`/`transformers` and
`sentence_transformers` respectively. Unit tests never load real model
weights — every test that touches `ChatService` or `RagService` injects a
fake or `unittest.mock.Mock` in place of these services. However, importing
`app.services.model_service` / `app.services.rag.embedding_service` (even
just for type hints) still triggers the real `import torch` /
`import transformers` / `import sentence_transformers` at module load time.

To keep this suite runnable without installing multi-gigabyte GPU
dependencies, we install minimal stand-in modules in `sys.modules` before
any `app.*` module is imported, for those three libraries only. This never
runs application code from these libraries: real behavior is always
replaced by fakes/mocks at the point of use, so the stubs only need to
exist, not do anything.

If your environment already has the real libraries installed, they load
normally and these stubs are skipped.
"""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


try:
    import torch  # noqa: F401
except ImportError:
    _stub_module("torch", cuda=_FakeCuda())

try:
    import transformers  # noqa: F401
except ImportError:
    _stub_module("transformers", AutoModelForCausalLM=object, AutoTokenizer=object)

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _stub_module("sentence_transformers", SentenceTransformer=object)
