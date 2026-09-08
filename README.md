# Qwen AI Chatbot API

A modular, locally hosted AI chatbot backend built with **FastAPI** and **Qwen3-1.7B**. The project combines conversational memory, configurable reasoning modes, web search, file-based context, and Retrieval-Augmented Generation (RAG) behind a clean service-oriented architecture.

The system is designed to keep model inference, business logic, retrieval, persistence, and HTTP concerns separated so individual components can be tested, replaced, or extended independently.

## Highlights

- **Local LLM inference** with Qwen3-1.7B through Hugging Face Transformers and PyTorch
- **Three reasoning modes** — fast, balanced, and deep
- **Conversation memory** persisted in SQLite
- **Full-text conversation search** using SQLite FTS5
- **Configurable assistant personalities** — neutral, friendly, formal, concise, technical, and humorous
- **Bilingual responses** — English and Persian
- **Web-augmented generation** using a pluggable search service
- **File context extraction** from `.txt`, `.md`, `.pdf`, and `.docx`
- **RAG pipeline** with chunking, multilingual embeddings, and vector similarity search
- **Document lifecycle management** — ingest, list, and delete
- **Lazy model loading** to avoid loading large model weights until inference is requested
- **Dependency injection** throughout the API layer for clean testing and component replacement
- **Typed API contracts** with Pydantic models and FastAPI-generated OpenAPI documentation
- **Unit and API tests** built with pytest and FastAPI's `TestClient`

---

## Architecture

The application follows a layered architecture with explicit responsibilities:

```text
                         ┌───────────────────────┐
                         │       FastAPI         │
                         │   API / HTTP Layer    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     ChatService       │
                         │   Application Logic   │
                         └───────┬───┬───┬───────┘
                                 │   │   │
              ┌──────────────────┘   │   └──────────────────┐
              ▼                      ▼                      ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │ ModelService │      │ WebSearch     │      │  RagService  │
      │ Qwen3 + HF   │      │ Service      │      │              │
      └──────────────┘      └──────────────┘      └──────┬───────┘
                                                          │
                                                  ┌───────┴────────┐
                                                  │ Embeddings +   │
                                                  │ Vector Store   │
                                                  └────────────────┘

                 ┌─────────────────┐       ┌────────────────────┐
                 │ HistoryService  │       │   FileService      │
                 │ SQLite + FTS5   │       │ TXT/MD/PDF/DOCX    │
                 └─────────────────┘       └────────────────────┘
```

### Core design principles

- **Separation of concerns:** HTTP endpoints do not contain model, storage, or retrieval logic.
- **Dependency inversion at service boundaries:** services depend on abstractions/responsibilities rather than exposing infrastructure details to the API.
- **Replaceable infrastructure:** the web search provider and vector storage implementation can be replaced without rewriting the chat orchestration layer.
- **Testability:** FastAPI dependencies can be overridden with mocks, allowing API tests to run without downloading or loading model weights.
- **Single responsibility:** model inference, file extraction, conversation persistence, RAG, and web search are isolated into dedicated services.

---

## Key Features

### 1. Local Qwen Chat

The chatbot runs Qwen3-1.7B locally through Hugging Face Transformers.

The model is **loaded lazily** on the first generation request rather than during application startup. This keeps startup lightweight and avoids unnecessary memory consumption when the API is only being used for health checks or non-generation operations.

Device selection supports:

- `auto` — automatically selects CUDA when available, otherwise CPU
- `cpu`
- `cuda`

Generation parameters are configurable through environment variables, including maximum output tokens and temperature.

### 2. Configurable Reasoning

The chat endpoint exposes three reasoning profiles:

| Mode | Thinking | Output budget |
|---|---|---|
| `fast` | Disabled | 0.5× configured maximum |
| `balanced` | Enabled | 1× configured maximum |
| `deep` | Enabled + additional reasoning instruction | 2× configured maximum |

This provides a simple API-level trade-off between response speed and reasoning depth without coupling clients to model-specific generation details.

### 3. Conversation Memory

Conversations and messages are persisted in SQLite.

Each conversation contains:

- Conversation ID
- Title
- Creation timestamp
- Last update timestamp
- Ordered user/assistant messages

A returned `conversation_id` can be supplied to subsequent chat requests to continue the same conversation.

Conversation titles are automatically normalized and limited to a readable maximum length.

### 4. Conversation Search

Conversation history includes **SQLite FTS5** full-text search.

Search results include:

- Conversation metadata
- Matching snippet
- Relevance ordering
- Conversation-level deduplication

The implementation also maintains the FTS index through SQLite triggers when messages are inserted, updated, or deleted.

### 5. Personality & Language Controls

Clients can select a predefined response style:

- `neutral`
- `friendly`
- `formal`
- `concise`
- `technical`
- `humorous`

Response language can be explicitly selected as:

- `english`
- `persian`

These options are translated into system-level instructions before the request reaches the model.

### 6. Web-Augmented Responses

The optional web search feature uses a dedicated `WebSearchService`.

When `web_search=true`:

1. The user's query is sent to the configured search backend.
2. Search results are normalized into a common internal format.
3. Results are formatted as context.
4. The context is injected into the model's system prompt.
5. The model generates the final response using the additional information.

Search failures are isolated from the main chat flow: a failed search does not prevent the chatbot from responding.

The search provider is encapsulated so it can be replaced later without changing `ChatService`.

### 7. File Context Extraction

`FileService` converts supported files into plain text:

| Format | Extraction |
|---|---|
| `.txt` | UTF-8 text |
| `.md` | UTF-8 text |
| `.pdf` | Text extraction with PyPDF |
| `.docx` | Paragraph extraction with python-docx |

Extracted content can be passed as chat context, making it possible to ask questions about a document without permanently adding it to the RAG index.

### 8. Retrieval-Augmented Generation

The RAG subsystem provides persistent document retrieval.

The ingestion pipeline is:

```text
Upload
  │
  ▼
File Extraction
  │
  ▼
Paragraph-aware Chunking
  │
  ▼
Overlapping Chunks
  │
  ▼
Multilingual Embeddings
  │
  ▼
SQLite Vector Store
```

At query time:

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Vector Similarity Search
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
Prompt Context
    │
    ▼
Qwen3
    │
    ▼
Response
```

The default embedding model is `intfloat/multilingual-e5-small`, with `query:` and `passage:` prefixes applied according to the model's expected retrieval format.

The default chunk configuration is:

- Chunk size: `800` characters
- Overlap: `150` characters
- Default retrieval count: `4`
- Maximum API retrieval count: `20`

Embeddings are normalized and stored as `float32` blobs. Similarity is calculated with a NumPy matrix multiplication, which is equivalent to cosine similarity for normalized vectors.

This lightweight design avoids requiring a dedicated vector database for small knowledge bases.

---

## API

Base path:

```text
/api/v1
```

### Health

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Application health check |

### Chat

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Generate a chatbot response |

The message is sent as `text/plain`.

Available query parameters include:

- `thinking_mode`
- `web_search`
- `rag`
- `rag_top_k`
- `personality`
- `language`
- `context`
- `conversation_id`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat?thinking_mode=balanced&personality=technical&language=english" \
  -H "Content-Type: text/plain" \
  --data "Explain dependency injection in FastAPI."
```

To continue an existing conversation:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat?conversation_id=1" \
  -H "Content-Type: text/plain" \
  --data "Can you give me an example?"
```

### File Extraction

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/files/extract` | Extract plain text from a supported file |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/files/extract" \
  -F "file=@document.pdf"
```

### RAG

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/rag/ingest` | Add a document to the RAG index |
| `GET` | `/api/v1/rag/documents` | List indexed documents |
| `DELETE` | `/api/v1/rag/documents/{document_id}` | Remove a document and its chunks |

Example ingestion:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/ingest" \
  -F "file=@knowledge-base.pdf"
```

Then enable retrieval during chat:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat?rag=true&rag_top_k=4" \
  -H "Content-Type: text/plain" \
  --data "What does the document say about the deployment process?"
```

### Conversation History

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/history` | List conversations |
| `GET` | `/api/v1/history?q=...` | Full-text search conversations |
| `GET` | `/api/v1/history/{conversation_id}` | Retrieve a complete conversation |
| `DELETE` | `/api/v1/history/{conversation_id}` | Delete a conversation |

Pagination is supported through `limit` and `offset` when listing conversations.

---

## Interactive API Documentation

Once the application is running, FastAPI provides:

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI schema:** `/openapi.json`

For example:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

The API uses typed Pydantic schemas and enum-based query parameters so the available chat options are directly discoverable through the generated OpenAPI documentation.

---

## Project Structure

```text
QwenChatbot/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── chat.py
│   │       │   ├── files.py
│   │       │   ├── history.py
│   │       │   └── rag.py
│   │       ├── dependencies.py
│   │       └── router.py
│   │
│   ├── core/
│   │   └── logging.py
│   │
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── history.py
│   │   └── rag.py
│   │
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── file_service.py
│   │   ├── history_service.py
│   │   ├── model_service.py
│   │   ├── web_search.py
│   │   └── rag/
│   │       ├── chunker.py
│   │       ├── embedding_service.py
│   │       ├── rag_service.py
│   │       └── vector_store.py
│   │
│   ├── config.py
│   └── main.py
│
├── tests/
│   ├── api/
│   └── services/
│
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

### Responsibility boundaries

- `api/` — HTTP routing, validation, response mapping, and dependency injection
- `schemas/` — request/response contracts and domain-facing enums
- `services/` — application and infrastructure behavior
- `services/rag/` — document retrieval pipeline
- `core/` — cross-cutting infrastructure such as logging
- `tests/` — isolated service tests and API-level tests
- `config.py` — environment-driven application configuration

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bluryAmirhosein/Qwen-ai-chatbot.git
cd QwenChatbot
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For the application:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

> PyTorch and the model runtime are part of the application dependency set. Choose the appropriate PyTorch build for the target machine when GPU-specific installation is required.

### 4. Configure environment variables

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

A typical configuration:

```env
MODEL_NAME=Qwen/Qwen3-1.7B
MODEL_CACHE_DIR=./model_cache
DEVICE=auto
MAX_NEW_TOKENS=512
TEMPERATURE=0.7

LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

For a machine with a dedicated model drive, the cache can be redirected:

```env
MODEL_CACHE_DIR=D:/ai-models/qwen3-1.7b
```

The model weights are downloaded automatically by Hugging Face on the first generation request.

### 5. Start the API

```bash
python -m app.main
```

Or directly with Uvicorn:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The default server is available at:

```text
http://127.0.0.1:8000
```

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

---

## Configuration

Application settings are managed through Pydantic Settings and can be supplied through environment variables or `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `8000` | Server port |
| `APP_NAME` | `Qwen Chatbot API` | FastAPI application name |
| `API_V1_PREFIX` | `/api/v1` | API version prefix |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | Hugging Face model |
| `MODEL_CACHE_DIR` | `./model_cache` | Model cache directory |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `MAX_NEW_TOKENS` | `512` | Base generation budget |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `LOG_LEVEL` | `INFO` | Application log level |
| `LOG_FILE` | `logs/app.log` | Log file path |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Maximum search results |
| `WEB_SEARCH_TIMEOUT` | `10` | Search timeout in seconds |
| `RAG_DB_PATH` | `./data/rag.db` | RAG SQLite database |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-small` | Embedding model |
| `RAG_CHUNK_SIZE` | `800` | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `150` | Chunk overlap |
| `RAG_TOP_K` | `4` | Default retrieval count |
| `HISTORY_DB_PATH` | `./data/history.db` | Conversation database |

Directories required by the application are created automatically when their services initialize.

---

## Testing

The project separates service-level and API-level tests.

Run the complete test suite:

```bash
pytest
```

Run with concise output:

```bash
pytest -q
```

The test suite validates areas including:

- Chat orchestration and prompt construction
- Conversation persistence and search
- File extraction
- Web search behavior
- RAG ingestion and retrieval
- Text chunking
- Vector storage
- API validation and HTTP responses
- Dependency injection and mocked service boundaries

The tests intentionally avoid downloading or loading real model weights. Model and embedding dependencies can be replaced with fakes/mocks, keeping the suite fast and suitable for CI environments.

---

## Operational Notes

### Model loading

The language model and embedding model are loaded only when their services are first used. This reduces application startup cost, but the first inference or first RAG operation can take longer because model weights may need to be downloaded and initialized.

### RAG storage strategy

The current vector store uses brute-force similarity search over normalized embeddings stored in SQLite.

This is intentionally simple and dependency-light for a small knowledge base. For a significantly larger corpus, the `SQLiteVectorStore` boundary can be replaced with an ANN/vector database implementation such as FAISS, sqlite-vec, or a dedicated vector database without changing the RAG orchestration layer.

### Concurrency

Model generation is protected by a lock so concurrent requests do not execute against the same model instance simultaneously. This keeps the current single-model architecture predictable, while horizontal scaling or multiple model workers can be introduced later when throughput requirements justify it.

### Data persistence

Two SQLite databases are used by default:

```text
data/history.db
data/rag.db
```

These files contain conversation history and indexed document data respectively and should be persisted when running the application in a container or managed environment.

---

## Production Considerations

The repository intentionally focuses on a compact, modular local-AI backend. Before exposing it to an untrusted public network, production hardening should include:

- Authentication and authorization
- Request rate limiting
- File size and upload limits
- Stronger input/content validation
- Secure handling of uploaded documents
- Observability and metrics
- Centralized persistent storage for horizontally scaled deployments
- A production-grade vector index for large corpora
- Model serving optimization and worker strategy based on target hardware
- Secret and configuration management appropriate to the deployment environment

These concerns are kept outside the core feature implementation so the current architecture can evolve without coupling the application layer to a specific deployment model.

---

## Technology Stack

- **Python**
- **FastAPI**
- **Pydantic / Pydantic Settings**
- **Uvicorn**
- **Qwen3-1.7B**
- **Hugging Face Transformers**
- **PyTorch**
- **Sentence Transformers**
- **multilingual-e5-small**
- **SQLite / SQLite FTS5**
- **NumPy**
- **PyPDF**
- **python-docx**
- **DDGS**
- **pytest**

---

## Design Goal

The project is built around a simple principle:

> **Keep the AI capabilities modular enough that the model, search provider, retrieval backend, and persistence strategy can evolve independently.**

The result is a lightweight backend that demonstrates practical LLM application engineering rather than treating the language model as the application itself: **API design, service boundaries, persistent conversations, retrieval, external context, testability, configuration, and operational separation are all first-class parts of the system.**
