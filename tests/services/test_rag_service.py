from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.rag.rag_service import RagService
from app.services.rag.vector_store import SearchResult


@pytest.fixture
def mocks():
    return {
        "file_service": MagicMock(),
        "embedding_service": MagicMock(),
        "vector_store": MagicMock(),
        "chunker": MagicMock(),
    }


@pytest.fixture
def service(mocks):
    return RagService(
        file_service=mocks["file_service"],
        embedding_service=mocks["embedding_service"],
        vector_store=mocks["vector_store"],
        chunker=mocks["chunker"],
    )


def test_ingest_file_runs_extract_chunk_embed_store_pipeline_in_order(service, mocks):
    fake_file = MagicMock(filename="report.docx")
    mocks["file_service"].extract_text.return_value = "full document text"
    mocks["chunker"].split.return_value = ["chunk 1", "chunk 2"]
    fake_embeddings = np.zeros((2, 4), dtype=np.float32)
    mocks["embedding_service"].embed_documents.return_value = fake_embeddings
    mocks["vector_store"].add_document.return_value = (42, 2)

    filename, document_id, chunk_count = service.ingest_file(fake_file)

    mocks["file_service"].extract_text.assert_called_once_with(fake_file)
    mocks["chunker"].split.assert_called_once_with("full document text")
    mocks["embedding_service"].embed_documents.assert_called_once_with(["chunk 1", "chunk 2"])
    mocks["vector_store"].add_document.assert_called_once_with(
        "report.docx", ["chunk 1", "chunk 2"], fake_embeddings
    )
    assert (filename, document_id, chunk_count) == ("report.docx", 42, 2)


def test_ingest_file_falls_back_to_unknown_filename(service, mocks):
    fake_file = MagicMock(filename=None)
    mocks["file_service"].extract_text.return_value = "text"
    mocks["chunker"].split.return_value = ["chunk"]
    mocks["embedding_service"].embed_documents.return_value = np.zeros((1, 4), dtype=np.float32)
    mocks["vector_store"].add_document.return_value = (1, 1)

    filename, _, _ = service.ingest_file(fake_file)

    assert filename == "unknown"


def test_ingest_file_raises_when_no_chunks_are_produced(service, mocks):
    fake_file = MagicMock(filename="empty.txt")
    mocks["file_service"].extract_text.return_value = ""
    mocks["chunker"].split.return_value = []

    with pytest.raises(ValueError):
        service.ingest_file(fake_file)

    # Nothing should have been embedded or stored for an empty document.
    mocks["embedding_service"].embed_documents.assert_not_called()
    mocks["vector_store"].add_document.assert_not_called()


def test_retrieve_embeds_query_and_searches_vector_store(service, mocks):
    query_embedding = np.array([1.0, 0.0], dtype=np.float32)
    mocks["embedding_service"].embed_query.return_value = query_embedding
    expected_results = [
        SearchResult(chunk_id=1, document_id=1, filename="doc.txt", content="hit", score=0.9)
    ]
    mocks["vector_store"].search.return_value = expected_results

    results = service.retrieve("what is X?", top_k=3)

    mocks["embedding_service"].embed_query.assert_called_once_with("what is X?")
    mocks["vector_store"].search.assert_called_once_with(query_embedding, top_k=3)
    assert results == expected_results


def test_retrieve_context_returns_none_when_nothing_found(service, mocks):
    mocks["embedding_service"].embed_query.return_value = np.zeros(2, dtype=np.float32)
    mocks["vector_store"].search.return_value = []

    assert service.retrieve_context("anything") is None


def test_retrieve_context_formats_multiple_sources_with_separators(service, mocks):
    mocks["embedding_service"].embed_query.return_value = np.zeros(2, dtype=np.float32)
    mocks["vector_store"].search.return_value = [
        SearchResult(chunk_id=1, document_id=1, filename="a.txt", content="alpha content", score=0.9),
        SearchResult(chunk_id=2, document_id=2, filename="b.txt", content="beta content", score=0.5),
    ]

    context = service.retrieve_context("query", top_k=2)

    assert context == (
        "[Source: a.txt]\nalpha content"
        "\n\n---\n\n"
        "[Source: b.txt]\nbeta content"
    )


def test_list_documents_and_delete_document_delegate_to_vector_store(service, mocks):
    mocks["vector_store"].list_documents.return_value = [{"id": 1, "filename": "a", "chunk_count": 3}]
    mocks["vector_store"].delete_document.return_value = True

    assert service.list_documents() == [{"id": 1, "filename": "a", "chunk_count": 3}]
    assert service.delete_document(1) is True
    mocks["vector_store"].delete_document.assert_called_once_with(1)
