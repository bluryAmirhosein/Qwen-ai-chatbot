import numpy as np
import pytest

from app.services.rag.vector_store import SQLiteVectorStore


@pytest.fixture
def store(tmp_path):
    return SQLiteVectorStore(db_path=str(tmp_path / "rag.db"))


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_add_document_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.add_document(
            filename="doc.txt",
            chunks=["a", "b"],
            embeddings=np.zeros((1, 4), dtype=np.float32),
        )


def test_add_document_returns_id_and_chunk_count(store):
    embeddings = np.vstack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])])
    document_id, chunk_count = store.add_document("doc.txt", ["chunk a", "chunk b"], embeddings)

    assert isinstance(document_id, int)
    assert chunk_count == 2


def test_search_ranks_by_cosine_similarity_to_query(store):
    # Three orthogonal-ish chunks; the query should rank the closest one first.
    store.add_document(
        "doc.txt",
        ["about cats", "about dogs", "about rocks"],
        np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]),
    )

    query_embedding = _unit([0.9, 0.1, 0])  # closest to "about cats"
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0].content == "about cats"
    assert results[0].score > results[1].score


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search(_unit([1, 0, 0]), top_k=4) == []


def test_search_respects_top_k(store):
    embeddings = np.vstack([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
    store.add_document("doc.txt", ["a", "b", "c"], embeddings)

    results = store.search(_unit([1, 0]), top_k=1)

    assert len(results) == 1
    assert results[0].content == "a"


def test_list_documents_reports_chunk_counts(store):
    embeddings = np.vstack([_unit([1, 0]), _unit([0, 1])])
    store.add_document("doc.txt", ["a", "b"], embeddings)

    docs = store.list_documents()

    assert docs == [{"id": docs[0]["id"], "filename": "doc.txt", "chunk_count": 2}]


def test_delete_document_removes_it_and_its_chunks(store):
    embeddings = np.vstack([_unit([1, 0]), _unit([0, 1])])
    document_id, _ = store.add_document("doc.txt", ["a", "b"], embeddings)

    deleted = store.delete_document(document_id)
    assert deleted is True
    assert store.list_documents() == []
    # The chunks should be gone too, not just orphaned.
    assert store.search(_unit([1, 0]), top_k=10) == []


def test_delete_document_returns_false_for_unknown_id(store):
    assert store.delete_document(999) is False
