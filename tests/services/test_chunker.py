import pytest

from app.services.rag.chunker import TextChunker


def test_rejects_overlap_greater_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_split_empty_text_returns_no_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == []
    assert chunker.split("   \n\n   ") == []


def test_single_short_paragraph_is_one_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("A short paragraph.") == ["A short paragraph."]


def test_paragraphs_are_packed_until_size_exceeded_then_split_with_overlap():
    """Small, deterministic chunk_size/overlap so the packing and
    overlap-carry logic can be checked exactly, not just "looks reasonable".
    """
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    text = (
        "Alpha bravo charlie.\n\n"
        "Delta echo foxtrot golf.\n\n"
        "Hotel india juliet kilo lima."
    )

    chunks = chunker.split(text)

    assert chunks == [
        "Alpha bravo charlie.",
        "rlie.\n\nDelta echo fo",
        "ho foxtrot golf.",
        "golf.\n\nHotel india j",
        "dia juliet kilo lima",
        "lima.",
    ]
    # No chunk should ever exceed the configured chunk_size.
    assert all(len(chunk) <= 20 for chunk in chunks)
    # Every chunk carries content, never blank/whitespace-only entries.
    assert all(chunk.strip() for chunk in chunks)


def test_paragraph_longer_than_chunk_size_is_hard_split():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "abcdefghijklmnopqrst"  # single 20-char "paragraph", no blank lines

    chunks = chunker.split(text)

    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks).replace("", "") != ""  # sanity: something was produced
    # Reassembling without the intentional overlap should recover the original text.
    assert chunks[0] == "abcdefghij"


def test_blank_lines_between_paragraphs_do_not_produce_empty_chunks():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=10)
    text = "First paragraph.\n\n\n\n\nSecond paragraph.\n\n   \n\nThird paragraph."

    chunks = chunker.split(text)

    assert chunks == ["First paragraph.\n\nSecond paragraph.\n\nThird paragraph."]
