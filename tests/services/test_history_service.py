import pytest

from app.services.history_service import HistoryService


@pytest.fixture
def history(tmp_path):
    return HistoryService(db_path=str(tmp_path / "history.db"))


def test_create_conversation_and_add_messages(history):
    conversation_id = history.create_conversation(title="hello there")
    history.add_message(conversation_id, "user", "hello there")
    history.add_message(conversation_id, "assistant", "hi, how can I help?")

    conversation = history.get_conversation(conversation_id)

    assert conversation["title"] == "hello there"
    assert [m["role"] for m in conversation["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in conversation["messages"]] == [
        "hello there",
        "hi, how can I help?",
    ]


def test_get_conversation_returns_none_for_unknown_id(history):
    assert history.get_conversation(999) is None


def test_get_messages_for_prompt_returns_role_content_dicts_only(history):
    conversation_id = history.create_conversation(title="t")
    history.add_message(conversation_id, "user", "first")
    history.add_message(conversation_id, "assistant", "second")

    messages = history.get_messages_for_prompt(conversation_id)

    assert messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]


def test_list_conversations_orders_most_recently_updated_first(history):
    first_id = history.create_conversation(title="first")
    second_id = history.create_conversation(title="second")
    # Touch the first conversation again so it becomes the most recently updated.
    history.add_message(first_id, "user", "ping")

    conversations = history.list_conversations()

    assert [c["id"] for c in conversations] == [first_id, second_id]


def test_list_conversations_respects_limit_and_offset(history):
    ids = [history.create_conversation(title=f"c{i}") for i in range(5)]

    page = history.list_conversations(limit=2, offset=1)

    assert len(page) == 2
    assert page[0]["id"] == ids[3]
    assert page[1]["id"] == ids[2]


def test_title_over_max_length_is_truncated_with_ellipsis(history):
    long_title = "word " * 40
    conversation_id = history.create_conversation(title=long_title)

    conversation = history.get_conversation(conversation_id)

    assert len(conversation["title"]) <= 60
    assert conversation["title"].endswith("…")


def test_blank_title_falls_back_to_default(history):
    conversation_id = history.create_conversation(title="   ")
    conversation = history.get_conversation(conversation_id)
    assert conversation["title"] == "New conversation"


def test_search_finds_conversation_by_message_content(history):
    conversation_id = history.create_conversation(title="python question")
    history.add_message(conversation_id, "user", "How do I reverse a list in Python?")
    history.add_message(conversation_id, "assistant", "Use list[::-1] or list.reverse().")

    results = history.search("reverse")

    assert len(results) == 1
    assert results[0]["id"] == conversation_id
    assert "reverse" in results[0]["snippet"].lower()


def test_search_deduplicates_conversations_with_multiple_matching_messages(history):
    conversation_id = history.create_conversation(title="dedupe test")
    history.add_message(conversation_id, "user", "tell me about rockets")
    history.add_message(conversation_id, "assistant", "rockets use combustion")

    results = history.search("rockets")

    assert len(results) == 1


def test_search_with_only_punctuation_returns_empty_list_instead_of_raising(history):
    history.create_conversation(title="anything")
    assert history.search("???") == []
    assert history.search("") == []


def test_search_does_not_match_unrelated_conversations(history):
    history.create_conversation(title="unrelated")
    results = history.search("nonexistent_keyword_xyz")
    assert results == []


def test_delete_conversation_removes_conversation_and_messages(history):
    conversation_id = history.create_conversation(title="to delete")
    history.add_message(conversation_id, "user", "will this be searchable after deletion?")

    deleted = history.delete_conversation(conversation_id)

    assert deleted is True
    assert history.get_conversation(conversation_id) is None
    # The FTS index entries must be cleaned up too, not just the row.
    assert history.search("searchable") == []


def test_delete_conversation_returns_false_for_unknown_id(history):
    assert history.delete_conversation(999) is False


def test_search_query_with_special_characters_does_not_raise(history):
    conversation_id = history.create_conversation(title="quotes")
    history.add_message(conversation_id, "user", 'He said "hello" to me.')

    # A literal double-quote in the search text must not break the FTS5 MATCH syntax.
    results = history.search('"hello"')

    assert isinstance(results, list)
