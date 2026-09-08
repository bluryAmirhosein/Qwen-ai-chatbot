from unittest.mock import MagicMock

import pytest

from app.schemas.chat import Language, Personality, ThinkingMode
from app.services.chat_service import ChatService, ConversationNotFoundError


@pytest.fixture
def mocks():
    return {
        "model_service": MagicMock(),
        "search_service": MagicMock(),
        "file_service": MagicMock(),
        "rag_service": MagicMock(),
        "history_service": MagicMock(),
    }


@pytest.fixture
def service(mocks):
    mocks["model_service"].generate.return_value = "the model's reply"
    return ChatService(
        model_service=mocks["model_service"],
        search_service=mocks["search_service"],
        file_service=mocks["file_service"],
        rag_service=mocks["rag_service"],
        history_service=mocks["history_service"],
    )


def _messages_passed_to_generate(mocks) -> list[dict]:
    args, kwargs = mocks["model_service"].generate.call_args
    return args[0]


# --- basic response shape -----------------------------------------------------

def test_get_response_returns_model_reply_and_conversation_id(service, mocks):
    mocks["history_service"].get_conversation.return_value = None
    mocks["history_service"].create_conversation.return_value = 7

    result = service.get_response(user_message="hi there")

    assert result.response == "the model's reply"
    assert result.conversation_id == 7


def test_get_response_works_with_no_history_service_configured():
    model_service = MagicMock()
    model_service.generate.return_value = "reply"
    service = ChatService(
        model_service=model_service,
        search_service=MagicMock(),
        file_service=MagicMock(),
        rag_service=MagicMock(),
        history_service=None,
    )

    result = service.get_response(user_message="hi")

    assert result.response == "reply"
    assert result.conversation_id is None


# --- conversation / history handling -------------------------------------------

def test_unknown_conversation_id_raises_not_found(service, mocks):
    mocks["history_service"].get_conversation.return_value = None

    with pytest.raises(ConversationNotFoundError):
        service.get_response(user_message="hi", conversation_id=123)


def test_existing_conversation_id_loads_prior_messages_into_prompt(service, mocks):
    mocks["history_service"].get_conversation.return_value = {"id": 5, "title": "t"}
    mocks["history_service"].get_messages_for_prompt.return_value = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    result = service.get_response(user_message="follow up", conversation_id=5)

    messages = _messages_passed_to_generate(mocks)
    assert messages[1] == {"role": "user", "content": "earlier question"}
    assert messages[2] == {"role": "assistant", "content": "earlier answer"}
    assert messages[-1] == {"role": "user", "content": "follow up"}
    # An existing conversation_id must never be replaced with a new one.
    mocks["history_service"].create_conversation.assert_not_called()
    assert result.conversation_id == 5


def test_new_conversation_is_created_when_no_conversation_id_given(service, mocks):
    mocks["history_service"].create_conversation.return_value = 99

    result = service.get_response(user_message="hello")

    mocks["history_service"].create_conversation.assert_called_once_with(title="hello")
    assert result.conversation_id == 99


def test_conversation_id_given_but_no_history_service_is_ignored_gracefully():
    model_service = MagicMock()
    model_service.generate.return_value = "reply"
    service = ChatService(
        model_service=model_service,
        search_service=MagicMock(),
        file_service=MagicMock(),
        rag_service=MagicMock(),
        history_service=None,
    )

    # Must not raise, and must not treat conversation_id as valid.
    result = service.get_response(user_message="hi", conversation_id=5)

    assert result.conversation_id is None


def test_successful_response_is_persisted_as_user_then_assistant_message(service, mocks):
    mocks["history_service"].get_conversation.return_value = None
    mocks["history_service"].create_conversation.return_value = 1

    service.get_response(user_message="hi")

    calls = mocks["history_service"].add_message.call_args_list
    assert calls[0].args == (1, "user", "hi")
    assert calls[1].args == (1, "assistant", "the model's reply")


# --- web search / rag / context injection --------------------------------------

def test_web_search_results_are_injected_into_the_system_prompt(service, mocks):
    mocks["search_service"].search.return_value = [{"title": "t", "url": "u", "snippet": "s"}]
    mocks["search_service"].format_for_prompt.return_value = "Web search results:\n1. t (u)\n   s"

    service.get_response(user_message="what's new?", web_search=True)

    mocks["search_service"].search.assert_called_once_with("what's new?")
    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "Web search results" in system_message


def test_empty_web_search_results_are_not_injected(service, mocks):
    mocks["search_service"].search.return_value = []
    mocks["search_service"].format_for_prompt.return_value = ""

    service.get_response(user_message="what's new?", web_search=True)

    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "Web search results" not in system_message


def test_web_search_is_skipped_when_flag_is_false(service, mocks):
    service.get_response(user_message="hi", web_search=False)
    mocks["search_service"].search.assert_not_called()


def test_rag_context_is_injected_with_requested_top_k(service, mocks):
    mocks["rag_service"].retrieve_context.return_value = "[Source: doc.txt]\nrelevant text"

    service.get_response(user_message="question", rag=True, rag_top_k=8)

    mocks["rag_service"].retrieve_context.assert_called_once_with("question", top_k=8)
    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "relevant text" in system_message


def test_rag_requested_but_no_rag_service_configured_does_not_crash():
    model_service = MagicMock()
    model_service.generate.return_value = "reply"
    service = ChatService(
        model_service=model_service,
        search_service=MagicMock(),
        file_service=MagicMock(),
        rag_service=None,
        history_service=None,
    )

    result = service.get_response(user_message="question", rag=True)

    assert result.response == "reply"


def test_explicit_context_argument_is_included_in_system_prompt(service, mocks):
    service.get_response(user_message="hi", context="some extracted file text")

    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "some extracted file text" in system_message


# --- thinking mode / personality / language ------------------------------------

@pytest.mark.parametrize(
    "mode,expected_thinking,expected_multiplier",
    [
        (ThinkingMode.FAST, False, 0.5),
        (ThinkingMode.BALANCED, True, 1.0),
        (ThinkingMode.DEEP, True, 2.0),
    ],
)
def test_thinking_mode_controls_model_generation_parameters(
    service, mocks, mode, expected_thinking, expected_multiplier
):
    service.get_response(user_message="hi", thinking_mode=mode)

    _, kwargs = mocks["model_service"].generate.call_args
    assert kwargs["enable_thinking"] is expected_thinking
    assert kwargs["max_new_tokens_multiplier"] == expected_multiplier


def test_deep_thinking_mode_adds_extra_instruction_to_system_prompt(service, mocks):
    service.get_response(user_message="hi", thinking_mode=ThinkingMode.DEEP)
    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "multiple angles" in system_message


def test_language_instruction_is_always_present_regardless_of_input_language(service, mocks):
    service.get_response(user_message="hi", language=Language.PERSIAN)
    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "Persian" in system_message


def test_personality_instruction_is_included(service, mocks):
    service.get_response(user_message="hi", personality=Personality.HUMOROUS)
    system_message = _messages_passed_to_generate(mocks)[0]["content"]
    assert "humorous" in system_message.lower()


# --- delegation -----------------------------------------------------------------

def test_extract_file_text_delegates_to_file_service(service, mocks):
    mocks["file_service"].extract_text.return_value = "extracted"
    fake_file = MagicMock()

    result = service.extract_file_text(fake_file)

    mocks["file_service"].extract_text.assert_called_once_with(fake_file)
    assert result == "extracted"
