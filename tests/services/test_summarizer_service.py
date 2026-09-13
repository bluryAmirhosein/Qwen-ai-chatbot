# tests/services/test_summarizer_service.py

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.summarizer_service import SummarizerService


@pytest.fixture
def mock_model_service():
    service = Mock()
    service.generate.return_value = "This is a generated summary."
    return service


@pytest.fixture
def settings():
    """Duck-typed stand-in: SummarizerService only reads .max_new_tokens."""
    return SimpleNamespace(max_new_tokens=512)


@pytest.fixture
def summarizer_service(mock_model_service, settings):
    return SummarizerService(model_service=mock_model_service, settings=settings)


class TestSummarize:
    def test_returns_expected_result_shape(self, summarizer_service, mock_model_service):
        mock_model_service.generate.return_value = "Short summary here."

        result = summarizer_service.summarize("Some long original text about a topic.", max_length=50)

        assert result == {
            "summary": "Short summary here.",
            "original_length": 7,  # word count of the input text
            "summary_length": 3,  # word count of the summary
        }

    def test_calls_model_service_with_general_system_prompt(self, summarizer_service, mock_model_service):
        summarizer_service.summarize("Some text to summarize.", max_length=100)

        args, kwargs = mock_model_service.generate.call_args
        messages = args[0]

        assert messages[0]["role"] == "system"
        assert "100" in messages[0]["content"]
        assert "summarization assistant" in messages[0]["content"]
        assert messages[1] == {"role": "user", "content": "Some text to summarize."}

    def test_passes_enable_thinking_false(self, summarizer_service, mock_model_service):
        summarizer_service.summarize("Some text.", max_length=50)

        _, kwargs = mock_model_service.generate.call_args
        assert kwargs["enable_thinking"] is False

    def test_strips_whitespace_from_input(self, summarizer_service, mock_model_service):
        summarizer_service.summarize("   text with padding   ", max_length=50)

        args, _ = mock_model_service.generate.call_args
        assert args[0][1]["content"] == "text with padding"

    def test_truncates_input_beyond_max_input_chars(self, summarizer_service, mock_model_service):
        long_text = "a" * (SummarizerService.MAX_INPUT_CHARS + 500)

        summarizer_service.summarize(long_text, max_length=50)

        args, _ = mock_model_service.generate.call_args
        user_content = args[0][1]["content"]
        assert len(user_content) == SummarizerService.MAX_INPUT_CHARS

    def test_does_not_truncate_input_within_limit(self, summarizer_service, mock_model_service):
        text = "a" * (SummarizerService.MAX_INPUT_CHARS - 100)

        summarizer_service.summarize(text, max_length=50)

        args, _ = mock_model_service.generate.call_args
        assert len(args[0][1]["content"]) == len(text)


class TestSummarizeByQuery:
    def test_returns_expected_result_shape(self, summarizer_service, mock_model_service):
        mock_model_service.generate.return_value = "Focused summary."

        result = summarizer_service.summarize_by_query(
            "Some long text with several topics in it.", query="topics", max_length=50
        )

        assert result["summary"] == "Focused summary."
        assert result["original_length"] == 8
        assert result["summary_length"] == 2

    def test_calls_model_service_with_query_focused_prompt(self, summarizer_service, mock_model_service):
        summarizer_service.summarize_by_query("Some text.", query="pricing", max_length=80)

        args, _ = mock_model_service.generate.call_args
        messages = args[0]

        assert messages[0]["role"] == "system"
        assert "80" in messages[0]["content"]
        assert "query" in messages[0]["content"].lower()
        assert "Key query: pricing" in messages[1]["content"]
        assert "Some text." in messages[1]["content"]

    def test_truncates_input_before_building_query_message(self, summarizer_service, mock_model_service):
        long_text = "b" * (SummarizerService.MAX_INPUT_CHARS + 200)

        summarizer_service.summarize_by_query(long_text, query="q", max_length=50)

        args, _ = mock_model_service.generate.call_args
        user_content = args[0][1]["content"]
        # message wraps the (already truncated) text, so it must not contain
        # more than MAX_INPUT_CHARS worth of the original filler text
        assert user_content.count("b") == SummarizerService.MAX_INPUT_CHARS


class TestTokensMultiplier:
    @pytest.mark.parametrize(
        "max_length_words, max_new_tokens, expected",
        [
            (150, 512, max(int(150 * 1.8), 64) / 512),
            (10, 512, max(int(10 * 1.8), 64) / 512),  # hits the 64-token floor
            (1000, 512, max(int(1000 * 1.8), 64) / 512),
        ],
    )
    def test_multiplier_formula(
        self, summarizer_service, mock_model_service, max_length_words, max_new_tokens, expected
    ):
        summarizer_service._settings.max_new_tokens = max_new_tokens

        summarizer_service.summarize("text", max_length=max_length_words)

        _, kwargs = mock_model_service.generate.call_args
        assert kwargs["max_new_tokens_multiplier"] == pytest.approx(expected)