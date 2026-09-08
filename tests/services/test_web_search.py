from unittest.mock import MagicMock, patch

from app.services.web_search import WebSearchService


def test_search_maps_ddgs_results_to_expected_shape():
    fake_results = [
        {"title": "Result A", "href": "https://a.example", "body": "snippet a"},
        {"title": "Result B", "href": "https://b.example", "body": "snippet b"},
    ]
    with patch("app.services.web_search.DDGS") as mock_ddgs_cls:
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = fake_results
        mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs

        service = WebSearchService(max_results=5, timeout=10)
        results = service.search("python tutorials")

    assert results == [
        {"title": "Result A", "url": "https://a.example", "snippet": "snippet a"},
        {"title": "Result B", "url": "https://b.example", "snippet": "snippet b"},
    ]
    mock_ddgs.text.assert_called_once_with("python tutorials", max_results=5)


def test_search_returns_empty_list_on_backend_failure_instead_of_raising():
    with patch("app.services.web_search.DDGS") as mock_ddgs_cls:
        mock_ddgs_cls.return_value.__enter__.side_effect = RuntimeError("network down")

        service = WebSearchService()
        results = service.search("anything")

    assert results == []


def test_search_handles_missing_fields_in_provider_response_gracefully():
    with patch("app.services.web_search.DDGS") as mock_ddgs_cls:
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [{}]  # no title/href/body keys at all
        mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs

        service = WebSearchService()
        results = service.search("anything")

    assert results == [{"title": "", "url": "", "snippet": ""}]


def test_format_for_prompt_with_no_results_returns_empty_string():
    assert WebSearchService.format_for_prompt([]) == ""


def test_format_for_prompt_numbers_and_includes_url_and_snippet():
    results = [
        {"title": "Result A", "url": "https://a.example", "snippet": "snippet a"},
        {"title": "Result B", "url": "https://b.example", "snippet": "snippet b"},
    ]

    formatted = WebSearchService.format_for_prompt(results)

    assert formatted.startswith("Web search results:")
    assert "1. Result A (https://a.example)" in formatted
    assert "2. Result B (https://b.example)" in formatted
    assert "snippet a" in formatted
    assert "snippet b" in formatted
