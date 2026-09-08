import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


class WebSearchService:
    """Thin wrapper around a web search backend.

    Kept as its own layer so the search provider can be swapped
    (DuckDuckGo -> Bing/Tavily/SerpAPI/etc.) without touching ChatService.
    """

    def __init__(self, max_results: int = 5, timeout: int = 10):
        self._max_results = max_results
        self._timeout = timeout

    def search(self, query: str) -> list[dict]:
        try:
            with DDGS(timeout=self._timeout) as ddgs:
                results = list(ddgs.text(query, max_results=self._max_results))
        except Exception:
            logger.exception("Web search failed for query: %s", query)
            return []

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]

    @staticmethod
    def format_for_prompt(results: list[dict]) -> str:
        if not results:
            return ""

        lines = ["Web search results:"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r['title']} ({r['url']})\n   {r['snippet']}")
        return "\n".join(lines)