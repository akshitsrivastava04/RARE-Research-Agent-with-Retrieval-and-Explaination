import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import List, Dict

class SearchService:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Performs a web search using DuckDuckGo.
        Returns a list of dictionaries with 'title', 'href', and 'body'.
        """
        results = []
        try:
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=self.max_results)
                for r in ddgs_gen:
                    results.append(r)
        except Exception as e:
            print(f"Search error: {e}")
        return results

    def crawl(self, url: str) -> str:
        """
        Fetches the content of a URL and extracts text using BeautifulSoup.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Break into lines and remove leading and trailing whitespace
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit to 5000 characters for context
        except Exception as e:
            print(f"Crawl error for {url}: {e}")
            return ""

    async def get_search_context(self, query: str) -> str:
        """
        Performs search and crawls top results to build a context string.
        """
        results = self.search(query)
        if not results:
            return ""

        context_parts = []
        for res in results:
            url = res.get('href')
            title = res.get('title')
            content = self.crawl(url)
            if content:
                context_parts.append(f"Source: {title} ({url})\nContent: {content}\n")
            else:
                context_parts.append(f"Source: {title} ({url})\nSnippet: {res.get('body')}\n")

        return "\n---\n".join(context_parts)
