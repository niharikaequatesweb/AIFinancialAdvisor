from  serpapi.google_search import GoogleSearch
import requests, urllib.parse
from bs4 import BeautifulSoup
from typing import List, Tuple

from config import settings
from cache import vector_cache
from embeddings import embed

MAX_PAGES = 5  # how many organic results to fetch
USER_AGENT = "Mozilla/5.0 (compatible; QueryAgent/1.0)"

# — helper to extract readable text —

def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    return "".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))[:4000]

# — main function —

def search_and_scrape(query: str) -> List[Tuple[str, str]]:
    """Perform similarity search in the vector store."""
    query_embedding = embed(query)
    results = vector_cache.search(query_embedding)
    pages: List[Tuple[str, str]] = []
    for result in results:
        if result and 'metadata' in result and 'source' in result['metadata'] and 'text' in result:
            pages.append((result['metadata']['source'], result['text']))
    return pages