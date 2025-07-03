from  serpapi.google_search import GoogleSearch
import requests, urllib.parse
from bs4 import BeautifulSoup
from typing import List, Tuple

from config import settings

MAX_PAGES = 5  # how many organic results to fetch
USER_AGENT = "Mozilla/5.0 (compatible; QueryAgent/1.0)"

# — helper to extract readable text —

def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    return "".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))[:4000]

# — main function —

async def search_and_scrape(query: str) -> List[Tuple[str, str]]:
    search = GoogleSearch({
        "q": query,
        "api_key": settings.serpapi_key,
        "num": MAX_PAGES,
        "hl": "en",
    })
    results = search.get_dict()
    # print(results)
    organic = results.get("organic_results", [])[:MAX_PAGES]
    print(f"Found {len(organic)} organic results for query: {query}")
    pages: List[Tuple[str, str]] = []
    for res in organic:
        url = res.get("link")
        if not url:
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            resp.raise_for_status()
            text = _extract_text(resp.text)
            if len(text) > 200:
                pages.append((url, text))
        except Exception:
            continue
    return pages