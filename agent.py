import asyncio
from typing import Optional, List, Tuple

from cache import vector_cache
from embeddings import embed
from search import search_and_scrape
from summarizer import summarize
from similarity import maybe_from_cache

INVALID_TRIGGERS = {
    "walk my pet", "add apples to grocery",  # extend as needed
}

def is_valid(query: str) -> bool:
    return not any(trigger in query.lower() for trigger in INVALID_TRIGGERS)

async def ask_agent(query: str) -> str:
    """Main orchestrator – fully async."""
    if not is_valid(query):
        return "This is not a valid query."

    # 1️⃣ check semantic cache
    score, cached_answer, vec = maybe_from_cache(query)
    if cached_answer:
        return f"(⚡ from cache, similarity={score:.2f})" + cached_answer

    # 2️⃣ live search + scrape (await!)
    pages: List[Tuple[str, str]] = await search_and_scrape(query)
    if not pages:
        return "I couldn’t find information on that topic."

    urls, texts = zip(*pages)  # unzip list of tuples

    # 3️⃣ summarise with open‑source HF model
    answer = await summarize(list(texts), list(urls))

    # 4️⃣ persist to vector cache
    vector_cache.add(vec, answer)
    return answer