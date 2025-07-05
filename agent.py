import asyncio
from typing import List, Tuple
import json

from cache import vector_cache
from embeddings import embed
from search import search_and_scrape
from summarizer import summarize
from similarity import maybe_from_cache


async def ask_agent(query: str, profile: dict) -> Tuple[str, List[str]]:
    """Main orchestrator – fully async."""
    # Check semantic cache
    score, cached_answer, vec = maybe_from_cache(query)
    if cached_answer:
        return f"(⚡ from cache, similarity={score:.2f})" + cached_answer, []

    # Live search + scrape
    pages: List[Tuple[str, str]] = search_and_scrape(query)
    if not pages:
        return "I couldn’t find information on that topic.", []

    urls, texts = zip(*pages)  # Unzip list of tuples

    # Summarize with open-source HF model
    answer = await summarize(list(texts), list(urls))

    # Persist to vector cache
    vector_cache.add(vec, answer)

    # Combine profile and query for LLM analysis
    llm_input = {
        "profile": profile,
        "query": query,
        "similarity_results": answer
    }
    llm_response = await analyze_with_llm(llm_input)

    return llm_response

async def analyze_with_llm(input_data: dict) -> str:
    """Send data to LLM for analysis and response generation."""
    # Simulate LLM call (replace with actual implementation)
    return f"Generated response based on profile: {input_data['profile']} and query: {input_data['query']}"

def process_finco_data():
    """Read finco.json and convert data to vector embeddings."""
    with open("data/finco.json", "r") as f:
        data = json.load(f)

    for item in data:
        text_representation = f"{item['Category']} {item['Sub_Category']} {item['Provider']} {item['Product_Name']} {item['USP']} {item['Key_Features']}"
        embedding = embed(text_representation,1)
        print(f"Embedding shape: {embedding.shape}")
        vector_cache.add(embedding, text_representation)

if __name__ == "__main__":
    process_finco_data()