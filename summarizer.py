"""Chunk‑aware summarizer leveraging the Hugging Face *Inference API*.
No local model download required. Each chunk (≈800 words) is summarised remotely,
then optionally compressed again if multiple partial summaries exist.
"""

import asyncio
from functools import lru_cache
from typing import List
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import settings

# ––––– helpers –––––––

def _split_into_chunks(text: str, max_words: int = 100) -> List[str]:
    """Split text into chunks with a maximum of `max_words` words."""
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

async def summarize(excerpts: List[str], urls: List[str]) -> str:
    merged_text = "".join(excerpts)
    chunks = _split_into_chunks(merged_text, max_words=50)  # Limit chunks to 50 words

    # Process each chunk sequentially
    partials = []
    for chunk in chunks:
        partial = await _summarize_chunk(chunk)
        partials.append(partial)

    # Compress again if we produced >1 partial summaries
    summary_text = " ".join(partials)
    if len(partials) > 1:
        summary_text = await _summarize_chunk(summary_text[:500])

    sources = "".join(f"[{i+1}] {u}" for i, u in enumerate(urls))
    return f"{summary_text.strip()}Sources:{sources}"

@lru_cache(maxsize=10)
def get_client() -> InferenceClient:
    return InferenceClient(
        provider="hf-inference",
        api_key=settings.hf_token)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(4))
async def _summarize_chunk(chunk: str) -> str:
    client = get_client()
    print("Summarizer Model:", settings.summarizer_model)
    print("Chunk length:", len(chunk))
    print("Summarizing chunk...")

    # Call the synchronous method directly
    response = client.summarization(
        text=chunk,
        model=settings.summarizer_model,
    )
    print("Chunk summarized.")

    # Response can be list or dict depending on the model
    if isinstance(response, list):
        response = response[0]
    return response.get("summary_text") or response.get("generated_text") or ""
