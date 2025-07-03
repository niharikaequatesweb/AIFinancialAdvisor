from sentence_transformers import SentenceTransformer
from functools import lru_cache

from config import settings

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(settings.embedding_model)

def embed(text: str):
    return get_embedder().encode(text, normalize_embeddings=True)