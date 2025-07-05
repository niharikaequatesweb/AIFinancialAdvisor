from sentence_transformers import SentenceTransformer
from functools import lru_cache

from config import settings

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(settings.embedding_model)

def embed(text: str):
    embedding = get_embedder().encode(text, normalize_embeddings=True)
    # Ensure 2D array for FAISS compatibility
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    return embedding