import numpy as np
from embeddings import embed
from cache import vector_cache

def maybe_from_cache(query: str):
    vec = embed(query).reshape(1, -1)
    score, answer = vector_cache.search(vec)
    return score, answer, vec