import pickle
from pathlib import Path
import faiss
import numpy as np
from typing import List, Tuple

from config import settings

VECTOR_DIM = 384  # dimensions for MiniLM‑L6

class VectorCache:
    def __init__(self):
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.answers: list[str] = []  # parallel list
        if settings.vector_db_path.exists():
            self.load()

    # ‑‑ persistence
    def save(self):
        faiss.write_index(self.index, str(settings.vector_db_path))
        with open(settings.cache_path, "wb") as f:
            pickle.dump(self.answers, f)

    def load(self):
        self.index = faiss.read_index(str(settings.vector_db_path))
        try:
            with open(settings.cache_path, "rb") as f:
                self.answers = pickle.load(f)
        except FileNotFoundError:
            self.answers = []

    # ‑‑ API
    def add(self, embedding: np.ndarray, answer: str):
        self.index.add(embedding.astype("float32"))
        self.answers.append(answer)
        self.save()

    def search(self, embedding: np.ndarray, k: int) -> List[Tuple[float, str]]:
        """Search for k most similar items and return list of (score, answer) tuples."""
        if self.index.ntotal == 0:
            print("Vector database is empty!")
            return []
        
        print(f"Searching {self.index.ntotal} products with k={k}, threshold={settings.similarity_threshold}")
        
        D, I = self.index.search(embedding.astype("float32"), k)  # inner‑product similarity
        results = []
        
        print(f"Raw search results - scores: {D[0][:5]}, indices: {I[0][:5]}")
        
        for i in range(len(D[0])):
            score = float(D[0][i])
            idx = int(I[0][i])
            if score >= 0.2:#settings.similarity_threshold:
                results.append((score, self.answers[idx]))
        
        print(f"Filtered results: {len(results)} products above threshold")
        return results

vector_cache = VectorCache()