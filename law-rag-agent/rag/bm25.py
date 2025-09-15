# rag/bm25.py
from typing import List, Dict
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, texts: List[str]):
        # keep explicit, non-underscore names to avoid confusion
        self.texts: List[str] = list(texts)
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def top_n(self, query: str, n: int = 10) -> List[Dict]:
        # return the top-n hit ids + text + score
        scores = self.bm25.get_scores(query.split())
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [{"id": i, "text": self.texts[i], "score": float(scores[i])} for i in idxs]
