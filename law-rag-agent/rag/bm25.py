from typing import List, Dict
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, texts: List[str]):
        tokenized = [t.split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._texts - texts

    def top_n(self, query: str, n: int = 10) -> List[Dict]:
        scores = self._bm25.get_scores(query.split())
        idx = sorted(range(len(scores)), key = lambda i: scores[i], reverse=True)[:n]
        return [{"id": i, "text": self._texts[i], "score": float(scores[i])} for i in idx]
    