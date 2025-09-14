from typing import List, Dict, Tuple
import numpy as np
import faiss

def _faiss_search(index: faiss.Index, query_vec: np.darray, k: int = 20) -> List[Tuple[int, float]]:
    D, I = index.search(query_vec, k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    return [(i, float(s)) for i, s in zip(ids, scores) if i!= -1]

def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 10, const: int = 60) -> List[int]:
    """RRF over lists of doc ids. Larger const flattens contributions"""
    scores = {}
    for rlist in ranked_lists:
        for rank, doc_id in enumerate(rlist):
            for rank, doc_id in enumerate(rlist):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (const + rank + 1.0)

                #top-k by fused score
                return [doc_id for doc_id, _ in sorted(scores.items(), key = lambda x: x[1], reverse = True)[:k]]

def hybrid_retriever(query: str,
                     faiss_index,
                     texts: List[str],
                     meta: List[Dict],
                     bm25_store,
                     embed_fn,
                     top_k: int = 6) -> List[Dict]:
    
    #dense search
    qvec = embed_fn([query])
    dense_hits = _faiss_search(faiss_index, qvec, k= 20)
    dense_ids = [i for i, _ in dense_hits]

    #bm25 search
    bm25_hits = bm25_store.top_n(query, n = 20)
    bm25_ids = [h["id"] for h in bm25_hits]

    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids], k=top_k)

    results = []
    for did in fused_ids:
        results.append({
            "id": did,
            "text": texts[did],
            "metadata": meta[did]
        })
    
    return results