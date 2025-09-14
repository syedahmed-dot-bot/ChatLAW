from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model(name: str = "sentence-transformers/all-MiniLM-l6-v2") -> SentenceTransformer:
    global _model 
    if _model is None:
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = get_model(model_name)
    arr = model.encode(texts, normalize_embeddings= True)
    return np.asarray(arr, dtype=np.float32)
