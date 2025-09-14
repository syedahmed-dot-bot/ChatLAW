import json 
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np

from rag.chunkers import pdf_to_text_blocks, chunk_blocks
from rag.embeddings import embed_texts
from rag.bm25 import BM25Store

ARTIFACTS = {
    "texts": "texts.json",
    "meta": "meta.json",
    "faiss": "index.json",
    "bm25": "bm25.json"
}

def build_from_pdfs (corpus_dir: Path, index_dir: Path, 
                     chunk_size: int = 900, overlap: int = 120, 
                     embedding_model = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    
    index_dir.mkdir(parents = True, exist_ok= True)
    texts: List[str] = []
    meta: List[Dict] = []

    for pdf in sorted(corpus_dir.glob("**/*.pdf")):
        blocks = pdf_to_text_blocks(str(pdf))
        chunks = chunk_blocks(blocks, size= chunk_size, overlap = overlap)

        for ch in chunks:
            texts.append(ch["text"])
            m = dict(ch["metadata"])
            m.update({"source": str(pdf.name)})
            meta.append(m)
    
    # Persist texts & metadata
            (index_dir / ARTIFACTS["texts"]).write_text(json.dumps(texts, indent=2))
            (index_dir / ARTIFACTS["meta"]).write_text(json.dumps(meta, indent=2))


    if not texts:
        print("No texts found to index")
        return
    
    vecs = embed_texts(texts, model_name = embedding_model)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(index_dir / ARTIFACTS["faiss"]))

    (index_dir / ARTIFACTS["bm25"]).write_text(json.dumps({"count": len(texts)}))
    print(f"Indexed {len(texts)} chunks from {corpus_dir}")

def load_indexes(index_dir: Path) -> Tuple[faiss.Index, List[str], List[Dict], BM25Store]:
    texts = json.loads((index_dir / ARTIFACTS["texts"]).read_text())
    meta = json.loads((index_dir / ARTIFACTS['meta']).read_text())
    index = faiss.read_index(str(index_dir / ARTIFACTS['faiss']))
    bm25 = BM25Store(texts)
    return index, texts, meta, bm25

