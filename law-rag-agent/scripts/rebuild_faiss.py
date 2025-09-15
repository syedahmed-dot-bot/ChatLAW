# scripts/rebuild_faiss.py
from pathlib import Path
import json
import faiss
from rag.embeddings import embed_texts

INDEX_DIR = Path("data/index").resolve()
texts_p = INDEX_DIR / "texts.json"
faiss_p = INDEX_DIR / "index.faiss"

print(f"[rebuild] Index dir: {INDEX_DIR}")
if not texts_p.exists():
    raise FileNotFoundError(f"Missing {texts_p}. Run ingest first.")

texts = json.loads(texts_p.read_text(encoding="utf-8"))
if not texts:
    raise ValueError("texts.json is empty")

print(f"[rebuild] Embedding {len(texts)} chunks...")
vecs = embed_texts(texts)  # normalized float32
dim = vecs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vecs)
faiss.write_index(index, str(faiss_p))
print(f"[rebuild] Wrote {faiss_p} (size={faiss_p.stat().st_size} bytes)")
