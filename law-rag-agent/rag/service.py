# rag/service.py
from pathlib import Path
import json
from typing import List, Dict, Optional, Union
import faiss
from rag.embeddings import embed_texts
from rag.bm25 import BM25Store

class RagRuntime:
    def __init__(self) -> None:
        self.faiss_index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.meta: List[Dict] = []
        self.bm25: Optional[BM25Store] = None

    def _artifact_paths(self, index_dir: Union[str, Path]) -> Dict[str, Path]:
        p = Path(index_dir).resolve()
        return {
            "index_dir": p,
            "texts":     p / "texts.json",
            "meta":      p / "meta.json",
            "faiss":     p / "index.faiss",
        }

    def load(self, index_dir: Union[str, Path]) -> None:
        paths = self._artifact_paths(index_dir)
        missing = [k for k, v in paths.items() if k != "index_dir" and not v.exists()]
        if missing:
            # Print detailed status so we see what's there vs missing
            print("[RAG] Artifact check:")
            for k, v in paths.items():
                if k == "index_dir": 
                    print(f"  {k}: {v}  (exists={v.exists()})")
                else:
                    print(f"  {k}: {v}  (exists={v.exists()})")
            raise FileNotFoundError(f"Missing artifacts: {missing}")

        self.texts = json.loads(paths["texts"].read_text(encoding="utf-8"))
        self.meta  = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.faiss_index = faiss.read_index(str(paths["faiss"]))
        self.bm25 = BM25Store(self.texts)

    def embed(self, texts: List[str]):
        return embed_texts(texts)

rag_runtime = RagRuntime()
