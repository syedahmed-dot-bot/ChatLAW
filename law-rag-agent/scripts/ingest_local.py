import argparse
from pathlib import Path
from rag.indexer import build_from_pdfs

def main():

    print("Starting ingestion---\n")
    
    p = argparse.ArgumentParser(description="Ingest PDFs into FAISS+BM25 index")
    p.add_argument("--corpus", type=str, default="data/corpus", help="Folder with PDFs")
    p.add_argument("--index", type=str, default="data/index", help="Folder to write index")
    p.add_argument("--chunk_size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    build_from_pdfs(
        corpus_dir=Path(args.corpus),
        index_dir=Path(args.index),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.model,
    )

if __name__ == "__main__":
    main()