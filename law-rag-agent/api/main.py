from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.schemas import (
    HealthResponse, SearchRequest, SearchResponse, RetrievedChunk,
    AnswerRequest, AnswerResponse
)
from pathlib import Path
from config.settings import settings
from rag.service import rag_runtime
from rag.retriever import hybrid_retrieve

DEFAULT_TOP_K = 6

app = FastAPI(title = settings.APP_NAME, version = settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins = settings.CORS_ALLOW_ORIGINS,
    allow_credentials = settings.CORS_ALLOW_CREDENTIALS,
    allow_methods = settings.CORS_ALLOW_METHODS,
    allow_headers = settings.CORS_ALLOW_HEADERS,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"[RAG] Loading index from: {settings.INDEX_DIR}")
        rag_runtime.load(settings.INDEX_DIR)
        print(f"Loaded index: {len(rag_runtime.texts)} chunks")
    except Exception as e:
        print(f"[WARN] RAG Index not loaded: {e}")
    yield

app = FastAPI(title=settings.APP_NAME,
              version=settings.APP_VERSION,
              lifespan=lifespan)

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    status = 'ok' if rag_runtime.faiss_index is not None else "degraded"
    return HealthResponse(status=status, service=settings.APP_NAME, version = settings.APP_VERSION)

def debug_index():
    p = Path(settings.INDEX_DIR).resolve()
    files = list(p.glob("*")) if p.exists() else []
    return {
        "cwd": str(Path.cwd()),
        "index_dir": str(p),
        "exists": p.exists(),
        "files": [str(f.name) for f in files][:20]  # first 20 files
    }
@app.post("/v1/search", response_model = SearchResponse, tags=["rag"])
def search(req: SearchRequest) -> SearchResponse:
    if rag_runtime.faiss_index is None:
        raise HTTPException(status_code=503, detail = 'Index not loaded. run ingest first')
    
    top_k = req.top_k or settings.TOP_K
    hits = hybrid_retrieve(
        req.query,
        rag_runtime.faiss_index,
        rag_runtime.texts,
        rag_runtime.meta,
        rag_runtime.bm25,
        rag_runtime.embed,
        top_k = top_k,
    )

    return SearchResponse(results=[RetrievedChunk(**h) for h in hits])


@app.post("/v1/answer", response_model=AnswerResponse, tags=["rag"])
def answer(req: AnswerRequest) -> AnswerResponse:
    if rag_runtime.faiss_index is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingest first.")

    top_k = req.top_k or settings.TOP_K
    hits = hybrid_retrieve(
        req.query,
        rag_runtime.faiss_index,
        rag_runtime.texts,
        rag_runtime.meta,
        rag_runtime.bm25,
        rag_runtime.embed,
        top_k=top_k,
    )

    if not hits:
        return AnswerResponse(answer="No relevant context found.", citations=[])

    paras, cites = [], []
    for h in hits:
        cite = f"[{h['source']}{f' p.{h['page']}' if h.get('page') else ''}]"
        paras.append(f"{h['text']}\n{cite}")
        cites.append(cite)

    stitched = "\n\n---\n\n".join(paras)

    return AnswerResponse(
        answer=f"(context-only draft)\nQ: {req.query}\n\n{stitched}",
        citations=list(dict.fromkeys(cites))
    )
