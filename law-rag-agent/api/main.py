from fastapi import FastAPI, HTTPException, Query
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
from agents.drafting import generate_answer
from groq import Groq

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

@app.get("/debug/groq_ping", tags=["meta"])
def groq_ping(model: str = Query(default=None, description="Override model (optional)")):
    mdl = model or settings.LLM_MODEL
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": "Say: ping"}],
            max_tokens=10,
            temperature=0.0,
        )
        return {
            "ok": True,
            "model": mdl,
            "content": resp.choices[0].message.content,
            "usage": getattr(resp, "usage", None),
        }
    except Exception as e:
        return {"ok": False, "model": mdl, "error": repr(e)}
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

    content = generate_answer(req.query, hits)

    # Collect unique citations (order-preserving)
    cites = []
    for h in hits:
        c = f"[{h['source']}{f' p.{h['page']}' if h.get('page') else ''}]"
        if c not in cites:
            cites.append(c)

    return AnswerResponse(answer=content, citations=cites)

