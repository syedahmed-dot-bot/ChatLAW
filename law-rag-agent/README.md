# Law RAG Agent (voice-ready) — backend-first

A backend-first, agentic **Lawyer–Client Conversational RAG** project.  
MVP goals:
- **RAG** over uploaded legal docs (NDA/MSA/etc.) with strict citations
- **Agentic** orchestration (intake → research → drafting → compliance)
- **Voice-ready** (ASR/TTS integration later)
- **Evaluation harness** (retrieval precision, groundedness)

> ⚠️ Educational use only — **not legal advice**.

## Roadmap (backend-first)
- [ ] Ingestion & indexing (PDF → chunks → embeddings + BM25)
- [ ] Hybrid retrieval (dense + BM25; reciprocal rank fusion)
- [ ] Orchestrator (LangGraph) with Intake/Research/Drafting/Compliance agents
- [ ] Answer formatting (citations + “insufficient context” guardrail)
- [ ] Eval harness (RAGAS; WER for ASR later)
- [ ] Audio (Whisper/Coqui) integration
- [ ] Streamlit UI integration (final step)

## Repo layout (initial)
```
law-rag-agent/
  api/            # FastAPI endpoints (to be added)
  agents/         # agent nodes (to be added)
  rag/            # chunking, embeddings, vector stores (to be added)
  tools/          # ASR, TTS, PII, deadline calc (to be added)
  config/         # settings + prompts
  data/
    corpus/       # put sample PDFs here
    index/        # vector/BM25 artifacts (gitignored)
  evals/          # evaluation datasets & scripts
  scripts/        # dev helper scripts
  tests/          # unit tests
```

## Quick start
1) Create a new GitHub repo (public recommended).  
2) Clone or initialize locally, then add these files and push (see commands in the chat message).  
3) Next steps (we’ll do step-by-step in this chat): add **FastAPI skeleton** and **ingestion script**.

## License
MIT (see `LICENSE`).
