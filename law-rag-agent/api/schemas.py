from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class RetrievedChunk(BaseModel):
    id: int
    text: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[RetrievedChunk]

class AnswerRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    citations: List[str] = []