# app/models/schemas/query.py
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    regime: str
    top_k: int = 5
    response_language: str | None = None  # None = auto-detect from question


class Citation(BaseModel):
    chunk_text: str
    title: str | None = None
    regime: str | None = None
    language: str | None = None
    doc_type: str | None = None
    source_url: str | None = None
    distance: float | None = None


class QueryResponse(BaseModel):
    answer: str
    regime: str
    citations: list[Citation]


class CompareQueryRequest(BaseModel):
    question: str
    regimes: list[str]
    top_k: int = 5
    response_language: str | None = None


class RegimeAnswer(BaseModel):
    regime: str
    answer: str
    citations: list[Citation]


class CompareQueryResponse(BaseModel):
    per_regime_answers: list[RegimeAnswer]
    synthesis: str