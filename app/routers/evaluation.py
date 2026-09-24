from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.dependencies import get_current_user
from app.models.user import User

from app.services.crag import crag_retrieve
from app.services.generation import generate_answer
from app.services.evaluation import evaluate_rag_response


router = APIRouter(
    prefix="/api/v1/evaluate",
    tags=["evaluation"]
)


class EvaluationRequest(BaseModel):
    question: str
    regime: str
    top_k: int = 5


@router.post("")
def evaluate_query(
    request: EvaluationRequest,
    current_user: User = Depends(get_current_user)
):

    # --------------------------------------------------
    # 1. CRAG RETRIEVAL
    # --------------------------------------------------

    chunks, crag_info = crag_retrieve(
        question=request.question,
        regime=request.regime,
        top_k=request.top_k
    )

    # --------------------------------------------------
    # 2. Extract context
    # --------------------------------------------------

    chunk_texts = [
        c["chunk_text"]
        for c in chunks
    ]

    # --------------------------------------------------
    # 3. Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        request.question,
        chunks
    )

    # --------------------------------------------------
    # 4. Evaluate generated answer
    # --------------------------------------------------

    scores = evaluate_rag_response(
        request.question,
        answer,
        chunk_texts
    )

    # --------------------------------------------------
    # 5. Return everything
    # --------------------------------------------------

    return {
        "question": request.question,

        "answer": answer,

        "crag": crag_info,

        "scores": scores,

        "retrieved_chunks": chunks
    }