

from fastapi import APIRouter, Depends
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.schemas.classify import ClassifyRequest, ClassifyResponse
from app.services.classification import classify_formulation

router = APIRouter(prefix="/api/v1/classify", tags=["classify"])

@router.post("", response_model=ClassifyResponse)
def classify(request: ClassifyRequest, current_user: User = Depends(get_current_user)):
    result = classify_formulation(request.description, request.regime)
    return ClassifyResponse(**result)
