# app/routers/regimes.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["regimes"])

REGIMES = [
    {"code": "IN-AYUSH", "label": "India — AYUSH", "language": "hi/en"},
    {"code": "US-FDA", "label": "United States — FDA", "language": "en"},
    {"code": "CN-NMPA", "label": "China — NMPA", "language": "zh/en"},
]
import requests
from app.config import settings

@router.get("/debug-hf")
def debug_hf_connectivity():
    url = f"https://router.huggingface.co/hf-inference/models/{settings.EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
            json={"inputs": ["test"], "options": {"wait_for_model": True}},
            timeout=30,
        )
        return {
            "status_code": response.status_code,
            "body": response.text[:500],
        }
    except Exception as e:
        return {"error": repr(e)}

@router.get("/regimes")
def get_regimes():
    return REGIMES
