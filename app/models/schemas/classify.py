from pydantic import BaseModel

class ClassifyRequest(BaseModel):
    description: str  # e.g. "tablet with turmeric and ashwagandha"
    regime: str

class ClassifyResponse(BaseModel):
    formulation_type: str
    reasoning: str
    ip_posture: str
    clarifying_questions: list[str]
