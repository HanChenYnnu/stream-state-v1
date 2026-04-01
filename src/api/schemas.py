from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    state: str = Field(..., description="State code S0-S5")
    state_name_zh: str
    confidence: float
    scores: dict
    features: dict
