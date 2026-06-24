from pydantic import BaseModel, Field

class LivestockTelemetryInput(BaseModel):
    animal_id: str = Field(..., example="COW-0941")
    heart_rate: float = Field(..., example=88.5)
    temperature: float = Field(..., example=41.2)
    activity_level: float = Field(..., example=12.0)

class LivestockAnomalyResponse(BaseModel):
    animal_id: str
    is_anomaly: bool
    anomaly_score: float
    recommendation: str