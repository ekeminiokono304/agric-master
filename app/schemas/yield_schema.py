from pydantic import BaseModel, Field

class YieldRequest(BaseModel):
    soil_type: str = Field(..., example="Loam")
    crop_type: str = Field(..., example="Maize")
    region: str = Field(..., example="North")
    rainfall: float = Field(..., example=1150.5)
    fertilizer_use: float = Field(..., example=145.0)

class YieldResponse(BaseModel):
    predicted_yield: float
    confidence_lower: float
    confidence_upper: float