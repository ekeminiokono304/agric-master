from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.models.manager import model_manager
from app.routers import (
    yield_prediction, disease_classifier, price_forecaster, 
    livestock_anomaly, drought_risk
)
from app.exceptions import model_prediction_exception_handler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize singleton and load models at startup
    _ = model_manager
    yield
    # Cleanup actions if any go here

app = FastAPI(
    title="AGRIC-MASTER AI Platform Backend",
    version="1.0.0",
    lifespan=lifespan
)

# Exception Handlers
app.add_exception_handler(Exception, model_prediction_exception_handler)

# Include Routers
app.include_router(yield_prediction.router, prefix="/predict/yield", tags=["Yield"])
app.include_router(disease_classifier.router, prefix="/predict/disease", tags=["Disease"])
app.include_router(price_forecaster.router, prefix="/predict/price", tags=["Price"])
app.include_router(livestock_anomaly.router, prefix="/predict/livestock", tags=["Livestock"])
app.include_router(drought_risk.router, prefix="/predict/drought", tags=["Drought"])

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    status = {}
    for k, v in model_manager.models.items():
        status[f"{k}_model_loaded"] = v is not None
    
    overall = "healthy" if all(model_manager.models.values()) else "degraded"
    if len(model_manager.models) == 0:
        overall = "unhealthy"
        
    return {"status": overall, "components": status}