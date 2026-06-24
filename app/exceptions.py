from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("agric-master")

async def model_prediction_exception_handler(request: Request, exc: Exception):
    logger.error(f"Prediction failed: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Model inference failed. Ensure model inputs are valid and artifacts are loaded."},
    )