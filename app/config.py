import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AGRIC-MASTER AI Platform"
    DEBUG: bool = False
    MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "artifacts")
    
    class Config:
        env_file = ".env"

settings = Settings()