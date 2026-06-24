import os
import joblib
import onnxruntime as ort
import logging
from app.config import settings

logger = logging.getLogger("agric-master")

class ModelManager:
    """Singleton pattern to load ML models once at startup with lazy loading capability."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.model_dir = settings.MODEL_DIR
        self.models = {}
        self.load_all_models()
        self._initialized = True

    def load_all_models(self):
        # 1. Crop Yield (LightGBM / Joblib)
        self.models['yield'] = self._load_joblib("crop_yield_model.pkl")
        
        # 2. Plant Disease (ONNX PyTorch CNN)
        self.models['disease'] = self._load_onnx("plant_disease_model.onnx")
        
        # 3. Price Forecaster (LightGBM Recursive / Joblib)
        self.models['price'] = self._load_joblib("price_forecaster_model.pkl")
        
        # 4. Livestock Anomaly (Isolation Forest / Joblib)
        self.models['livestock'] = self._load_joblib("livestock_anomaly_model.pkl")
        
        # 5. Drought Risk (LightGBM Classifier / Joblib)
        self.models['drought'] = self._load_joblib("drought_risk_model.pkl")

    def _load_joblib(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            try:
                logger.info(f"Loading joblib model: {filename}")
                return joblib.load(path)
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                return None
        logger.warning(f"Model file missing: {filename}. Endpoint will return 503.")
        return None

    def _load_onnx(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            try:
                logger.info(f"Loading ONNX runtime session: {filename}")
                # Optimize for single-thread/CPU usage on free hosting
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                return ort.InferenceSession(path, sess_options=opts)
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                return None
        logger.warning(f"Model file missing: {filename}. Endpoint will return 503.")
        return None

    def get_model(self, name: str):
        model = self.models.get(name)
        if model is None:
            raise RuntimeError(f"Model '{name}' is not loaded or unavailable.")
        return model

model_manager = ModelManager()