from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.disease_schema import DiseaseResponse, DiseaseClassProba
from app.models.manager import model_manager
from PIL import Image
import numpy as np
import io

router = APIRouter()
CLASSES = ["Healthy Tomato", "Tomato Late Blight", "Corn Common Rust"]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img).astype(np.float32) / 255.0
    # Normalize channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_arr = (img_arr - mean) / std
    img_arr = img_arr.transpose(2, 0, 1) # HWC to CHW
    return np.expand_dims(img_arr, axis=0).astype(np.float32)

@router.post("/", response_model=DiseaseResponse)
async def classify_disease(file: UploadFile = File(...)):
    try:
        ort_session = model_manager.get_model('disease')
    except Exception:
        raise HTTPException(status_code=503, detail="Disease classification model unavailable.")

    try:
        contents = await file.read()
        tensor = preprocess_image(contents)
        
        # ONNX Inference Run
        inputs = {ort_session.get_inputs()[0].name: tensor}
        outs = ort_session.run(None, inputs)[0][0]
        
        # Softmax conversion
        exp_outs = np.exp(outs - np.max(outs))
        probs = exp_outs / exp_outs.sum()
        
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        
        # Derive demo severity parameters based on probability confidence scaling
        severity = "High" if top_prob > 0.85 else "Moderate" if top_prob > 0.5 else "Low Risk"
        if "Healthy" in CLASSES[top_idx]:
            severity = "Healthy Field Profile"

        prob_list = [
            DiseaseClassProba(disease_class=CLASSES[i], probability=float(probs[i]))
            for i in range(len(CLASSES))
        ]
        
        return DiseaseResponse(
            top_prediction=CLASSES[top_idx],
            severity_assessment=severity,
            probabilities=prob_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image verification breakdown: {str(e)}")