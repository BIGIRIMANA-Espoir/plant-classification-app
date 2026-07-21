import io
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Plant Classification API",
    description="API de classification d'images pour amarante et légumes",
    version="1.0.0"
)

# Configuration CORS pour autoriser la communication avec le Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["amarante", "legumes"]
MODEL_PATH = Path("backend/plant_model.pth")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("plant_model.pth")

# Chargement du modèle au démarrage de l'API
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASSES))

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
else:
    raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")

def preprocess_single_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de classification de plantes fonctionnelle"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni doit être une image.")

    try:
        contents = await file.read()
        tensor = preprocess_single_image(contents)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = CLASSES[predicted_idx.item()]
        score = round(confidence.item() * 100, 2)

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{score}%",
            "probabilities": {
                CLASSES[i]: f"{round(probabilities[i].item() * 100, 2)}%"
                for i in range(len(CLASSES))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")