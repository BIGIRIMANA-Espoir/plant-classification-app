import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os

app = FastAPI(
    title="Plant Classification API",
    description="API & Interface de classification d'images"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mot de passe simple pour l'accès
APP_PASSWORD = "admin"  # Tu pourras le changer ici !

# Configuration du modèle
class_names = ['amarante', 'legumes']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    # Utilisation de MobileNetV2 pour correspondre au fichier plant_model.pth
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model_path = "plant_model.pth"
    if not os.path.exists(model_path):
        model_path = "backend/plant_model.pth"
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Modèle chargé avec succès.")
    else:
        print("⚠️ Fichier du modèle non trouvé.")
        
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route pour la page principale HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join("backend", "index.html")
    if not os.path.exists(index_path):
        index_path = "index.html"
    return FileResponse(index_path)

# Route de vérification du mot de passe
@app.post("/verify-password")
async def verify_password(password: str = Form(...)):
    if password == APP_PASSWORD:
        return {"status": "ok", "message": "Accès autorisé"}
    raise HTTPException(status_code=401, detail="Mot de passe incorrect")

# Route de prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    pred_class = class_names[predicted_idx.item()]
    conf_score = float(confidence.item()) * 100
    
    # Seuil de reconnaissance (si confiance < 50%, objet considéré comme inconnu)
    is_known = conf_score >= 50.0

    return {
        "filename": file.filename,
        "prediction": pred_class if is_known else "Inconnu / Non reconnu",
        "is_known": is_known,
        "confidence": f"{conf_score:.2f}%",
        "probabilities": {
            class_names[i]: f"{float(probabilities[i].item()) * 100:.2f}%"
            for i in range(len(class_names))
        }
    }