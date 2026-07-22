import io
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI(
    title="Plant Classification API",
    description="API & Interface de classification d'images"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mot de passe d'accès
APP_PASSWORD = "admin"

# Reconstitution dynamique des chemins de fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Les 3 classes exactes
class_names = ['amarante', 'legumes', 'maracoudja']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # Recherche prioritaire dans le même dossier que main.py
    model_path = os.path.join(BASE_DIR, "plant_model.pth")
    if not os.path.exists(model_path):
        model_path = "plant_model.pth"
        
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Modèle chargé avec succès pour les 3 classes.")
        except Exception as e:
            print(f"Erreur de chargement des poids du modèle : {e}")
    else:
        print(f"Fichier '{model_path}' non trouvé.")
        
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocessing des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Servir le frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(BASE_DIR, "..", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Page web non trouvée.")

# Vérification du mot de passe
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
    
    # Seuil de rigueur professionnelle fixé à 85% minimum
    CONFIDENCE_THRESHOLD = 85.0
    is_known = conf_score >= CONFIDENCE_THRESHOLD

    return {
        "filename": file.filename,
        "prediction": pred_class.upper() if is_known else "Inconnu!",
        "is_known": is_known,
        "confidence": f"{conf_score:.2f}%",
        "probabilities": {
            class_names[i]: f"{float(probabilities[i].item()) * 100:.2f}%"
            for i in range(len(class_names))
        }
    }