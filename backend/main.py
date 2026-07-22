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

# Configuration des 3 classes exactes
class_names = ['amarante', 'legumes', 'maracoudja']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    # Architecture MobileNetV2 configurée pour 3 classes
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model_path = "plant_model.pth"
    if not os.path.exists(model_path):
        model_path = "backend/plant_model.pth"
        
    if os.path.exists(model_path):
        # Chargement sécurisé avec gestion d'erreurs
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(" Modèle chargé avec succès pour 3 classes.")
        except Exception as e:
            print(f" Erreur de correspondance des poids : {e}")
            print(" Assurez-vous d'avoir ré-entraîné le modèle avec les 3 classes.")
    else:
        print(" Fichier du modèle non trouvé.")
        
    model.to(device)
    model.eval()
    return model

model = load_model()

# Transformations d'images (Normalisation Standard ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route Frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join("backend", "index.html")
    if not os.path.exists(index_path):
        index_path = "index.html"
    return FileResponse(index_path)

# Route Authentification
@app.post("/verify-password")
async def verify_password(password: str = Form(...)):
    if password == APP_PASSWORD:
        return {"status": "ok", "message": "Accès autorisé"}
    raise HTTPException(status_code=401, detail="Mot de passe incorrect")

# Route de prédiction avec rigueur élevée
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image validée.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    pred_class = class_names[predicted_idx.item()]
    conf_score = float(confidence.item()) * 100
    
    # En dessous de 85%, le résultat est jugé trop incertain.
    CONFIDENCE_THRESHOLD = 85.0
    is_known = conf_score >= CONFIDENCE_THRESHOLD

    return {
        "filename": file.filename,
        "prediction": pred_class.upper() if is_known else "Inconnu / Certitude insuffisante",
        "is_known": is_known,
        "confidence": f"{conf_score:.2f}%",
        "probabilities": {
            class_names[i]: f"{float(probabilities[i].item()) * 100:.2f}%"
            for i in range(len(class_names))
        }
    }