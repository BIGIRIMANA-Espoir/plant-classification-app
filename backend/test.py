import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from preprocessing import ImagePreprocessor

def evaluate_test_set():
    base_dir = Path("Plant-Classification-1")
    if not base_dir.exists():
        base_dir = Path("backend/Plant-Classification-1")

    # 1. Préparation des données de test
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    preprocessor.fit_classes(base_dir / "train")

    X_test, y_test = preprocessor.process_split(base_dir / "test")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 2. Reconstitution de la structure du modèle
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(preprocessor.class_to_idx))

    # 3. Chargement des poids entraînés
    model_path = Path("backend/plant_model.pth")
    if not model_path.exists():
        model_path = Path("plant_model.pth")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 4. Évaluation
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = outputs.max(1)
        
        correct = predicted.eq(y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = (correct / total) * 100

    print("\n" + "="*45)
    print("      RÉSULTATS DE L'ÉVALUATION TEST")
    print("="*45)
    print(f"Total d'images de test : {total}")
    print(f"Prédictions correctes : {correct}/{total}")
    print(f"Précision sur le Test : {accuracy:.2f}%")
    print("="*45 + "\n")

    # Détail par image
    idx_to_class = preprocessor.idx_to_class
    print("Détail des prédictions :")
    for i in range(total):
        vrai = idx_to_class[y_test_tensor[i].item()]
        predit = idx_to_class[predicted[i].item()]
        status = " OK" if vrai == predit else " ERREUR"
        print(f"  - Image {i+1:02d}: Réel = {vrai:<10} | Prédit = {predit:<10} [{status}]")

if __name__ == "__main__":
    evaluate_test_set()