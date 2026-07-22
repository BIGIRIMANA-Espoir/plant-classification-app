import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Chemin vers le modèle sauvegardé
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.pth")
DATA_DIR = r"D:\BIRIMANA\MQTT_Scripts\dataset_plantes"

def evaluate_test_set():
    device = torch.device("cpu")

    # 1. Charger le dataset pour détecter automatiquement les 3 classes
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dossier introuvable : {DATA_DIR}")

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_names = dataset.classes
    num_classes = len(class_names)

    print(f"Classes chargées ({num_classes}) : {class_names}")

    # 2. Instancier le modèle MobileNetV2
    model = models.mobilenet_v2(weights=None)
    
    # Ajuster la couche de sortie à 3 classes AVANT de charger les poids
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # 3. Charger le fichier .pth
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model = model.to(device)
    model.eval()

    # 4. Évaluation sur les images
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = (correct / total) * 100
    print(f"\n Précision du modèle évaluée sur le test set : {accuracy:.2f}% ({correct}/{total} images correctes)")

if __name__ == '__main__':
    evaluate_test_set()