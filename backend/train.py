import os
# Réduction des threads pour garantir la stabilité de la RAM CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Hyperparamètres optimisés pour un entraînement stable
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = r"D:\BIRIMANA\MQTT_Scripts\dataset_plantes"

def main():
    device = torch.device("cpu")
    print(f"Calcul exécuté sur : {device}")

    # Transformations d'images standards pour MobileNetV2
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dossier introuvable : {DATA_DIR}")

    # Chargement des données
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    class_names = dataset.classes
    print(f"Classes détectées ({len(class_names)}) : {class_names}")
    print(f"Total d'images à traiter : {len(dataset)}")

    # Transfer Learning : MobileNetV2 pré-entraîné
    model = models.mobilenet_v2(weights="DEFAULT")
    
    # Geler les premières couches pour un entraînement CPU ultra-rapide et stable
    for param in model.parameters():
        param.requires_grad = False

    # Adapter la couche de sortie à nos 3 classes (amarante, legumes, maracoudja)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)

    print("\n Début de l'entraînement...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = (correct_preds.double() / total_preds) * 100

        print(f"Époque {epoch+1:02d}/{NUM_EPOCHS} | Perte: {epoch_loss:.4f} | Précision: {epoch_acc:.2f}%")

    # Sauvegarde du modèle dans le dossier backend
    output_path = os.path.join(os.path.dirname(__file__), "plant_model.pth")
    torch.save(model.state_dict(), output_path)
    print(f"\n Modèle sauvegardé avec succès dans : '{output_path}'")

if __name__ == '__main__':
    main()