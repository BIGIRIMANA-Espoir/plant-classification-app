import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Hyperparamètres
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DATA_DIR = "dataset"  # Dossier contenant 'amarante', 'legumes', 'maracoudja'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Transformations avec augmentation de données (Rigueur Pro)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Résistance aux changements de lumière
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Chargement des données
dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
print(f"Classes détectées ({len(class_names)}) : {class_names}")

# 4. Modèle MobileNetV2 Pré-entraîné
model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)

# Adapter la sortie aux 3 classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 5. Perte et Optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Boucle d'entraînement
print(" Début de l'entraînement...")
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

    print(f"Époque {epoch+1}/{NUM_EPOCHS} - Perte: {epoch_loss:.4f} - Précision: {epoch_acc:.2f}%")

# 7. Sauvegarde du modèle entraîné
torch.save(model.state_dict(), "plant_model.pth")
print(" Entraînement terminé ! Modèle sauvegardé sous 'plant_model.pth'")
