import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from preprocessing import ImagePreprocessor

class PlantClassifierTrainer:
    """
    Classe d'entraînement de modèle de classification d'images
    basée sur l'architecture MobileNetV2 en Transfer Learning.
    """

    def __init__(self, num_classes: int = 2, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Chargement du modèle pré-entraîné MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Gel des couches de base (extraction de caractéristiques)
        for param in self.model.parameters():
            param.requires_grad = False

        # Remplacement de la tête de classification
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

    def prepare_dataloader(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        # Conversion du format (N, H, W, C) vers le format PyTorch (N, C, H, W)
        X_transposed = X.permute(0, 3, 1, 2)
        dataset = TensorDataset(X_transposed, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, save_path: str = "backend/plant_model.pth"):
        print(f"Début de l'entraînement sur : {self.device}\n")
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Phase d'entraînement
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

            train_loss = running_loss / total_train
            train_acc = (correct_train / total_train) * 100

            # Phase de validation
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            val_loss = val_loss / total_val
            val_acc = (correct_val / total_val) * 100

            print(f"Époque [{epoch:02d}/{epochs:02d}] "
                  f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% "
                  f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

            # Sauvegarde du meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)

        print(f"\nEntraînement terminé. Meilleure précision de validation : {best_val_acc:.2f}%")
        print(f"Modèle sauvegardé dans : {save_path}")


if __name__ == "__main__":
    base_dir = Path("Plant-Classification-1")
    if not base_dir.exists():
        base_dir = Path("backend/Plant-Classification-1")

    # 1. Chargement des données
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    preprocessor.fit_classes(base_dir / "train")

    X_train, y_train = preprocessor.process_split(base_dir / "train")
    X_val, y_val = preprocessor.process_split(base_dir / "valid")

    # 2. Conversion en Tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 3. Initialisation du Trainer
    trainer = PlantClassifierTrainer(num_classes=len(preprocessor.class_to_idx))

    train_loader = trainer.prepare_dataloader(X_train_tensor, y_train_tensor, batch_size=16, shuffle=True)
    val_loader = trainer.prepare_dataloader(X_val_tensor, y_val_tensor, batch_size=16, shuffle=False)

    # 4. Lancement de l'entraînement
    trainer.train(train_loader, val_loader, epochs=10)