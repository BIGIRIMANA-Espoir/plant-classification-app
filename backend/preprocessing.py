import os
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Module de prétraitement des images pour les modèles de Deep Learning.
    Gère le redimensionnement, la normalisation et la conversion en tableaux.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    def fit_classes(self, train_dir: Path) -> None:
        """
        Mappe dynamiquement les noms de dossiers de classes en identifiants numériques.
        """
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Charge une image, la convertit en RGB, la redimensionne et la normalise [0, 1].
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array

    def process_split(self, split_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Traite un sous-dossier complet (train, valid ou test) et retourne
        les matrices X (images) et y (labels).
        """
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for class_name, class_idx in self.class_to_idx.items():
            class_path = split_dir / class_name
            if not class_path.exists():
                continue

            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        img_arr = self.load_and_preprocess_image(img_file)
                        X_list.append(img_arr)
                        y_list.append(class_idx)
                    except Exception as err:
                        print(f"[ERREUR] Échec de chargement pour {img_file.name}: {err}")

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


if __name__ == "__main__":
    base_dir = Path("Plant-Classification-1")
    if not base_dir.exists():
        base_dir = Path("backend/Plant-Classification-1")

    preprocessor = ImagePreprocessor(target_size=(224, 224))
    preprocessor.fit_classes(base_dir / "train")

    print("\n--- TEST DU PRÉTRAITEMENT DES DONNÉES ---")
    print(f"Classes détectées : {preprocessor.class_to_idx}")

    for split in ["train", "valid", "test"]:
        split_path = base_dir / split
        if split_path.exists():
            X, y = preprocessor.process_split(split_path)
            print(f"Split '{split}' -> Images (X): {X.shape}, Étiquettes (y): {y.shape}")
    print("------------------------------------------\n")