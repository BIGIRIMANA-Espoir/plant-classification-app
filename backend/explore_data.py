import os
from pathlib import Path
import pandas as pd
from PIL import Image

def analyze_dataset(dataset_path: str):
    base_path = Path(dataset_path)
    
    if not base_path.exists():
        print(f"[ERREUR] Le dossier {base_path.resolve()} n'existe pas.")
        return

    splits = ['train', 'valid', 'test']
    records = []

    print("\n--- DEBUT DE L'ANALYSE DU DATASET ---\n")

    for split in splits:
        split_dir = base_path / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        is_valid = True
                        width, height, mode = None, None, None
                        try:
                            with Image.open(img_file) as img:
                                width, height = img.size
                                mode = img.mode
                        except Exception:
                            is_valid = False

                        records.append({
                            'Split': split,
                            'Class': class_name,
                            'Filename': img_file.name,
                            'Width': width,
                            'Height': height,
                            'Mode': mode,
                            'Valid': is_valid
                        })

    df = pd.DataFrame(records)

    if df.empty:
        print("[AVERTISSEMENT] Aucune image n'a été trouvée dans le dossier.")
        return

    print("=" * 50)
    print("RAPPORT D'ANALYSE QUALITE DES DONNEES")
    print("=" * 50)
    print(f"Total d'images     : {len(df)}")
    print(f"Nombre de classes  : {df['Class'].nunique()}")
    print(f"Classes trouvées   : {list(df['Class'].unique())}")
    print(f"Images corrompues  : {len(df[~df['Valid']])}")
    print("-" * 50)
    print("Répartition par Split et Classe :")
    print(pd.crosstab(df['Class'], df['Split'], margins=True))
    print("=" * 50 + "\n")

if __name__ == "__main__":
    # Vérifie d'abord à la racine, sinon dans backend/
    path_root = Path("Plant-Classification-1")
    path_backend = Path("backend/Plant-Classification-1")

    if path_root.exists():
        analyze_dataset("Plant-Classification-1")
    elif path_backend.exists():
        analyze_dataset("backend/Plant-Classification-1")
    else:
        print("[ERREUR] Dossier Plant-Classification-1 introuvable.")