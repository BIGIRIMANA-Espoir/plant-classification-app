# Utiliser une image Python officielle légère
FROM python:3.12-slim

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage direct des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier de dépendances et installer les paquets
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le dossier backend et le modèle entraîné
COPY backend/ ./backend/
COPY backend/plant_model.pth ./plant_model.pth

# Exposer le port de FastAPI
EXPOSE 8000

# Commande pour démarrer l'API Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]