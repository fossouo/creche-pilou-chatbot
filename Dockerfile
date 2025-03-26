FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt séparément pour profiter du cache des couches Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p knowledge_base templates static config data_sources

# Copier les fichiers du projet
COPY app.py .
COPY config/ config/
COPY scripts/ scripts/
COPY templates/ templates/
# Au lieu de copier le dossier static, nous l'avons déjà créé ci-dessus
# COPY static/ static/
COPY data_sources/ data_sources/

# Exposer le port sur lequel l'application s'exécutera
EXPOSE 5000

# Configuration des variables d'environnement
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV ANTHROPIC_API_KEY=""
ENV CLAUDE_MODEL="claude-3-opus-20240229"
ENV MAX_TOKENS=1000
ENV VECTOR_SEARCH_ENABLED=false

# Exécuter l'application au démarrage
# Au lieu de traiter les documents au build time, nous le ferons au runtime
CMD ["sh", "-c", "python scripts/process_documents.py && python app.py"]
