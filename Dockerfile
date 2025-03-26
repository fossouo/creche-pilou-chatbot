FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du projet
COPY . .

# Exposer le port sur lequel l'application s'exécutera
EXPOSE 5000

# Configuration des variables d'environnement
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV ANTHROPIC_API_KEY=""
ENV CLAUDE_MODEL="claude-3-opus-20240229"
ENV MAX_TOKENS=1000
ENV VECTOR_SEARCH_ENABLED=true

# Préparer les données au démarrage (traitement des PDF)
RUN mkdir -p knowledge_base
RUN python scripts/process_documents.py

# Exécuter l'application
CMD ["python", "app.py"]
