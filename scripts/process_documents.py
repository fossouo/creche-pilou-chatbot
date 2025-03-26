#!/usr/bin/env python3
"""
Script pour traiter les documents PDF et les convertir en base de connaissances vectorielle.
Ce script peut être utilisé localement ou dans le cadre d'un pipeline CI/CD pour 
mettre à jour la base de connaissances du chatbot.
"""

import os
import json
import argparse
import unicodedata
import re
from typing import List, Dict, Any
import logging
from pathlib import Path

import PyPDF2
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dossiers et chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CONFIG_DIR = PROJECT_ROOT / "config"

# Assurez-vous que les dossiers existent
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

def normalize_filename(filename: str) -> str:
    """
    Normalise un nom de fichier en supprimant les accents et en remplaçant les espaces.
    
    Args:
        filename: Nom de fichier à normaliser
        
    Returns:
        Nom de fichier normalisé
    """
    # Supprimer les accents
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    # Remplacer les espaces et caractères spéciaux par des underscores
    filename = re.sub(r'[^\w\.\-]', '_', filename)
    return filename

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrait le texte d'un fichier PDF.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Texte extrait du PDF
    """
    logger.info(f"Traitement du fichier PDF: {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            logger.info(f"Extraction réussie: {len(text)} caractères extraits")
            return text
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Divise le texte en morceaux de taille similaire avec un chevauchement.
    
    Args:
        text: Texte à diviser
        chunk_size: Taille approximative de chaque morceau
        overlap: Nombre de caractères de chevauchement entre les morceaux
        
    Returns:
        Liste des morceaux de texte
    """
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def process_pdf_files() -> List[Dict[str, Any]]:
    """
    Traite tous les fichiers PDF dans le dossier des sources de données.
    
    Returns:
        Liste de dictionnaires contenant les informations des documents traités
    """
    documents = []
    
    # Vérifier si des PDF sont présents
    pdf_files = list(DATA_SOURCES_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans {DATA_SOURCES_DIR}")
        # Ajouter un document factice pour éviter les erreurs
        documents.append({
            "source": "document_exemple.pdf",
            "chunk_id": 0,
            "text": "Ce document est un exemple. Aucun PDF n'a été trouvé dans le dossier data_sources.",
            "metadata": {
                "filename": "document_exemple.pdf",
                "size": 0,
                "modified": 0
            }
        })
        return documents
    
    logger.info(f"Traitement de {len(pdf_files)} fichiers PDF trouvés")
    
    for file_path in pdf_files:
        try:
            logger.info(f"Traitement du fichier: {file_path.name}")
            text = extract_text_from_pdf(file_path)
            
            if text:
                chunks = chunk_text(text)
                logger.info(f"Fichier {file_path.name} divisé en {len(chunks)} morceaux")
                
                for i, chunk in enumerate(chunks):
                    doc = {
                        "source": file_path.name,
                        "chunk_id": i,
                        "text": chunk,
                        "metadata": {
                            "filename": file_path.name,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path)
                        }
                    }
                    documents.append(doc)
            else:
                logger.warning(f"Aucun texte extrait du fichier {file_path.name}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path.name}: {str(e)}")
    
    logger.info(f"Total de {len(documents)} morceaux extraits de tous les PDF")
    return documents

def create_document_embeddings(documents: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Crée des embeddings pour les documents à l'aide d'un modèle de transformer.
    
    Args:
        documents: Liste de dictionnaires contenant les informations du document
        model_name: Nom du modèle SentenceTransformer à utiliser
        
    Returns:
        Dictionnaire avec les documents et leurs embeddings
    """
    logger.info(f"Création des embeddings avec le modèle {model_name}")
    
    try:
        # Import ici pour éviter les problèmes de dépendances si le module n'est pas disponible
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        
        for doc in documents:
            doc["embedding"] = model.encode(doc["text"]).tolist()
        
        logger.info(f"Embeddings créés pour {len(documents)} documents")
        return {"documents": documents, "model": model_name}
    except Exception as e:
        logger.error(f"Erreur lors de la création des embeddings: {str(e)}")
        # En cas d'erreur, retourner les documents sans embeddings
        return {"documents": documents, "model": None}

def save_knowledge_base(embedded_docs: Dict[str, Any], output_path: Path) -> None:
    """
    Sauvegarde la base de connaissances avec les embeddings.
    
    Args:
        embedded_docs: Dictionnaire contenant les documents avec embeddings
        output_path: Chemin de sortie pour la base de connaissances
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Base de connaissances sauvegardée à {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la base de connaissances: {str(e)}")

def update_config_with_sources(sources: List[str]) -> None:
    """
    Met à jour le fichier de configuration avec la liste des sources.
    
    Args:
        sources: Liste des noms de fichiers sources
    """
    config_path = CONFIG_DIR / "sources.json"
    
    try:
        config = {
            "sources": sources,
            "last_updated": os.path.getmtime(config_path) if config_path.exists() else None
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Configuration mise à jour avec {len(sources)} sources")
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la configuration: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Traitement des documents PDF pour le chatbot")
    parser.add_argument("--output", type=str, default=str(KNOWLEDGE_BASE_DIR / "vector_knowledge_base.json"),
                        help="Chemin de sortie pour la base de connaissances")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Modèle SentenceTransformer à utiliser")
    
    args = parser.parse_args()
    
    # Traiter les fichiers PDF
    documents = process_pdf_files()
    
    if not documents:
        logger.warning("Aucun document trouvé ou traité.")
        return
    
    # Créer les embeddings
    embedded_docs = create_document_embeddings(documents, args.model)
    
    # Sauvegarder la base de connaissances
    save_knowledge_base(embedded_docs, Path(args.output))
    
    # Mettre à jour la configuration
    sources = list(set(doc["metadata"]["filename"] for doc in documents))
    update_config_with_sources(sources)

if __name__ == "__main__":
    main()
