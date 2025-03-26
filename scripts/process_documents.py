#!/usr/bin/env python3
"""
Script pour traiter les documents PDF et les convertir en base de connaissances vectorielle.
Ce script peut être utilisé localement ou dans le cadre d'un pipeline CI/CD pour 
mettre à jour la base de connaissances du chatbot.
"""

import os
import json
import argparse
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
                text += page.extract_text()
            
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
    
    for file_path in pdf_files:
        text = extract_text_from_pdf(file_path)
        
        if text:
            chunks = chunk_text(text)
            
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
    
    logger.info(f"Total de {len(documents)} morceaux extraits de tous les PDF")
    return documents

def save_knowledge_base(documents: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Sauvegarde la base de connaissances.
    
    Args:
        documents: Liste des documents
        output_path: Chemin de sortie pour la base de connaissances
    """
    try:
        # Version simplifiée sans embeddings pour éviter les dépendances complexes
        data = {
            "documents": documents,
            "model": None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
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
    parser.add_argument("--output", type=str, default=str(KNOWLEDGE_BASE_DIR / "knowledge_base.json"),
                        help="Chemin de sortie pour la base de connaissances")
    
    args = parser.parse_args()
    
    # Traiter les fichiers PDF
    documents = process_pdf_files()
    
    if not documents:
        logger.warning("Aucun document trouvé ou traité.")
        return
    
    # Sauvegarder la base de connaissances
    save_knowledge_base(documents, Path(args.output))
    
    # Mettre à jour la configuration
    sources = list(set(doc["metadata"]["filename"] for doc in documents))
    update_config_with_sources(sources)

if __name__ == "__main__":
    main()
