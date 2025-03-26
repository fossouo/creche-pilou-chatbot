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
import hashlib
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

def get_file_hash(file_path: Path) -> str:
    """
    Génère un hash pour le fichier.
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        Hash du fichier
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

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
            
            total_pages = len(reader.pages)
            logger.info(f"Nombre total de pages: {total_pages}")
            
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    logger.debug(f"Page {page_num+1}/{total_pages} traitée: {len(page_text)} caractères")
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction de la page {page_num+1}: {str(e)}")
            
            if not text.strip():
                logger.warning(f"Aucun texte extrait de {pdf_path}")
                return ""
                
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

def process_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Traite un seul fichier PDF.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Liste de dictionnaires contenant les informations du document
    """
    documents = []
    original_filename = pdf_path.name
    file_hash = get_file_hash(pdf_path)
    
    try:
        logger.info(f"Traitement du fichier: {original_filename} (hash: {file_hash})")
        
        # Vérifier si ce fichier a déjà été traité
        kb_path = KNOWLEDGE_BASE_DIR / f"kb_{file_hash}.json"
        if kb_path.exists():
            logger.info(f"Fichier déjà traité précédemment, chargement depuis {kb_path}")
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("documents", [])
        
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"Aucun texte extrait du fichier {original_filename}")
            return []
        
        chunks = chunk_text(text)
        logger.info(f"Fichier {original_filename} divisé en {len(chunks)} morceaux")
        
        for i, chunk in enumerate(chunks):
            doc = {
                "source": original_filename,
                "chunk_id": i,
                "text": chunk,
                "file_hash": file_hash,
                "metadata": {
                    "filename": original_filename,
                    "size": os.path.getsize(pdf_path),
                    "modified": os.path.getmtime(pdf_path)
                }
            }
            documents.append(doc)
        
        logger.info(f"Traitement du fichier {original_filename} terminé avec succès")
        return documents
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier {original_filename}: {str(e)}")
        return []

def create_document_embeddings(documents: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Crée des embeddings pour les documents à l'aide d'un modèle de transformer.
    
    Args:
        documents: Liste de dictionnaires contenant les informations du document
        model_name: Nom du modèle SentenceTransformer à utiliser
        
    Returns:
        Dictionnaire avec les documents et leurs embeddings
    """
    if not documents:
        logger.warning("Aucun document à encoder")
        return {"documents": [], "model": None}
    
    logger.info(f"Création des embeddings avec le modèle {model_name} pour {len(documents)} documents")
    
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

def merge_knowledge_bases() -> Dict[str, Any]:
    """
    Fusionne toutes les bases de connaissances individuelles en une seule.
    
    Returns:
        Base de connaissances fusionnée
    """
    logger.info("Fusion des bases de connaissances individuelles")
    
    all_documents = []
    model_name = None
    
    # Trouver tous les fichiers de base de connaissances
    kb_files = list(KNOWLEDGE_BASE_DIR.glob("kb_*.json"))
    logger.info(f"Nombre de bases de connaissances trouvées: {len(kb_files)}")
    
    if not kb_files:
        logger.warning("Aucune base de connaissances individuelle trouvée")
        return {"documents": [], "model": None}
    
    # Fusionner les documents
    for kb_file in kb_files:
        try:
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                all_documents.extend(kb_data.get("documents", []))
                if not model_name and kb_data.get("model"):
                    model_name = kb_data.get("model")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de {kb_file}: {str(e)}")
    
    logger.info(f"Fusion terminée, {len(all_documents)} documents au total")
    return {"documents": all_documents, "model": model_name}

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
            "last_updated": os.path.getmtime(config_path) if config_path.exists() else None,
            "total_files": len(sources)
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
    
    # Liste des fichiers PDF
    pdf_files = list(DATA_SOURCES_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans {DATA_SOURCES_DIR}")
        return
    
    logger.info(f"Nombre de fichiers PDF trouvés: {len(pdf_files)}")
    
    # Traiter chaque PDF individuellement
    all_sources = []
    for pdf_path in pdf_files:
        documents = process_single_pdf(pdf_path)
        if documents:
            # Créer les embeddings
            file_hash = documents[0].get("file_hash", "unknown")
            embedded_docs = create_document_embeddings(documents, args.model)
            
            # Sauvegarder la base de connaissances individuelle
            kb_path = KNOWLEDGE_BASE_DIR / f"kb_{file_hash}.json"
            save_knowledge_base(embedded_docs, kb_path)
            
            all_sources.append(pdf_path.name)
    
    # Fusionner les bases de connaissances individuelles
    merged_kb = merge_knowledge_bases()
    
    # Sauvegarder la base de connaissances fusionnée
    save_knowledge_base(merged_kb, Path(args.output))
    
    # Mettre à jour la configuration
    update_config_with_sources(all_sources)
    
    logger.info(f"Traitement terminé: {len(all_sources)}/{len(pdf_files)} fichiers traités avec succès")

if __name__ == "__main__":
    main()
