import os
import json
import logging
import time
from typing import List, Dict, Any

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Chargement des configurations depuis les variables d'environnement
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1000))
VECTOR_SEARCH_ENABLED = os.environ.get("VECTOR_SEARCH_ENABLED", "false").lower() == "true"

# Chargement de la base de connaissances
KNOWLEDGE_BASE_PATH = os.path.join("knowledge_base", "vector_knowledge_base.json")
FALLBACK_KNOWLEDGE_PATH = os.path.join("config", "knowledge_base.json")

def load_knowledge_base():
    """Charge la base de connaissances à partir du fichier JSON"""
    try:
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif os.path.exists(FALLBACK_KNOWLEDGE_PATH):
            logger.warning("Base vectorielle non trouvée, utilisation de la base de connaissances de repli")
            with open(FALLBACK_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.error("Aucune base de connaissances trouvée")
            return {"documents": [], "model": None}
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la base de connaissances: {str(e)}")
        return {"documents": [], "model": None}

# Chargement des données
knowledge_base = load_knowledge_base()

# Initialisation du modèle d'embeddings si disponible
if knowledge_base.get("model"):
    sentence_model = SentenceTransformer(knowledge_base["model"])
else:
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_local_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Recherche des documents similaires dans la base de connaissances locale"""
    if not knowledge_base.get("documents"):
        logger.warning("Aucun document dans la base de connaissances")
        return []
    
    try:
        # Encodage de la requête
        query_embedding = sentence_model.encode([query])[0]
        
        # Calcul des similarités avec tous les documents
        similarities = []
        for doc in knowledge_base["documents"]:
            embedding = doc.get("embedding")
            if embedding:
                # Calcul de similarité cosinus
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((doc, float(similarity)))
        
        # Tri par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Sélection des top_k documents
        results = []
        for doc, score in similarities[:top_k]:
            doc_copy = doc.copy()
            doc_copy["score"] = score
            # Suppression de l'embedding pour alléger la réponse
            if "embedding" in doc_copy:
                del doc_copy["embedding"]
            results.append(doc_copy)
        
        logger.info(f"Recherche effectuée: {len(results)} résultats trouvés pour '{query}'")
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche locale: {str(e)}")
        return []

def get_profile_instructions(profile: str, detail_level: str = "moyen") -> str:
    """
    Génère les instructions spécifiques au profil utilisateur.
    
    Args:
        profile: Profil de l'utilisateur (cap, directrice, gestionnaire, coordinatrice)
        detail_level: Niveau de détail souhaité (bas, moyen, haut)
    
    Returns:
        Instructions pour adapter la réponse au profil
    """
    # Instructions communes
    common_instructions = """
- Utilise un ton professionnel mais chaleureux.
- Structure ta réponse avec des paragraphes courts et clairs.
- Lorsque tu cites la réglementation, sois précis sur les articles et références.
    """
    
    # Instructions spécifiques par profil
    profile_instructions = {
        "cap": f"""Tu t'adresses à un(e) professionnel(le) CAP Petite Enfance qui travaille directement avec les enfants.
- Utilise un langage simple et accessible, évite le jargon technique.
- Focalise-toi sur les aspects pratiques et concrets du quotidien en crèche.
- Mets en avant les consignes de sécurité et les bonnes pratiques.
- Ne mentionne pas les références légales en détail, sauf si explicitement demandé.
- Propose des exemples concrets qui aident à comprendre l'application des règles.
- Explique l'essentiel sans entrer dans les complexités administratives.
        """,
        
        "directrice": f"""Tu t'adresses à une directrice de crèche (EJE ou Auxiliaire de puériculture avec responsabilité de direction).
- Fournis des informations complètes et précises avec les références réglementaires.
- Mets en avant les responsabilités légales et administratives.
- Explique les implications pratiques pour l'organisation de la structure.
- Donne des conseils sur la façon d'assurer la conformité réglementaire.
- Inclus les nuances et exceptions qui peuvent s'appliquer.
- N'hésite pas à citer les textes officiels et à donner des références précises.
        """,
        
        "gestionnaire": f"""Tu t'adresses à un(e) gestionnaire de réseau de micro-crèches qui s'occupe des aspects administratifs et financiers.
- Niveau de détail: {detail_level.upper()}.
{'- Fournis uniquement les informations essentielles et synthétiques.' if detail_level == 'bas' else ''}
{'- Offre un équilibre entre informations clés et contexte explicatif.' if detail_level == 'moyen' else ''}
{'- Fournis une analyse approfondie avec toutes les nuances et références.' if detail_level == 'haut' else ''}
- Insiste sur les aspects économiques, financiers et de gestion.
- Mentionne les exigences réglementaires qui ont un impact sur le modèle économique.
- Aborde les questions de conformité et de responsabilité juridique.
- Évoque les aspects de management et d'organisation.
        """,
        
        "coordinatrice": f"""Tu t'adresses à une coordinatrice qui supervise plusieurs structures et assure le lien entre terrain et gestion.
- Fournis des informations complètes et détaillées.
- Propose une vision globale et stratégique.
- Mets en avant les aspects de coordination entre différents établissements.
- Insiste sur les enjeux de qualité pédagogique et de cohérence des pratiques.
- Aborde les aspects réglementaires en détail avec leurs implications concrètes.
- Donne des pistes pour accompagner les équipes dans l'application des règles.
        """
    }
    
    # Instructions pour l'affichage des références
    reference_instructions = {
        "cap": "- Ne cite pas les références des sources, sauf si explicitement demandé.",
        "directrice": "- Cite systématiquement les références précises des documents réglementaires.",
        "gestionnaire": f"- {'Ne mentionne pas les références sauf si explicitement demandé.' if detail_level == 'bas' else 'Cite les principales références.' if detail_level == 'moyen' else 'Cite systématiquement les références précises des documents réglementaires.'}",
        "coordinatrice": "- Cite systématiquement les références précises des documents réglementaires."
    }
    
    # Construction des instructions finales
    profile_specific = profile_instructions.get(profile, profile_instructions["directrice"])
    reference_specific = reference_instructions.get(profile, "- Cite les références lorsque c'est pertinent.")
    
    return f"""Instructions pour adapter la réponse au profil {profile.upper()}:
{profile_specific}
{reference_specific}
{common_instructions}
"""

def query_anthropic(query: str, context: str, profile: str, detail_level: str = "moyen") -> Dict[str, Any]:
    """Interroge l'API Anthropic Claude avec un contexte"""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Obtenir les instructions spécifiques au profil
    profile_instructions = get_profile_instructions(profile, detail_level)
    
    # Construction du système prompt incluant le contexte et les instructions de profil
    system_prompt = f"""Tu es un assistant spécialisé dans les questions sur les crèches et l'accueil des jeunes enfants en France.
Ton rôle est de fournir des informations précises sur la réglementation et le fonctionnement des établissements d'accueil du jeune enfant.

{profile_instructions}

Utilise les informations suivantes pour répondre aux questions de l'utilisateur:

{context}

Si tu ne trouves pas la réponse dans le contexte fourni, base-toi sur les connaissances générales que tu as sur les règlements des crèches en France.
Assure-toi que ta réponse est adaptée au profil de l'utilisateur, tant dans le contenu que dans la forme."""
    
    logger.info(f"Envoi d'une requête à l'API Anthropic (profil: {profile}, niveau de détail: {detail_level})")
    
    payload = {
        "model": CLAUDE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "system": system_prompt,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        logger.info(f"Appel de l'API Anthropic avec le modèle {CLAUDE_MODEL}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Vérification du statut de la réponse
        if response.status_code != 200:
            logger.error(f"Erreur API Anthropic: {response.status_code} - {response.text}")
            return {"error": f"Erreur API: {response.status_code}", "details": response.text}
        
        # Traitement de la réponse
        data = response.json()
        logger.info(f"Réponse reçue de l'API Anthropic: {len(str(data))} caractères")
        return data
    except requests.exceptions.Timeout:
        logger.error("Timeout lors de l'appel à l'API Anthropic")
        return {"error": "L'appel à l'API a expiré"}
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API Anthropic: {str(e)}")
        return {"error": str(e)}

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Point d'entrée API pour le chat"""
    data = request.json
    user_message = data.get('message', '')
    user_profile = data.get('profile', 'directrice')  # Profil par défaut
    detail_level = data.get('detailLevel', 'moyen')   # Niveau de détail par défaut
    
    logger.info(f"Nouvelle requête - Profil: {user_profile}, Niveau de détail: {detail_level}")
    
    if not user_message:
        return jsonify({"answer": "Désolé, je n'ai pas compris votre question."})
    
    # Recherche des documents pertinents
    start_time = time.time()
    relevant_docs = search_local_knowledge(user_message, top_k=3)
    search_time = time.time() - start_time
    
    # Construction du contexte à partir des documents
    context = ""
    if relevant_docs:
        context = "Contexte pertinent extrait des documents réglementaires:\n\n"
        for i, doc in enumerate(relevant_docs):
            source = doc.get("source", "base de connaissances")
            score = doc.get("score", 0.0)
            text = doc.get("text", "")
            context += f"[Document {i+1} de {source} (pertinence: {score:.2f})]\n{text}\n\n"
    else:
        context = "Aucun document pertinent n'a été trouvé dans la base de connaissances."
    
    # Interrogation de l'API Anthropic avec le profil utilisateur
    start_time = time.time()
    api_response = query_anthropic(user_message, context, user_profile, detail_level)
    api_time = time.time() - start_time
    
    # Traitement de la réponse
    if "error" in api_response:
        logger.error(f"Erreur API: {api_response['error']}")
        answer = f"Désolé, une erreur s'est produite lors de la génération de la réponse. Veuillez réessayer."
    else:
        try:
            # Extraction de la réponse du modèle
            answer = api_response["content"][0]["text"]
            logger.info(f"Réponse générée avec succès: {len(answer)} caractères")
        except (KeyError, IndexError) as e:
            logger.error(f"Erreur dans le format de réponse d'Anthropic: {str(e)}")
            answer = "Désolé, une erreur s'est produite lors de la génération de la réponse."
    
    # Logs de performance
    logger.info(f"Recherche: {search_time:.2f}s, API: {api_time:.2f}s, Documents trouvés: {len(relevant_docs)}")
    
    # Retourner la réponse avec des informations supplémentaires
    return jsonify({
        "answer": answer,
        "sources": [doc.get("source") for doc in relevant_docs],
        "performance": {
            "search_time": search_time,
            "api_time": api_time
        }
    })

@app.route('/static/<path:path>')
def send_static(path):
    """Sert les fichiers statiques"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Afficher les informations de configuration
    logger.info(f"Démarrage du chatbot avec la configuration suivante:")
    logger.info(f"Modèle Claude: {CLAUDE_MODEL}")
    logger.info(f"Clé API configurée: {'Oui' if ANTHROPIC_API_KEY else 'Non'}")
    logger.info(f"Nombre de documents dans la base de connaissances: {len(knowledge_base.get('documents', []))}")
    
    # Démarrer l'application
    app.run(debug=True, host='0.0.0.0', port=5000)
