import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)

# Charger le modèle de traitement de langage
model = SentenceTransformer('all-MiniLM-L6-v2')

# Charger la base de connaissances
with open('config/knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# Prétraiter toutes les questions
all_questions = []
all_answers = []
all_categories = []

for category in knowledge_base["categories"]:
    for qa_pair in category["questions"]:
        all_questions.append(qa_pair["question"])
        all_answers.append(qa_pair["answer"])
        all_categories.append(category["name"])

# Encoder toutes les questions
question_embeddings = model.encode(all_questions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"answer": "Désolé, je n'ai pas compris votre question."})
    
    # Encoder la question de l'utilisateur
    user_embedding = model.encode([user_message])
    
    # Calculer la similarité avec toutes les questions
    similarities = np.dot(user_embedding, question_embeddings.T)[0]
    
    # Trouver l'index de la question la plus similaire
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    # Définir un seuil de similarité
    threshold = 0.6
    
    if best_similarity >= threshold:
        answer = all_answers[best_idx]
        category = all_categories[best_idx]
        return jsonify({
            "answer": answer,
            "category": category,
            "confidence": float(best_similarity)
        })
    else:
        return jsonify({
            "answer": "Désolé, je n'ai pas trouvé de réponse à votre question. Pouvez-vous la reformuler ou être plus précis?",
            "confidence": float(best_similarity)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
