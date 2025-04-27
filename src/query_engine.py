from sentence_transformers import SentenceTransformer
import numpy as np
from memory import load_memory

# Carica modello e memoria
model = SentenceTransformer('all-MiniLM-L6-v2')
memory = load_memory("../data/memory.json")

def get_best_matches(query, memory, top_k=3):
    """
    Confronta la domanda con la memoria e restituisce la frase più simile.
    """
    query_embedding = model.encode([query])[0]

    similarities = []
    for concept, vector in memory:
        similarity = cosine_similarity(query_embedding, vector)
        similarities.append((concept, similarity))

    # Ordina per similarità decrescente
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def build_composed_response(matches):
    """
    Crea una risposta composta usando le frasi trovate.
    """
    if not matches:
        return "❌ Non ho travo alcuna risposta utile"

    risposta = "🧠 Per rispondere alla tua domanda, posso dirti che:\n"
    for concept, score in matches:
        risposta += f"- {concept}\n"

    return risposta


# Test se eseguito direttamente
if __name__ == "__main__":
    print("Fai una domanda alla tua IA:")
    query = input(">> ")

    matches = get_best_matches(query, memory, top_k=3)

    SOGLIA = 0.50

    matches_filtrati = [m for m in matches if m[1] >= SOGLIA]

    risposta_composta = build_composed_response(matches_filtrati)

    print("\n" + risposta_composta)