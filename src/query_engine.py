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

def show_menu():
    """
    Mostra il menù iniziale e restituisce la scelta dell'utente.
    """
    print("\n Benvenuto nella tua IA personale!")
    print("Cosa vuoi fare?\n")
    print("[1] Fai una domanda")
    print("[2] Aggiungi una nuova frase alla memoria")
    print("[3] Esci")

    scelta = input(">> Inserisci il numero dell'opzione: ").strip()
    return scelta

if __name__ == "__main__":
    while True:
        scelta = show_menu()

        if scelta == "1":
            query = input("\nScrivi la tua domanda: ")
            matches = get_best_matches(query, memory, top_k=3)

            SOGLIA = 0.50
            matches_filtrati = [m for m in matches if m[1] >= SOGLIA]

            risposta = build_composed_response(matches_filtrati)
            print("\n" + risposta)

        elif scelta == "2":
            print("⚠️ Funzionalità in sviluppo: aggiunta dinamica delle frasi non ancora attiva.")

        elif scelta == "3":
            print("👋 Ciao! Alla prossima.")
            break

        else:
            print("❌ Scelta non valida. Inserisci 1, 2 o 3.")