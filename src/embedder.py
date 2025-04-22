from sentence_transformers import SentenceTransformer
from data_loader import load_concepts

# Inizializza il modello di embedding (scarica al primo utilizzo)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_concepts(concepts):
    """
    Riceve una lista di frasi e restituisce i relativi vettori (embedding).
    """
    embeddings = model.encode(concepts)
    return list(zip(concepts, embeddings))

# Test se eseguito direttamente
if __name__ == "__main__":
    file_path = "../data/concetti_base.txt"
    concepts = load_concepts(file_path)
    embedded_concepts = embed_concepts(concepts)

    print("Embedding creati:")
    for concept, vector in embedded_concepts:
        print(f"\nFrase: {concept}\nVettore (primi 5 valori): {vector[:5]}")