import json
import numpy as np

def save_memory(embedded_concepts, file_path):
    """
    Salva le frasi e i loro vettori in formato JSON.
    """
    memory = []
    for concept, vector in embedded_concepts:
        memory.append({
            "text": concept,
            "vector": vector.tolist()  # da numpy array a lista standard per JSON
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def load_memory(file_path):
    """
    Carica i dati salvati dalla memoria.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        memory = json.load(f)

    # Convertiamo i vettori da lista a numpy array
    return [(entry["text"], np.array(entry["vector"])) for entry in memory]


# ESEMPIO DI TEST
if __name__ == "__main__":
    from data_loader import load_concepts
    from embedder import embed_concepts

    concepts = load_concepts("../data/concetti_base.txt")
    embedded = embed_concepts(concepts)

    # Salviamo la memoria
    save_memory(embedded, "../data/memory.json")
    print("Memoria salvata.")

    # La ricarichiamo per testare
    loaded = load_memory("../data/memory.json")
    print(f"Ho caricato {len(loaded)} concetti dalla memoria.")