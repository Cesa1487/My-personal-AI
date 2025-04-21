import os

def load_concepts(file_path):
    """
    Legge un file di testo e restituisce una lista di frasi pulite.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file '{file_path}' non esiste.")

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Pulizia di base: togliamo spazi inutili e righe vuote
    concepts = [line.strip() for line in lines if line.strip() != ""]

    return concepts


# ESEMPIO DI TEST (in futuro se spostato si può rimuovere)
if __name__ == "__main__":
    file_path = "../data/concetti_base.txt"
    concepts = load_concepts(file_path)

    print("Concetti caricati:")
    for i, concept in enumerate(concepts, start=1):
        print(f"{i}. {concept}")