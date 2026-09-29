import json
import pickle
from pathlib import Path
import numpy as np
import faiss

# ----------------------------
# CONFIG
# ----------------------------

EMBEDDINGS_FILE = Path("data/openagenda_embeddings.json")
FAISS_INDEX_FILE = Path("data/faiss_events.index")
METADATA_FILE = Path("data/faiss_events_metadata.pkl")

# ----------------------------
# CHARGEMENT DES CHUNKS + EMBEDDINGS
# ----------------------------

def load_embeddings(path: Path):
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


# ----------------------------
# CONSTRUCTION DE L'INDEX FAISS
# ----------------------------

def build_faiss_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Construit un index FAISS à partir des chunks contenant un champ 'embedding'.
    Retourne :
      - index FAISS
      - liste de metadata (une par vecteur, dans le même ordre)
    """
    # Extraire les embeddings et les metadata dans le même ordre
    embeddings = []
    metadata = []

    for chunk in chunks:
        emb = chunk.get("embedding")
        if emb is None:
            continue
        embeddings.append(emb)
        # On garde les infos utiles pour la recherche / l'affichage
        meta = {
            "id_chunk": chunk.get("id_chunk"),
            "document_id": chunk.get("document_id"),
            "text": chunk.get("text"),
            "metadata": chunk.get("metadata", {}),
        }
        metadata.append(meta)

    # Convertir en numpy array float32 (requis par FAISS)
    X = np.array(embeddings, dtype=np.float32)
    dimension = X.shape[1]

    # Index Flat avec inner product (cosine similarity après normalisation L2)
    index = faiss.IndexFlatIP(dimension)

    # Normalisation L2 des vecteurs (pour que inner product = cosine similarity)
    faiss.normalize_L2(X)

    # Ajout des vecteurs à l'index
    index.add(X)

    print(f"Index FAISS créé : {index.ntotal} vecteurs, dimension {dimension}")
    return index, metadata


# ----------------------------
# SAUVEGARDE INDEX + METADATA
# ----------------------------

def save_index_and_metadata(
    index: faiss.Index,
    metadata: list[dict],
    index_path: Path,
    metadata_path: Path,
):
    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)
    print(f"Index sauvegardé dans {index_path}")
    print(f"Metadata sauvegardées dans {metadata_path}")


# ----------------------------
# MAIN
# ----------------------------

def main():
    # 1. Charger les chunks avec embeddings
    chunks = load_embeddings(EMBEDDINGS_FILE)
    print(f"{len(chunks)} chunks chargés depuis {EMBEDDINGS_FILE}")

    # 2. Construire l'index FAISS
    index, metadata = build_faiss_index(chunks)

    # 3. Sauvegarder index + metadata
    save_index_and_metadata(index, metadata, FAISS_INDEX_FILE, METADATA_FILE)

    print("\nPour faire une recherche sémantique, utilise search_events.py")


if __name__ == "__main__":
    main()