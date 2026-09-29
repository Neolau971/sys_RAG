import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------

FAISS_INDEX_FILE = Path("data/faiss_events.index")
METADATA_FILE = Path("data/faiss_events_metadata.pkl")
MODEL_EMBED = "mistral-embed"

# ----------------------------
# CHARGEMENT INDEX + METADATA
# ----------------------------

def load_index_and_metadata(
    index_path: Path,
    metadata_path: Path,
) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ----------------------------
# EMBEDDING D'UNE REQUÊTE
# ----------------------------

def embed_query(text: str) -> np.ndarray:
    client = Mistral()
    response = client.embeddings.create(model=MODEL_EMBED, inputs=[text])
    emb = response.data[0].embedding
    # Convertir en numpy array float32 de shape (1, dim)
    return np.array([emb], dtype=np.float32)


# ----------------------------
# RECHERCHE SÉMANTIQUE
# ----------------------------

def search_events(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Recherche les k chunks les plus proches de la requête (similarité cosinus).
    Retourne une liste de résultats avec :
      - score (similarité cosinus)
      - text
      - metadata complètes
    """
    # 1. Embedder la requête
    q_vec = embed_query(query)

    # 2. Normaliser L2 (comme pour les vecteurs de l'index)
    faiss.normalize_L2(q_vec)

    # 3. Recherche des k plus proches voisins
    distances, indices = index.search(q_vec, k)

    # distances: shape (1, k), indices: shape (1, k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for dist, idx in zip(distances, indices, strict=True):
        if idx == -1:
            # FAISS renvoie -1 s'il n'y a pas assez de vecteurs
            continue
        meta = metadata[idx]
        results.append({
            "score": float(dist),  # similarité cosinus
            "text": meta["text"],
            "metadata": meta["metadata"],
        })

    return results


# ----------------------------
# EXEMPLE D'UTILISATION
# ----------------------------

def main():
    # Charger index + metadata
    index, metadata = load_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
    print(f"Index chargé : {index.ntotal} vecteurs")

    # Exemple de requête
    query = "concerts et festivals de musique à Montpellier cet été"
    results = search_events(query, index, metadata, k=5)

    print(f"\nRequête : {query}\n")
    for i, res in enumerate(results, start=1):
        meta = res["metadata"]
        title = meta.get("title", "Sans titre")
        city = meta.get("city", "")
        date_start = meta.get("firstdate_begin", "")
        print(f"{i}. {title} – {city} ({date_start})")
        print(f"   Score : {res['score']:.4f}")
        # Affiche un extrait du chunk
        excerpt = res["text"][:200].replace("\n", " ")
        print(f"   Extrait : {excerpt}...")
        print()


if __name__ == "__main__":
    main()