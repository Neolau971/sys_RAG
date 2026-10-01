import json
import uuid
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------

EMBEDDINGS_FILE = Path("data/openagenda_embeddings.json")
FAISS_FOLDER = Path("data/langchain_faiss")

# Cet objet est nécessaire lorsque LangChain devra vectoriser
# une future requête utilisateur lors du retrieval.
EMBEDDING_MODEL = "mistral-embed"


def load_chunks_with_embeddings(path: Path) -> list[dict]:
    """Charge les chunks enrichis avec leurs embeddings Mistral."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Exécute d'abord create_chunks_and_embed.py."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_vector_store_from_existing_embeddings(chunks: list[dict]) -> FAISS:
    """
    Construit un vector store LangChain FAISS à partir d'embeddings déjà calculés.

    Aucun appel API d'embedding n'est lancé dans cette fonction.
    """
    documents = []
    vectors = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        embedding = chunk.get("embedding")

        if not text or not embedding:
            continue

        chunk_metadata = chunk.get("metadata", {})
        metadata = {
            "id_chunk": chunk.get("id_chunk", ""),
            "document_id": chunk.get("document_id", ""),
            "source": chunk_metadata.get("source", ""),
            "source_url": chunk_metadata.get("source_url", ""),
            "title": chunk_metadata.get("title", ""),
            "city": chunk_metadata.get("city", ""),
            "department": chunk_metadata.get("department", ""),
            "region": chunk_metadata.get("region", ""),
            "keywords": ", ".join(chunk_metadata.get("keywords", [])),
            "firstdate_begin": chunk_metadata.get("firstdate_begin", ""),
            "lastdate_end": chunk_metadata.get("lastdate_end", ""),
            "chunk_index": chunk_metadata.get("chunk_index", 0),
            "total_chunks": chunk_metadata.get("total_chunks", 1),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )
        vectors.append(embedding)

    if not documents:
        raise ValueError(
            "Aucun chunk avec un texte et un embedding valide n'a été trouvé."
        )

    # FAISS exige une matrice float32 : (nombre_de_chunks, dimension).
    vectors_array = np.asarray(vectors, dtype=np.float32)

    if vectors_array.ndim != 2:
        raise ValueError(
            "Les embeddings ne forment pas une matrice 2D. "
            "Vérifie openagenda_embeddings.json."
        )

    dimension = vectors_array.shape[1]

    # Même convention que ton index manuel : produit scalaire sur vecteurs L2 normalisés
    # = similarité cosinus.
    faiss.normalize_L2(vectors_array)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)

    # LangChain requiert :
    # - un docstore qui associe un ID interne à chaque Document ;
    # - un mapping FAISS_position -> ID interne du document.
    docstore_data = {}
    index_to_docstore_id = {}

    for position, document in enumerate(documents):
        docstore_id = str(uuid.uuid4())
        docstore_data[docstore_id] = document
        index_to_docstore_id[position] = docstore_id

    docstore = InMemoryDocstore(docstore_data)

    # L'objet embeddings sert UNIQUEMENT lors des futures recherches :
    # question texte -> embedding Mistral -> recherche dans FAISS.
    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        max_concurrent_requests=1,
    )

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        normalize_L2=False,
        distance_strategy="MAX_INNER_PRODUCT",
    )

    print(
        f"Index LangChain créé : {index.ntotal} vecteurs, "
        f"dimension {dimension}."
    )

    return vector_store


def main():
    chunks = load_chunks_with_embeddings(EMBEDDINGS_FILE)

    print(f"{len(chunks)} chunks chargés depuis {EMBEDDINGS_FILE}.")

    vector_store = build_vector_store_from_existing_embeddings(chunks)

    FAISS_FOLDER.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_FOLDER))

    print(f"Index LangChain sauvegardé dans : {FAISS_FOLDER}")
    print("Fichiers attendus : index.faiss et index.pkl")


if __name__ == "__main__":
    main()