import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "openagenda_chunks.json"
FAISS_FOLDER = DATA_DIR / "langchain_faiss"

EMBEDDING_MODEL = "mistral-embed"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de chunks introuvable : {path}. "
            "Exécute d’abord le script de création des chunks."
        )

    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError(f"Aucun chunk trouvé dans {path}.")

    return chunks


def build_documents(chunks: list[dict]) -> list[Document]:
    documents = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        metadata = chunk.get("metadata") or {}

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "event_id": str(metadata.get("event_id") or ""),
                    "title": str(metadata.get("title") or ""),
                    "city": str(metadata.get("city") or ""),
                    "date_from": str(metadata.get("date_from") or ""),
                    "date_to": str(metadata.get("date_to") or ""),
                    "categories": str(metadata.get("categories") or ""),
                    "url": str(metadata.get("url") or ""),
                    "chunk_index": int(metadata.get("chunk_index") or 0),
                    "total_chunks": int(metadata.get("total_chunks") or 1),
                },
            )
        )

    if not documents:
        raise ValueError("Aucun document valide à indexer.")

    return documents


def build_faiss_index() -> dict:
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY non définie. "
            "Vérifie ton fichier .env ou tes variables d’environnement."
        )

    chunks = load_chunks(CHUNKS_FILE)
    documents = build_documents(chunks)

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        max_concurrent_requests=1,
    )

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    if FAISS_FOLDER.exists():
        shutil.rmtree(FAISS_FOLDER)

    FAISS_FOLDER.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_FOLDER))

    return {
        "chunks_read": len(chunks),
        "documents_indexed": len(documents),
        "vectors_indexed": vector_store.index.ntotal,
        "embedding_dimension": vector_store.index.d,
        "index_folder": str(FAISS_FOLDER),
    }


def main() -> None:
    result = build_faiss_index()

    print(
        f"Index FAISS construit : "
        f"{result['vectors_indexed']} vecteurs, "
        f"dimension {result['embedding_dimension']}."
    )
    print(f"Documents indexés : {result['documents_indexed']}")
    print(f"Dossier : {result['index_folder']}")


if __name__ == "__main__":
    main()