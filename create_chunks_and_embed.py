import json
from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()

from mistralai.client import Mistral
import tiktoken

# ----------------------------
# CONFIG
# ----------------------------
TOKENIZER = tiktoken.get_encoding("cl100k_base")

INPUT_FILE = Path("data/openagenda_rag_documents.json")
OUTPUT_CHUNKS_FILE = Path("data/openagenda_chunks.json")
OUTPUT_EMBEDDINGS_FILE = Path("data/openagenda_embeddings.json")

CHUNK_SIZE_TOKENS = 400      # taille cible d'un chunk (en tokens approx)
CHUNK_OVERLAP_TOKENS = 50    # recouvrement entre chunks
MODEL_EMBED = "mistral-embed"

# ----------------------------
# UTILITAIRES DE CHUNKING
# ----------------------------
def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens dans un texte avec le tokenizer officiel Mistral.
    """
    # encode renvoie une liste de tokens (ints)

def extract_description_from_page_content(page_content: str) -> str:
    """
    Extrait uniquement le texte de la description depuis page_content.
    Retourne une chaîne (éventuellement vide).
    """
    lines = page_content.splitlines()
    in_description = False
    desc_lines = []

    for line in lines:
        if line.startswith("Description :"):
            in_description = True
            # Prendre ce qu'il y a après "Description :"
            desc_lines.append(line[len("Description :"):].strip())
            continue

        if in_description:
            # Dès qu'on tombe sur une autre ligne de champ, on arrête
            if any(
                line.startswith(prefix)
                for prefix in [
                    "Événement :",
                    "Date de début :",
                    "Date de fin :",
                    "Ville :",
                    "Département :",
                    "Région :",
                    "Adresse :",
                    "Mots-clés :",
                    "Source :",
                ]
            ):
                break
            # Sinon, on considère que c'est la suite de la description
            desc_lines.append(line.strip())

    return " ".join(desc_lines).strip()

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def split_text_into_sentences(text: str) -> list[str]:
    """
    Découpe un texte en phrases (approximatif, mais souvent suffisant).
    On split sur . ! ? suivis d'un espace ou fin de chaîne.
    """
    import re
    # On garde les délimiteurs pour pouvoir reconstruire proprement
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Découpe un texte en chunks de taille ~chunk_size tokens,
    avec un recouvrement de chunk_overlap tokens.
    
    Ici on fait une approximation en comptant les mots comme tokens.
    Pour un vrai comptage tokens, il faudrait utiliser le tokenizer du modèle.
    """
    sentences = split_text_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence)

        # Si la phrase seule est plus grande que chunk_size, on la met dans son propre chunk
        if s_tokens >= chunk_size:
            # D'abord, on ferme le chunk en cours s'il y en a un
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0
            # Puis on découpe la phrase en sous-chunks si nécessaire (très grossier)
            words = sentence.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_words = words[start:end]
                chunks.append(" ".join(chunk_words))
                start += chunk_size - chunk_overlap
            continue

        # Si ajouter cette phrase dépasse chunk_size, on ferme le chunk actuel
        if current_tokens + s_tokens > chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            # On garde un recouvrement : on repart avec les dernières phrases
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                if overlap_tokens + count_tokens(s) <= chunk_overlap:
                    overlap_sentences.append(s)
                    overlap_tokens += count_tokens(s)
                else:
                    break
            current_chunk_sentences = list(reversed(overlap_sentences))
            current_tokens = overlap_tokens

        # On ajoute la phrase au chunk courant
        current_chunk_sentences.append(sentence)
        current_tokens += s_tokens

    # Chunk restant
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


# ----------------------------
# CHARGEMENT DES DOCUMENTS
# ----------------------------

def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# CRÉATION DES CHUNKS
# ----------------------------

def build_chunks(documents: list[dict]) -> list[dict]:
    """
    Pour chaque document RAG, crée plusieurs chunks avec :
    - id_chunk: identifiant unique
    - document_id: lien vers le document d'origine
    - text: le chunk
    - metadata: copie partielle des metadata du document
    """
    all_chunks = []

    for doc in documents:
        doc_id = doc.get("id")
        metadata = doc.get("metadata", {})
        page_content = doc.get("page_content", "")
        text = extract_description_from_page_content(page_content)

        if not text:
            continue

        text_chunks = chunk_text(text)

        for i, chunk_text_content in enumerate(text_chunks):
            chunk = {
                "id_chunk": f"{doc_id}__chunk_{i}",
                "document_id": doc_id,
                "text": chunk_text_content,
                "metadata": {
                    "source": metadata.get("source"),
                    "source_url": metadata.get("source_url"),
                    "title": metadata.get("title"),
                    "city": metadata.get("city"),
                    "department": metadata.get("department"),
                    "region": metadata.get("region"),
                    "keywords": metadata.get("keywords"),
                    "firstdate_begin": metadata.get("firstdate_begin"),
                    "lastdate_end": metadata.get("lastdate_end"),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            }
            all_chunks.append(chunk)

    return all_chunks


# ----------------------------
# EMBEDDINGS AVEC MISTRAL API
# ----------------------------

def get_mistral_embeddings(
    texts: list[str],
    model: str = MODEL_EMBED,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Appelle l'API Mistral pour obtenir les embeddings d'une liste de textes.
    Gère le batching pour éviter d'envoyer trop de textes en une fois.
    """
    client = Mistral()  # utilise MISTRAL_API_KEY depuis l'environnement

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, inputs=batch)
        # response.data est une liste d'objets avec .embedding
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Embeddings batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    return all_embeddings


# ----------------------------
# MAIN
# ----------------------------

def main():
    # 1. Charger les documents RAG
    documents = load_documents(INPUT_FILE)
    print(f"{len(documents)} documents RAG chargés.")

    # 2. Créer les chunks
    chunks = build_chunks(documents)
    print(f"{len(chunks)} chunks créés.")

    # 3. Sauvegarder les chunks bruts
    with OUTPUT_CHUNKS_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks sauvegardés dans {OUTPUT_CHUNKS_FILE}")

    # 4. Calculer les embeddings
    texts = [c["text"] for c in chunks]
    embeddings = get_mistral_embeddings(texts)

    # 5. Attacher les embeddings aux chunks
    for chunk, emb in zip(chunks, embeddings, strict=True):
        chunk["embedding"] = emb

    # 6. Sauvegarder les chunks avec embeddings
    with OUTPUT_EMBEDDINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks + embeddings sauvegardés dans {OUTPUT_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()