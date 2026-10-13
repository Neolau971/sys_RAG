import json
from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv

from mistralai.client import Mistral
import tiktoken

# ----------------------------
# CONFIG
# ----------------------------
TOKENIZER = tiktoken.get_encoding("cl100k_base")  # estimation only

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "openagenda_rag_documents.json"
OUTPUT_CHUNKS_FILE = DATA_DIR / "openagenda_chunks.json"

CHUNK_SIZE_TOKENS = 2_000
CHUNK_OVERLAP_TOKENS = 0
MODEL_EMBED = "mistral-embed"


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

def infer_event_type(
    title: str,
    description: str,
    keywords: str,
) -> str:
    """
    Déduit un type d'événement à partir du titre, de la description
    et des mots-clés.

    Le résultat est volontairement conservateur : si aucun indice
    fiable n'est trouvé, on retourne 'événement'.
    """

    text = " ".join(
        [
            title.lower(),
            description.lower(),
            keywords.lower(),
        ]
    )

    rules = [
        ("exposition", ["exposition", "exposition collective", "rétrospective"]),
        ("visite guidée", ["visite guidée", "visite commentée", "visite libre"]),
        ("atelier", ["atelier", "ateliers"]),
        ("conférence", ["conférence", "conférences", "rencontre", "table ronde"]),
        ("concert", ["concert", "live", "tournée", "scène"]),
        ("spectacle", ["spectacle", "théâtre", "pièce de théâtre", "danse"]),
        ("festival", ["festival", "festival de"]),
        ("marché", ["marché", "pop-up market", "salon"]),
        ("projection", ["projection", "cinéma", "film"]),
        ("formation", ["formation", "job dating", "recrute", "s'informer"]),
    ]

    for event_type, indicators in rules:
        if any(indicator in text for indicator in indicators):
            return event_type

    return "événement"

def build_event_text(doc: dict) -> str:
    metadata = doc.get("metadata", {})
    page_content = doc.get("page_content", "")

    description = extract_description_from_page_content(page_content)

    title = metadata.get("title") or "Sans titre"
    city = metadata.get("city") or "Non renseignée"
    date_from = (
        metadata.get("firstdate_begin_date")
        or metadata.get("firstdate_begin")
        or "Non renseignée"
    )
    date_to = (
        metadata.get("lastdate_end_date")
        or metadata.get("lastdate_end")
        or date_from
    )
    categories = (
        metadata.get("keywords_text")
        or metadata.get("keywords")
        or "Non renseignées"
    )
    url = metadata.get("source_url") or "Non disponible"

    event_type = infer_event_type(
        title=title,
        description=description,
        keywords=categories,
    )

    return (
        f"Titre : {title}\n"
        f"Type : {event_type}\n"
        f"Ville : {city}\n"
        f"Date de début : {date_from}\n"
        f"Date de fin : {date_to}\n"
        f"Catégories : {categories}\n"
        f"URL : {url}\n"
        f"Description : {description}"
    )

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
    all_chunks: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id")
        metadata = doc.get("metadata", {})
        page_content = doc.get("page_content", "")

        if not doc_id or not page_content:
            continue

        # 1. Construire d'abord le texte complet de l'événement.
        event_text = build_event_text(doc)

        if count_tokens(event_text) <= CHUNK_SIZE_TOKENS:
            text_chunks = [event_text]
        else:
            text_chunks = chunk_text(
                event_text,
                chunk_size=CHUNK_SIZE_TOKENS,
                chunk_overlap=CHUNK_OVERLAP_TOKENS,
            )

        title = metadata.get("title") or "Sans titre"
        city = metadata.get("city") or "Non renseignée"
        date_from = (
            metadata.get("firstdate_begin_date")
            or metadata.get("firstdate_begin")
            or "Non renseignée"
        )
        date_to = (
            metadata.get("lastdate_end_date")
            or metadata.get("lastdate_end")
            or date_from
        )
        keywords = (
            metadata.get("keywords_text")
            or metadata.get("keywords")
            or ""
        )
        url = metadata.get("source_url") or ""

        event_type = infer_event_type(
        title=title,
        description=extract_description_from_page_content(page_content),
        keywords=keywords,
)

        for chunk_index, text in enumerate(text_chunks):
            all_chunks.append(
                {
                    "id_chunk": f"{doc_id}__chunk_{chunk_index}",
                    "document_id": str(doc_id),
                    "text": text,
                    "metadata": {
                        "event_id": str(doc_id),
                        "title": title,
                        "event_type": event_type,
                        "city": city,
                        "date_from": date_from,
                        "date_to": date_to,
                        "categories": keywords,
                        "url": url,
                        "chunk_index": chunk_index,
                        "total_chunks": len(text_chunks),
                    },
                }
            )

    return all_chunks


# ----------------------------
# EMBEDDINGS AVEC MISTRAL API
# ----------------------------

def get_mistral_embeddings(
    texts: list[str],
    model: str = MODEL_EMBED,
    batch_size: int = 64,
) -> list[list[float]]:
    if not texts:
        return []

    client = Mistral()
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=model,
            inputs=batch,
        )

        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Nombre d’embeddings reçu ({len(response.data)}) "
                f"différent du nombre de textes envoyés ({len(batch)})."
            )

        all_embeddings.extend(item.embedding for item in response.data)

        print(
            f"Embeddings : {min(start + batch_size, len(texts))}/{len(texts)}"
        )

    return all_embeddings


# ----------------------------
# MAIN
# ----------------------------

def main():
    documents = load_documents(INPUT_FILE)
    print(f"{len(documents)} documents RAG chargés.")

    chunks = build_chunks(documents)
    print(f"{len(chunks)} chunks créés.")

    if not chunks:
        raise RuntimeError("Aucun chunk généré. Vérifie les documents source.")

    OUTPUT_CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CHUNKS_FILE.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Chunks sauvegardés dans {OUTPUT_CHUNKS_FILE}")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_mistral_embeddings(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"{len(embeddings)} embeddings pour {len(chunks)} chunks."
        )

    print(
        "Embeddings générés. "
        "Passe maintenant à scripts/build_faiss_index.py."
    )


if __name__ == "__main__":
    main()