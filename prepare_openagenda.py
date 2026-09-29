import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


DATASET_ID = "evenements-publics-openagenda"
BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    f"catalog/datasets/{DATASET_ID}/records"
)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

PAGE_SIZE = 100
MAX_RECORDS = 5_000
TARGET_CITY = "Montpellier"


def clean_text(value: object) -> str:
    """Nettoie une valeur textuelle sans perdre son contenu utile."""

    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        value = ", ".join(
            str(item)
            for item in value
            if item is not None
        )

    if isinstance(value, dict):
        value = " ".join(
            str(item)
            for item in value.values()
            if item is not None
        )

    if not isinstance(value, str) and pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_datetime(value: object) -> str | None:
    """Convertit vers un datetime ISO UTC normalisé, ou retourne None."""
    if value is None or pd.isna(value) or value == "":
        return None

    dt = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(dt):
        return None

    return dt.isoformat()


def fetch_recent_events() -> list[dict]:
    """Extrait les événements récents localisés à Montpellier."""

    cutoff = datetime.now(timezone.utc) - timedelta(days=365)

    where_clause = (
        f"lastdate_end >= date'{cutoff.date().isoformat()}' "
        f"AND location_city = '{TARGET_CITY}'"
    )

    params = {
        "where": where_clause,
        "order_by": "lastdate_end DESC",
        "limit": PAGE_SIZE,
        "offset": 0,
    }

    records = []

    while len(records) < MAX_RECORDS:
        response = requests.get(BASE_URL, params=params, timeout=30)
        print("Paramètres API :", params)
        response.raise_for_status()

        payload = response.json()
        batch = payload.get("results", [])

        if not batch:
            break

        records.extend(batch)

        print(
            f"{len(records)} événements récupérés à {TARGET_CITY} "
            f"(offset={params['offset']})"
        )

        if len(batch) < PAGE_SIZE:
            break

        params["offset"] += PAGE_SIZE

    return records[:MAX_RECORDS]


def get_string_list(value: object) -> list[str]:
    """Uniformise listes, chaînes et valeurs manquantes en liste de textes propres."""

    if value is None:
        return []

    # Cas fréquent pour keywords_fr : une liste JSON.
    if isinstance(value, (list, tuple, set)):
        return [
            cleaned
            for item in value
            if (cleaned := clean_text(item))
        ]

    # Cas d'une valeur manquante Pandas/NumPy scalaire.
    if pd.isna(value):
        return []

    # Cas d'une chaîne simple éventuellement séparée par des virgules.
    text = clean_text(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def build_rag_document(event: dict) -> dict | None:
    """
    Construit :
    - page_content : texte conçu pour l'embedding / le LLM
    - metadata : filtres structurés et attribution de la source
    """
    title = clean_text(event.get("title_fr") or event.get("title"))
    description = clean_text(
        event.get("description_fr") or event.get("description")
    )

    firstdate_begin = parse_datetime(event.get("firstdate_begin"))
    lastdate_end = parse_datetime(event.get("lastdate_end"))

    city = clean_text(event.get("location_city"))
    department = clean_text(event.get("location_department"))
    region = clean_text(event.get("location_region"))
    country = clean_text(event.get("location_countrycode"))

    address = clean_text(event.get("location_address"))
    keywords = get_string_list(event.get("keywords_fr"))
    uid = clean_text(event.get("uid") or event.get("event_uid"))
    url = clean_text(event.get("canonicalurl") or event.get("url"))

    # Ne pas indexer un enregistrement sans contenu textuel exploitable.
    if not title and not description:
        return None

    # Fallback d'identifiant stable.
    event_id = uid or url or f"{title}|{firstdate_begin}|{city}"

    keywords_text = ", ".join(keywords) if keywords else "Non renseignés"

    content_parts = [
        f"Événement : {title or 'Sans titre'}",
        f"Description : {description or 'Description non renseignée'}",
        f"Date de début : {firstdate_begin or 'Non renseignée'}",
        f"Date de fin : {lastdate_end or 'Non renseignée'}",
        f"Ville : {city or 'Non renseignée'}",
        f"Département : {department or 'Non renseigné'}",
        f"Région : {region or 'Non renseignée'}",
        f"Adresse : {address or 'Non renseignée'}",
        f"Mots-clés : {keywords_text}",
        f"Source : {url or 'OpenAgenda'}",
    ]

    return {
        "id": event_id,
        "page_content": "\n".join(content_parts),
        "metadata": {
            "source": "OpenAgenda",
            "source_url": url,
            "title": title,
            "city": city,
            "department": department,
            "region": region,
            "country_code": country,
            "keywords": keywords,
            "firstdate_begin": firstdate_begin,
            "lastdate_end": lastdate_end,
            "event_id": event_id,
        },
    }


def deduplicate_documents(documents: list[dict]) -> list[dict]:
    """Supprime les doublons selon l'identifiant stable de l'événement."""
    seen = set()
    unique_documents = []

    for document in documents:
        event_id = document["id"]

        if event_id in seen:
            continue

        seen.add(event_id)
        unique_documents.append(document)

    return unique_documents


def main():
    raw_events = fetch_recent_events()

    with open(OUTPUT_DIR / "openagenda_raw.json", "w", encoding="utf-8") as file:
        json.dump(raw_events, file, ensure_ascii=False, indent=2)

    documents = [
        document
        for event in raw_events
        if (document := build_rag_document(event)) is not None
    ]

    documents = deduplicate_documents(documents)

    with open(
        OUTPUT_DIR / "openagenda_rag_documents.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)

    metadata_rows = [
        {"id": doc["id"], **doc["metadata"]}
        for doc in documents
    ]

    pd.DataFrame(metadata_rows).to_csv(
        OUTPUT_DIR / "openagenda_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nÉvénements bruts récupérés : {len(raw_events)}")
    print(f"Documents RAG valides : {len(documents)}")
    print("Fichiers créés :")
    print("- data/openagenda_raw.json")
    print("- data/openagenda_rag_documents.json")
    print("- data/openagenda_metadata.csv")


if __name__ == "__main__":
    main()