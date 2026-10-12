from __future__ import annotations
import os
from dotenv import load_dotenv

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


@dataclass(frozen=True)
class SearchFilters:
    """
    Représente les règles métier réellement appliquées à la recherche.

    Les dates sont inclusives :
    date_from <= firstdate_begin <= date_to.
    """

    city: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class RAGService:
    """
    Service métier RAG.

    Responsabilités :
    - charger l'index ;
    - rechercher des candidats FAISS ;
    - appliquer des filtres ville/date ;
    - générer une réponse augmentée ;
    - retourner les sources utilisées.
    """

    def __init__(
        self,
        faiss_folder: Path,
        embedding_model: str = "mistral-embed",
        chat_model: str = "mistral-small-2603",
        candidate_multiplier: int = 20,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")

        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "MISTRAL_API_KEY est absente ou vide. "
                "Vérifie le fichier .env à la racine du projet."
            )
        
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.candidate_multiplier = candidate_multiplier

        self.embeddings = MistralAIEmbeddings(
            model=embedding_model,
            max_concurrent_requests=1,
        )

        self.vector_store = FAISS.load_local(
            str(faiss_folder),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.llm = ChatMistralAI(
            model=chat_model,
            temperature=0.0,
            max_retries=2,
            timeout=60,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Tu es un assistant spécialisé dans les événements culturels.\n"
                        "Réponds uniquement à partir des sources fournies dans le contexte.\n\n"
                        "Règles obligatoires :\n"
                        "- N’invente jamais un événement, une date, un lieu, un artiste ou une URL.\n"
                        "- N’ajoute aucune information absente du contexte.\n"
                        "- Cite uniquement les événements présents dans le contexte.\n"
                        "- Si le contexte ne permet pas de répondre, réponds exactement :\n"
                        "\"Je ne dispose pas d'assez d'informations dans les événements indexés.\"\n"
                        "- Pour chaque événement, indique uniquement les informations présentes "
                        "dans le contexte : titre, dates si disponibles, catégories si disponibles "
                        "et URL si disponible."
                    ),
                ),
                (
                    "human",
                    "Contexte :\n{context}\n\nQuestion : {question}",
                ),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    # -------------------------------------------------
    # Parsing et inférence des filtres métier
    # -------------------------------------------------

    @staticmethod
    def parse_openagenda_date(value: str | None) -> date | None:
        """
        Convertit une date ISO OpenAgenda vers date.

        Exemple :
        '2026-06-21T13:30:00+00:00' -> date(2026, 6, 21)
        """
        if not value:
            return None

        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def normalize_city(value: str | None) -> str | None:
        """Normalise une ville pour une comparaison sans sensibilité à la casse."""
        if not value:
            return None

        normalized = " ".join(value.strip().casefold().split())
        return normalized or None

    @staticmethod
    def detect_city_from_question(question: str) -> str | None:
        """
        Détecte une ville seulement parmi une liste maîtrisée.

        Comme ton index est aujourd'hui centré sur Montpellier, cette détection
        suffit. Quand plusieurs villes seront indexées, il sera préférable de :
        - charger les villes distinctes depuis tes metadata ;
        - faire un matching contrôlé ;
        - ou exposer city explicitement dans le body API.
        """
        normalized_question = question.casefold()

        if "montpellier" in normalized_question:
            return "Montpellier"

        return None

    @staticmethod
    def detect_date_range_from_question(
        question: str,
        reference_date: date | None = None,
    ) -> tuple[date | None, date | None]:
        """
        Traduit quelques expressions françaises usuelles en période.

        Périmètre volontairement simple :
        - aujourd'hui
        - demain
        - ce week-end / ce weekend
        - cet été / cette été / en été

        Pour un interpréteur plus général ("dans deux semaines", "mi-juillet"),
        il faudra un parseur temporel plus complet ou du function calling.
        """
        today = reference_date or date.today()
        normalized_question = question.casefold()

        if "aujourd'hui" in normalized_question:
            return today, today

        if "demain" in normalized_question:
            tomorrow = today + timedelta(days=1)
            return tomorrow, tomorrow

        if (
            "ce week-end" in normalized_question
            or "ce weekend" in normalized_question
        ):
            # weekday() : lundi=0 ... dimanche=6
            days_until_saturday = (5 - today.weekday()) % 7
            saturday = today + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            return saturday, sunday

        if (
            "cet été" in normalized_question
            or "cette été" in normalized_question
            or "en été" in normalized_question
        ):
            # Si la date actuelle est dans l'été, on prend l'été courant.
            # Sinon, on prend le prochain été à venir.
            summer_start_this_year = date(today.year, 6, 21)
            summer_end_this_year = date(today.year, 9, 22)

            if today <= summer_end_this_year:
                year = today.year
            else:
                year = today.year + 1

            return date(year, 6, 21), date(year, 9, 22)

        return None, None

    def resolve_filters(
        self,
        question: str,
        city: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> SearchFilters:
        """
        Fusionne les filtres explicites et ceux déduits de la question.

        Règle :
        1. Filtres API explicites prioritaires.
        2. Sinon, tentative de détection depuis le texte.
        """
        inferred_city = self.detect_city_from_question(question)
        inferred_date_from, inferred_date_to = (
            self.detect_date_range_from_question(question)
        )

        return SearchFilters(
            city=city or inferred_city,
            date_from=date_from or inferred_date_from,
            date_to=date_to or inferred_date_to,
        )

    # -------------------------------------------------
    # Retrieval + application des règles métier
    # -------------------------------------------------

    @staticmethod
    def parse_simple_date(value: str | None) -> date | None:
        """Convertit une date YYYY-MM-DD en objet date."""

        if not value:
            return None

        try:
            return date.fromisoformat(value[:10])
        except (ValueError, TypeError):
            return None


    def document_matches_filters(
        self,
        document: Document,
        filters: SearchFilters,
    ) -> bool:
        """Retourne True si un document respecte tous les filtres actifs."""

        metadata = document.metadata

        if filters.city:
            event_city = self.normalize_city(metadata.get("city"))
            requested_city = self.normalize_city(filters.city)

            if event_city != requested_city:
                return False

        if filters.date_from or filters.date_to:
            event_start = self.parse_simple_date(
                metadata.get("date_from")
            )
            event_end = self.parse_simple_date(
                metadata.get("date_to")
            ) or event_start

            if event_start is None:
                return False

            # Intersection : l'événement doit chevaucher la période demandée.
            if filters.date_from and event_end < filters.date_from:
                return False

            if filters.date_to and event_start > filters.date_to:
                return False

        return True

    @staticmethod
    def deduplicate_by_event(
        documents: list[Document],
    ) -> list[Document]:
        """Évite plusieurs chunks du même événement dans le contexte final."""

        seen_event_ids: set[str] = set()
        unique_documents: list[Document] = []

        for document in documents:
            event_id = document.metadata.get("event_id")

            if not event_id:
                continue

            if event_id in seen_event_ids:
                continue

            seen_event_ids.add(event_id)
            unique_documents.append(document)

        return unique_documents

    def retrieve_documents(
        self,
        question: str,
        k: int,
        filters: SearchFilters,
        generation_k: int = 5,
    ) -> list[Document]:
        candidate_k = min(
            max(k * self.candidate_multiplier, k),
            self.vector_store.index.ntotal,
        )

        search_query = self.build_search_query(
            question=question,
            filters=filters,
        )

        candidates = self.vector_store.similarity_search(
            search_query,
            k=candidate_k,
        )

        filtered_documents = [
            document
            for document in candidates
            if self.document_matches_filters(document, filters)
        ]

        unique_documents = self.deduplicate_by_event(filtered_documents)

        if unique_documents:
            return unique_documents[:k]

        # Fallback : recherche par période lorsque la recherche
        # vectorielle ne trouve pas les bons documents.
        date_documents = self.find_documents_by_date(
            filters=filters,
            limit=max(k * 4, k),
        )

        unique_date_documents = self.deduplicate_by_event(date_documents)

        return unique_date_documents[:k]

    # -------------------------------------------------
    # Génération et formatage de la réponse
    # -------------------------------------------------

    @staticmethod
    def format_documents(documents: list[Document]) -> str:
        """Transforme les documents récupérés en contexte explicite pour le LLM."""

        parts = []

        for rank, document in enumerate(documents, start=1):
            parts.append(
                f"[Source {rank}]\n"
                f"{document.page_content}"
            )

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def documents_to_sources(
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """Construit les sources structurées de la réponse API."""

        return [
            {
                "rank": rank,
                "event_id": document.metadata.get("event_id"),
                "title": document.metadata.get("title") or "Sans titre",
                "city": document.metadata.get("city"),
                "date_from": document.metadata.get("date_from"),
                "date_to": document.metadata.get("date_to"),
                "categories": document.metadata.get("categories"),
                "source_url": document.metadata.get("url"),
                "excerpt": document.page_content[:300].replace("\n", " "),
            }
            for rank, document in enumerate(documents, start=1)
        ]
    
    def ask_with_trace(
        self,
        question: str,
        k: int = 5,
        city: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict[str, Any]:
        """
        Exécute le pipeline RAG et retourne les éléments nécessaires à Ragas.

        À utiliser uniquement pour l'évaluation offline, pas directement
        dans l'endpoint public.
        """
        filters = self.resolve_filters(
            question=question,
            city=city,
            date_from=date_from,
            date_to=date_to,
        )

        documents = self.retrieve_documents(
            question=question,
            k=k,
            filters=filters,
            generation_k=5,
        )

        contexts = [document.page_content for document in documents]

        if not documents:
            return {
                "question": question,
                "answer": (
                    "Je ne dispose pas d'assez d'informations dans les "
                    "événements indexés."
                ),
                "contexts": [],
                "sources": [],
                "applied_filters": {
                    "city": filters.city,
                    "date_from": (
                        filters.date_from.isoformat()
                        if filters.date_from
                        else None
                    ),
                    "date_to": (
                        filters.date_to.isoformat()
                        if filters.date_to
                        else None
                    ),
                },
            }

        context = self.format_documents(documents)

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "sources": self.documents_to_sources(documents),
            "applied_filters": {
                "city": filters.city,
                "date_from": (
                    filters.date_from.isoformat()
                    if filters.date_from
                    else None
                ),
                "date_to": (
                    filters.date_to.isoformat()
                    if filters.date_to
                    else None
                ),
            },
        }

    def debug_metadata(self, k: int = 10) -> list[dict[str, Any]]:
        docs = self.vector_store.similarity_search(
            "événements Montpellier",
            k=k,
        )

        return [
            {
                "event_id": doc.metadata.get("event_id"),
                "title": doc.metadata.get("title"),
                "city": doc.metadata.get("city"),
                "date_from": doc.metadata.get("date_from"),
                "date_to": doc.metadata.get("date_to"),
            }
            for doc in docs
        ] 

    def debug_date_retrieval(
        self,
        requested_date: date,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """Diagnostique le retrieval avant et après filtrage par date."""

        filters = SearchFilters(
            city="Montpellier",
            date_from=requested_date,
            date_to=requested_date,
        )

        candidate_k = min(
            max(k * self.candidate_multiplier, k),
            self.vector_store.index.ntotal,
        )

        candidates = self.vector_store.similarity_search(
            "événements Montpellier",
            k=candidate_k,
        )

        results = []

        for document in candidates:
            event_start = self.parse_simple_date(
                document.metadata.get("date_from")
            )
            event_end = self.parse_simple_date(
                document.metadata.get("date_to")
            ) or event_start

            matches = (
                event_start is not None
                and event_end >= requested_date
                and event_start <= requested_date
            )

            results.append(
                {
                    "event_id": document.metadata.get("event_id"),
                    "title": document.metadata.get("title"),
                    "date_from": document.metadata.get("date_from"),
                    "date_to": document.metadata.get("date_to"),
                    "matches": matches,
                }
            )

        return results  

    def find_documents_by_date(
        self,
        filters: SearchFilters,
        limit: int = 20,
    ) -> list[Document]:
        """
        Recherche de secours : parcourt les documents dont la période
        chevauche la période demandée.

        À utiliser lorsque la recherche vectorielle ne renvoie aucun
        document après application des filtres.
        """

        matching_documents: list[Document] = []

        for docstore_id, document in self.vector_store.docstore._dict.items():
            if not self.document_matches_filters(document, filters):
                continue

            matching_documents.append(document)

            if len(matching_documents) >= limit:
                break

        return self.deduplicate_by_event(matching_documents)

    def build_search_query(
        self,
        question: str,
        filters: SearchFilters,
    ) -> str:
        """Construit une requête enrichie pour la recherche vectorielle."""

        parts = [question]

        if filters.city:
            parts.append(f"Ville : {filters.city}")

        if filters.date_from:
            parts.append(
                f"Date de début : {filters.date_from.isoformat()}"
            )

        if filters.date_to:
            parts.append(
                f"Date de fin : {filters.date_to.isoformat()}"
            )

        return "\n".join(parts) 

    def ask(
        self,
        question: str,
        k: int = 5,
        city: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict[str, Any]:
        """
        Point d'entrée métier unique de l'API.

        L'endpoint FastAPI ne connaît pas les détails de recherche : il transmet
        des paramètres validés et reçoit une réponse + les sources.
        """
        filters = self.resolve_filters(
            question=question,
            city=city,
            date_from=date_from,
            date_to=date_to,
        )

        documents = self.retrieve_documents(
            question=question,
            k=k,
            filters=filters,
            generation_k=5,
        )

        if not documents:
            return {
                "answer": (
                    "Je ne dispose pas d'assez d'informations dans les "
                    "événements indexés."
                ),
                "sources": [],
                "applied_filters": {
                    "city": filters.city,
                    "date_from": (
                        filters.date_from.isoformat()
                        if filters.date_from
                        else None
                    ),
                    "date_to": (
                        filters.date_to.isoformat()
                        if filters.date_to
                        else None
                    ),
                },
            }

        context = self.format_documents(documents)

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return {
            "answer": answer,
            "sources": self.documents_to_sources(documents),
            "applied_filters": {
                "city": filters.city,
                "date_from": (
                    filters.date_from.isoformat()
                    if filters.date_from
                    else None
                ),
                "date_to": (
                    filters.date_to.isoformat()
                    if filters.date_to
                    else None
                ),
            },
        }