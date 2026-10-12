from datetime import date

import pytest
from langchain_core.documents import Document

from app import rag_service as mod


@pytest.fixture
def service(monkeypatch):
    """Construit le service sans clé Mistral, FAISS ni connexion réseau."""

    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeIndex:
        ntotal = 10
        d = 768

    class FakeVectorStore:
        index = FakeIndex()

    class FakeFAISS:
        @staticmethod
        def load_local(*args, **kwargs):
            return FakeVectorStore()

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

    class FakePrompt:
        def __or__(self, other):
            return self

    class FakeChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return FakePrompt()

    class FakeParser:
        def __ror__(self, other):
            return FakeChain()

    class FakeChain:
        def invoke(self, inputs):
            return "Réponse simulée"

        def __or__(self, other):
            return self

    monkeypatch.setattr(mod, "MistralAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(mod, "FAISS", FakeFAISS)
    monkeypatch.setattr(mod, "ChatMistralAI", FakeLLM)
    monkeypatch.setattr(mod, "ChatPromptTemplate", FakeChatPromptTemplate)
    monkeypatch.setattr(mod, "StrOutputParser", FakeParser)

    return mod.RAGService(faiss_folder=mod.Path("index-factice"))


def make_document(
    event_id="event-1",
    city="Montpellier",
    date_from="2026-06-21",
    date_to="2026-06-21",
    title="Concert",
    content="Concert de jazz",
):
    return Document(
        page_content=content,
        metadata={
            "event_id": event_id,
            "city": city,
            "date_from": date_from,
            "date_to": date_to,
            "title": title,
            "categories": "musique",
            "url": "https://example.com/event",
        },
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-06-21T13:30:00+00:00", date(2026, 6, 21)),
        ("2026-06-21T13:30:00Z", date(2026, 6, 21)),
        ("pas-une-date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_openagenda_date(service, value, expected):
    assert service.parse_openagenda_date(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" Montpellier  ", "montpellier"),
        ("MONTPELLIER", "montpellier"),
        ("Montpellier   Centre", "montpellier centre"),
        ("  ", None),
        (None, None),
    ],
)
def test_normalize_city(service, value, expected):
    assert service.normalize_city(value) == expected


def test_detect_city_from_question(service):
    assert (
        service.detect_city_from_question("Quels concerts à Montpellier ?")
        == "Montpellier"
    )
    assert service.detect_city_from_question("Événements à Nîmes ?") is None


def test_detect_date_aujourdhui(service):
    reference = date(2026, 6, 21)

    assert service.detect_date_range_from_question(
        "Que faire aujourd'hui ?",
        reference_date=reference,
    ) == (reference, reference)


def test_detect_date_demain(service):
    assert service.detect_date_range_from_question(
        "Que faire demain ?",
        reference_date=date(2026, 6, 21),
    ) == (date(2026, 6, 22), date(2026, 6, 22))


def test_detect_date_weekend(service):
    # Le 19 juin 2026 est un vendredi.
    assert service.detect_date_range_from_question(
        "Que faire ce week-end ?",
        reference_date=date(2026, 6, 19),
    ) == (date(2026, 6, 20), date(2026, 6, 21))


def test_detect_date_ete_apres_ete(service):
    assert service.detect_date_range_from_question(
        "Événements en été",
        reference_date=date(2026, 10, 11),
    ) == (date(2027, 6, 21), date(2027, 9, 22))


def test_resolve_filters_priorise_filtres_explicites(service):
    filters = service.resolve_filters(
        question="Événements à Montpellier",
        city="Nîmes",
        date_from=date(2026, 7, 1),
    )

    assert filters.city == "Nîmes"
    assert filters.date_from == date(2026, 7, 1)
    assert filters.date_to is None


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (mod.SearchFilters(city="montpellier"), True),
        (mod.SearchFilters(city="Nîmes"), False),
        (
            mod.SearchFilters(date_from=date(2026, 6, 21)),
            True,
        ),
        (
            mod.SearchFilters(date_to=date(2026, 6, 20)),
            False,
        ),
        (
            mod.SearchFilters(
                date_from=date(2026, 6, 22),
                date_to=date(2026, 6, 25),
            ),
            False,
        ),
    ],
)
def test_document_matches_filters(service, filters, expected):
    document = make_document()

    assert service.document_matches_filters(document, filters) is expected


def test_document_matches_periode_chevauchement(service):
    document = make_document(
        date_from="2026-06-20",
        date_to="2026-06-23",
    )
    filters = mod.SearchFilters(
        date_from=date(2026, 6, 22),
        date_to=date(2026, 6, 25),
    )

    assert service.document_matches_filters(document, filters)


def test_document_avec_date_invalide_ne_correspond_pas(service):
    document = make_document(date_from="date invalide")

    assert not service.document_matches_filters(
        document,
        mod.SearchFilters(date_from=date(2026, 6, 21)),
    )


def test_deduplicate_by_event(service):
    documents = [
        make_document(event_id="event-1", content="chunk 1"),
        make_document(event_id="event-1", content="chunk 2"),
        make_document(event_id="event-2", content="autre événement"),
        Document(page_content="sans identifiant", metadata={}),
    ]

    result = service.deduplicate_by_event(documents)

    assert [doc.metadata["event_id"] for doc in result] == [
        "event-1",
        "event-2",
    ]
    assert result[0].page_content == "chunk 1"


def test_format_documents(service):
    result = service.format_documents(
        [
            make_document(event_id="event-1", content="Premier"),
            make_document(event_id="event-2", content="Second"),
        ]
    )

    assert "[Source 1]\nPremier" in result
    assert "[Source 2]\nSecond" in result
    assert "\n\n---\n\n" in result


def test_documents_to_sources(service):
    source = service.documents_to_sources([make_document()])[0]

    assert source["rank"] == 1
    assert source["event_id"] == "event-1"
    assert source["title"] == "Concert"
    assert source["source_url"] == "https://example.com/event"
    assert source["excerpt"] == "Concert de jazz"


def test_build_search_query(service):
    query = service.build_search_query(
        question="Quels concerts ?",
        filters=mod.SearchFilters(
            city="Montpellier",
            date_from=date(2026, 6, 21),
            date_to=date(2026, 6, 22),
        ),
    )

    assert "Quels concerts ?" in query
    assert "Ville : Montpellier" in query
    assert "Date de début : 2026-06-21" in query
    assert "Date de fin : 2026-06-22" in query


def test_find_documents_by_date(service):
    doc1 = make_document(event_id="event-1", date_from="2026-06-21")
    doc2 = make_document(event_id="event-2", date_from="2026-06-22")
    duplicate = make_document(event_id="event-1", content="autre chunk")

    class FakeDocstore:
        _dict = {
            "1": doc1,
            "2": doc2,
            "3": duplicate,
        }

    service.vector_store.docstore = FakeDocstore()

    result = service.find_documents_by_date(
        filters=mod.SearchFilters(
            date_from=date(2026, 6, 21),
            date_to=date(2026, 6, 21),
        ),
        limit=20,
    )

    assert [doc.metadata["event_id"] for doc in result] == ["event-1"]


def test_retrieve_documents_filtre_et_dedoublonne(service, monkeypatch):
    candidates = [
        make_document(event_id="event-1"),
        make_document(event_id="event-1", content="autre chunk"),
        make_document(event_id="event-2", city="Nîmes"),
    ]

    calls = {}

    def fake_similarity_search(query, k):
        calls["query"] = query
        calls["k"] = k
        return candidates

    service.vector_store.similarity_search = fake_similarity_search

    result = service.retrieve_documents(
        question="Concert à Montpellier",
        k=5,
        filters=mod.SearchFilters(city="Montpellier"),
    )

    assert len(result) == 1
    assert result[0].metadata["event_id"] == "event-1"
    assert calls["k"] == service.vector_store.index.ntotal
    assert "Ville : Montpellier" in calls["query"]


def test_ask_retourne_reponse_et_sources(service, monkeypatch):
    document = make_document()
    monkeypatch.setattr(
        service,
        "retrieve_documents",
        lambda **kwargs: [document],
    )

    class FakeChain:
        def invoke(self, inputs):
            assert "Concert de jazz" in inputs["context"]
            assert inputs["question"] == "Que faire ?"
            return "Réponse simulée"

    service.chain = FakeChain()

    result = service.ask("Que faire ?")

    assert result["answer"] == "Réponse simulée"
    assert result["sources"][0]["event_id"] == "event-1"
    assert result["applied_filters"] == {
        "city": None,
        "date_from": None,
        "date_to": None,
    }


def test_ask_sans_document_retourne_reponse_de_secours(service, monkeypatch):
    monkeypatch.setattr(
        service,
        "retrieve_documents",
        lambda **kwargs: [],
    )

    result = service.ask("Que faire demain ?")

    assert result["sources"] == []
    assert "Je ne dispose pas d'assez d'informations" in result["answer"]
    assert result["applied_filters"]["date_from"] is not None


def test_ask_with_trace_retourne_contextes(service, monkeypatch):
    document = make_document()
    monkeypatch.setattr(
        service,
        "retrieve_documents",
        lambda **kwargs: [document],
    )

    class FakeChain:
        def invoke(self, inputs):
            return "Réponse simulée"

    service.chain = FakeChain()

    result = service.ask_with_trace("Que faire ?")

    assert result["answer"] == "Réponse simulée"
    assert result["contexts"] == ["Concert de jazz"]
    assert result["sources"][0]["event_id"] == "event-1"

def test_retrieve_documents_utilise_fallback_date(service, monkeypatch):
    class FakeDocstore:
        _dict = {
            "1": make_document(event_id="event-1"),
        }

    service.vector_store.docstore = FakeDocstore()
    service.vector_store.similarity_search = lambda query, k: []

    result = service.retrieve_documents(
        question="Concert le 21 juin",
        k=5,
        filters=mod.SearchFilters(
            city="Montpellier",
            date_from=date(2026, 6, 21),
            date_to=date(2026, 6, 21),
        ),
    )

    assert len(result) == 1
    assert result[0].metadata["event_id"] == "event-1"