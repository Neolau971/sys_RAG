import pandas as pd
import pytest
import requests

from app import prepare_openagenda as oa


def test_clean_text_nettoie_html_et_espaces():
    assert oa.clean_text("  <b>Bonjour</b>\n   Montpellier  ") == "Bonjour Montpellier"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (["danse", None, "musique"], "danse, musique"),
        ({"fr": "Théâtre", "en": "Theatre"}, "Théâtre Theatre"),
        (float("nan"), ""),
    ],
)
def test_clean_text_valeurs_variees(value, expected):
    assert oa.clean_text(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (["2025-01-01"], None),
        ("date invalide", None),
        ("2025-01-01T12:00:00+01:00", "2025-01-01T11:00:00+00:00"),
    ],
)
def test_parse_datetime(value, expected):
    assert oa.parse_datetime(value) == expected


def test_get_string_list_separe_une_chaine():
    assert oa.get_string_list("  danse, musique , théâtre ") == [
        "danse",
        "musique",
        "théâtre",
    ]


def test_get_string_list_nettoie_une_liste():
    assert oa.get_string_list(["danse", None, "<b>musique</b>", " "]) == [
        "danse",
        "musique",
    ]


def test_get_string_list_valeur_manquante():
    assert oa.get_string_list(pd.NA) == []


def test_build_rag_document_construit_document():
    event = {
        "uid": "event-123",
        "title_fr": "Concert <b>jazz</b>",
        "description_fr": "Un concert à Montpellier.",
        "firstdate_begin": "2025-06-10T20:00:00+02:00",
        "lastdate_end": "2025-06-10T22:00:00+02:00",
        "location_city": "Montpellier",
        "location_department": "Hérault",
        "location_region": "Occitanie",
        "location_countrycode": "FR",
        "location_address": "Place de la Comédie",
        "keywords_fr": ["musique", "jazz"],
        "canonicalurl": "https://example.com/event-123",
    }

    document = oa.build_rag_document(event)

    assert document is not None
    assert document["id"] == "event-123"
    assert "Événement : Concert jazz" in document["page_content"]
    assert document["metadata"]["city"] == "Montpellier"
    assert document["metadata"]["keywords"] == ["musique", "jazz"]
    assert document["metadata"]["firstdate_begin"] == "2025-06-10T18:00:00+00:00"


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"title_fr": "   ", "description_fr": None},
    ],
)
def test_build_rag_document_ignore_sans_titre_ni_description(event):
    assert oa.build_rag_document(event) is None


def test_build_rag_document_utilise_le_titre_et_url_de_secours():
    document = oa.build_rag_document(
        {
            "title": "Événement",
            "url": "https://example.com/evenement",
        }
    )

    assert document is not None
    assert document["metadata"]["title"] == "Événement"
    assert document["metadata"]["source_url"] == "https://example.com/evenement"
    assert document["id"] == "https://example.com/evenement"


def test_deduplicate_documents_conserve_le_premier():
    documents = [
        {"id": "a", "page_content": "premier"},
        {"id": "a", "page_content": "doublon"},
        {"id": "b", "page_content": "autre"},
    ]

    result = oa.deduplicate_documents(documents)

    assert result == [
        {"id": "a", "page_content": "premier"},
        {"id": "b", "page_content": "autre"},
    ]


def test_fetch_recent_events_recupere_plusieurs_pages(monkeypatch):
    appels = []

    class FakeResponse:
        def __init__(self, results):
            self._results = results

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": self._results}

    def fake_get(url, params, timeout):
        appels.append(params.copy())

        if params["offset"] == 0:
            # PAGE_SIZE éléments forcent fetch_recent_events à demander la page suivante.
            return FakeResponse([{"uid": str(i)} for i in range(oa.PAGE_SIZE)])

        return FakeResponse([{"uid": "dernier"}])

    monkeypatch.setattr(oa.requests, "get", fake_get)

    result = oa.fetch_recent_events()

    assert len(result) == oa.PAGE_SIZE + 1
    assert len(appels) == 2
    assert appels[0]["offset"] == 0
    assert appels[1]["offset"] == oa.PAGE_SIZE
    assert appels[0]["where"].find(oa.TARGET_CITY) >= 0


def test_fetch_recent_events_propage_erreur_http(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("Erreur HTTP")

    monkeypatch.setattr(
        oa.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(requests.HTTPError):
        oa.fetch_recent_events()


def test_main_sans_evenement_leve_erreur(monkeypatch):
    monkeypatch.setattr(oa, "fetch_recent_events", lambda: [])

    with pytest.raises(RuntimeError, match="Aucun événement"):
        oa.main()