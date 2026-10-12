from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import main as api

from fastapi import FastAPI
from fastapi.testclient import TestClient

test_app = FastAPI()
test_app.router.routes = api.app.router.routes

@pytest.fixture(autouse=True)
def reset_rebuild_state():
    api.rebuild_state.update(
        {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "chunks_read": None,
            "vectors_indexed": None,
            "embedding_dimension": None,
            "error": None,
        }
    )
    yield

def test_health_service_absent():
    api.app.state.rag_service = None

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 503


def test_health_service_present():
    class FakeIndex:
        ntotal = 42

    class FakeVectorStore:
        index = FakeIndex()

    class FakeService:
        chat_model = "modele-chat-test"
        embedding_model = "modele-embedding-test"
        vector_store = FakeVectorStore()

    api.app.state.rag_service = FakeService()

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "chat_model": "modele-chat-test",
        "embedding_model": "modele-embedding-test",
        "indexed_vectors": 42,
    }


def test_ask_service_absent():
    api.app.state.rag_service = None

    client = TestClient(api.app)
    response = client.post(
        "/ask",
        json={"question": "Que faire à Montpellier ?"},
    )

    assert response.status_code == 503


def test_ask_succes(monkeypatch):
    class FakeService:
        def ask(self, **kwargs):
            assert kwargs["question"] == "Que faire à Montpellier ?"
            return {
                "answer": "Un concert est prévu.",
                "sources": [
                    {
                        "rank": 1,
                        "event_id": "event-1",
                        "title": "Concert",
                        "city": "Montpellier",
                        "date_from": "2026-06-21",
                        "date_to": "2026-06-21",
                        "categories": "musique",
                        "source_url": "https://example.com/event",
                        "excerpt": "Un concert de jazz.",
                    }
                ],
                "applied_filters": {
                    "city": None,
                    "date_from": None,
                    "date_to": None,
                },
            }

    api.app.state.rag_service = FakeService()

    client = TestClient(api.app)
    response = client.post(
        "/ask",
            json={
                "question": "Que faire à Montpellier ?",
                "k": 3,
            },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Un concert est prévu."
    assert response.json()["question"] == "Que faire à Montpellier ?"
    assert response.json()["sources"][0]["event_id"] == "event-1"


def test_ask_erreur_service_devient_502():
    class FakeService:
        def ask(self, **kwargs):
            raise RuntimeError("échec simulé")

    api.app.state.rag_service = FakeService()

    client = TestClient(api.app)
    response = client.post(
        "/ask",
        json={"question": "Que faire à Montpellier ?"},
    )

    assert response.status_code == 502
    assert "échec simulé" in response.json()["detail"]


def test_rebuild_refuse_cle_invalide(monkeypatch):
    class FakeIndex:
        ntotal = 0

    class FakeVectorStore:
        index = FakeIndex()

    class FakeService:
        chat_model = "modele-chat-test"
        embedding_model = "modele-embedding-test"
        vector_store = FakeVectorStore()

    api.app.state.rag_service = FakeService()

    monkeypatch.setattr(api, "REBUILD_API_KEY", "secret-test")
    api.rebuild_state["status"] = "idle"

    client = TestClient(api.app)
    response = client.post("/rebuild")

    assert response.status_code == 403


def test_rebuild_refuse_si_deja_en_cours(monkeypatch):
    monkeypatch.setattr(api, "REBUILD_API_KEY", None)
    api.rebuild_state["status"] = "running"

    with TestClient(api.app) as client:
        response = client.post("/rebuild")

    assert response.status_code == 409


