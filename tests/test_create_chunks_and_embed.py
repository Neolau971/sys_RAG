import json

import pytest

from app import create_chunks_and_embed as mod


def test_extract_description():
    page_content = (
        "Événement : Concert\n"
        "Description : Un concert à Montpellier.\n"
        "Le groupe joue du jazz.\n"
        "Date de début : 2025-06-10\n"
        "Ville : Montpellier"
    )

    assert mod.extract_description_from_page_content(page_content) == (
        "Un concert à Montpellier. Le groupe joue du jazz."
    )


def test_extract_description_absente():
    assert mod.extract_description_from_page_content(
        "Événement : Concert\nVille : Montpellier"
    ) == ""


def test_split_text_into_sentences():
    assert mod.split_text_into_sentences(
        "Bonjour. Comment ça va ? Très bien !"
    ) == ["Bonjour.", "Comment ça va ?", "Très bien !"]


def test_chunk_text_texte_vide():
    assert mod.chunk_text("") == []


def test_chunk_text_regroupe_les_phrases():
    # On rend le comptage déterministe pour tester la logique de découpage.
    def fake_count_tokens(text):
        return len(text.split())

    sentences = ["un deux.", "trois quatre.", "cinq six."]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "count_tokens", fake_count_tokens)

    try:
        result = mod.chunk_text(
            " ".join(sentences),
            chunk_size=4,
            chunk_overlap=0,
        )
    finally:
        monkeypatch.undo()

    assert result == ["un deux. trois quatre.", "cinq six."]


def test_build_event_text_utilise_metadata():
    doc = {
        "metadata": {
            "title": "Concert jazz",
            "city": "Montpellier",
            "firstdate_begin_date": "2025-06-10",
            "lastdate_end_date": "2025-06-10",
            "keywords_text": "musique, jazz",
            "source_url": "https://example.com/evenement",
        },
        "page_content": "Description : Un concert de jazz.",
    }

    text = mod.build_event_text(doc)

    assert "Titre : Concert jazz" in text
    assert "Ville : Montpellier" in text
    assert "Catégories : musique, jazz" in text
    assert "Description : Un concert de jazz." in text


def test_build_event_text_valeurs_par_defaut():
    text = mod.build_event_text({})

    assert "Titre : Sans titre" in text
    assert "Ville : Non renseignée" in text
    assert "Date de début : Non renseignée" in text
    assert "URL : Non disponible" in text


def test_load_documents(tmp_path):
    path = tmp_path / "documents.json"
    documents = [{"id": "event-1", "page_content": "Contenu"}]
    path.write_text(json.dumps(documents), encoding="utf-8")

    assert mod.load_documents(path) == documents


def test_load_documents_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.load_documents(tmp_path / "absent.json")


def test_build_chunks_ignore_documents_invalides():
    documents = [
        {"page_content": "Contenu sans identifiant"},
        {"id": "event-vide", "page_content": ""},
    ]

    assert mod.build_chunks(documents) == []


def test_build_chunks_cree_un_chunk(monkeypatch):
    # Force le cas simple : un seul chunk, sans dépendre du tokenizer réel.
    monkeypatch.setattr(mod, "count_tokens", lambda text: 10)

    documents = [
        {
            "id": "event-1",
            "page_content": "Description : Un concert.",
            "metadata": {
                "title": "Concert",
                "city": "Montpellier",
                "keywords_text": "musique",
                "source_url": "https://example.com/event-1",
            },
        }
    ]

    chunks = mod.build_chunks(documents)

    assert len(chunks) == 1
    assert chunks[0]["id_chunk"] == "event-1__chunk_0"
    assert chunks[0]["document_id"] == "event-1"
    assert chunks[0]["metadata"]["title"] == "Concert"
    assert chunks[0]["metadata"]["city"] == "Montpellier"
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["total_chunks"] == 1


def test_build_chunks_decoupe_un_long_document(monkeypatch):
    monkeypatch.setattr(mod, "CHUNK_SIZE_TOKENS", 2)
    monkeypatch.setattr(mod, "CHUNK_OVERLAP_TOKENS", 0)

    # Le texte formé par build_event_text est considéré comme trop long.
    monkeypatch.setattr(mod, "count_tokens", lambda text: 10)

    def fake_chunk_text(text, chunk_size, chunk_overlap):
        return ["premier morceau", "second morceau"]

    monkeypatch.setattr(mod, "chunk_text", fake_chunk_text)

    documents = [
        {
            "id": "event-2",
            "page_content": "Description : Une description.",
            "metadata": {"title": "Événement"},
        }
    ]

    chunks = mod.build_chunks(documents)

    assert len(chunks) == 2
    assert chunks[0]["id_chunk"] == "event-2__chunk_0"
    assert chunks[1]["id_chunk"] == "event-2__chunk_1"
    assert chunks[0]["metadata"]["total_chunks"] == 2


def test_get_mistral_embeddings_sans_textes():
    assert mod.get_mistral_embeddings([]) == []


def test_get_mistral_embeddings_par_lots(monkeypatch):
    appels = []

    class FakeItem:
        def __init__(self, embedding):
            self.embedding = embedding

    class FakeResponse:
        def __init__(self, embeddings):
            self.data = [FakeItem(item) for item in embeddings]

    class FakeEmbeddings:
        def create(self, model, inputs):
            appels.append({"model": model, "inputs": inputs})
            return FakeResponse([[float(i)] for i in range(len(inputs))])

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(mod, "Mistral", FakeClient)

    result = mod.get_mistral_embeddings(
        ["texte 1", "texte 2", "texte 3"],
        model="modele-test",
        batch_size=2,
    )

    assert len(result) == 3
    assert len(appels) == 2
    assert appels[0]["model"] == "modele-test"
    assert appels[0]["inputs"] == ["texte 1", "texte 2"]
    assert appels[1]["inputs"] == ["texte 3"]


def test_get_mistral_embeddings_erreur_si_nombre_incorrect(monkeypatch):
    class FakeResponse:
        data = []

    class FakeEmbeddings:
        def create(self, model, inputs):
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(mod, "Mistral", FakeClient)

    with pytest.raises(RuntimeError, match="différent du nombre de textes"):
        mod.get_mistral_embeddings(["un texte"])