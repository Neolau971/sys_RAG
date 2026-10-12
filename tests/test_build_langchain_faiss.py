import json

import pytest
from langchain_core.documents import Document

from app import build_langchain_faiss as mod


def test_load_chunks_lit_le_json(tmp_path):
    path = tmp_path / "chunks.json"
    chunks = [{"id_chunk": "event-1__chunk_0", "text": "Concert"}]
    path.write_text(json.dumps(chunks), encoding="utf-8")

    assert mod.load_chunks(path) == chunks


def test_load_chunks_fichier_absent(tmp_path):
    path = tmp_path / "absent.json"

    with pytest.raises(FileNotFoundError, match="Fichier de chunks introuvable"):
        mod.load_chunks(path)


@pytest.mark.parametrize("content", ["[]", ""])
def test_load_chunks_json_vide(tmp_path, content):
    path = tmp_path / "chunks.json"
    path.write_text(content or "[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Aucun chunk trouvé"):
        mod.load_chunks(path)


def test_load_chunks_json_invalide(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text("{json invalide", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        mod.load_chunks(path)


def test_build_documents_convertit_et_nettoie():
    chunks = [
        {
            "text": "  Description de l’événement  ",
            "metadata": {
                "event_id": "event-1",
                "title": "Concert",
                "city": "Montpellier",
                "chunk_index": 2,
                "total_chunks": 3,
            },
        },
        {
            "text": "   ",
            "metadata": {"event_id": "event-2"},
        },
    ]

    documents = mod.build_documents(chunks)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Description de l’événement"
    assert documents[0].metadata["event_id"] == "event-1"
    assert documents[0].metadata["title"] == "Concert"
    assert documents[0].metadata["city"] == "Montpellier"
    assert documents[0].metadata["chunk_index"] == 2
    assert documents[0].metadata["total_chunks"] == 3


def test_build_documents_valeurs_par_defaut():
    documents = mod.build_documents([{"text": "Contenu"}])

    assert len(documents) == 1
    assert documents[0].metadata["event_id"] == ""
    assert documents[0].metadata["title"] == ""
    assert documents[0].metadata["chunk_index"] == 0
    assert documents[0].metadata["total_chunks"] == 1


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [{"text": "   "}],
        [{"text": ""}, {"text": None}],
    ],
)
def test_build_documents_sans_document_valide(chunks):
    with pytest.raises(ValueError, match="Aucun document valide"):
        mod.build_documents(chunks)


def test_build_documents_metadata_numeriques_invalides():
    chunks = [
        {
            "text": "Contenu",
            "metadata": {"chunk_index": "pas-un-entier"},
        }
    ]

    with pytest.raises(ValueError):
        mod.build_documents(chunks)


def test_build_faiss_index_refuse_sans_cle(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY non définie"):
        mod.build_faiss_index()


def test_build_faiss_index_cree_et_sauvegarde_index(
    monkeypatch,
    tmp_path,
):
    chunks = [
        {
            "text": "Description de l’événement",
            "metadata": {
                "event_id": "event-1",
                "title": "Concert",
                "chunk_index": 0,
                "total_chunks": 1,
            },
        }
    ]

    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

    faiss_folder = tmp_path / "faiss_index"
    faiss_folder.mkdir()
    (faiss_folder / "ancien_fichier").write_text("ancien", encoding="utf-8")

    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")
    monkeypatch.setattr(mod, "CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(mod, "FAISS_FOLDER", faiss_folder)

    embeddings_args = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            embeddings_args.update(kwargs)

    class FakeIndex:
        ntotal = 1
        d = 768

    class FakeVectorStore:
        index = FakeIndex()

        def __init__(self):
            self.saved_path = None

        def save_local(self, path):
            self.saved_path = path

    fake_vector_store = FakeVectorStore()
    from_documents_args = {}

    def fake_from_documents(documents, embedding):
        from_documents_args["documents"] = documents
        from_documents_args["embedding"] = embedding
        return fake_vector_store

    monkeypatch.setattr(mod, "MistralAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(mod.FAISS, "from_documents", fake_from_documents)

    result = mod.build_faiss_index()

    assert embeddings_args == {
        "model": mod.EMBEDDING_MODEL,
        "max_concurrent_requests": 1,
    }
    assert len(from_documents_args["documents"]) == 1
    assert from_documents_args["embedding"] is not None
    assert fake_vector_store.saved_path == str(faiss_folder)

    assert result == {
        "chunks_read": 1,
        "documents_indexed": 1,
        "vectors_indexed": 1,
        "embedding_dimension": 768,
        "index_folder": str(faiss_folder),
    }

    # L'ancien dossier a été supprimé avant la sauvegarde.
    assert not (faiss_folder / "ancien_fichier").exists()


def test_main_affiche_le_resultat(monkeypatch, capsys):
    monkeypatch.setattr(
        mod,
        "build_faiss_index",
        lambda: {
            "vectors_indexed": 12,
            "embedding_dimension": 768,
            "documents_indexed": 10,
            "index_folder": "/tmp/faiss_index",
        },
    )

    mod.main()

    output = capsys.readouterr().out

    assert "12 vecteurs" in output
    assert "dimension 768" in output
    assert "Documents indexés : 10" in output
    assert "Dossier : /tmp/faiss_index" in output