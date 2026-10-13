import os
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import time
from datetime import datetime, timezone
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status, Header
from fastapi.middleware.cors import CORSMiddleware

from app.rag_service import RAGService
from app.schemas import (
    AskRequest,
    AskResponse,
    RebuildResponse,
    RebuildStatusResponse,
)
from app.build_langchain_faiss import build_faiss_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
FAISS_FOLDER = BASE_DIR / "data" / "langchain_faiss"
REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")

rebuild_lock = threading.Lock()

rebuild_state = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "chunks_read": None,
    "vectors_indexed": None,
    "embedding_dimension": None,
    "error": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Chargement unique des ressources lourdes au démarrage.

    app.state.rag_service sera utilisé par les endpoints.
    """
    if not FAISS_FOLDER.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {FAISS_FOLDER}. "
            "Exécute d'abord : python -m app.build_langchain_faiss"
        )

    logger.info("Chargement du service RAG...")

    app.state.rag_service = RAGService(
        faiss_folder=FAISS_FOLDER,
        embedding_model="mistral-embed",
        chat_model="mistral-small-2603",
    )

    logger.info(
        "Service RAG prêt : %s vecteurs.",
        app.state.rag_service.vector_store.index.ntotal,
    )

    yield

    logger.info("Arrêt de l'API RAG.")


app = FastAPI(
    title="OpenAgenda RAG API",
    description=(
        "API de questions/réponses augmentées par les événements OpenAgenda "
        "indexés dans FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Utile si une future interface React/Vue/Streamlit tourne sur un autre port.
# En production, remplace "*" par le domaine exact de ton frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_rebuild(app: FastAPI) -> None:
    """
    Reconstruit l'index dans une tâche de fond puis recharge
    RAGService en mémoire lorsque les fichiers sont prêts.
    """
    started_at = datetime.now(timezone.utc)

    with rebuild_lock:
        rebuild_state.update(
            {
                "status": "running",
                "started_at": started_at,
                "finished_at": None,
                "chunks_read": None,
                "vectors_indexed": None,
                "embedding_dimension": None,
                "error": None,
            }
        )

        try:
            logger.info("Reconstruction de l'index FAISS démarrée.")

            result = build_faiss_index()

            # Recharger l'index fraîchement écrit pour que /ask l'utilise
            # sans redémarrer l'API.
            app.state.rag_service = RAGService(
                faiss_folder=FAISS_FOLDER,
                embedding_model="mistral-embed",
                chat_model="mistral-small-2603",
            )

            rebuild_state.update(
                {
                    "status": "completed",
                    "finished_at": datetime.now(timezone.utc),
                    "chunks_read": result["chunks_read"],
                    "vectors_indexed": result["vectors_indexed"],
                    "embedding_dimension": result["embedding_dimension"],
                    "error": None,
                }
            )

            logger.info(
                "Reconstruction terminée : %s vecteurs indexés.",
                result["vectors_indexed"],
            )

        except Exception as error:
            logger.exception("Échec de la reconstruction de l'index FAISS.")

            rebuild_state.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc),
                    "error": str(error),
                }
            )

@app.get(
    "/health",
    summary="Vérifier la disponibilité du service RAG",
    description=(
        "Vérifie que l'API est démarrée, que l'index FAISS est chargé "
        "et que le service RAG est prêt à traiter des questions.\n\n"
        "Retourne également les modèles utilisés pour la génération "
        "et les embeddings, ainsi que le nombre de vecteurs indexés."
    ),
    response_description="État du service RAG.",
    tags=["Monitoring"],
)
def health() -> dict:
    """Vérifie que l'API et l'index sont chargés."""
    service: RAGService | None = getattr(app.state, "rag_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Le service RAG n'est pas encore disponible.",
        )

    return {
        "status": "ok",
        "chat_model": service.chat_model,
        "embedding_model": service.embedding_model,
        "indexed_vectors": service.vector_store.index.ntotal,
    }

@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lancer la reconstruction de l'index FAISS",
    description=(
        "Lance la reconstruction asynchrone de l'index LangChain FAISS "
        "à partir des chunks présents dans `data/openagenda_chunks.json`.\n\n"
        "Les embeddings Mistral sont générés pendant la reconstruction.\n\n"
        "Une fois terminée, l'index est sauvegardé puis le service RAG "
        "est rechargé sans redémarrage de l'API.\n\n"
        "Si `REBUILD_API_KEY` est définie, la requête doit inclure "
        "l'en-tête `X-Rebuild-Key`."
    ),
    response_description="Reconstruction acceptée et démarrée.",
    tags=["Administration"],
)
def rebuild_index(
    background_tasks: BackgroundTasks,
    request: Request,
    x_rebuild_key: str | None = Header(default=None),
) -> RebuildResponse:
    """
    Lance la reconstruction asynchrone de l'index LangChain FAISS.

    La source est data/openagenda_chunks.json :
    les embeddings Mistral sont générés pendant la reconstruction.
    """
    if REBUILD_API_KEY and x_rebuild_key != REBUILD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé d'administration invalide.",
        )
    
    if rebuild_lock.locked() or rebuild_state["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Une reconstruction est déjà en cours. "
                "Consulte GET /rebuild/status."
            ),
        )

    background_tasks.add_task(run_rebuild, request.app)

    return RebuildResponse(
        status="started",
        message=(
            "Reconstruction lancée en arrière-plan depuis "
            "openagenda_chunks.json. "
            "Consulte GET /rebuild/status pour suivre son état."
        ),
        started_at=datetime.now(timezone.utc),
    )


@app.get(
    "/rebuild/status",
    response_model=RebuildStatusResponse,
    summary="Consulter l'état de la reconstruction",
    description=(
        "Retourne l'état de la dernière reconstruction de l'index FAISS.\n\n"
        "Statuts possibles :\n"
        "- `idle` : aucune reconstruction en cours.\n"
        "- `running` : reconstruction en cours.\n"
        "- `completed` : reconstruction terminée avec succès.\n"
        "- `failed` : reconstruction échouée."
    ),
    response_description="État détaillé de la dernière reconstruction.",
    tags=["Administration"],
)
def get_rebuild_status() -> RebuildStatusResponse:
    """Retourne l'état de la dernière reconstruction demandée."""
    return RebuildStatusResponse(**rebuild_state)

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question sur les événements indexés",
    description=(
        "Reçoit une question en langage naturel et renvoie une réponse "
        "générée à partir des événements OpenAgenda indexés dans FAISS.\n\n"
        "Le service applique les filtres métier fournis — ville et période — "
        "recherche les événements pertinents, puis génère une réponse "
        "uniquement à partir des contextes récupérés.\n\n"
        "La réponse inclut les sources utilisées afin de garantir "
        "la traçabilité des informations citées."
    ),
    response_description="Réponse générée et sources associées.",
    tags=["RAG"],
)
def ask(request: AskRequest, fastapi_request: Request) -> AskResponse:
    """
    Reçoit une question JSON et renvoie :
    - une réponse générée à partir du contexte récupéré ;
    - les chunks/sources utilisés, pour la traçabilité.
    """
    service: RAGService | None = getattr(
        fastapi_request.app.state,
        "rag_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Le service RAG n'est pas prêt.",
        )

    started_at = time.perf_counter()

    try:
        result = service.ask(
            question=request.question,
            k=request.k,
            city=request.city,
            date_from=request.date_from,
            date_to=request.date_to,
        )
    except Exception as error:
        logger.exception("Erreur durant le pipeline RAG.")
        raise HTTPException(
            status_code=502,
            detail=f"Erreur durant l'appel au service RAG : {error}",
        ) from error

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

    logger.info(
        "Question traitée en %s ms | k=%s | question=%r",
        duration_ms,
        request.k,
        request.question,
    )

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
    )