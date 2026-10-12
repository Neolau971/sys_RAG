import json
from datetime import date
from pathlib import Path
from typing import Any
import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import pandas as pd
from datasets import Dataset as HFDataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

from app.rag_service import RAGService

os.environ["RAGAS_DO_NOT_TRACK"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_FOLDER = DATA_DIR / "langchain_faiss"

EVALUATION_FILE = DATA_DIR / "evaluation" / "openagenda_eval_set.json"
RESULTS_DIR = DATA_DIR / "evaluation" / "results"

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-2603"
TOP_K = 20


def parse_optional_date(value: str | None) -> date | None:
    """Convertit YYYY-MM-DD vers date, ou None."""
    return date.fromisoformat(value) if value else None


def load_evaluation_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Jeu d'évaluation introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_ragas_rows(
    service: RAGService,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Exécute le RAG pour chaque cas et produit :
    - rows_ragas : format attendu par Ragas
    - traces : détails utiles pour diagnostiquer les mauvais scores
    """
    rows_ragas = []
    traces = []

    for position, case in enumerate(cases, start=1):
        question = case["question"]

        print(f"[{position}/{len(cases)}] Évaluation : {question}")

        trace = service.ask_with_trace(
            question=question,
            k=case.get("k", TOP_K),
            city=case.get("city"),
            date_from=parse_optional_date(case.get("date_from")),
            date_to=parse_optional_date(case.get("date_to")),
        )

        rows_ragas.append(
            {
                "user_input": question,
                "response": trace["answer"],
                "retrieved_contexts": trace["contexts"],
                "reference": case["ground_truth"],
            }
        )

        traces.append(
            {
                "id": case.get("id"),
                "question": question,
                "ground_truth": case["ground_truth"],
                "answer": trace["answer"],
                "contexts": trace["contexts"],
                "sources": trace["sources"],
                "applied_filters": trace["applied_filters"],
            }
        )

    return rows_ragas, traces


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY non définie. Vérifie ton fichier .env ou tes variables d’environnement."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_evaluation_cases(EVALUATION_FILE)

    service = RAGService(
        faiss_folder=FAISS_FOLDER,
        embedding_model=EMBEDDING_MODEL,
        chat_model=CHAT_MODEL,
    )

    rows_ragas, traces = build_ragas_rows(service, cases)

    # Normaliser contexts en liste si besoin
    for row in rows_ragas:
        if isinstance(row["retrieved_contexts"], str):
            row["retrieved_contexts"] = [row["retrieved_contexts"]]

    ragas_dataset = HFDataset.from_list(rows_ragas)

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.0,
        max_retries=3,
        timeout=120,
    )

    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    llm_wrapper = LangchainLLMWrapper(llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    metrics_to_run = [
        ("faithfulness", faithfulness),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]

    metric_frames: list[pd.DataFrame] = []

    for metric_name, metric in metrics_to_run:
        print(f"\nÉvaluation de {metric_name}...")

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[metric],
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
            show_progress=True,
        )

        result_df = result.to_pandas()
        metric_frames.append(result_df[[metric_name]])

        print(f"\n--- {metric_name} ---")
        print(result_df[[metric_name]])

    final_df = pd.concat(metric_frames, axis=1)
    final_df = pd.concat(
        [
            ragas_dataset.to_pandas(),
            final_df,
        ],
        axis=1,
    )

    print("\n--- Scores moyens Ragas ---")
    print(
        final_df[
            [
                "faithfulness",
                "context_precision",
                "context_recall",
            ]
        ]
        .mean(numeric_only=True)
    )

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    scores_file = RESULTS_DIR / f"ragas_scores_{timestamp}.csv"
    traces_file = RESULTS_DIR / f"ragas_traces_{timestamp}.json"

    final_df.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig",
    )

    with traces_file.open("w", encoding="utf-8") as file:
        json.dump(traces, file, ensure_ascii=False, indent=2)

    print(f"\nScores détaillés : {scores_file}")
    print(f"Traces RAG : {traces_file}")


if __name__ == "__main__":
    main()