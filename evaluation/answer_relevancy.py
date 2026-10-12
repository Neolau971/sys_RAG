import os
from pathlib import Path

os.environ["RAGAS_DO_NOT_TRACK"] = "1"

from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from mistralai.client import Mistral
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

dataset = HFDataset.from_list(
    [
        {
            "user_input": "Existe-t-il une exposition de photographie à Montpellier ?",
            "response": "Oui. Une rétrospective consacrée à Lucien Hervé est prévue au Pavillon Populaire.",
            "retrieved_contexts": [
                "En hiver 2026-2027, le Pavillon Populaire consacrera une rétrospective à Lucien Hervé."
            ],
            "reference": "Oui. Une rétrospective consacrée à Lucien Hervé est prévue au Pavillon Populaire.",
        }
    ]
)

mistral_client = Mistral()

judge_llm = llm_factory(
    model="mistral-small",
    provider="mistral",
    client=mistral_client,
)

metric = AnswerRelevancy(
    llm=judge_llm,
    strictness=1,
)

result = evaluate(
    dataset=dataset,
    metrics=[metric],
    llm=judge_llm,
    show_progress=True,
)

print(result.to_pandas())