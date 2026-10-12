from datetime import date, datetime
from pydantic import BaseModel, Field, model_validator
from typing import Literal


class AskRequest(BaseModel):
    """Corps JSON attendu par POST /ask."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1_000,
        examples=[
            "Quels sont les concerts de musique à Montpellier cet été ?"
        ],
    )
    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre maximum de chunks à récupérer dans FAISS.",
    )

    city: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        examples=["Montpellier"],
        description=(
            "Ville à filtrer explicitement. Elle est prioritaire sur la ville "
            "éventuellement mentionnée dans la question."
        ),
    )

    date_from: date | None = Field(
        default=None,
        examples=["2026-06-21"],
        description="Date de début incluse du filtre.",
    )

    date_to: date | None = Field(
        default=None,
        examples=["2026-09-22"],
        description="Date de fin incluse du filtre.",
    )

    @model_validator(mode="after")
    def validate_date_range(self):
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError(
                "date_from doit être antérieure ou égale à date_to."
            )

        return self


class Source(BaseModel):
    """Information de traçabilité d'un chunk utilisé par la réponse."""

    rank: int
    event_id: str | None = None
    title: str
    city: str | None = None
    firstdate_begin: str | None = None
    lastdate_end: str | None = None
    source_url: str | None = None
    excerpt: str


class AskResponse(BaseModel):
    """Réponse renvoyée par POST /ask."""

    question: str
    answer: str
    sources: list[Source]

class RebuildResponse(BaseModel):
    """Réponse immédiate retournée quand une reconstruction est lancée."""

    status: Literal["started", "running", "completed", "failed"]
    message: str
    started_at: datetime | None = None


class RebuildStatusResponse(BaseModel):
    """État actuel de la dernière reconstruction connue."""

    status: Literal["idle", "running", "completed", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    chunks_read: int | None = None
    vectors_indexed: int | None = None
    embedding_dimension: int | None = None
    error: str | None = None