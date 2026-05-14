"""FastAPI service exposing TANDEM as two endpoints (BCT hackathon submission).

Task A (POST /simulator/predict):    user persona + product details  → predicted review + rating
Task B (POST /recommender/recommend): user persona + candidate items → ranked top-k

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Run in Docker:
    docker compose up                 # uses .env for GROQ_API_KEY
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.simulator import predict
from src.llm.client import GroqClient

app = FastAPI(
    title="TANDEM",
    description="Two-agent LLM recommendation with cultural overlay (BCT 2026)",
    version="0.1.0",
)

# Lazy-initialised client; reads GROQ_API_KEY from env on first request.
_client: GroqClient | None = None


def _get_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient(cache_path="cache/llm_responses.jsonl")
    return _client


# ---- Schemas ---------------------------------------------------------------

class HistoryItem(BaseModel):
    item_id: str
    rating: float = 0.0
    summary: str = ""
    review_text: str = ""
    timestamp: int = 0


class Persona(BaseModel):
    persona_id: str = "p_request"
    history_window: list[HistoryItem] = Field(default_factory=list)
    preference_summary: str = ""
    default_name: str = "User"
    naija_name: str = "Nigerian User"
    ethnic_hint: str = "Nigerian"
    religious_hint: str = ""


class Item(BaseModel):
    item_id: str
    title: str
    brand: str = ""
    description: str = ""
    category: str = ""


class SimulatorRequest(BaseModel):
    persona: Persona
    item: Item
    condition: Literal["overlay-off", "noise-on", "cultural-on"] = "cultural-on"
    architecture: Literal["decomposed", "monolithic"] = "decomposed"


class SimulatorResponse(BaseModel):
    rating: float
    review: str
    model: str
    cached: bool


class RecommenderRequest(BaseModel):
    persona: Persona
    candidates: list[Item]
    top_k: int = 10
    condition: Literal["overlay-off", "noise-on", "cultural-on"] = "cultural-on"


class RankedItem(BaseModel):
    item_id: str
    rating: float
    review: str


class RecommenderResponse(BaseModel):
    ranked: list[RankedItem]
    cached_hits: int
    api_calls: int


# ---- Endpoints --------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "TANDEM", "version": "0.1.0"}


@app.post("/simulator/predict", response_model=SimulatorResponse)
def simulator_predict(req: SimulatorRequest) -> SimulatorResponse:
    """Task A — predict the user's review and rating for the candidate item."""
    persona_dict = req.persona.model_dump()
    item_dict = req.item.model_dump()
    rec = predict(
        client=_get_client(),
        persona=persona_dict,
        item=item_dict,
        condition=req.condition,
        architecture=req.architecture,
    )
    return SimulatorResponse(
        rating=rec["predicted_rating"],
        review=rec["predicted_review"],
        model=rec["model"],
        cached=rec["cached"],
    )


@app.post("/recommender/recommend", response_model=RecommenderResponse)
def recommender_recommend(req: RecommenderRequest) -> RecommenderResponse:
    """Task B — rank the candidate items for the persona by predicted rating."""
    if not req.candidates:
        raise HTTPException(400, "candidates list is empty")

    client = _get_client()
    persona_dict = req.persona.model_dump()
    predictions: list[dict] = []
    cached_hits = 0
    for item in req.candidates:
        rec = predict(
            client=client,
            persona=persona_dict,
            item=item.model_dump(),
            condition=req.condition,
            architecture="decomposed",
        )
        if rec.get("cached"):
            cached_hits += 1
        predictions.append(rec)

    predictions.sort(
        key=lambda r: (-r["predicted_rating"], -len(r.get("predicted_review", ""))),
    )
    top = predictions[: req.top_k]
    return RecommenderResponse(
        ranked=[
            RankedItem(
                item_id=r["item_id"],
                rating=r["predicted_rating"],
                review=r["predicted_review"],
            )
            for r in top
        ],
        cached_hits=cached_hits,
        api_calls=len(predictions) - cached_hits,
    )
