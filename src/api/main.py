"""FastAPI service exposing TANDEM as four endpoints (BCT hackathon submission).

Task A      POST /simulator/predict            persona + product   -> review + rating
Task B      POST /recommender/recommend        persona + cands     -> deterministic top-k
Agentic     POST /recommender/recommend-agentic persona + cands    -> plan/score/reflect top-k
Multi-turn  POST /recommender/converse          session+message    -> conversational top-k

The /recommender/recommend endpoint is the deterministic argsort that is
load-bearing for the H7 architectural ablation. /recommender/recommend-agentic
is the second mode that satisfies the brief's "agentic workflows that
reason before recommending" requirement without disturbing the ablation
baseline.

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Run in Docker:
    docker compose up                 # uses .env for GROQ_API_KEY
"""
from __future__ import annotations

import threading
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.recommender_agentic import recommend_agentic
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


# ---- Agentic recommender (plan -> score -> reflect) -------------------------

class AgenticRequest(BaseModel):
    persona: Persona
    candidates: list[Item]
    top_k: int = 5
    reflect_window: int = 5
    condition: Literal["overlay-off", "noise-on", "cultural-on"] = "cultural-on"


class AgenticRankedItem(BaseModel):
    item_id: str
    rank: int
    title: str
    predicted_rating: float
    predicted_review: str
    reason: str


class AgenticResponse(BaseModel):
    priorities: list[str]
    ranked: list[AgenticRankedItem]
    trace: dict


@app.post("/recommender/recommend-agentic", response_model=AgenticResponse)
def recommender_recommend_agentic(req: AgenticRequest) -> AgenticResponse:
    """Agentic Task B mode. Plans the user's priorities, scores each candidate
    via the simulator, then reflects + re-ranks the top-N with one-sentence
    justifications. Total LLM calls: C scoring + 2 reasoning (plan, reflect).

    The deterministic /recommender/recommend endpoint remains canonical for
    H7-ablation purposes; this is the additional reasoning mode the BCT
    brief asks for.
    """
    if not req.candidates:
        raise HTTPException(400, "candidates list is empty")
    if req.top_k < 1 or req.top_k > 20:
        raise HTTPException(400, "top_k must be between 1 and 20")
    result = recommend_agentic(
        client=_get_client(),
        persona=req.persona.model_dump(),
        candidates=[c.model_dump() for c in req.candidates],
        top_k=req.top_k,
        reflect_window=req.reflect_window,
        condition=req.condition,
    )
    return AgenticResponse(
        priorities=result["priorities"],
        ranked=[AgenticRankedItem(**r) for r in result["ranked"]],
        trace=result["trace"],
    )


# ---- Multi-turn conversational recommender ----------------------------------

# Sessions live in process memory. The free-tier HF Space restarts on sleep, so
# session state is best-effort and ephemeral by design --- judges hitting the
# live demo see this behaviour explicitly documented in /docs.
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


class ConverseRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Omit on the first turn; the server returns one. Pass it back on subsequent turns.",
    )
    persona: Persona | None = None
    candidates: list[Item] | None = None
    message: str | None = Field(
        default=None,
        description="User feedback for turns 2+, e.g. 'I want something with shea butter, not synthetic'.",
    )
    top_k: int = 5
    condition: Literal["overlay-off", "noise-on", "cultural-on"] = "cultural-on"


class ConverseResponse(BaseModel):
    session_id: str
    turn: int
    refined_preferences: list[str]
    ranked: list[RankedItem]
    assistant_reply: str


def _converse_rank(client: GroqClient, persona: dict, candidates: list[dict], top_k: int,
                   condition: str) -> list[dict]:
    out: list[dict] = []
    for it in candidates:
        rec = predict(client=client, persona=persona, item=it, condition=condition,
                      architecture="decomposed")
        out.append(rec)
    out.sort(key=lambda r: (-r["predicted_rating"], -len(r.get("predicted_review", ""))))
    return out[:top_k]


@app.post("/recommender/converse", response_model=ConverseResponse)
def recommender_converse(req: ConverseRequest) -> ConverseResponse:
    """Conversational recommendation with server-side session state.

    First turn: send `persona` and `candidates`. Server returns a `session_id`
    plus top-k.

    Subsequent turns: send the same `session_id` and a free-text `message`
    (e.g. "I want something with shea butter, not synthetic"). The server
    appends the feedback to the persona's preference summary and re-ranks
    the original candidate pool through the simulator, returning the new
    top-k. The candidate pool is fixed at session creation.

    Sessions are in-process and ephemeral; expect them to vanish on HF
    Space sleep.
    """
    client = _get_client()

    if req.session_id is None:
        if req.persona is None or not req.candidates:
            raise HTTPException(400, "first turn requires `persona` and `candidates`")
        sid = uuid.uuid4().hex[:12]
        persona_dict = req.persona.model_dump()
        candidates_dict = [c.model_dump() for c in req.candidates]
        top = _converse_rank(client, persona_dict, candidates_dict, req.top_k, req.condition)
        state = {
            "persona": persona_dict,
            "candidates": candidates_dict,
            "refined": [],
            "turn": 1,
            "top_k": req.top_k,
            "condition": req.condition,
        }
        with _sessions_lock:
            _sessions[sid] = state
        reply = (
            f"Started session {sid}. Top {len(top)} recommendations ranked. "
            f"Send a follow-up message to refine — e.g. 'I want something with shea butter, "
            f"not synthetic' or 'show me halal-friendly options only'."
        )
        return ConverseResponse(
            session_id=sid, turn=1, refined_preferences=[],
            ranked=[RankedItem(item_id=r["item_id"], rating=r["predicted_rating"],
                               review=r["predicted_review"]) for r in top],
            assistant_reply=reply,
        )

    with _sessions_lock:
        state = _sessions.get(req.session_id)
    if state is None:
        raise HTTPException(404, f"session {req.session_id} not found (sessions are ephemeral)")
    if not req.message:
        raise HTTPException(400, "subsequent turns require `message`")

    msg = req.message.strip()[:300]
    state["refined"].append(msg)
    state["turn"] += 1
    persona = dict(state["persona"])
    persona["preference_summary"] = (
        (persona.get("preference_summary") or "")
        + " | conversational refinement: " + " ; ".join(state["refined"])
    ).strip()
    persona["persona_id"] = f"{state['persona'].get('persona_id', 'session')}-t{state['turn']}"
    top = _converse_rank(client, persona, state["candidates"], state["top_k"], state["condition"])

    reply = (
        f"Turn {state['turn']}: applied feedback '{msg[:80]}'. "
        f"Re-ranked top {len(top)}. Send another message to keep refining."
    )
    return ConverseResponse(
        session_id=req.session_id, turn=state["turn"],
        refined_preferences=list(state["refined"]),
        ranked=[RankedItem(item_id=r["item_id"], rating=r["predicted_rating"],
                           review=r["predicted_review"]) for r in top],
        assistant_reply=reply,
    )
