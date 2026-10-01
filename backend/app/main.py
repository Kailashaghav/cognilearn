import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import perception

app = FastAPI(title="CogniLearn API", version="0.2.1")

# Exact origins (comma-separated), e.g. "https://cognilearn.vercel.app"
_extra = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Matches the production URL and every Vercel preview/deployment URL of this project,
# e.g. https://cognilearn-7vi9itxcc-kailashaghavs-projects.vercel.app
_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://cognilearn[a-z0-9-]*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *_extra],
    allow_origin_regex=_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeaturesIn(BaseModel):
    session_id: str = "demo"
    blink_left: float = Field(0, ge=0, le=1)
    blink_right: float = Field(0, ge=0, le=1)
    smile: float = Field(0, ge=0, le=1)
    brow_down: float = Field(0, ge=0, le=1)
    brow_up: float = Field(0, ge=0, le=1)
    jaw_open: float = Field(0, ge=0, le=1)
    yaw: float = 0
    pitch: float = 0
    face_found: bool = True


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "cognilearn", "version": app.version}


@app.post("/api/perception/analyze")
def analyze(body: FeaturesIn):
    data = body.model_dump()
    sid = data.pop("session_id")
    return perception.analyze(sid, perception.FaceFeatures(**data))


@app.get("/api/perception/history/{session_id}")
def history(session_id: str, n: int = 120):
    return {"session_id": session_id, "items": perception.history(session_id, n)}


@app.post("/api/perception/text")
def text(body: TextIn):
    return perception.text_sentiment(body.text)
