"""Perception module: turns raw face features into attention + emotion.

Inputs come from MediaPipe FaceLandmarker blendshapes (0..1) and head pose (degrees).
Rule-based on purpose for Step 2: transparent, explainable, no training data needed.
Step 6 compares this against a learned model.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict

EMA_ALPHA = 0.35  # smoothing: higher = reacts faster


@dataclass
class FaceFeatures:
    blink_left: float = 0.0
    blink_right: float = 0.0
    smile: float = 0.0        # mean of mouthSmileLeft/Right
    brow_down: float = 0.0    # mean of browDownLeft/Right
    brow_up: float = 0.0      # browInnerUp
    jaw_open: float = 0.0
    yaw: float = 0.0          # degrees, left/right head turn
    pitch: float = 0.0        # degrees, up/down head tilt
    face_found: bool = True


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def attention_score(f: FaceFeatures) -> float:
    """1.0 = looking at the screen, eyes open. Drops with head turn and closed eyes."""
    if not f.face_found:
        return 0.0
    yaw_pen = _clamp((abs(f.yaw) - 15) / 30)       # free up to 15 deg, zero at 45 deg
    pitch_pen = _clamp((abs(f.pitch) - 15) / 30)
    eyes_closed = _clamp(((f.blink_left + f.blink_right) / 2 - 0.4) / 0.4)
    return _clamp(1.0 - 0.7 * yaw_pen - 0.3 * pitch_pen - 0.6 * eyes_closed)


def classify_emotion(f: FaceFeatures, attention: float) -> Dict:
    """Returns {label, confidence, reasons}. Reasons feed the explainability panel."""
    if not f.face_found:
        return {"label": "away", "confidence": 1.0, "reasons": ["No face detected"]}

    blink = (f.blink_left + f.blink_right) / 2
    if f.smile > 0.45:
        return {"label": "happy", "confidence": _clamp(f.smile),
                "reasons": [f"Smile strength {f.smile:.2f} > 0.45"]}
    if f.brow_down > 0.35 or (f.brow_up > 0.45 and f.jaw_open < 0.2):
        strength = max(f.brow_down, f.brow_up)
        return {"label": "confused", "confidence": _clamp(strength + 0.2),
                "reasons": [f"Brow tension {strength:.2f} (furrowed or raised brows)"]}
    if blink > 0.5 or attention < 0.35:
        return {"label": "bored", "confidence": _clamp(1 - attention),
                "reasons": [f"Attention {attention:.2f} is low", f"Eye closure {blink:.2f}"]}
    if attention > 0.7:
        return {"label": "engaged", "confidence": attention,
                "reasons": [f"Attention {attention:.2f} > 0.70 with a relaxed face"]}
    return {"label": "neutral", "confidence": 0.5, "reasons": ["No strong signal"]}


@dataclass
class SessionState:
    attention_ema: float = 1.0
    history: list = field(default_factory=list)


_sessions: Dict[str, SessionState] = {}


def analyze(session_id: str, f: FaceFeatures) -> Dict:
    s = _sessions.setdefault(session_id, SessionState())
    raw = attention_score(f)
    s.attention_ema = EMA_ALPHA * raw + (1 - EMA_ALPHA) * s.attention_ema
    emo = classify_emotion(f, s.attention_ema)
    result = {
        "attention": round(s.attention_ema, 3),
        "attention_raw": round(raw, 3),
        "emotion": emo["label"],
        "confidence": round(emo["confidence"], 3),
        "reasons": emo["reasons"],
    }
    s.history.append(result)
    del s.history[:-600]  # keep last ~2 minutes at 5 Hz
    return result


def history(session_id: str, n: int = 120):
    return _sessions.get(session_id, SessionState()).history[-n:]


# ---------- Text sentiment (lexicon fallback, optional transformer) ----------
_POS = {"good", "great", "clear", "understand", "got", "easy", "love", "nice", "makes", "sense", "thanks", "helpful", "cool"}
_NEG = {"confused", "lost", "hard", "difficult", "hate", "boring", "stuck", "dont", "don't", "cant", "can't", "unclear", "tired", "frustrated", "why"}
_pipe = None


def _transformer():
    global _pipe
    if _pipe is None:
        from transformers import pipeline  # lazy import: heavy
        _pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _pipe


def text_sentiment(text: str) -> Dict:
    if os.getenv("COGNILEARN_USE_TRANSFORMERS") == "1":
        out = _transformer()(text[:512])[0]
        score = out["score"] if out["label"] == "POSITIVE" else -out["score"]
        return {"sentiment": round(score, 3), "label": out["label"].lower(), "engine": "distilbert"}
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return {"sentiment": round(score, 3), "label": label, "engine": "lexicon"}
