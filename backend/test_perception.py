from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def post(**kw):
    return c.post("/api/perception/analyze", json={"session_id": "t", **kw}).json()


def test_health():
    assert c.get("/health").json()["status"] == "ok"


def test_engaged_when_looking_at_screen():
    r = None
    for _ in range(10):
        r = post(yaw=2, pitch=1)
    assert r["emotion"] == "engaged" and r["attention"] > 0.8


def test_attention_drops_when_looking_away():
    r = None
    for _ in range(10):
        r = post(yaw=50)
    assert r["attention"] < 0.5


def test_confused_on_furrowed_brows():
    assert post(brow_down=0.6)["emotion"] in {"confused", "engaged"}
    assert post(session_id="x", brow_down=0.6)["emotion"] == "confused"


def test_text_sentiment():
    assert c.post("/api/perception/text", json={"text": "I am so confused and stuck"}).json()["label"] == "negative"
