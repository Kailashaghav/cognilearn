# CogniLearn — Emotion-Aware Adaptive Study Companion

A cognitive computing mini project: perceive the learner (attention + emotion),
understand their notes (RAG + knowledge graph), and adapt teaching (bandit policy).

## Status
- [x] Step 1: repo + backend/frontend scaffold
- [x] Step 2: perception module (webcam -> attention/emotion, text sentiment)
- [ ] Step 3: knowledge module (RAG + knowledge graph)
- [ ] Step 4: adaptation engine (Thompson-sampling bandit)
- [ ] Step 5: explainability dashboard
- [ ] Step 6: evaluation
- [ ] Step 7: deploy + report

## Run locally

Backend (Terminal 1):
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Frontend (Terminal 2):
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 and allow camera access.

## Privacy
Video never leaves the browser. MediaPipe runs client-side and only
numeric face features (~20 numbers, 5x per second) go to the backend.

## Architecture
Browser (MediaPipe FaceLandmarker) -> features -> FastAPI perception -> attention score + emotion label
