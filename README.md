# Feeling AI MVP

This workspace now includes:

- `backend`: FastAPI API for text + photo analysis
- `frontend`: React app UI built with Vite

## 1) Run Backend

```bash
source .venv/bin/activate
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Backend URL: http://127.0.0.1:8000

## 2) Run Frontend

```bash
cd frontend
cp .env.example .env
npm run dev
```

Frontend URL: http://127.0.0.1:5173

## Deploy Public Website (Render)

This repo is configured for one-service deployment where the backend serves the built frontend from the same URL.

1. In Render, create a new `Blueprint` deployment from this repo:
  - `https://github.com/spiralwynd-stack/Synthesis`
2. Render will detect `render.yaml` and provision the `feeling-ai` web service.
3. In Render service settings, set environment variable:
  - `STABILITY_API_KEY=your_key_here`
4. Deploy. Render will build the Docker image, compile the frontend, and serve the app publicly.

After deployment, your public site is the Render service URL.

## What Works Now

- Text analysis endpoint (`/analyze-text`)
- Photo palette extraction endpoint (`/analyze-photo`)
- Combined endpoint (`/analyze`) for text + photo
- Emotion image endpoint (`/generate-emotion-image`) powered by Stability AI
- UI form to submit text and photo and show:
  - top themes
  - dominant emotions
  - extracted color palette
  - generated background prompt

## Stability AI Setup

Create `backend/.env` with:

```bash
STABILITY_API_KEY=your_key_here
```

The frontend now shows the analysis first, then a separate button to generate one image per session from the extracted emotions. The optional style notes field lets you steer the visual tone in text.

## Next Upgrade Steps

1. Add proper NLP models (spaCy, transformers, sentence-transformers)
2. Add person segmentation + clothing-only color extraction
3. Add image generation + compositing pipeline
4. Save results and project history
