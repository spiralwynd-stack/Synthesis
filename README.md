# Feeling AI MVP

This workspace now includes:

- `backend`: FastAPI API for text + photo analysis
- `frontend`: React app UI built with Vite

## 1) Run Backend

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --reload --app-dir .
```

Backend URL: http://127.0.0.1:8000

## 2) Run Frontend

```bash
cd frontend
cp .env.example .env
npm run dev
```

Frontend URL: http://127.0.0.1:5173

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
