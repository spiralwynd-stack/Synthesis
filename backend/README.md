# Feeling AI Backend

## Run

1. Create and activate venv:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install deps:
   - `pip install -r requirements.txt`
3. Start API:
   - `uvicorn app:app --reload`

API runs on `http://127.0.0.1:8000`.

## Endpoints

- `GET /health`
- `POST /analyze-text`
- `POST /analyze-photo`
- `POST /analyze` (text + photo multipart)
