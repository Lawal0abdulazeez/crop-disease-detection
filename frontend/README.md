# Crop Disease Detection — Frontend

React + Vite UI for the FastAPI backend.

## Local development

1. Start the API (from `backend/`):

```bash
uv run uvicorn app.main:app --reload --port 8000
```

2. Start the UI:

```bash
npm install
npm run dev
```

Open http://localhost:5173 — requests to `/api` are proxied to the backend.

## Production build

```bash
# Point at your deployed API
set VITE_API_URL=https://your-api.onrender.com
npm run build
```

Static files are in `dist/`.
