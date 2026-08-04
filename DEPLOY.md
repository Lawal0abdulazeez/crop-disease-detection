# Frontend, local testing & Render deployment

## Local test (recommended before deploy)

### Option A — separate processes (fastest for development)

**Terminal 1 — API**

```powershell
cd backend
uv sync --extra ml --extra api
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ensure `backend/models/checkpoints/best_model.pt` exists (from smoke or full training)  
and `backend/data/metadata/class_names.json` exists (created during training/prepare).

**Terminal 2 — React**

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**  
Vite proxies `/api/*` → `http://127.0.0.1:8000`.

Checklist:

1. Status badges show **API online** and **Model loaded**
2. Upload a leaf from `backend/data/splits/test/...`
3. **Detect disease** returns class + confidence + treatment tips
4. Swagger still works: http://localhost:8000/docs

### Option B — Docker Compose

From repo root (with checkpoint already on disk):

```powershell
docker compose up --build
```

- Frontend: http://localhost:3000  
- API: http://localhost:8000/docs  

Volumes mount your local `models/checkpoints` and `data/metadata`.

---

## Render deployment

### 1. Host the model file

`best_model.pt` is gitignored (too large / binary). Host it publicly, for example:

- Hugging Face model repo (raw file URL)
- S3 / R2 with public read
- Google Drive **direct** download link

Also host `class_names.json` if possible (`CLASS_NAMES_URL`).

### 2. Deploy API (Docker)

In [Render](https://dashboard.render.com):

1. **New → Web Service** → connect this GitHub repo
2. Root directory: `backend`
3. Runtime: **Docker**
4. Dockerfile path: `./Dockerfile`
5. Instance: at least **Starter** (free tier often OOMs on torch)
6. Environment variables:

| Key | Value |
|-----|--------|
| `MODEL_URL` | Public URL to `best_model.pt` |
| `MODEL_FILENAME` | `best_model.pt` |
| `CLASS_NAMES_URL` | Public URL to `class_names.json` (optional) |
| `PORT` | `8000` (Render may set this automatically) |

Health check path: `/health`

After deploy, open `https://<your-api>.onrender.com/docs` and try `/predict`.

### 3. Deploy frontend (Static Site)

1. **New → Static Site** → same repo
2. Root directory: `frontend`
3. Build command: `npm install && npm run build`
4. Publish directory: `dist`
5. Environment variable at **build** time:

| Key | Value |
|-----|--------|
| `VITE_API_URL` | `https://<your-api>.onrender.com` (no trailing slash) |

Rewrite rule: `/*` → `/index.html` (SPA).

Redeploy the static site whenever you change `VITE_API_URL`.

### 4. Blueprint (optional)

`render.yaml` at repo root can create both services. You still must set `MODEL_URL` and `VITE_API_URL` in the dashboard (`sync: false`).

---

## CORS

Backend already allows all origins (`allow_origins=["*"]`). Fine for this project; tighten later if needed.

## After full training

1. Replace `best_model.pt` on your host / local volume  
2. No frontend or API code changes required  
3. Restart API (or redeploy) so it reloads weights  

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| API offline in UI | Backend not running or wrong `VITE_API_URL` |
| Model not loaded | Missing checkpoint / `MODEL_URL` failed |
| 503 on predict | Checkpoint not found on server |
| CORS errors | Use Vite proxy locally or set full API URL in prod |
| Render OOM / crash | Use Starter+ plan; CPU torch only |
