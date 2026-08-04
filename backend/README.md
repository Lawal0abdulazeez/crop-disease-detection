# Crop Disease Detection Backend

## Setup

```bash
cd backend
uv venv
uv sync --extra ml --extra api
```

## Dataset

PlantVillage is **not** in the repo. Use Kaggle:

1. `backend/.env`:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

(Use plain `KEY=value` lines only — no quotes or extra text, or python-dotenv will warn.)

2. Download + prepare:

```bash
uv run python scripts/download_dataset.py
uv run python -m scripts.prepare_dataset
```

## Training

In `app/core/config.py`:

```python
TRAINING_MODE = "smoke"   # or "debug" | "full"
```

```bash
uv run python -m scripts.train
```

Checkpoints → `models/checkpoints/best_model.pt` and `last_model.pt`  
History/plots → `outputs/history/`, `outputs/plots/`

**Plug-and-play:** After full training later, the same `best_model.pt` is used by evaluate, predict, and the API with no code changes.

## Evaluation (Milestone 4)

Uses your existing smoke (or full) checkpoint:

```bash
uv run python -m scripts.evaluate
uv run python -m scripts.evaluate --checkpoint last
```

Outputs → `outputs/evaluation/`:
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `predictions.json`

## CLI Predict

```bash
uv run python -m scripts.predict path/to/leaf.jpg
uv run python -m scripts.predict path/to/leaf.jpg --checkpoint best --top-k 5
```

## FastAPI (Milestone 5)

```bash
uv sync --extra api --extra ml
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + model loaded |
| GET | `/classes` | Class names |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Single image upload |
| POST | `/batch-predict` | Multiple images |

Example:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@leaf.jpg"
```

## Milestone Status

| Milestone | Status |
|-----------|--------|
| 1. Foundation | ✅ |
| 2. Data pipeline | ✅ |
| 3. Training | ✅ (smoke verified; full when ready) |
| 4. Evaluation | ✅ |
| 5. FastAPI | ✅ |
| 6. Frontend + Deploy | ⬜ Next |
