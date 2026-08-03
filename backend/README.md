# Crop Disease Detection Backend

## Setup

This project uses `uv` for Python dependency management.

```bash
cd backend
uv venv
uv sync --extra ml
```

For API and explainability later:

```bash
uv sync --extra api --extra ml --extra explainability
```

## Dataset

The PlantVillage dataset is **not** included in the repository. Download it locally with the Kaggle API.

1. Create a `.env` file in `backend/` with:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

2. Download:

```bash
uv run python scripts/download_dataset.py
```

3. Prepare splits (train / val / test):

```bash
uv run python -m scripts.prepare_dataset
```

Data will live under `backend/data/` (gitignored).

## Training

### 1. Choose training mode (in `app/core/config.py`)

```python
TRAINING_MODE = "full"   # options: "smoke" | "debug" | "full"
```

| Mode   | Purpose                         | Epochs | Batches limited? |
|--------|---------------------------------|--------|------------------|
| smoke  | Pipeline check                  | 1      | Yes (very few)   |
| debug  | Quick sanity check              | 2      | Yes              |
| full   | Real training on full dataset   | 20     | No               |

### 2. Run training

```bash
# From the backend/ directory
uv run python -m scripts.train
```

Checkpoints are saved to `backend/models/checkpoints/`:
- `best_model.pt`
- `last_model.pt`
- periodic `epoch_XXX.pt`

History and plots go to:
- `backend/outputs/history/training_history.json`
- `backend/outputs/plots/` (loss, accuracy, LR curves)

Logs: `backend/logs/training.log`

### Resume training

In `scripts/train.py` set `resume=True` when calling `trainer.train(...)`, or modify the call after a quick edit.

## Milestone Status

| Milestone | Status |
|-----------|--------|
| 1. Project Foundation | ✅ Completed |
| 2. Data Pipeline | ✅ Completed |
| 3. Model Development & Training | ✅ Training pipeline ready (run on your machine) |
| 4. Evaluation & Explainability | ⬜ Pending |
| 5. FastAPI | ⬜ Pending |
| 6. Frontend + Deployment | ⬜ Pending |

## Notes

- Requires Python ≥ 3.11
- GPU (CUDA) is used automatically when available; otherwise CPU
- `NUM_CLASSES` is detected from the dataset at runtime (no hard-coding)
