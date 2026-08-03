# Crop Disease Detection Backend

## Setup

This project uses `uv` for Python dependency management.

```powershell
uv sync
```

Install the full backend and ML stack when you are ready for training/API work:

```powershell
uv sync --extra api --extra ml --extra explainability
```

## Download PlantVillage

The dataset downloader exports Hugging Face's predefined train/test split into
`data/raw/plantvillage/`.

For a quick smoke test:

```powershell
uv run python scripts/download_plantvillage.py --max-samples 10
```

For the full color dataset:

```powershell
uv run python scripts/download_plantvillage.py
```

Available configurations:

```powershell
uv run python scripts/download_plantvillage.py --config color
uv run python scripts/download_plantvillage.py --config grayscale
uv run python scripts/download_plantvillage.py --config segmented
```

✅ Milestone 1
Project Setup

✓ Project Structure
✓ Configuration
✓ Utilities
✓ Logging

----------------------------

✅ Milestone 2
Dataset Pipeline

✓ Dataset Download
✓ Dataset Preparation
✓ Dataset Splitting
✓ Metadata Generation
✓ Dataloader

----------------------------

🚧 Milestone 3

✓ EfficientNet
✓ Model Factory
✓ Optimizer
✓ Scheduler
✓ Loss
✓ Metrics
✓ Callbacks
✓ Trainer
✓ Train Script

Pending

• Integration Testing
• Evaluator
How to Run
# create virtual environment
uv venv

# install dependencies
uv sync

# prepare dataset
uv run python -m scripts.prepare_dataset

# train
uv run python -m scripts.train
Dataset

Mention that the dataset is not included in the repository.

Example:

The PlantVillage dataset is downloaded locally using the Kaggle API and is intentionally excluded from version control.

Roadmap
✔ Milestone 1

✔ Milestone 2

🚧 Milestone 3

⬜ Milestone 4
FastAPI API

⬜ Milestone 5
Deployment