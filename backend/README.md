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
