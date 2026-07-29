import os
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment variables
load_dotenv()

# Set Kaggle credentials from .env
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Authenticate
api = KaggleApi()
api.authenticate()

# Output folder
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download and unzip
api.dataset_download_files(
    "emmarex/plantdisease",
    path=str(OUTPUT_DIR),
    unzip=True
)

print("✅ Dataset downloaded successfully!")