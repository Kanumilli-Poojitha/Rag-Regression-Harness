"""Environment-driven settings. Secrets come only from the environment / .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional at import time
    pass

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "config"
PROMPTS_PATH = ROOT / "prompts" / "judge_prompts.json"
PINS_PATH = CONFIG_DIR / "eval_pins.json"
EVAL_SET_PATH = DATASET_DIR / "eval_set.json"
CORPUS_PATH = DATASET_DIR / "corpus.json"


def qdrant_url() -> str:
    url = os.environ.get("QDRANT_URL")
    if url:
        return url
    return f"http://{os.environ.get('QDRANT_HOST', 'localhost')}:{os.environ.get('QDRANT_PORT', '6333')}"
