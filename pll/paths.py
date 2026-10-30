"""Where data, results and figures live. Everything is relative to the repository root."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PLL_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# Mess3 transformer of Shai et al. (2024): 4 layers, width 64, checkpoint after 4,075,724,800 tokens.
MESS3_CHECKPOINT = DATA_DIR / "mess3_transformer" / "4075724800.pt"

# Wikidata birth-year (property P569) entities from Heinzerling & Inui (2024).
BIRTH_YEAR_FILES = [
    DATA_DIR / "wikidata" / f"numprop_P569.quantile_9.sample_0.{split}.jsonl"
    for split in ("train", "test")
]

# Ungated mirror of the Llama-2-7B base weights.
LLAMA_MODEL_ID = "NousResearch/Llama-2-7b-hf"


def ensure_output_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
