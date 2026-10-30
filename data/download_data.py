"""Fetch the two external inputs: the Mess3 transformer checkpoint and the Wikidata birth-year prompts.

  python data/download_data.py

Llama-2-7B itself is downloaded by `transformers` the first time a Llama script runs (~13 GB).
"""
import shutil
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).resolve().parent

# Shai et al. (2024) training run 20241205175736_23: 4 layers, d_model 64, Mess3 with x=0.05, alpha=0.85.
MESS3_REPO = "SimplexAI/quantum-representations"
MESS3_FILES = ["models/20241205175736_23/4075724800.pt", "models/20241205175736_23/run_config.yaml"]

# Heinzerling & Inui (2024), property P569 (date of birth).
WIKIDATA_BASE = "https://raw.githubusercontent.com/bheinzerling/numeric-property-repr/main/data/wikidata/"
WIKIDATA_FILES = ["numprop_P569.quantile_9.sample_0.train.jsonl", "numprop_P569.quantile_9.sample_0.test.jsonl"]


def fetch_mess3_checkpoint():
    target_dir = DATA_DIR / "mess3_transformer"
    target_dir.mkdir(exist_ok=True)
    snapshot_dir = snapshot_download(MESS3_REPO, repo_type="dataset", allow_patterns=MESS3_FILES)
    for relative in MESS3_FILES:
        source = Path(snapshot_dir) / relative
        destination = target_dir / source.name
        shutil.copyfile(source, destination)
        print(f"  {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def fetch_birth_year_prompts():
    target_dir = DATA_DIR / "wikidata"
    target_dir.mkdir(exist_ok=True)
    for filename in WIKIDATA_FILES:
        destination = target_dir / filename
        # requests ships its own CA bundle; the standard library fails on python.org macOS installs without certificates
        response = requests.get(WIKIDATA_BASE + filename, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        print(f"  {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    print("Mess3 transformer checkpoint:")
    fetch_mess3_checkpoint()
    print("Wikidata birth-year prompts:")
    fetch_birth_year_prompts()
