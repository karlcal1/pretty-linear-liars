"""Llama-2-7B birth year: intended vs. actual decade shift for 400 base/source interchanges (Figure 1, bottom).

Directions: OLS and intervention-optimal ridge (fit on the first 70% of prompts), the DAS direction
trained by llm_sweep.py, and a random direction as a control.
"""
import json

import numpy as np
import torch

from pll.llama import (PROBE_LAYER, RIDGE_LAMBDA, decades_of, digit_token_ids, expected_decade, interchange_delta,
                       interchange_metrics, load_birth_year_entities, load_llama, name_final_indices, standardize)
from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import ols_weights, orient_toward_target, ridge_weights, unit

N_PAIRS = 400


def main():
    ensure_output_dirs()
    np.random.seed(0)
    torch.manual_seed(0)
    cached = np.load(RESULTS_DIR / "llm_activations.npz")
    activations = cached["activations"]
    n_train = int(cached["n_train"])
    names, years = load_birth_year_entities()
    decades = decades_of(years)
    target = standardize(years)
    X = activations - activations.mean(axis=0)
    train = np.arange(n_train)

    def oriented(direction):
        return orient_toward_target(unit(direction), X[train], target[train])

    directions = {
        "ridge": oriented(ridge_weights(X[train], target[train], RIDGE_LAMBDA)),
        "OLS": oriented(ols_weights(X[train], target[train])),
        "DAS": oriented(cached["u_das"]),
        "random": oriented(np.random.randn(X.shape[1])),
    }
    base_idx = np.random.choice(len(names), N_PAIRS)
    source_idx = np.random.choice(len(names), N_PAIRS)
    intended = (decades[source_idx] - decades[base_idx]).astype(float)

    tokenizer, model = load_llama()
    digit_ids = digit_token_ids(tokenizer)
    positions = name_final_indices(tokenizer, names)
    base_names = [names[i] for i in base_idx]
    base_positions = [positions[i] for i in base_idx]
    baseline = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER)

    results = {"layer": PROBE_LAYER, "n_pairs": N_PAIRS, "intended": intended.tolist(), "actual": {}, "summary": {}}
    for name, direction in directions.items():
        delta = interchange_delta(activations, direction, base_idx, source_idx)
        shifted = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER, delta, base_positions)
        actual = shifted - baseline
        results["actual"][name] = actual.tolist()
        results["summary"][name] = interchange_metrics(intended, actual)
        print(f"  {name:6s}: {results['summary'][name]}", flush=True)
    with open(RESULTS_DIR / "llm_persample.json", "w") as handle:
        json.dump(results, handle)
    print("wrote results/llm_persample.json")


if __name__ == "__main__":
    main()
