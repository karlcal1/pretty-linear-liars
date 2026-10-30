"""Llama-2-7B birth year: intervention loss and probe loss of a ridge probe at each layer (Figure 6).

3000 prompts; at each layer a ridge probe (lambda = 1e4) is fit in-sample and tested on 200 fixed
base/source interchanges.  Layer 8 minimizes the intervention loss and is the site used elsewhere.
"""
import json

import numpy as np
import torch

from pll.llama import (RIDGE_LAMBDA, cache_activations, decades_of, digit_token_ids, expected_decade,
                       interchange_delta, interchange_metrics, load_birth_year_entities, load_llama, standardize)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.probes import best_affine_mse, ridge_weights, unit

N_ENTITIES = smoke(48, 3000)
N_PAIRS = smoke(12, 200)
LAYERS = smoke([0, 8], [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28])


def main():
    ensure_output_dirs()
    np.random.seed(0)
    torch.manual_seed(0)
    tokenizer, model = load_llama()
    digit_ids = digit_token_ids(tokenizer)
    names, years = load_birth_year_entities(N_ENTITIES)
    decades = decades_of(years)
    target = standardize(years)
    base_idx = np.random.choice(len(names), N_PAIRS)
    source_idx = np.random.choice(len(names), N_PAIRS)
    intended = (decades[source_idx] - decades[base_idx]).astype(float)
    base_names = [names[i] for i in base_idx]

    results = {"n": len(names), "n_pairs": N_PAIRS, "ridge_lambda": RIDGE_LAMBDA, "layers": LAYERS,
               "probe_loss": [], "decodability": [], "interv_mse": [], "slope": []}
    for layer in LAYERS:
        activations, positions = cache_activations(model, tokenizer, names, layer)
        X = activations - activations.mean(axis=0)
        direction = unit(ridge_weights(X, target, RIDGE_LAMBDA))
        # in-sample best scale-and-bias readout along the direction; equals 1 - corr^2 for a standardized target
        probe_loss = best_affine_mse(direction, X, target, X, target)
        decodability = float(abs(np.corrcoef(X @ direction, target)[0, 1]))
        base_positions = [positions[i] for i in base_idx]
        baseline = expected_decade(model, tokenizer, base_names, digit_ids, layer)
        delta = interchange_delta(activations, direction, base_idx, source_idx)
        shifted = expected_decade(model, tokenizer, base_names, digit_ids, layer, delta, base_positions)
        metrics = interchange_metrics(intended, shifted - baseline)
        results["probe_loss"].append(probe_loss)
        results["decodability"].append(decodability)
        results["interv_mse"].append(metrics["interv_mse"])
        results["slope"].append(metrics["slope"])
        print(f"  layer {layer:2d}: probe_loss={probe_loss:.3f} |corr|={decodability:.3f} "
              f"interv_mse={metrics['interv_mse']:.2f} slope={metrics['slope']:+.3f}", flush=True)
    with open(RESULTS_DIR / "llm_layersweep.json", "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote results/llm_layersweep.json")


if __name__ == "__main__":
    main()
