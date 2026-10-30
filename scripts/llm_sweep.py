"""Llama-2-7B birth year: ridge sweep over the penalty, with DAS and OLS as reference directions.

For each lambda the ridge direction (fit on the first 70% of the 8000 prompts) is scored by
  * probe loss: in-sample MSE of the regression readout refit on all 8000 prompts (standardized target);
    for DAS, which has no regression weights, the best scale-and-bias readout along its direction,
  * variance along the direction,
  * intervention loss: MSE between the post-interchange expected decade and the source's decade on
    200 fixed base/source pairs,
  * signed cosine to the DAS direction and to the OLS direction.
Feeds Figure 2 (right), Figure 7 (right) and the Llama rows of Table 1.  Also caches the activations
(results/llm_activations.npz) for the model-free follow-ups.
"""
import json

import numpy as np
import torch

from pll.llama import (LAMBDA_GRID, PROBE_LAYER, cache_activations, decades_of, digit_token_ids,
                       expected_decade, interchange_delta, interchange_metrics, load_birth_year_entities,
                       load_llama, standardize, train_das_direction)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.probes import best_affine_mse, ols_weights, orient_toward_target, ridge_weights, unit

N_PAIRS = smoke(12, 200)
TRAIN_FRACTION = 0.7


def main():
    ensure_output_dirs()
    np.random.seed(0)
    torch.manual_seed(0)
    tokenizer, model = load_llama()
    digit_ids = digit_token_ids(tokenizer)
    names, years = load_birth_year_entities()
    decades = decades_of(years)
    target = standardize(years)
    print(f"{len(names)} people born 1900-1999, layer {PROBE_LAYER}", flush=True)

    activations, positions = cache_activations(model, tokenizer, names, PROBE_LAYER)
    mean = activations.mean(axis=0)
    X = activations - mean
    covariance = (X.T @ X) / len(X)
    n_train = int(TRAIN_FRACTION * len(names))
    train = np.arange(n_train)

    def oriented(direction):
        return orient_toward_target(unit(direction), X[train], target[train])

    u_ols = oriented(ols_weights(X[train], target[train]))
    print("training DAS ...", flush=True)
    u_das = oriented(train_das_direction(model, tokenizer, names, decades, activations, positions,
                                         PROBE_LAYER, digit_ids, seed=0))

    base_idx = np.random.choice(len(names), N_PAIRS)
    source_idx = np.random.choice(len(names), N_PAIRS)
    base_names = [names[i] for i in base_idx]
    base_positions = [positions[i] for i in base_idx]
    baseline = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER)
    intended = (decades[source_idx] - decades[base_idx]).astype(float)

    def readout_mse(weights):
        return float(np.mean((X @ weights - target) ** 2))

    def score(direction, probe_loss):
        delta = interchange_delta(activations, direction, base_idx, source_idx)
        shifted = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER, delta, base_positions)
        metrics = interchange_metrics(intended, shifted - baseline)
        metrics["probe_loss"] = probe_loss
        metrics["variance"] = float(direction @ covariance @ direction)
        metrics["cos_DAS"] = float(direction @ u_das)
        metrics["cos_OLS"] = float(direction @ u_ols)
        return metrics

    results = {"layer": PROBE_LAYER, "n": len(names), "n_pairs": N_PAIRS, "lambdas": [], "ridge": [],
               "OLS": score(u_ols, readout_mse(ols_weights(X, target))),
               "DAS": score(u_das, best_affine_mse(u_das, X, target, X, target))}
    for ridge_lambda in LAMBDA_GRID:
        direction = oriented(ridge_weights(X[train], target[train], ridge_lambda))
        metrics = score(direction, readout_mse(ridge_weights(X, target, ridge_lambda)))
        results["lambdas"].append(ridge_lambda)
        results["ridge"].append(metrics)
        print(f"  lambda={ridge_lambda:.0e}: probe_loss={metrics['probe_loss']:.3f} variance={metrics['variance']:.3f} "
              f"interv_mse={metrics['interv_mse']:.2f} cos(DAS)={metrics['cos_DAS']:+.3f}", flush=True)
    best = int(np.argmin([metrics["interv_mse"] for metrics in results["ridge"]]))
    results["selected_lambda"] = LAMBDA_GRID[best]
    print(f"intervention-optimal lambda = {results['selected_lambda']:.0e}")
    print(f"OLS: {results['OLS']}")
    print(f"DAS: {results['DAS']}")
    with open(RESULTS_DIR / "llm_sweep.json", "w") as handle:
        json.dump(results, handle, indent=1)
    np.savez(RESULTS_DIR / "llm_activations.npz", activations=activations, target=target, decades=decades,
             u_das=u_das, u_ols=u_ols, n_train=n_train)
    print("wrote results/llm_sweep.json and results/llm_activations.npz")


if __name__ == "__main__":
    main()
