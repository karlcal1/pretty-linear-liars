"""Mess3 HMM: probe loss, variance and intervention loss of the belief probe across ridge penalties.

Feeds Figure 2 (left) and the OLS / ridge rows of Table 1.  Intervention protocol: from three start
sequences (one near each simplex corner) steer toward every point of a uniform grid over the simplex.
"""
import json

import numpy as np

from pll.mess3 import corner_start_sequences, grid_intervention_loss, load_testbed, simplex_grid
from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import fit_belief_probe, mean_variance_along_rows, steering_map, weighted_probe_loss

LAMBDAS = [0.0] + [10.0 ** exponent for exponent in range(-7, 6)]
GRID_RESOLUTION = 30


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    covariance = data.weighted_covariance()
    start_tokens, start_beliefs = corner_start_sequences(T)
    grid = simplex_grid(GRID_RESOLUTION)
    print(f"{len(data.sequences)} prefixes, {len(grid)} grid targets x {len(start_beliefs)} starts", flush=True)

    results = {"lambdas": [], "probe_loss": [], "variance": [], "interv_mse": [], "interv_rmse": []}
    for ridge_lambda in LAMBDAS:
        W, b = fit_belief_probe(data.activations, data.beliefs, data.probabilities, ridge_lambda)
        probe_loss = weighted_probe_loss(W, b, data.activations, data.beliefs, data.probabilities)
        variance = mean_variance_along_rows(W, covariance)
        interv_mse, interv_rmse = grid_intervention_loss(
            model, T, steering_map(W), start_tokens, start_beliefs, grid)
        results["lambdas"].append(ridge_lambda)
        results["probe_loss"].append(probe_loss)
        results["variance"].append(variance)
        results["interv_mse"].append(interv_mse)
        results["interv_rmse"].append(interv_rmse)
        print(f"  lambda={ridge_lambda:>8.1e}  probe_loss={probe_loss:.3e}  variance={variance:.3e}  "
              f"interv_mse={interv_mse:.3e}", flush=True)

    best = int(np.argmin(results["interv_mse"]))
    results["selected_lambda"] = results["lambdas"][best]
    print(f"intervention-optimal lambda = {results['selected_lambda']}")
    with open(RESULTS_DIR / "hmm_sweep.json", "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote results/hmm_sweep.json")


if __name__ == "__main__":
    main()
