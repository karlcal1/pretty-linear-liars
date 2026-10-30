"""Mess3 HMM: Distributed Alignment Search, and how ridge probes rotate onto it.

DAS learns an orthogonal rotation of activation space and interchanges two rotated coordinates
(the two degrees of freedom of the belief simplex) between a base and a source sequence, trained to
minimize the squared error against the Bayes-optimal next-token distribution of the source belief.
The unembedding is linear from the probe site, so the forward pass is ``(a + delta) @ W_U + b_U``.

Outputs (Table 1 DAS row, Figure 7 left, and the DAS steering map used by Figures 1 and 3):
  results/hmm_das.json, results/hmm_das_plane.npy, results/hmm_das_steering.npy
"""
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from pll.mess3 import (corner_start_sequences, grid_intervention_loss, load_testbed,
                       next_token_distribution, simplex_grid)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.probes import fit_belief_probe, principal_angles_deg, steering_map

LAMBDAS = smoke([0.0, 0.1], [0.0] + [10.0 ** exponent for exponent in range(-7, 6)])
SELECTED_LAMBDA = 0.1
N_SEEDS = smoke(1, 8)
STEPS = smoke(30, 4000)
BATCH = 1024
LEARNING_RATE = 1e-2
SUBSPACE_DIM = 2


def train_rotation(model, data, T, seed):
    """One DAS run. Returns the orthogonal matrix Q [64, 64] and the final training loss."""
    torch.manual_seed(seed)
    rotation = nn.utils.parametrizations.orthogonal(nn.Linear(64, 64, bias=False))
    optimizer = torch.optim.Adam(rotation.parameters(), lr=LEARNING_RATE)
    W_U = model.W_U.detach().float()
    b_U = model.b_U.detach().float()
    activations = torch.tensor(data.activations, dtype=torch.float32)
    bayes_targets = torch.tensor(next_token_distribution(data.beliefs, T), dtype=torch.float32)
    sampling_probs = data.probabilities / data.probabilities.sum()
    mask = torch.zeros(64)
    mask[:SUBSPACE_DIM] = 1.0
    n = len(data.sequences)
    loss_value = None
    for _ in range(STEPS):
        base_idx = np.random.choice(n, BATCH, p=sampling_probs)
        source_idx = np.random.choice(n, BATCH, p=sampling_probs)
        base = activations[base_idx]
        source = activations[source_idx]
        Q = rotation.weight
        # move only the first SUBSPACE_DIM rotated coordinates from source into base
        delta = (((source - base) @ Q.T) * mask) @ Q
        predicted = torch.softmax((base + delta) @ W_U + b_U, dim=-1)
        loss = ((bayes_targets[source_idx] - predicted) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
    return rotation.weight.detach().numpy(), loss_value


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    covariance = data.weighted_covariance()
    sqrt_weights = np.sqrt(data.probabilities)[:, None]

    best_Q, best_loss, best_seed = None, None, None
    for seed in range(N_SEEDS):
        Q, loss_value = train_rotation(model, data, T, seed)
        print(f"  DAS seed {seed}: final train loss {loss_value:.2e}", flush=True)
        if best_loss is None or loss_value < best_loss:
            best_Q, best_loss, best_seed = Q, loss_value, seed
    plane = best_Q[:SUBSPACE_DIM]  # [2, 64], orthonormal rows spanning the DAS plane
    print(f"best seed {best_seed} (loss {best_loss:.2e})")

    # Belief readout confined to the DAS plane (weighted least squares), and its pinv steering map.
    coordinates = data.activations @ plane.T
    readout_in_plane = LinearRegression().fit(coordinates * sqrt_weights, data.beliefs * sqrt_weights)
    W_das = readout_in_plane.coef_ @ plane  # [3, 64]
    das_steering = steering_map(W_das)
    np.save(RESULTS_DIR / "hmm_das_plane.npy", plane)
    np.save(RESULTS_DIR / "hmm_das_steering.npy", das_steering)

    # Probe loss of the plane (weighted fit with intercept) and variance along its two directions.
    decoder = LinearRegression().fit(coordinates, data.beliefs, sample_weight=data.probabilities)
    predictions = decoder.predict(coordinates)
    probe_loss = float((data.probabilities[:, None] * (data.beliefs - predictions) ** 2).sum()
                       / data.probabilities.sum())
    variance = float(np.mean([row @ covariance @ row for row in plane]))

    # Intervention loss under the same grid protocol as the ridge sweep.
    start_tokens, start_beliefs = corner_start_sequences(T)
    interv_mse, _ = grid_intervention_loss(model, T, das_steering, start_tokens, start_beliefs,
                                           simplex_grid(smoke(6, 30)))

    # Principal angles between the ridge/OLS steering subspaces and the DAS plane, across lambda.
    W_ols, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    ols_steering = steering_map(W_ols)
    rotation_curve = {"lambdas": [], "angle_to_DAS": [], "angle_to_OLS": []}
    for ridge_lambda in LAMBDAS:
        W, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, ridge_lambda)
        steering = steering_map(W)
        rotation_curve["lambdas"].append(ridge_lambda)
        rotation_curve["angle_to_DAS"].append(float(principal_angles_deg(steering, plane).max()))
        rotation_curve["angle_to_OLS"].append(float(principal_angles_deg(steering, ols_steering).max()))
    selected = LAMBDAS.index(SELECTED_LAMBDA)

    results = {
        "best_seed": best_seed, "train_loss": best_loss,
        "probe_loss": probe_loss, "variance": variance, "interv_mse": interv_mse,
        "angle_ridge_to_DAS": rotation_curve["angle_to_DAS"][selected],
        "angle_OLS_to_DAS": rotation_curve["angle_to_DAS"][0],
        "selected_lambda": SELECTED_LAMBDA, "rotation_curve": rotation_curve,
    }
    print(f"DAS: probe_loss={probe_loss:.3e} variance={variance:.3f} interv_mse={interv_mse:.3e} | "
          f"angle ridge({SELECTED_LAMBDA})->DAS={results['angle_ridge_to_DAS']:.1f} deg, "
          f"OLS->DAS={results['angle_OLS_to_DAS']:.1f} deg")
    with open(RESULTS_DIR / "hmm_das.json", "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote results/hmm_das.json")


if __name__ == "__main__":
    main()
