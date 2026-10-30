"""Mess3 HMM: expected vs. actual change in next-token probabilities under steering (Figure 1, top).

20 random start sequences are steered toward each simplex corner at 20 strengths with the OLS,
intervention-optimal ridge, and DAS steering maps (all three use the same pseudoinverse protocol).
Each steering step gives three points, one per emission probability.
"""
import json

import numpy as np

from pll.mess3 import (MAX_LENGTH, load_testbed, next_token_distribution, output_distribution,
                       sample_sequences, beliefs_along_sequences, steered_output_distribution)
from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import fit_belief_probe, steering_map

SELECTED_LAMBDA = 0.1
N_STARTS = 20
STRENGTHS = np.linspace(0, 0.5, 21)


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    W_ols, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    W_ridge, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, SELECTED_LAMBDA)
    steering = {
        "OLS": steering_map(W_ols),
        "ridge": steering_map(W_ridge),
        "DAS": np.load(RESULTS_DIR / "hmm_das_steering.npy"),
    }
    starts = sample_sequences(T, MAX_LENGTH, N_STARTS)
    start_beliefs = beliefs_along_sequences(starts.numpy(), T)[:, -1]
    corners = np.eye(3)

    results = {"selected_lambda": SELECTED_LAMBDA, "methods": {}}
    for name, steering_matrix in steering.items():
        expected, actual = [], []
        for i in range(N_STARTS):
            tokens = starts[i:i + 1]
            belief = start_beliefs[i]
            baseline_output = output_distribution(model, tokens)[0]
            baseline_expected = next_token_distribution(belief[None], T)[0]
            for corner in corners:
                for strength in STRENGTHS:
                    target = (1 - strength) * belief + strength * corner
                    delta = (target - belief) @ steering_matrix
                    steered = steered_output_distribution(model, tokens, delta)[0]
                    expected.extend(next_token_distribution(target[None], T)[0] - baseline_expected)
                    actual.extend(steered - baseline_output)
        expected = np.array(expected)
        actual = np.array(actual)
        slope = float(np.polyfit(expected, actual, 1)[0])
        results["methods"][name] = {"expected": expected.tolist(), "actual": actual.tolist(), "slope": slope}
        print(f"  {name:5s}: slope={slope:+.3f} ({len(expected)} points)", flush=True)
    with open(RESULTS_DIR / "hmm_persample.json", "w") as handle:
        json.dump(results, handle)
    print("wrote results/hmm_persample.json")


if __name__ == "__main__":
    main()
