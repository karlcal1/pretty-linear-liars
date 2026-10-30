"""Mess3 HMM: probe mass and causal operativity of each principal component (Figure 5, top).

Operativity of a direction = mean L1 change of the next-token distribution when the direction's
coordinate is swapped from a random source sequence into a random base sequence (natural magnitudes).
"""
import numpy as np
from scipy.stats import spearmanr

from pll.mess3 import (MAX_LENGTH, last_token_activations, load_testbed, output_distribution,
                       sample_sequences, steered_output_distribution)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.probes import fit_belief_probe, principal_components, probe_mass_per_component

SELECTED_LAMBDA = 0.1
POOL_SIZE = smoke(50, 600)
N_PAIRS = smoke(50, 1000)


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    variances, directions = principal_components(data.weighted_covariance())
    W_ols, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    W_ridge, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, SELECTED_LAMBDA)
    mass_ols = probe_mass_per_component(W_ols, directions)
    mass_ridge = probe_mass_per_component(W_ridge, directions)

    pool = sample_sequences(T, MAX_LENGTH, POOL_SIZE)
    baseline = output_distribution(model, pool)
    pool_activations = last_token_activations(model, pool)
    operativity = np.zeros(len(variances))
    for component in range(len(variances)):
        direction = directions[:, component]
        projections = pool_activations @ direction
        base_idx = np.random.randint(0, POOL_SIZE, N_PAIRS)
        source_idx = np.random.randint(0, POOL_SIZE, N_PAIRS)
        swap = projections[source_idx] - projections[base_idx]
        delta = swap[:, None] * direction[None, :]
        steered = steered_output_distribution(model, pool[base_idx], delta)
        operativity[component] = np.mean(np.abs(steered - baseline[base_idx]).sum(axis=1))

    np.savez(RESULTS_DIR / "hmm_operativity.npz", operativity=operativity, mass_ols=mass_ols,
             mass_ridge=mass_ridge, variance=variances)
    heaviest = int(np.argmax(mass_ols))
    print(f"OLS: {mass_ols[heaviest]:.3f} of mass on one PC with variance {variances[heaviest]:.1e}")
    print(f"spearman(operativity, variance)={spearmanr(operativity, variances).correlation:+.2f}  "
          f"spearman(mass_ols, op)={spearmanr(mass_ols, operativity).correlation:+.2f}  "
          f"spearman(mass_ridge, op)={spearmanr(mass_ridge, operativity).correlation:+.2f}")
    print("wrote results/hmm_operativity.npz")


if __name__ == "__main__":
    main()
