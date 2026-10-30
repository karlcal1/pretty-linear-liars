"""Llama-2-7B birth year: probe mass and causal operativity per principal component (Figure 5, bottom).

Operativity of a PC = mean |change in expected decade| when its coordinate is swapped from a random
source prompt into a random base prompt.  Measured for the 100 PCs carrying the most combined
OLS + ridge probe mass plus 20 random low-mass PCs ("dust"); 150 swaps each.
"""
import time

import numpy as np
import torch

from pll.llama import (PROBE_LAYER, RIDGE_LAMBDA, digit_token_ids, expected_decade, load_birth_year_entities,
                       load_llama, name_final_indices)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.probes import principal_components, ridge_weights, unit

N_TOP = smoke(4, 100)
N_DUST = smoke(2, 20)
N_SWAPS = smoke(8, 150)
POOL = smoke(16, 300)


def main():
    ensure_output_dirs()
    start = time.time()
    cached = np.load(RESULTS_DIR / "llm_activations.npz")
    activations = cached["activations"].astype(np.float64)
    target = cached["target"]
    X = activations - activations.mean(axis=0)
    variances, directions = principal_components((X.T @ X) / len(X))
    u_ols = unit(cached["u_ols"].astype(np.float64))
    u_ridge = unit(ridge_weights(X, target - target.mean(), RIDGE_LAMBDA))  # in-sample, all prompts
    mass_ols = (directions.T @ u_ols) ** 2
    mass_ridge = (directions.T @ u_ridge) ** 2
    mass_ols /= mass_ols.sum()
    mass_ridge /= mass_ridge.sum()

    rng = np.random.default_rng(0)
    top = np.argsort(mass_ols + mass_ridge)[::-1][:N_TOP]
    rest = np.setdiff1d(np.arange(len(variances)), top)
    dust = rng.choice(rest, N_DUST, replace=False)
    components = np.concatenate([top, dust])

    tokenizer, model = load_llama()
    digit_ids = digit_token_ids(tokenizer)
    names, _ = load_birth_year_entities()
    positions = name_final_indices(tokenizer, names)
    swap_rng = np.random.default_rng(1)
    base_pool = swap_rng.integers(0, len(names), POOL)
    pool_baseline = expected_decade(model, tokenizer, [names[i] for i in base_pool], digit_ids, PROBE_LAYER)
    baseline_of = {int(index): value for index, value in zip(base_pool, pool_baseline)}

    operativity = np.zeros(len(components))
    for j, component in enumerate(components):
        direction = directions[:, component]
        projections = activations @ direction
        base_idx = swap_rng.choice(base_pool, N_SWAPS)
        source_idx = swap_rng.integers(0, len(names), N_SWAPS)
        delta = (projections[source_idx] - projections[base_idx])[:, None] * direction[None, :]
        shifted = expected_decade(model, tokenizer, [names[i] for i in base_idx], digit_ids, PROBE_LAYER,
                                  delta, [positions[i] for i in base_idx])
        baseline = np.array([baseline_of[int(i)] for i in base_idx])
        operativity[j] = np.mean(np.abs(shifted - baseline))
        if j % 20 == 0:
            print(f"  {j}/{len(components)} components, {time.time() - start:.0f}s", flush=True)
    np.savez(RESULTS_DIR / "llm_operativity.npz", components=components, operativity=operativity,
             mass_ols=mass_ols[components], mass_ridge=mass_ridge[components], variance=variances[components],
             mass_ols_all=mass_ols, mass_ridge_all=mass_ridge)
    print(f"shown PCs hold {mass_ridge[components].sum():.1%} of ridge mass and {mass_ols[components].sum():.1%} of OLS mass")
    print("wrote results/llm_operativity.npz")


if __name__ == "__main__":
    main()
