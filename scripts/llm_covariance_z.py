"""Llama-2-7B birth year: standardized covariance of every principal component with the target (Figure 8).

Split-half estimate: a random half of the prompts (seeded permutation) fixes the principal directions;
the other half supplies each direction's standard deviation and covariance with the (standardized) birth year.
z_i = sqrt(n) cov_i / (sigma_i sigma_f) is the covariance in units of its sampling standard error.
Model-free: reads the activations cached by llm_sweep.py.
"""
import numpy as np

from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import principal_components


def main():
    ensure_output_dirs()
    cached = np.load(RESULTS_DIR / "llm_activations.npz")
    activations = cached["activations"].astype(np.float64)
    target = cached["target"].astype(np.float64)
    permutation = np.random.default_rng(0).permutation(len(activations))
    half = len(activations) // 2
    first = activations[permutation[:half]]
    second = activations[permutation[half:]]
    target_second_half = target[permutation[half:]]

    centered_first = first - first.mean(axis=0)
    _, directions = principal_components((centered_first.T @ centered_first) / len(first))

    projections = (second - second.mean(axis=0)) @ directions
    target_second = target_second_half - target_second_half.mean()
    n = len(second)
    covariance = (projections * target_second[:, None]).mean(axis=0)
    sigma = projections.std(axis=0)
    z = np.sqrt(n) * covariance / (sigma * target_second.std())
    np.save(RESULTS_DIR / "llm_covariance_z.npy", z)
    print(f"{int((np.abs(z) <= 3).sum())} of {len(z)} PCs within 3 standard errors of zero")
    print("wrote results/llm_covariance_z.npy")


if __name__ == "__main__":
    main()
