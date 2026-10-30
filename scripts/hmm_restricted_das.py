"""Mess3 HMM: DAS confined to the principal components that carry the OLS probe's mass (Figure 9, left).

A 2-D steering map is trained inside the span of the heaviest-weight OLS components (enough of them
to hold 99% / 99.9% of the OLS mass) and evaluated with the trajectory protocol of hmm_persample.py.
If a combination of those components were causally operative, this search would find it.
"""
import numpy as np
import torch
import torch.nn as nn

from pll.mess3 import (MAX_LENGTH, beliefs_along_sequences, load_testbed, next_token_distribution,
                       output_distribution, sample_sequences, steered_output_distribution, PROBE_SITE)
from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import fit_belief_probe, principal_components, probe_mass_per_component

N_STARTS = 20
STRENGTHS = np.linspace(0, 0.5, 21)
STEPS = 600
MASS_FRACTIONS = (0.99, 0.999)

# orthonormal basis of the simplex tangent plane (belief shifts sum to zero)
TANGENT_BASIS = np.linalg.svd(np.eye(3) - np.ones((3, 3)) / 3)[0][:, :2]


def train_restricted_steering(model, T, subspace, starts, start_beliefs):
    """Learn a [64, 2] steering map M inside ``subspace`` [64, k]: delta = (target - start) @ TANGENT_BASIS @ M.T."""
    subspace_t = torch.tensor(subspace, dtype=torch.float32)
    coefficients = nn.Parameter(torch.randn(subspace.shape[1], 2) * 0.1)
    optimizer = torch.optim.Adam([coefficients], lr=2e-2)
    emission_given_state = torch.tensor(T.sum(axis=2), dtype=torch.float32)  # [k, s]
    start_beliefs_t = torch.tensor(start_beliefs, dtype=torch.float32)
    tangent_t = torch.tensor(TANGENT_BASIS, dtype=torch.float32)
    corners = torch.eye(3)
    for _ in range(STEPS):
        corner = np.random.randint(0, 3)
        strength = np.random.uniform(0.05, 0.5)
        targets = (1 - strength) * start_beliefs_t + strength * corners[corner]
        tangent_coords = (targets - start_beliefs_t) @ tangent_t
        delta = tangent_coords @ (subspace_t @ coefficients).T

        def add_delta(activation, hook, delta=delta):
            activation[:, -1, :] = activation[:, -1, :] + delta
            return activation

        logits = model.run_with_hooks(starts, fwd_hooks=[(PROBE_SITE, add_delta)])
        steered = torch.softmax(logits[:, -1, :], dim=-1)
        expected = targets @ emission_given_state.T
        loss = ((steered - expected) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (subspace_t @ coefficients).detach().numpy()


def evaluate_trajectories(model, T, steering_2d, starts, start_beliefs):
    """Per-sample expected vs. actual change in emission probabilities, as in hmm_persample.py."""
    expected, actual, squared_errors = [], [], []
    for i in range(N_STARTS):
        tokens = starts[i:i + 1]
        belief = start_beliefs[i]
        baseline_output = output_distribution(model, tokens)[0]
        baseline_expected = next_token_distribution(belief[None], T)[0]
        for corner in np.eye(3):
            for strength in STRENGTHS:
                target = (1 - strength) * belief + strength * corner
                delta = ((target - belief) @ TANGENT_BASIS) @ steering_2d.T
                steered = steered_output_distribution(model, tokens, delta)[0]
                target_output = next_token_distribution(target[None], T)[0]
                expected.extend(target_output - baseline_expected)
                actual.extend(steered - baseline_output)
                squared_errors.append(((target_output - steered) ** 2).sum())
    expected = np.array(expected)
    actual = np.array(actual)
    slope = float(np.polyfit(expected, actual, 1)[0])
    return expected, actual, slope, float(np.mean(squared_errors))


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    variances, directions = principal_components(data.weighted_covariance())
    W_ols, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    mass = probe_mass_per_component(W_ols, directions)
    by_mass = np.argsort(mass)[::-1]
    cumulative = np.cumsum(mass[by_mass])

    starts = sample_sequences(T, MAX_LENGTH, N_STARTS)
    start_beliefs = beliefs_along_sequences(starts.numpy(), T)[:, -1]
    saved = {}
    for fraction in MASS_FRACTIONS:
        n_components = int(np.searchsorted(cumulative, fraction) + 1)
        chosen = by_mass[:n_components]
        variance_share = float(variances[chosen].sum() / variances.sum())
        steering_2d = train_restricted_steering(model, T, directions[:, chosen], starts, start_beliefs)
        expected, actual, slope, interv_mse = evaluate_trajectories(model, T, steering_2d, starts, start_beliefs)
        tag = str(fraction).replace("0.", "")
        saved[f"expected_{tag}"] = expected
        saved[f"actual_{tag}"] = actual
        saved[f"slope_{tag}"] = slope
        saved[f"interv_mse_{tag}"] = interv_mse
        saved[f"n_components_{tag}"] = n_components
        saved[f"variance_share_{tag}"] = variance_share
        print(f"  {fraction:.1%} of OLS mass: {n_components} PCs holding {variance_share:.1e} of variance | "
              f"restricted-DAS slope={slope:+.4f} interv_mse={interv_mse:.4f}", flush=True)
    np.savez(RESULTS_DIR / "hmm_restricted_das.npz", **saved)
    print("wrote results/hmm_restricted_das.npz")


if __name__ == "__main__":
    main()
