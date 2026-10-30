"""Figure 3: belief-simplex reconstructions by the OLS probe, the intervention-optimal ridge probe,
and a belief readout confined to the DAS plane (from hmm_das.py), next to the ground truth.

120,000 sampled sequences; positions 2-8 are decoded and rasterized, colouring each pixel by the mean
true belief of the points that land in it.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from pll.mess3 import (MAX_LENGTH, PROBE_SITE, beliefs_along_sequences, load_testbed, sample_sequences,
                       simplex_to_2d)
from pll.paths import RESULTS_DIR, ensure_output_dirs, smoke
from pll.plotting import save_figure
from pll.probes import fit_belief_probe, weighted_probe_loss

SELECTED_LAMBDA = 0.1
N_SAMPLED_SEQUENCES = smoke(2000, 120000)
X_RANGE = (-0.18, 1.18)
Y_RANGE = (-0.12, 1.00)


def rasterize(points, colors, width=440, height=370):
    """Mean colour of the points falling in each pixel; white where empty."""
    x_bins = np.linspace(X_RANGE[0], X_RANGE[1], width + 1)
    y_bins = np.linspace(Y_RANGE[0], Y_RANGE[1], height + 1)
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[y_bins, x_bins])
    image = np.ones((height, width, 3))
    for channel in range(3):
        sums, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[y_bins, x_bins], weights=colors[:, channel])
        filled = counts > 0
        image[..., channel][filled] = sums[filled] / counts[filled]
    return np.clip(image, 0, 1)


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    W_ols, b_ols = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    W_ridge, b_ridge = fit_belief_probe(data.activations, data.beliefs, data.probabilities, SELECTED_LAMBDA)
    plane = np.load(RESULTS_DIR / "hmm_das_plane.npy")
    das_fit = LinearRegression().fit(data.activations @ plane.T, data.beliefs, sample_weight=data.probabilities)
    W_das = das_fit.coef_ @ plane
    b_das = das_fit.intercept_
    probes = {"Unregularized (OLS)": (W_ols, b_ols),
              rf"Intervention-Optimal Ridge, $\lambda={SELECTED_LAMBDA:g}$": (W_ridge, b_ridge),
              "DAS": (W_das, b_das)}

    sequences = sample_sequences(T, MAX_LENGTH, N_SAMPLED_SEQUENCES)
    true_beliefs = beliefs_along_sequences(sequences.numpy(), T)
    activations = []
    for start in range(0, N_SAMPLED_SEQUENCES, 256):
        with __import__("torch").no_grad():
            _, cache = model.run_with_cache(sequences[start:start + 256])
        activations.append(cache[PROBE_SITE].numpy())
    activations = np.concatenate(activations)
    positions = range(1, MAX_LENGTH)  # skip the first token, whose belief barely moves
    flat_activations = np.concatenate([activations[:, p, :] for p in positions])
    flat_beliefs = np.concatenate([true_beliefs[:, p, :] for p in positions])
    colors = np.clip(flat_beliefs, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 4.2), facecolor="white")
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    panels = [("Ground Truth", simplex_to_2d(flat_beliefs))]
    for title, (W, b) in probes.items():
        loss = weighted_probe_loss(W, b, data.activations, data.beliefs, data.probabilities)
        panels.append((f"{title}\n(Probe Loss $=$ {loss:.2e})", simplex_to_2d(flat_activations @ W.T + b)))
        print(f"  {title.splitlines()[0]}: probe loss {loss:.2e}")
    for ax, (title, points) in zip(axes, panels):
        ax.imshow(rasterize(points, colors), extent=[*X_RANGE, *Y_RANGE], origin="lower", interpolation="nearest")
        ax.plot(outline[:, 0], outline[:, 1], color="#333333", lw=1.0)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_aspect("equal")
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)
        ax.axis("off")
    fig.suptitle("Belief Simplex Reconstruction (Mess3 HMM)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig3_simplex.png")


if __name__ == "__main__":
    main()
