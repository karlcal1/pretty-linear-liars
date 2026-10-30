"""Figure 4: intervention loss across the belief simplex for OLS vs. intervention-optimal ridge.

From each of three start sequences (one near each corner), steer toward every point of a dense grid
and plot the root-squared error between steered and Bayes-optimal next-token distributions.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from pll.mess3 import (corner_start_sequences, load_testbed, next_token_distribution, simplex_grid,
                       simplex_to_2d, steered_output_distribution)
from pll.paths import ensure_output_dirs, smoke
from pll.plotting import save_figure
from pll.probes import fit_belief_probe, steering_map

SELECTED_LAMBDA = 0.1
GRID_RESOLUTION = smoke(6, 34)


def error_field(model, T, steering, start_tokens, start_belief, grid):
    errors = []
    for target in grid:
        delta = (target - start_belief) @ steering
        steered = steered_output_distribution(model, start_tokens, delta)[0]
        expected = next_token_distribution(target[None], T)[0]
        errors.append(np.sqrt(((expected - steered) ** 2).sum()))
    return np.array(errors)


def main():
    ensure_output_dirs()
    model, T, data = load_testbed(seed=0)
    W_ols, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, 0.0)
    W_ridge, _ = fit_belief_probe(data.activations, data.beliefs, data.probabilities, SELECTED_LAMBDA)
    steering = {"Unregularized (OLS)": steering_map(W_ols),
                "Intervention-Optimal\nRidge": steering_map(W_ridge)}
    start_tokens, start_beliefs = corner_start_sequences(T)
    grid = simplex_grid(GRID_RESOLUTION)
    xy = simplex_to_2d(grid)
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])

    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.0), facecolor="white")
    for row, (label, steering_matrix) in enumerate(steering.items()):
        for start in range(3):
            ax = axes[row, start]
            errors = error_field(model, T, steering_matrix, start_tokens[start:start + 1],
                                 start_beliefs[start], grid)
            contour = ax.tricontourf(triangulation, errors, levels=np.linspace(0, 0.5, 26),
                                     cmap="RdYlBu_r", extend="max")
            ax.plot(outline[:, 0], outline[:, 1], color="#333333", lw=1.0)
            start_xy = simplex_to_2d(start_beliefs[start][None])[0]
            ax.plot(start_xy[0], start_xy[1], marker="*", ms=11, color="white", mec="#222222", mew=0.8)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Start Near $s_{start}$", fontsize=11.5, pad=1)
        axes[row, 0].text(-0.10, 0.5, label, rotation=90, va="center", ha="center",
                          transform=axes[row, 0].transAxes, fontsize=11.5, fontweight="bold")
    fig.suptitle("Intervention Loss Across the Belief Simplex (Mess3 HMM)", fontsize=13, fontweight="bold", y=0.97)
    fig.subplots_adjust(hspace=-0.05, wspace=0.02, left=0.05, right=0.85, top=0.90, bottom=0.01)
    colorbar_axis = fig.add_axes([0.875, 0.14, 0.022, 0.66])
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label("Intervention Loss", fontsize=11)
    save_figure(fig, "fig4_heatmap.png")


if __name__ == "__main__":
    main()
