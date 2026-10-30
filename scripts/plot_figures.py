"""Draw every figure that is assembled from saved results (Figures 1, 2, 5, 6, 7, 8, 9).

Figures 3 and 4 are drawn directly by scripts/hmm_simplex.py and scripts/hmm_heatmap.py.
Any figure whose inputs are missing is skipped with a message.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

from pll.paths import RESULTS_DIR
from pll.plotting import (COLOR, LABEL, METHOD_ORDER, expected_vs_actual_panel, lambda_axis, lambda_positions,
                          save_figure)


def load_json(name):
    with open(RESULTS_DIR / name) as handle:
        return json.load(handle)


def figure_1_intervention_effect():
    hmm = load_json("hmm_persample.json")["methods"]
    llm = load_json("llm_persample.json")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.0), facecolor="white")
    for column, method in enumerate(METHOD_ORDER):
        expected_vs_actual_panel(axes[0, column], hmm[method]["expected"], hmm[method]["actual"],
                                 COLOR[method], LABEL[method])
        expected_vs_actual_panel(axes[1, column], llm["intended"], llm["actual"][method],
                                 COLOR[method], LABEL[method])
        axes[0, column].set_xlabel("Expected $\\Delta P$(Emission)", fontsize=9.5)
        axes[1, column].set_xlabel("Intended Decade Shift", fontsize=9.5)
    axes[0, 0].set_ylabel("Mess3 HMM\n\nActual $\\Delta P$(Emission)", fontsize=11)
    axes[1, 0].set_ylabel("Llama-2-7B\n\nActual Output Decade Shift", fontsize=11)
    legend = [Line2D([0], [0], color=COLOR["fit_line"], ls=(0, (5, 3)), lw=2.0, label="Line of Best Fit"),
              Line2D([0], [0], color=COLOR["diagonal"], ls=":", lw=1.6, label="Perfect ($y = x$)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=10, columnspacing=2.0,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Actual vs. Expected Intervention Effect", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    save_figure(fig, "fig1_intervention_effect.png")


def sweep_panel(ax, lambdas, probe_loss, interv_mse, selected_lambda, title):
    ols_x = lambda_axis(ax, lambdas)
    x = lambda_positions(lambdas, ols_x)
    order = np.argsort(x)
    probe_loss = np.clip(np.array(probe_loss, dtype=float), 1e-6, None)
    ax.plot(x[order], probe_loss[order], "o-", color=COLOR["probe_loss"], lw=2.1, ms=4.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Probe Loss (MSE)", color=COLOR["probe_loss"], fontsize=10.5)
    ax.tick_params(axis="y", colors=COLOR["probe_loss"])
    twin = ax.twinx()
    twin.plot(x[order], np.array(interv_mse)[order], "s-", color=COLOR["OLS"], lw=2.1, ms=4.5, zorder=3)
    twin.set_ylabel("Intervention Loss (MSE)", color=COLOR["OLS"], fontsize=10.5)
    twin.tick_params(axis="y", colors=COLOR["OLS"])
    twin.set_ylim(bottom=0)
    ax.axvline(selected_lambda, ls=(0, (4, 3)), color=COLOR["fit_line"], lw=1.1, zorder=1)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=5)
    ax.grid(alpha=0.12, which="major", axis="x")
    ax.spines["top"].set_visible(False)
    twin.spines["top"].set_visible(False)


def figure_2_lambda_sweep():
    hmm = load_json("hmm_sweep.json")
    llm = load_json("llm_sweep.json")
    llm_lambdas = [0.0] + llm["lambdas"]
    llm_probe_loss = [llm["OLS"]["probe_loss"]] + [row["probe_loss"] for row in llm["ridge"]]
    llm_interv = [llm["OLS"]["interv_mse"]] + [row["interv_mse"] for row in llm["ridge"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor="white")
    sweep_panel(axes[0], hmm["lambdas"], hmm["probe_loss"], hmm["interv_mse"], hmm["selected_lambda"], "Mess3 HMM")
    sweep_panel(axes[1], llm_lambdas, llm_probe_loss, llm_interv, llm["selected_lambda"], "Llama-2-7B")
    legend = [Line2D([0], [0], color=COLOR["probe_loss"], marker="o", lw=2.1, label="Probe Loss (MSE)"),
              Line2D([0], [0], color=COLOR["OLS"], marker="s", lw=2.1, label="Intervention Loss (MSE)"),
              Line2D([0], [0], color=COLOR["fit_line"], ls=(0, (4, 3)), lw=1.1, label=r"Intervention-Selected $\lambda$")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(r"Probe Loss & Intervention Loss vs. Ridge Penalty $\lambda$", fontsize=13, fontweight="bold", y=0.965)
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    save_figure(fig, "fig2_lambda_sweep.png")


def bubble_frame(ax, operativity, mass_ols, mass_ridge, title):
    operativity = np.asarray(operativity, dtype=float)
    smallest_positive = operativity[operativity > 0].min()
    has_inert = (operativity <= 0).any()
    x = np.where(operativity <= 0, smallest_positive / 3.0, operativity)
    scale = 1700.0 / max(mass_ols.max(), mass_ridge.max())
    for y in (0, 1):
        ax.axhline(y, color="#ededed", lw=1.0, zorder=0)
    ax.scatter(x, np.ones_like(x), s=np.clip(mass_ols * scale, 4.0, None), c=COLOR["OLS"], edgecolors="white",
               lw=0.6, alpha=0.85, zorder=3, clip_on=False)
    ax.scatter(x, np.zeros_like(x), s=np.clip(mass_ridge * scale, 4.0, None), c=COLOR["ridge"], edgecolors="white",
               lw=0.6, alpha=0.85, zorder=3, clip_on=False)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Intervention-\nOptimal Ridge", "OLS"], fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlim(x.min() / (5.0 if has_inert else 1.5), operativity.max() * 1.9)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bcbcbc")
    ax.tick_params(left=False, labelsize=9)


def figure_5_probe_mass():
    hmm = np.load(RESULTS_DIR / "hmm_operativity.npz")
    llm = np.load(RESULTS_DIR / "llm_operativity.npz")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), facecolor="white")
    bubble_frame(axes[0], hmm["operativity"], hmm["mass_ols"], hmm["mass_ridge"], "Mess3 HMM")
    bubble_frame(axes[1], llm["operativity"], llm["mass_ols"], llm["mass_ridge"], "Llama-2-7B")
    axes[1].set_xlabel("Causal Operativity (Mean Output Change per Intervention)", fontsize=11)
    fig.suptitle("Probe Mass by Causal Operativity", fontsize=14, fontweight="bold", y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(left=0.16)
    save_figure(fig, "fig5_probe_mass.png", dpi=150)


def figure_6_layer_sweep():
    sweep = load_json("llm_layersweep.json")
    layers = np.array(sweep["layers"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor="white")
    ax.plot(layers, sweep["interv_mse"], "o-", color=COLOR["OLS"], lw=2.2, ms=5)
    ax.set_ylabel("Intervention Loss (MSE)", color=COLOR["OLS"], fontsize=11)
    ax.tick_params(axis="y", colors=COLOR["OLS"])
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylim(bottom=0)
    twin = ax.twinx()
    twin.plot(layers, sweep["probe_loss"], "s--", color=COLOR["probe_loss"], lw=2.2, ms=5)
    twin.set_ylabel("Probe Loss (MSE)", color=COLOR["probe_loss"], fontsize=11)
    twin.tick_params(axis="y", colors=COLOR["probe_loss"])
    twin.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    twin.spines["top"].set_visible(False)
    ax.grid(alpha=0.12, which="both", axis="x")
    ax.set_title("Intervention Loss & Probe Loss vs. Layer (Llama-2-7B)", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    save_figure(fig, "fig6_layer_sweep.png")


def angle_panel(ax, lambdas, angle_to_das, angle_to_ols, selected_lambda, title):
    ols_x = lambda_axis(ax, lambdas)
    x = lambda_positions(lambdas, ols_x)
    order = np.argsort(x)
    ax.plot(x[order], np.array(angle_to_das)[order], "o-", color=COLOR["ridge"], lw=2.1, ms=4.5, zorder=3)
    ax.plot(x[order], np.array(angle_to_ols)[order], "s-", color=COLOR["OLS"], lw=2.1, ms=4.5, zorder=3)
    ax.axvline(selected_lambda, ls=(0, (4, 3)), color=COLOR["fit_line"], lw=1.1, zorder=1)
    ax.set_ylim(-4, 94)
    ax.axhline(90, color="#cccccc", lw=0.7, zorder=0)
    ax.axhline(0, color="#cccccc", lw=0.7, zorder=0)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=5)
    ax.grid(alpha=0.12, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_7_principal_angle():
    hmm = load_json("hmm_das.json")
    curve = hmm["rotation_curve"]
    llm = load_json("llm_sweep.json")
    llm_lambdas = [0.0] + llm["lambdas"]
    cos_das = [llm["OLS"]["cos_DAS"]] + [row["cos_DAS"] for row in llm["ridge"]]
    cos_ols = [1.0] + [row["cos_OLS"] for row in llm["ridge"]]
    to_degrees = lambda cosines: np.degrees(np.arccos(np.clip(cosines, -1, 1)))  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), facecolor="white", sharey=True)
    angle_panel(axes[0], curve["lambdas"], curve["angle_to_DAS"], curve["angle_to_OLS"], hmm["selected_lambda"], "Mess3 HMM")
    angle_panel(axes[1], llm_lambdas, to_degrees(cos_das), to_degrees(cos_ols), llm["selected_lambda"], "Llama-2-7B")
    axes[0].set_ylabel(r"Principal Angle to Reference ($^\circ$)", fontsize=10.5)
    legend = [Line2D([0], [0], color=COLOR["ridge"], marker="o", lw=2.1, label="Angle to DAS"),
              Line2D([0], [0], color=COLOR["OLS"], marker="s", lw=2.1, label="Angle to OLS"),
              Line2D([0], [0], color=COLOR["fit_line"], ls=(0, (4, 3)), lw=1.1, label=r"Intervention-Selected $\lambda$")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9.5, handlelength=1.8,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle(r"Principal Angle to DAS & OLS Directions vs. Ridge Penalty $\lambda$", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    save_figure(fig, "fig7_principal_angle.png")


def figure_8_covariance_z():
    z = np.load(RESULTS_DIR / "llm_covariance_z.npy")
    middle = np.linspace(-4, 4, 49)
    bins = np.unique(np.concatenate([np.arange(-40, -4, 2), middle, np.arange(6, 42, 2)]))
    peak = int(np.histogram(z[np.abs(z) <= 4], bins=middle)[0].max())
    within_three = int((np.abs(z) <= 3).sum())
    fig, (linear_ax, log_ax) = plt.subplots(2, 1, figsize=(8.0, 5.4), facecolor="white", sharex=True)
    for ax in (linear_ax, log_ax):
        ax.axvspan(-3, 3, color="#000000", alpha=0.06, lw=0, zorder=0)
        ax.hist(z, bins=bins, color=COLOR["cloud"], edgecolor="white", linewidth=0.4, zorder=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9.5)
        ax.set_ylabel("Number of PCs", fontsize=11)
    linear_ax.set_ylim(0, peak * 1.12)
    linear_ax.text(0.5, 0.90, f"{within_three} of {len(z)} PCs within 3 standard errors of zero",
                   transform=linear_ax.transAxes, fontsize=9.5, ha="center", color="#555555")
    linear_ax.text(0.015, 0.90, "Linear Scale", transform=linear_ax.transAxes, fontsize=9, color="#999999", style="italic")
    log_ax.set_yscale("log")
    log_ax.set_ylim(0.7, peak * 2.4)
    log_ax.set_yticks([1, 10, 100])
    log_ax.set_yticklabels(["1", "10", "100"])
    log_ax.text(0.015, 0.90, "Log Scale", transform=log_ax.transAxes, fontsize=9, color="#999999", style="italic")
    log_ax.set_xscale("symlog", linthresh=4, linscale=1.6)
    log_ax.set_xlim(-45, 45)
    log_ax.xaxis.set_major_locator(FixedLocator([-40, -10, -4, -2, 0, 2, 4, 10, 40]))
    log_ax.xaxis.set_major_formatter(FixedFormatter(["-40", "-10", "-4", "-2", "0", "2", "4", "10", "40"]))
    log_ax.set_xlabel("Standardized Covariance $z$", fontsize=11)
    fig.suptitle("Birth-Year Covariance $z$ per Principal Component (Llama-2-7B, Layer 8)", fontsize=12.5,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig8_covariance_z.png", dpi=150)


def figure_9_restricted_das(hmm_mass_tag="999"):
    hmm = np.load(RESULTS_DIR / "hmm_restricted_das.npz")
    llm = np.load(RESULTS_DIR / "llm_restricted_das.npz")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), facecolor="white")
    expected_vs_actual_panel(axes[0], hmm[f"expected_{hmm_mass_tag}"], hmm[f"actual_{hmm_mass_tag}"],
                             COLOR["cloud"], "Mess3 HMM", point_size=24)
    expected_vs_actual_panel(axes[1], llm["intended"], llm["actual"], COLOR["cloud"], "Llama-2-7B", point_size=24)
    axes[0].set_xlabel("Expected $\\Delta P$(Emission)", fontsize=10)
    axes[0].set_ylabel("Actual $\\Delta P$(Emission)", fontsize=10)
    axes[1].set_xlabel("Intended Decade Shift", fontsize=10)
    axes[1].set_ylabel("Actual Output Decade Shift", fontsize=10)
    for ax in axes:
        ax.set_title(ax.get_title(), color="black")
    legend = [Line2D([0], [0], color=COLOR["fit_line"], ls=(0, (5, 3)), lw=1.6, label="Line of Best Fit"),
              Line2D([0], [0], color=COLOR["diagonal"], ls=":", lw=1.4, label="Perfect ($y = x$)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Actual vs. Expected Intervention Effect (Restricted DAS)", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    mass_fraction = float("0." + hmm_mass_tag)
    print(f"  HMM restricted DAS uses {int(hmm[f'n_components_{hmm_mass_tag}'])} PCs "
          f"({mass_fraction:.1%} of OLS mass)")
    save_figure(fig, "fig9_restricted_das.png", dpi=150)


FIGURES = {
    "1": figure_1_intervention_effect, "2": figure_2_lambda_sweep, "5": figure_5_probe_mass,
    "6": figure_6_layer_sweep, "7": figure_7_principal_angle, "8": figure_8_covariance_z,
    "9": figure_9_restricted_das,
}


def main(which):
    for key in which:
        try:
            FIGURES[key]()
        except FileNotFoundError as missing:
            print(f"figure {key}: skipped, missing input {missing.filename}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(FIGURES))
