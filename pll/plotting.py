"""Shared plotting style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pll.paths import FIGURES_DIR  # noqa: E402

COLOR = {
    "OLS": "#D64545",
    "ridge": "#2E8B57",
    "DAS": "#8E44AD",
    "probe_loss": "#3B6EA5",
    "cloud": "#9db8d2",
    "fit_line": "#666666",
    "diagonal": "#bbbbbb",
}
LABEL = {"OLS": "OLS", "ridge": "Intervention-Optimal Ridge", "DAS": "DAS"}
METHOD_ORDER = ["OLS", "ridge", "DAS"]


def save_figure(fig, filename, dpi=200):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {path}")


def lambda_axis(ax, lambdas):
    """Log x-axis over ridge penalties with an extra 'OLS' tick (lambda = 0) at the left.

    Returns the x position used for lambda = 0.
    """
    positive = [value for value in lambdas if value > 0]
    x_min = min(positive) / 3.0
    x_max = max(positive) * 3.0
    ols_x = x_min * 1.5
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    first_decade = int(np.ceil(np.log10(ols_x * 3)))
    last_decade = int(np.floor(np.log10(x_max)))
    decades = list(range(first_decade, last_decade + 1))
    ax.set_xticks([ols_x] + [10.0 ** exponent for exponent in decades])
    labels = ["OLS"]
    for exponent in decades:
        labels.append(rf"$10^{{{exponent}}}$" if exponent % 2 == 0 else "")
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel(r"Ridge Penalty $\lambda$", fontsize=10.5)
    return ols_x


def lambda_positions(lambdas, ols_x):
    """x positions for a list of lambdas, mapping lambda = 0 onto the OLS tick."""
    values = np.array(lambdas, dtype=float)
    return np.where(values == 0, ols_x, values)


def expected_vs_actual_panel(ax, expected, actual, color, title, point_size=7):
    """Scatter of actual against expected intervention effect, with y = x and a best-fit line."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    limit = max(np.abs(np.r_[expected, actual]).max(), 1e-9) * 1.08
    ax.plot([-limit, limit], [-limit, limit], ls=":", color=COLOR["diagonal"], lw=1.4, zorder=1)
    ax.axhline(0, color="#dddddd", lw=0.7, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.7, zorder=0)
    ax.scatter(expected, actual, s=point_size, alpha=0.28, color=color, edgecolors="none", zorder=2)
    slope, intercept = np.polyfit(expected, actual, 1)
    ends = np.array([-limit, limit])
    ax.plot(ends, slope * ends + intercept, ls=(0, (5, 3)), color=COLOR["fit_line"], lw=2.0, zorder=3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=(10.5 if len(title) > 6 else 12.5), color=color, fontweight="bold", pad=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.10)
