"""Regression probes, steering maps, principal angles, and probe mass."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def fit_belief_probe(activations, beliefs, probabilities, ridge_lambda):
    """Probability-weighted regression probe from activations [n, d] to beliefs [n, 3].

    Rows of both activations and targets are scaled by sqrt(probability) and handed to
    scikit-learn with its default intercept.  (The intercept is therefore fit in the scaled
    space; this is the exact procedure behind the numbers in the paper.)
    Returns the weight matrix W [3, d] and intercept b [3]; the readout is ``activation @ W.T + b``.
    """
    sqrt_weights = np.sqrt(probabilities)[:, None]
    if ridge_lambda == 0:
        fit = LinearRegression().fit(activations * sqrt_weights, beliefs * sqrt_weights)
    else:
        fit = Ridge(alpha=ridge_lambda).fit(activations * sqrt_weights, beliefs * sqrt_weights)
    return fit.coef_, fit.intercept_


def steering_map(W):
    """Pseudoinverse map from a belief shift to an activation edit: delta_a = delta_belief @ steering_map."""
    return np.linalg.pinv(W.T)


def weighted_probe_loss(W, b, activations, beliefs, probabilities):
    """Probability-weighted squared error of the readout, summed over the three belief coordinates."""
    predictions = activations @ W.T + b
    squared_errors = ((beliefs - predictions) ** 2).sum(axis=1)
    return float((probabilities * squared_errors).sum() / probabilities.sum())


def mean_variance_along_rows(W, covariance):
    """Mean activation variance along the (normalized) rows of W."""
    variances = []
    for row in W:
        norm = np.linalg.norm(row)
        if norm > 1e-9:
            unit = row / norm
            variances.append(float(unit @ covariance @ unit))
    return float(np.mean(variances))


def orthonormal_basis(rows):
    """Orthonormal basis (columns) of the span of the given row vectors."""
    normalized = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    basis, _ = np.linalg.qr(normalized.T)
    return basis


def principal_angles_deg(rows_a, rows_b):
    """Principal angles (degrees, ascending) between the spans of two sets of row vectors."""
    cosines = np.linalg.svd(orthonormal_basis(rows_a).T @ orthonormal_basis(rows_b), compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.degrees(np.arccos(np.sort(cosines)[::-1]))


def probe_mass_per_component(W, principal_directions):
    """Share of the probe's squared weight on each principal direction (columns of ``principal_directions``)."""
    squared_weights = np.linalg.norm(W @ principal_directions, axis=0) ** 2
    return squared_weights / squared_weights.sum()


def principal_components(covariance):
    """Eigen-decomposition of a covariance matrix, sorted by decreasing variance."""
    variances, directions = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    return variances[order], directions[:, order]


# ---- single-direction probes (scalar target), used on Llama-2-7B ----

def ridge_weights(X, y, ridge_lambda):
    """Ridge solution on centered X and target y (no intercept)."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge_lambda * np.eye(d), X.T @ y)


def ols_weights(X, y):
    weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return weights


def unit(vector):
    return vector / np.linalg.norm(vector)


def orient_toward_target(direction, X, y):
    """Flip the sign so the projection correlates positively with the target."""
    if np.corrcoef(X @ direction, y)[0, 1] >= 0:
        return direction
    return -direction


def best_affine_mse(direction, X_fit, y_fit, X_eval, y_eval):
    """MSE of the best scale-and-bias readout along a direction (fit on one set, evaluated on another)."""
    slope, intercept = np.polyfit(X_fit @ direction, y_fit, 1)
    predictions = slope * (X_eval @ direction) + intercept
    return float(np.mean((predictions - y_eval) ** 2))
