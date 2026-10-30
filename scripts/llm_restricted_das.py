"""Llama-2-7B birth year: DAS confined to the 100 heaviest-weight OLS principal components (Figure 9, right).

If any combination of the directions the OLS probe leans on were causally operative, a DAS search
restricted to their span would find it.  Evaluated on 150 base/source interchanges.
"""
import numpy as np
import torch

from pll.llama import (PROBE_LAYER, decades_of, digit_token_ids, expected_decade, interchange_delta,
                       interchange_metrics, load_birth_year_entities, load_llama, name_final_indices,
                       DEVICE, train_das_direction)
from pll.paths import RESULTS_DIR, ensure_output_dirs
from pll.probes import principal_components, unit

N_COMPONENTS = 100
N_PAIRS = 150
STEPS = 120


def main():
    ensure_output_dirs()
    np.random.seed(0)
    torch.manual_seed(0)
    cached = np.load(RESULTS_DIR / "llm_activations.npz")
    activations = cached["activations"]
    X = activations.astype(np.float64) - activations.astype(np.float64).mean(axis=0)
    variances, directions = principal_components((X.T @ X) / len(X))
    mass_ols = (directions.T @ unit(cached["u_ols"].astype(np.float64))) ** 2
    mass_ols /= mass_ols.sum()
    heaviest = np.argsort(mass_ols)[::-1][:N_COMPONENTS]
    subspace = directions[:, heaviest].astype(np.float32)
    variance_share = float(variances[heaviest].sum() / variances.sum())
    print(f"{N_COMPONENTS} PCs hold {mass_ols[heaviest].sum():.1%} of OLS mass, {variance_share:.2e} of variance")

    names, years = load_birth_year_entities()
    decades = decades_of(years)
    tokenizer, model = load_llama()
    digit_ids = digit_token_ids(tokenizer)
    positions = name_final_indices(tokenizer, names)

    projector = torch.tensor(subspace, device=DEVICE)
    direction = train_das_direction(model, tokenizer, names, decades, activations, positions, PROBE_LAYER,
                                    digit_ids, seed=0, steps=STEPS, projector=projector)

    rng = np.random.default_rng(0)
    base_idx = rng.integers(0, len(names), N_PAIRS)
    source_idx = rng.integers(0, len(names), N_PAIRS)
    base_names = [names[i] for i in base_idx]
    base_positions = [positions[i] for i in base_idx]
    baseline = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER)
    delta = interchange_delta(activations, direction, base_idx, source_idx)
    shifted = expected_decade(model, tokenizer, base_names, digit_ids, PROBE_LAYER, delta, base_positions)
    intended = (decades[source_idx] - decades[base_idx]).astype(float)
    metrics = interchange_metrics(intended, shifted - baseline)
    no_intervention_mse = float(np.mean((baseline - decades[source_idx]) ** 2))
    print(f"restricted DAS: {metrics} | no-intervention MSE vs source decade = {no_intervention_mse:.2f}")
    np.savez(RESULTS_DIR / "llm_restricted_das.npz", intended=intended, actual=shifted - baseline,
             slope=metrics["slope"], interv_mse=metrics["interv_mse"], n_components=N_COMPONENTS,
             variance_share=variance_share, no_intervention_mse=no_intervention_mse)
    print("wrote results/llm_restricted_das.npz")


if __name__ == "__main__":
    main()
