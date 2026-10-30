# Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful Features

Code for every figure and Table 1 of the paper (Vilhelmsson Emneby, Yocum, Allen; NeurIPS 2026 Workshop
*Interpretability as a Science*), plus the Feature Fields reproduction notebook in `feature_fields/`.

Two testbeds:

* **Mess3 HMM** – the pre-trained 4-layer, width-64 transformer of Shai et al. (2024); the probe target is the
  belief state over hidden states, and all prefixes of length 1–8 are enumerated exactly. CPU, ~10 min.
* **Llama-2-7B** – birth year of 8000 Wikidata people (Heinzerling & Inui, 2024), prompt
  `"{name} was born in 19"`, residual stream at layer 8. GPU (Apple-silicon MPS or CUDA), ~2 h.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data        # Mess3 checkpoint (1 MB, Hugging Face) + birth-year prompts (13 MB, GitHub)
make smoke       # every script at toy sizes, ~5 min; downloads Llama-2-7B (13 GB) on first use
```

## Run

```bash
make hmm         # Mess3 experiments -> results/hmm_*, figures 3 and 4
make llm         # Llama-2-7B experiments -> results/llm_*
make figures     # figures 1, 2, 5, 6, 7, 8, 9 from results/
make table       # Table 1 -> results/table1.md
```

`results/` already holds the numbers behind the paper, so `make figures` and `make table` work without running
any model. The HMM files were regenerated with this code and match the paper to the printed digit; the Llama
files are the original runs' outputs. The only missing result file is the 130 MB Llama activation cache
(`results/llm_activations.npz`), which `scripts/llm_sweep.py` writes and the other Llama scripts read.

## What produces what

| Paper | Computation (`scripts/`) | Drawn by | Results |
|---|---|---|---|
| Fig. 1 | `hmm_persample.py`, `llm_persample.py` | `plot_figures.py` | `hmm_persample.json`, `llm_persample.json` |
| Fig. 2 | `hmm_sweep.py`, `llm_sweep.py` | `plot_figures.py` | `hmm_sweep.json`, `llm_sweep.json` |
| Fig. 3 | `hmm_simplex.py` (DAS plane from `hmm_das.py`) | itself | – |
| Fig. 4 | `hmm_heatmap.py` | itself | – |
| Fig. 5 | `hmm_operativity.py`, `llm_operativity.py` | `plot_figures.py` | `hmm_operativity.npz`, `llm_operativity.npz` |
| Fig. 6 | `llm_layersweep.py` | `plot_figures.py` | `llm_layersweep.json` |
| Fig. 7 | `hmm_das.py`, `llm_sweep.py` | `plot_figures.py` | `hmm_das.json`, `llm_sweep.json` |
| Fig. 8 | `llm_covariance_z.py` (model-free) | `plot_figures.py` | `llm_covariance_z.npy` |
| Fig. 9 | `hmm_restricted_das.py`, `llm_restricted_das.py` | `plot_figures.py` | `hmm_restricted_das.npz`, `llm_restricted_das.npz` |
| Table 1 | `hmm_sweep.py`, `hmm_das.py`, `llm_sweep.py` | `make_table.py` | `table1.md` |

Shared code in `pll/`: `mess3.py` (HMM, exact beliefs, the transformer, steering hooks), `probes.py`
(weighted OLS/ridge, pseudoinverse steering maps, principal angles, probe mass), `llama.py` (model, prompts,
activation cache, expected-decade readout, interchange, DAS), `plotting.py`, `paths.py`.

## Notes on the procedure

1. HMM probes weight each prefix by its probability: rows of activations and targets are scaled by
   `sqrt(probability)` before scikit-learn's `Ridge`/`LinearRegression` (default intercept, fit in the scaled
   space). HMM probe loss is the weighted squared error summed over the three belief coordinates, as in
   Table 1 and Figure 3; the published Figure 2 HMM panel used the per-coordinate mean (3x smaller).
2. Llama directions used for interventions and angles are fit on the first 70% of the 8000 prompts. The probe
   losses in Table 1 and Figure 2 are the in-sample MSE of the regression readout refit on all 8000 prompts;
   for DAS it is the best scale-and-bias readout along its direction. Figure 6 uses `1 - corr^2` (the same
   best-affine readout) and a fixed penalty of 1e4 at every layer.
3. Figure 9, HMM panel: the caption says 7 principal components (99.9% of OLS mass); the published figure was
   drawn from the 99% set (4 components). `hmm_restricted_das.py` saves both, `plot_figures.py` draws 99.9%.
4. Seeds and sampling order match the original runs, so HMM numbers reproduce exactly. Llama runs in bfloat16
   on the GPU, so expect differences in the last printed digit.

## Feature Fields

`feature_fields/figure5c_beta_feature_fields.ipynb` reproduces Figure 5c of *Neural Manifold Geometry Encodes
Feature Fields* (Yocum, Allen, Olshausen, Russell): a 2-layer transformer is trained to map (a, b) to the
integral of Beta(phi; a, b) cos(7 pi phi); one linear "field probe" per phi recovers the density Beta(phi; a, b)
from the activations; PCA of the probe manifold gives a basis, and the field is drawn with 1 to 30 basis
vectors. The last two cells are representability tests: at the best-probed layer, linear probes are fit for
other target functions of (a, b) that the model has no reason to compute, including non-monotonic transforms
of the density, and their decodability is compared with the true Beta density. That is the "decodability is
cheap" point of Pretty Linear Liars. Self-contained (NumPy, SciPy, PyTorch, scikit-learn, Matplotlib), with
outputs saved in the notebook.

## Data sources

* Mess3 transformer: run `20241205175736_23` in the Hugging Face dataset `SimplexAI/quantum-representations`
  (Shai et al., *Transformers represent belief state geometry in their residual stream*, NeurIPS 2024).
* Birth years: `numprop_P569.*` from <https://github.com/bheinzerling/numeric-property-repr>
  (Heinzerling & Inui, *Monotonic representation of numeric attributes in language models*, ACL 2024).
