# Pretty Linear Liars

Code for the figures and Table 1 of *Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful
Features* (Vilhelmsson Emneby, Yocum, Allen; NeurIPS 2026 Workshop).

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data      # Mess3 checkpoint (Hugging Face) + birth-year prompts (GitHub), 14 MB
make smoke     # every script at toy sizes, a few minutes; downloads Llama-2-7B (13 GB) once
```

## Run

```bash
make hmm       # Mess3 HMM, CPU ~15 min        -> results/hmm_*, figures 3 and 4
make llm       # Llama-2-7B, MPS or CUDA ~2 h  -> results/llm_*
make figures   # figures 1, 2, 5-9 from results/
make table     # Table 1 -> results/table1.md
```

`results/` ships the numbers behind the paper, so `make figures` and `make table` need no model. The HMM files
were regenerated with this code and match the paper exactly; the Llama files are the original runs' outputs.

## Layout

| Paper | Script in `scripts/` | Result file |
|---|---|---|
| Fig. 1 | `hmm_persample.py`, `llm_persample.py` | `*_persample.json` |
| Fig. 2, Table 1 | `hmm_sweep.py`, `hmm_das.py`, `llm_sweep.py` | `*_sweep.json`, `hmm_das.json` |
| Fig. 3 | `hmm_simplex.py` | draws itself |
| Fig. 4 | `hmm_heatmap.py` | draws itself |
| Fig. 5 | `hmm_operativity.py`, `llm_operativity.py` | `*_operativity.npz` |
| Fig. 6 | `llm_layersweep.py` | `llm_layersweep.json` |
| Fig. 7 | `hmm_das.py`, `llm_sweep.py` | `hmm_das.json`, `llm_sweep.json` |
| Fig. 8 | `llm_covariance_z.py` | `llm_covariance_z.npy` |
| Fig. 9 | `hmm_restricted_das.py`, `llm_restricted_das.py` | `*_restricted_das.npz` |

`plot_figures.py` and `make_table.py` assemble the figures and Table 1 from those files. Shared code in `pll/`:
`mess3.py` (HMM, exact beliefs, transformer, steering hooks), `probes.py` (probability-weighted OLS/ridge,
pseudoinverse steering maps, principal angles, probe mass), `llama.py` (model, prompts, expected-decade readout,
interchange, DAS), `plotting.py`, `paths.py`.

## Where the code differs from the paper text

- Fig. 2, HMM panel: the paper plotted per-coordinate probe loss; here it is summed over the three belief
  coordinates, as in Table 1 and Fig. 3.
- Fig. 6: a fixed ridge penalty of 1e4 at every layer (the caption says intervention-optimized); probe loss is
  1 - corr^2.
- Fig. 9, HMM panel: drawn from the 99.9% / 7-component set as in the caption; the published panel used the
  99% / 4-component set. Both are saved.
- Llama directions are fit on the first 70% of the 8000 prompts; probe losses are in-sample on all 8000.

## Feature Fields

`feature_fields/figure5c_beta_feature_fields.ipynb` is a preliminary one-day reproduction of Fig. 5c of
*Neural Manifold Geometry Encodes Feature Fields* (Yocum et al.), not the authors' code. Self-contained, outputs saved.

## Data

- Mess3 transformer: run `20241205175736_23` of `SimplexAI/quantum-representations` (Shai et al., 2024).
- Birth years: `numprop_P569.*` from <https://github.com/bheinzerling/numeric-property-repr> (Heinzerling & Inui, 2024).
