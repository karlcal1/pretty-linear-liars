# Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful Features

Code to reproduce every figure and Table 1 of the paper (Vilhelmsson Emneby, Yocum, Allen; NeurIPS 2026
Workshop *Interpretability as a Science*).

Two testbeds:

* **Mess3 HMM** – the pre-trained 4-layer, width-64 transformer of Shai et al. (2024). The belief state
  over hidden states is the probe target; all prefixes of length 1–8 are enumerated exactly. Runs on CPU.
* **Llama-2-7B** – birth year of 8000 Wikidata people (Heinzerling & Inui, 2024), prompt
  `"{name} was born in 19"`, residual stream at layer 8. Needs a GPU (Apple-silicon MPS or CUDA) and ~16 GB
  of memory for the bfloat16 weights.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: uv venv --python 3.13 && uv pip install -r requirements.txt
make data        # Mess3 checkpoint (1 MB, Hugging Face) + birth-year prompts (13 MB, GitHub)
```

Llama-2-7B (`NousResearch/Llama-2-7b-hf`, ~13 GB) is downloaded by `transformers` the first time a Llama
script runs.

## Run

```bash
make hmm         # ~10 min on a laptop CPU  -> results/hmm_*, figures/fig3_*.png, figures/fig4_*.png
make llm         # ~2 h on an M-series Mac  -> results/llm_*
make figures     # figures 1, 2, 5, 6, 7, 8, 9 from results/
make table       # Table 1 -> results/table1.md
```

`results/` ships with the outputs of our run, so `make figures` and `make table` work without any model.
The only result file not included is the 130 MB cache of Llama activations (`results/llm_activations.npz`),
which `scripts/llm_sweep.py` writes and the other Llama scripts read.

## What produces what

| Paper | Computation | Drawn by | Result files |
|---|---|---|---|
| Fig. 1 | `hmm_persample.py`, `llm_persample.py` | `plot_figures.py` | `hmm_persample.json`, `llm_persample.json` |
| Fig. 2 | `hmm_sweep.py`, `llm_sweep.py` | `plot_figures.py` | `hmm_sweep.json`, `llm_sweep.json` |
| Fig. 3 | `hmm_simplex.py` (uses the DAS plane from `hmm_das.py`) | itself | – |
| Fig. 4 | `hmm_heatmap.py` | itself | – |
| Fig. 5 | `hmm_operativity.py`, `llm_operativity.py` | `plot_figures.py` | `hmm_operativity.npz`, `llm_operativity.npz` |
| Fig. 6 | `llm_layersweep.py` | `plot_figures.py` | `llm_layersweep.json` |
| Fig. 7 | `hmm_das.py`, `llm_sweep.py` | `plot_figures.py` | `hmm_das.json`, `llm_sweep.json` |
| Fig. 8 | `llm_covariance_z.py` (model-free, from the activation cache) | `plot_figures.py` | `llm_covariance_z.npy` |
| Fig. 9 | `hmm_restricted_das.py`, `llm_restricted_das.py` | `plot_figures.py` | `hmm_restricted_das.npz`, `llm_restricted_das.npz` |
| Table 1 | `hmm_sweep.py`, `hmm_das.py`, `llm_sweep.py` | `make_table.py` | `table1.md` |

All scripts live in `scripts/`; the Makefile runs them in dependency order. Shared code is in `pll/`:

```
pll/mess3.py     Mess3 process, exact belief enumeration, the Shai et al. transformer, steering hooks
pll/probes.py    weighted OLS/ridge probes, pseudoinverse steering maps, principal angles, probe mass
pll/llama.py     Llama-2-7B loading, prompts, activation caching, expected-decade readout, interchange, DAS
pll/plotting.py  colours and shared panel styles
pll/paths.py     where data, results and figures live
```

## Notes on the procedure

These are the details that matter if you want to match the paper's numbers exactly, or that differ from how
the paper's text describes them.

1. **HMM probes are weighted by prefix probability** by scaling the rows of activations and targets by
   `sqrt(probability)` before calling scikit-learn's `Ridge`/`LinearRegression` with its default intercept.
   The intercept is therefore fit in the scaled space. This is the procedure behind the published numbers.
2. **HMM probe loss** is the weighted squared error summed over the three belief coordinates (as in Table 1
   and the Figure 3 titles). The Figure 2 HMM panel in the published PDF used the per-coordinate average,
   which is exactly 3x smaller; here Figure 2 uses the same definition as Table 1.
3. **Llama directions** (OLS, ridge at each lambda) used for interventions and angles are fit on the first
   70% of the 8000 prompts. The probe losses reported in Table 1 and Figure 2 are the in-sample MSE of the
   regression readout refit on all 8000 prompts; for DAS, which has no regression weights, it is the best
   scale-and-bias readout along its direction.
4. **Figure 6** uses the same ridge penalty (1e4) at every layer, with 3000 prompts and 200 interchanges.
   The paper's caption describes the per-layer probe as intervention-optimized; the code never swept lambda
   per layer. The probe loss per layer is the in-sample readout MSE (standardized target).
5. **Figure 9, HMM panel**: the paper's caption says the restricted search used 7 principal components
   holding over 99.9% of the OLS mass. The published figure was drawn from the 99% set (4 components).
   `hmm_restricted_das.py` saves both; `plot_figures.py` draws the 99.9% set. Both are flat.
6. **HMM principal angles** are computed between the DAS plane and the span of the three rows of the
   pseudoinverse steering map. Restricting to the two-dimensional plane of edits that keep beliefs on the
   simplex changes the angles by at most 0.1 degree (28.9 vs 29.0).
7. **Intervention loss** on the HMM is the squared error summed over the three next-token probabilities,
   averaged over 496 grid targets from each of three start sequences. On Llama it is the squared error
   between the post-interchange expected decade and the source's decade digit.
8. **Randomness.** All seeds are fixed and the sampling order matches the original runs, so the HMM
   numbers reproduce to the printed precision. Llama runs in bfloat16 on the GPU, so expect differences in
   the last printed digit there.

## Data sources

* Mess3 transformer: training run `20241205175736_23` in the Hugging Face dataset
  `SimplexAI/quantum-representations` (Shai et al., *Transformers represent belief state geometry in their
  residual stream*, NeurIPS 2024).
* Birth years: `data/wikidata/numprop_P569.*` from <https://github.com/bheinzerling/numeric-property-repr>
  (Heinzerling & Inui, *Monotonic representation of numeric attributes in language models*, ACL 2024).
