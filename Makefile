# Reproduce every figure and Table 1 of "Pretty Linear Liars".
#   make data      download the Mess3 checkpoint and the birth-year prompts
#   make hmm       Mess3 HMM experiments (CPU, ~15 min) -> results/hmm_*, figures 3 and 4
#   make llm       Llama-2-7B experiments (Apple-silicon GPU or CUDA, ~2 h) -> results/llm_*
#   make figures   figures 1, 2, 5, 6, 7, 8, 9 from results/
#   make table     Table 1 from results/
#   make all       everything above
PY ?= .venv/bin/python
export PYTHONPATH := .

.PHONY: all data hmm llm figures table

all: data hmm llm figures table

data:
	$(PY) data/download_data.py

hmm:
	$(PY) scripts/hmm_sweep.py
	$(PY) scripts/hmm_das.py
	$(PY) scripts/hmm_persample.py
	$(PY) scripts/hmm_operativity.py
	$(PY) scripts/hmm_restricted_das.py
	$(PY) scripts/hmm_heatmap.py
	$(PY) scripts/hmm_simplex.py

llm:
	$(PY) scripts/llm_sweep.py
	$(PY) scripts/llm_persample.py
	$(PY) scripts/llm_layersweep.py
	$(PY) scripts/llm_operativity.py
	$(PY) scripts/llm_restricted_das.py
	$(PY) scripts/llm_covariance_z.py

figures:
	$(PY) scripts/plot_figures.py

table:
	$(PY) scripts/make_table.py
