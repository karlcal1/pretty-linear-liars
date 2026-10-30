# Targets are described in README.md.
PY ?= .venv/bin/python
export PYTHONPATH := .

.PHONY: all data hmm llm figures table smoke

all: data hmm llm figures table

smoke:
	PLL_SMOKE=1 $(MAKE) hmm llm figures table

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
