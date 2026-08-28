# Network Identification in International Inflation

This repository contains the code and data for the MSc Statistics dissertation *Cross-Sectional and Network Information in International Inflation: Bayesian GNAR with Stochastic Volatility*.

## Reproducing and inspecting the report results

### `experiments/run_all.ipynb`

`run_all.ipynb` is the main entry point for reproducing the complete experiment pipeline: it creates a versioned run, executes each experiment notebook in a fresh kernel, saves executed notebook copies, and finishes by running `load_results.ipynb`.

Set `QUICK = True` for a reduced end-to-end execution check, or `QUICK = False` for the full production analysis used to generate report results; the full run takes approximately **five hours** because several models use four-chain NUTS sampling.

### `experiments/load_results.ipynb`

`load_results.ipynb` is the results-of-record notebook: it loads a completed experimental run, checks forecast alignment and result provenance, constructs the final scoreboards and paired comparisons, and generates the report-facing figures without refitting substantive models.

For a reader checking the numerical results reported in the dissertation, this is the most direct notebook to inspect; it can also be pointed at a specific `GNAR_RUN_VERSION` to reconstruct outputs from an existing saved run.

The repository contains one full production run, `20260827_054747`, which is the exact run used to obtain the results reported in the dissertation.

## Data

All data required to reproduce the dissertation are included under `data/` as CSV files, so no further data acquisition is required.

The three source files used by `preprocessing.ipynb` are:

- `data/OECD/OECD-raw-COICOP1999.csv` — OECD monthly CPI source data.
- `data/CEPII/Gravity.csv` — CEPII gravity/distance data used to construct the geographic network.
- `data/Comtrade/comtrade-Top24-XandM-2010-2021.csv` — UN Comtrade bilateral trade data used to construct the export and import networks.

`preprocessing.ipynb` verifies or creates the four analysis-ready files consumed by the experiment code:

- `data/OECD/inflation.csv`
- `data/CEPII/W_geo.csv`
- `data/Comtrade/W_ex.csv`
- `data/Comtrade/W_im.csv`

## Other project files

- `preprocessing.ipynb` — constructs the 23-country year-on-year inflation panel and the geographic, export and import network matrices.
- `experiments/0_preliminaries.ipynb` — performs the pre-2016 lag-order, network, sparsity and stage-depth specification search used to fix the primary GNAR design.
- `experiments/1_core_model.ipynb` — fits the principal high-order BGNAR-SV models, the matched constant-variance model, and the main internal cross-sectional and network comparators.
- `experiments/2_benchmark_decomposition.ipynb` — fits the external benchmark suite: country AR-SV, BVAR(1), AR-GARCH, random walk and UCSV models.
- `experiments/4_calibration.ipynb` — evaluates forecast calibration using PIT, predictive-interval and coverage diagnostics from saved forecast distributions.
- `experiments/5_supplementary_nulls.ipynb` — contains supplementary prior and stochastic-volatility specification sensitivity checks for the selected high-order model.
- `experiments/6_common_factor_diagnostic.ipynb` — computes PCA/common-component summaries and selected-design collinearity diagnostics.
- `experiments/7_sparsity_sweep.ipynb` — evaluates high-order geographic, export and import network robustness over the sparsity path, including PC1-adjusted comparisons.
- `experiments/8_ragnar_search.ipynb` — runs the supplementary low-order RaGNAR-style random-network search with its own validation and evaluation split.
- `experiments/9_common_component.ipynb` — fits the PC1-adjusted AR, uniform, geographic stage-1 and geographic stage-2 BGNAR-SV comparison.
- `experiments/10_simulation_study.ipynb` — runs the low-order simulation study comparing raw and PC1-adjusted GNAR under common-factor confounding, planted network effects and geographic densification.
- `experiments/11_rolling_origins.ipynb` — performs the high-order conjugate rolling-origin analysis at direct horizons 1, 6 and 12 months.
- `experiments/shared_utils.py` — contains the shared data loading, GNAR design, priors, estimation, forecasting, scoring, stability, saving and utility functions used across the notebooks.

Notebook 3 in `experiments/` is intentionally absent because its experiments were removed from the final dissertation design.

## Saved results

Each `run_all.ipynb` execution is assigned a `GNAR_RUN_VERSION`; summary result objects, aligned forecast bundles and executed notebook copies are stored respectively under `experiments/results/`, `experiments/outputs/` and `experiments/runs/` using that run version.

The repository includes `experiments/results/20260827_054747/`, `experiments/outputs/20260827_054747/` and `experiments/runs/20260827_054747/`, corresponding to the production run used for the dissertation.

These saved results are fully reproducible by running `run_all.ipynb` with `QUICK = False`.

## Reproducibility notes

The code was run under Python 3.13; the package versions present in the environment used for the final production run are recorded in the repository-level `requirements.txt`.

Fixed random seeds are used throughout so that results are reproducible in a matching software environment.

The MCMC-fitted models used PyMC's `forkserver` multiprocessing context; if unavailable (e.g. on Windows), the only requirement is to change `mp_ctx="forkserver"` in `experiments/shared_utils.py` to a supported option such as `spawn`. 
