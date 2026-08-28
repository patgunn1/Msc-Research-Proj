"""Shared utilities for the factor-adjusted Bayesian GNAR inflation project.

Holds path and configuration constants, data loading, GNAR stage-weight and design
construction, the PyMC stochastic-volatility models, the closed-form conjugate
Bayesian GNAR, forecasting routines, scoring, the Diebold-Mariano test, a
full-system stationarity check, and the RaGNAR-style random-network search.

Nothing here is dataset-specific beyond the file paths: swapping the inflation CSV
and re-pointing DATA_DIR is enough to re-run the whole collection, provided saved
forecasts are regenerated.
"""

import os
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Paths (created on import)
# ---------------------------------------------------------------------------
# Resolve the project root relative to this file so the repository is portable.
# Layout assumed: <root>/experiments/shared_utils.py, with data under <root>/data.
# Set the GNAR_PROJECT_ROOT environment variable to override.
_ENV_ROOT = os.environ.get("GNAR_PROJECT_ROOT")
HOME = Path(_ENV_ROOT).expanduser() if _ENV_ROOT else Path(__file__).resolve().parent.parent
DATA_DIR = HOME / "data"
EXP_DIR = HOME / "experiments"
RUN_VERSION = os.environ.get("GNAR_RUN_VERSION", "manual").strip()
if not RUN_VERSION:
    raise ValueError("GNAR_RUN_VERSION must be non-empty.")
RESULTS = EXP_DIR / "results" / RUN_VERSION
OUTPUTS = EXP_DIR / "outputs" / RUN_VERSION

for _d in (DATA_DIR, EXP_DIR, RESULTS, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)


def quick_mode():
    """Return whether the reduced execution-check configuration is active."""
    return os.environ.get("GNAR_QUICK", "0").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPLIT_DATE = "2016-01-01"
CRISIS_START = "2020-03-01"
CRISIS_END = "2021-06-30"
POST2022_START = "2022-01-01"

# rw_centre is the prior mean of the own lag-1 coefficient. For levels data a
# random-walk centre of 1.0 is standard. This project demeans year-on-year
# inflation on the training window (see load_data) and the series are
# mean-reverting, so a unit-root centre over-persists. 0.8 leans towards
# stationarity while staying weakly informative. Override to 1.0 for levels.
MINN = dict(lambda1=0.2, lambda2=0.5, lambda3=1.0, rw_centre=0.8)

# Sampler settings shared by every NUTS fit. random_seed is fixed so results are
# reproducible across runs on a matching software stack.
NUTS_KW = dict(
    draws=1000,
    tune=2000,
    chains=4,
    target_accept=0.95,
    return_inferencedata=True,
    idata_kwargs={"log_likelihood": True},
    random_seed=42,
    mp_ctx="forkserver",
)

# Network file locations. N and the country list are inferred from the inflation
# CSV, not hard-coded.
INFLATION_CSV = DATA_DIR / "OECD" / "inflation.csv"
NETWORK_FILES = {
    "geographic": DATA_DIR / "CEPII" / "W_geo.csv",
    "export": DATA_DIR / "Comtrade" / "W_ex.csv",
    "import": DATA_DIR / "Comtrade" / "W_im.csv",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_network(path, countries):
    """Read a weight-matrix CSV and align it to the given country order."""
    # Read and coerce the axis labels to strings for a reliable reindex.
    W = pd.read_csv(path, index_col=0)
    W.index = W.index.astype(str)
    W.columns = W.columns.astype(str)

    # Fail loudly if any country in the panel is absent from this matrix.
    missing = [c for c in countries if c not in W.index or c not in W.columns]
    if missing:
        raise ValueError(f"{path.name}: missing countries {missing}")

    # Reorder to the canonical country order and validate the supplied weights.
    W = W.loc[countries, countries].to_numpy(dtype=float)
    if not np.all(np.isfinite(W)):
        raise ValueError(f"{path.name}: network contains non-finite weights.")
    if np.any(W < 0):
        raise ValueError(f"{path.name}: network contains negative weights.")

    np.fill_diagonal(W, 0.0)
    if np.any(W.sum(axis=1) <= 0):
        raise ValueError(f"{path.name}: at least one country has no positive outgoing weight.")
    return W


def load_data(network="geographic"):
    """Load the inflation panel and candidate networks.

    Demeans each series on the training window only, splits at SPLIT_DATE, and
    returns a dict with the train/test frames, the full and train arrays, the
    test-start index, the country list, N, the selected network, and all networks.
    The country set and N are inferred from the inflation CSV columns.
    """
    # Load the panel, validate its monthly structure and infer the country set.
    infl = pd.read_csv(INFLATION_CSV, index_col=0, parse_dates=True)
    infl = infl.sort_index()
    infl.columns = infl.columns.astype(str)

    if infl.index.has_duplicates:
        raise ValueError("Inflation data contain duplicate dates.")
    if infl.columns.duplicated().any():
        raise ValueError("Inflation data contain duplicate country columns.")
    if not np.all(np.isfinite(infl.to_numpy(dtype=float))):
        raise ValueError("Inflation data contain missing or non-finite observations.")

    months = infl.index.to_period("M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    if not months.equals(expected):
        raise ValueError("Inflation data are not a complete consecutive monthly panel.")

    countries = list(infl.columns)
    N = len(countries)

    # Demean using training-window means only, so the test window sees no future
    # information through the centring.
    train_mask = infl.index < pd.Timestamp(SPLIT_DATE)
    if not train_mask.any() or train_mask.all():
        raise ValueError("SPLIT_DATE does not produce non-empty training and test samples.")
    means = infl.loc[train_mask].mean()
    infl = infl - means

    # Split into train and test, recording the integer index where the test starts.
    train = infl.loc[train_mask]
    test = infl.loc[~train_mask]
    test_start = int(train_mask.sum())

    # Load every candidate network aligned to the same country order.
    networks = {name: _read_network(p, countries) for name, p in NETWORK_FILES.items()}
    if network not in networks:
        raise ValueError(f"unknown network {network!r}; have {list(networks)}")

    return dict(
        train=train,
        test=test,
        Y_full=infl,
        Y_train=train,
        test_start=test_start,
        countries=countries,
        N=N,
        W=networks[network],
        networks=networks,
        train_means=means,
    )


# ---------------------------------------------------------------------------
# Stage weights, design and priors
# ---------------------------------------------------------------------------
def _as_square_array(W):
    """Validate and return W as a square, non-negative, zero-diagonal array."""
    W = W.to_numpy(dtype=float) if hasattr(W, "to_numpy") else np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square; received shape {W.shape}.")
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains non-finite values.")
    if np.any(W < 0):
        raise ValueError("GNAR weights must be non-negative.")
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    return W


def _row_normalise(M):
    """Row-normalise, leaving all-zero rows unchanged."""
    M = np.asarray(M, dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    return np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)


def knn_sparsify(W, k):
    """Keep each row's k strongest off-diagonal links and row-normalise."""
    W = _as_square_array(W)
    n = W.shape[0]

    if k is None or k >= n - 1:
        return _row_normalise(W)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer or None.")

    out = np.zeros_like(W)
    for i in range(n):
        row = W[i].copy()
        row[i] = -np.inf
        keep = np.argsort(row, kind="stable")[-k:]
        out[i, keep] = W[i, keep]
    return _row_normalise(out)


def compute_stage_weights(W, max_stage):
    """Return row-normalised weights for exact graph-distance stages 1..max_stage.

    Stage 1 preserves the supplied edge weights; higher stages use equal weights
    over the nodes whose shortest-path distance is exactly that stage. Under a 
    complete network higher-stage weight matrices are all zero because every 
    other node is already reachable at stage 1.
    """
    if not isinstance(max_stage, (int, np.integer)) or max_stage < 0:
        raise ValueError("max_stage must be a non-negative integer.")
    W = _as_square_array(W)
    N = W.shape[0]
    if max_stage == 0:
        return []

    # Boolean adjacency drives the shortest-path frontier expansion below.
    adjacency = W > 0
    np.fill_diagonal(adjacency, False)

    # reached tracks every node at distance < current stage (seeded with self);
    # frontier holds the nodes discovered at the previous stage.
    stage_weights, reached, frontier = [], np.eye(N, dtype=bool), None
    for stage in range(1, max_stage + 1):
        # Stage 1 is direct adjacency; later stages are nodes one hop beyond the
        # previous frontier that have not been reached at any earlier stage.
        if stage == 1:
            exact_stage = adjacency.copy()
        else:
            exact_stage = ((frontier.astype(int) @ adjacency.astype(int)) > 0) & ~reached
            np.fill_diagonal(exact_stage, False)

        # Stage 1 keeps observed weights; higher stages use uniform weights, since
        # no direct edge (hence no observed weight) exists beyond distance one.
        raw = np.where(exact_stage, W, 0.0) if stage == 1 else exact_stage.astype(float)
        stage_weights.append(_row_normalise(raw))

        # Advance the frontier and mark the newly reached nodes.
        reached |= exact_stage
        frontier = exact_stage
    return stage_weights


def _validate_gnar_spec(p, stages):
    """Check that p is a positive integer and stages has length p."""
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("p must be a positive integer.")
    if len(stages) != p:
        raise ValueError(f"stages must have length p={p}; received {len(stages)}.")
    if any(not isinstance(s, (int, np.integer)) or s < 0 for s in stages):
        raise ValueError("Each stage order must be a non-negative integer.")


def _canonical_mode(mode):
    """Validate and return the design mode."""
    allowed = {"ar_only", "global_gnar", "local_alpha_gnar"}
    if mode not in allowed:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(allowed)}.")
    return mode


def _feature_matrix_at(Y, origin, stage_weights, p, stages, include_network):
    """Construct the N x K regressor matrix available at one forecast origin."""
    # Own lags: the observation vectors at origin, origin-1, ..., origin-p+1.
    columns = [Y[origin - lag + 1] for lag in range(1, p + 1)]

    # Network terms: stage-weighted neighbour aggregates at each lag and stage.
    if include_network:
        for lag in range(1, p + 1):
            y_lag = Y[origin - lag + 1]
            for stage in range(1, stages[lag - 1] + 1):
                columns.append(stage_weights[stage - 1] @ y_lag)
    return np.stack(columns, axis=1)


def build_design(Y, W, p, stages, mode="global_gnar", h=1):
    """Construct a direct-h GNAR design (stacked over countries and time).

    stages[j-1] is the highest neighbour stage included at lag j, so stages=[1, 1]
    gives stage-1 neighbours at both lags. The global and local-alpha models share
    the same regressors; coefficient heterogeneity is introduced by the model, not
    the design.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a T x N array.")
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError("h must be a positive integer.")
    _validate_gnar_spec(p, stages)
    mode = _canonical_mode(mode)

    # Build stage weights once; suppress them entirely in the AR-only mode.
    include_network = mode != "ar_only"
    stage_weights = compute_stage_weights(W, max(stages) if include_network else 0)

    # The first target that has p lags and an h-ahead origin available.
    T = Y.shape[0]
    first_target = p + h - 1
    if first_target >= T:
        raise ValueError("Not enough observations for the requested p and h.")

    # One N x K feature block per target time, later stacked over countries.
    rows_X, rows_y = [], []
    for target in range(first_target, T):
        origin = target - h
        rows_X.append(_feature_matrix_at(Y, origin, stage_weights, p, stages, include_network))
        rows_y.append(Y[target])
    X, y = np.concatenate(rows_X, axis=0), np.concatenate(rows_y, axis=0)

    # Column names mirror the own-then-network construction order exactly.
    names = [f"own_lag{lag}" for lag in range(1, p + 1)]
    if include_network:
        names += [f"net_lag{lag}_stage{stage}" for lag in range(1, p + 1)
                  for stage in range(1, stages[lag - 1] + 1)]
    return X, y, names


def build_forecast_features(Y_history, W, p, stages, mode="global_gnar",
                            stage_weights=None):
    """Construct the N x K regressors for the next one-step forecast."""
    Y_history = np.asarray(Y_history, dtype=float)
    if Y_history.ndim != 2:
        raise ValueError("Y_history must be a T x N array.")
    if Y_history.shape[0] < p:
        raise ValueError("Insufficient history for the requested lag order.")
    _validate_gnar_spec(p, stages)
    mode = _canonical_mode(mode)
    include_network = mode != "ar_only"

    # Reuse caller-supplied stage weights when available; the online forecasters
    # precompute them once and pass them in to avoid rebuilding per step.
    if stage_weights is None:
        stage_weights = compute_stage_weights(W, max(stages) if include_network else 0)
    return _feature_matrix_at(Y_history, Y_history.shape[0] - 1, stage_weights, p, stages, include_network)


def build_time_index(Y, p, h=1):
    """Return the target time indices aligned with build_design."""
    T = np.asarray(Y).shape[0]
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError("h must be a positive integer.")
    return np.arange(p + h - 1, T, dtype=int)


def variance_inflation_factor(X, column):
    """Return the VIF and auxiliary R-squared for one design column."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    if not 0 <= column < X.shape[1]:
        raise IndexError("column is outside the design matrix.")

    target = X[:, column]
    others = np.delete(X, column, axis=1)
    design = np.column_stack([np.ones(len(X)), others])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef

    ss_total = np.sum((target - target.mean()) ** 2)
    if ss_total <= 0:
        raise ValueError("Cannot compute a VIF for a constant column.")

    r_squared = 1.0 - np.sum(resid**2) / ss_total
    vif = 1.0 / max(1.0 - r_squared, np.finfo(float).eps)
    return float(vif), float(r_squared)


def ar1_forecast(series, horizon=1):
    """Fit an AR(1) with intercept and return its horizon-step forecast and persistence."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or len(series) < 3:
        raise ValueError("series must contain at least three observations.")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    design = np.column_stack([np.ones(len(series) - 1), series[:-1]])
    intercept, phi = np.linalg.lstsq(design, series[1:], rcond=None)[0]
    forecast = float(series[-1])
    for _ in range(horizon):
        forecast = float(intercept + phi * forecast)
    return forecast, float(phi)


def _extract_lag(name):
    """Parse the lag number out of a design-column name."""
    match = re.search(r"lag(\d+)", name)
    if match is None:
        raise ValueError(f"Cannot extract a lag number from {name!r}.")
    return int(match.group(1))


def minnesota_prior(names, minn=MINN):
    """Minnesota-style prior means and standard deviations for design columns.

    The own lag-1 coefficient is centred at rw_centre and all others at zero.
    Variances decay with lag and network columns are shrunk harder than own-lag
    columns by the factor lambda2.
    """
    mu, sd = np.zeros(len(names)), np.ones(len(names))
    for j, name in enumerate(names):
        # Lag drives the variance decay for both own and network columns.
        lag = _extract_lag(name)
        decay = lag ** minn["lambda3"]
        if name.startswith("own_"):
            # Random-walk belief on own lag 1; zero-centred elsewhere.
            mu[j] = minn["rw_centre"] if lag == 1 else 0.0
            sd[j] = minn["lambda1"] / decay
        elif name.startswith("net_"):
            # Extra cross-shrinkage on network coefficients via lambda2.
            sd[j] = minn["lambda1"] * minn["lambda2"] / decay
        else:
            raise ValueError(f"Unknown design-column type: {name!r}.")
    return mu, sd


# ---------------------------------------------------------------------------
# Stochastic-volatility models (PyMC)
# ---------------------------------------------------------------------------
def build_gnar_sv(X, y, time_idx, names, N, minn=MINN, student_t=False):
    """GNAR with a single shared AR(1) log-variance path (stochastic volatility).

    The log-variance follows a non-centred AR(1) with mean m_h, persistence phi
    and innovation scale sigma_h, giving one volatility value per distinct target
    time point, shared across countries at that time. Observation noise is Gaussian
    by default and Student-t if requested.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor import scan as pt_scan

    # Minnesota prior on the GNAR coefficients; the design must stack N countries
    # under each of n_time target times, in that order.
    mu0, sd0 = minnesota_prior(names, minn=minn)
    time_idx = np.asarray(time_idx)
    n_time = len(time_idx)
    n_rows = X.shape[0]
    if n_rows != n_time * N:
        raise ValueError(f"Expected n_time*N={n_time * N} design rows; received {n_rows}.")

    # Map each stacked design row to its target-time index, so a per-time volatility
    # can be broadcast back over the countries sharing that time.
    row_time = np.repeat(np.arange(n_time), N)

    with pm.Model() as model:
        # Conditional mean: the linear GNAR predictor.
        beta = pm.Normal("beta", mu=mu0, sigma=sd0, shape=len(names))
        mean = pt.dot(X, beta)

        # AR(1) log-variance law, parameterised non-centrally through z.
        m_h = pm.Normal("m_h", mu=-2.0, sigma=1.0)
        phi = pm.Uniform("phi", lower=-0.99, upper=0.99)
        sigma_h = pm.HalfNormal("sigma_h", sigma=0.5)
        z = pm.Normal("h_innov", mu=0.0, sigma=1.0, shape=n_time)

        def step(z_t, h_prev, m, ph, s):
            return m + ph * (h_prev - m) + s * z_t

        # Seed the recursion at the AR(1) stationary spread, then scan the path.
        h0 = m_h + sigma_h * z[0] / pt.sqrt(1 - phi**2)
        h_seq, _ = pt_scan(
            fn=step,
            sequences=[z[1:]],
            outputs_info=[h0],
            non_sequences=[m_h, phi, sigma_h],
        )
        h = pm.Deterministic("h", pt.concatenate([h0[None], h_seq]))

        # Convert log-variance to a per-row standard deviation and attach the likelihood.
        sigma_t = pt.exp(h / 2.0)
        sigma_obs = sigma_t[row_time]
        if student_t:
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
            pm.StudentT("y_obs", nu=nu, mu=mean, sigma=sigma_obs, observed=y)
        else:
            pm.Normal("y_obs", mu=mean, sigma=sigma_obs, observed=y)

    return model


def build_gnar_sv_local_alpha(X, y, time_idx, names, N, minn=MINN, student_t=False):
    """GNAR-SV with country-specific domestic coefficients and shared network effects.

    Each country's own-lag coefficients are drawn hierarchically around a global
    mean; the network coefficients remain common across countries.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor import scan as pt_scan

    mu0, sd0 = minnesota_prior(names, minn=minn)
    n_time, n_rows, k = len(np.asarray(time_idx)), X.shape[0], len(names)
    if n_rows != n_time * N:
        raise ValueError(f"Expected n_time*N={n_time * N} design rows; received {n_rows}.")

    # Own-lag columns must precede network columns for the block split below.
    own_idx = np.array([j for j, name in enumerate(names) if name.startswith("own_")], dtype=int)
    net_idx = np.array([j for j, name in enumerate(names) if name.startswith("net_")], dtype=int)
    if not np.array_equal(own_idx, np.arange(len(own_idx))):
        raise ValueError("Own-lag columns must appear first in the design matrix.")
    if not np.array_equal(net_idx, np.arange(len(own_idx), k)):
        raise ValueError("Network columns must follow the own-lag columns.")

    # Row-to-time and row-to-country maps for the stacked design.
    row_time = np.repeat(np.arange(n_time), N)
    row_country = np.tile(np.arange(N), n_time)

    with pm.Model() as model:
        # Hierarchical own-lag coefficients: a per-country offset around the global
        # mean, non-centred through z_alpha.
        alpha_global = pm.Normal("alpha_global", mu=mu0[own_idx], sigma=sd0[own_idx], shape=len(own_idx))
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=0.25, shape=len(own_idx))
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, shape=(N, len(own_idx)))
        alpha = pm.Deterministic("alpha", alpha_global[None, :] + tau_alpha[None, :] * z_alpha)

        # Shared network coefficients, tiled across countries and concatenated with
        # the per-country own-lag block to form a full N x K coefficient matrix.
        if len(net_idx):
            beta_network = pm.Normal("beta_network", mu=mu0[net_idx], sigma=sd0[net_idx], shape=len(net_idx))
            beta_matrix = pt.concatenate([alpha, pt.tile(beta_network[None, :], (N, 1))], axis=1)
        else:
            beta_matrix = alpha
        beta = pm.Deterministic("beta", beta_matrix)

        # Each row uses its own country's coefficient vector.
        mean = (X * beta[row_country]).sum(axis=1)

        # Shared AR(1) log-variance, identical in form to build_gnar_sv.
        m_h = pm.Normal("m_h", mu=-2.0, sigma=1.0)
        phi = pm.Uniform("phi", lower=-0.99, upper=0.99)
        sigma_h = pm.HalfNormal("sigma_h", sigma=0.5)
        z = pm.Normal("h_innov", 0.0, 1.0, shape=n_time)

        def step(z_t, h_prev, m, ph, s):
            return m + ph * (h_prev - m) + s * z_t

        h0 = m_h + sigma_h * z[0] / pt.sqrt(1 - phi**2)
        h_seq, _ = pt_scan(fn=step, sequences=[z[1:]], outputs_info=[h0],
                           non_sequences=[m_h, phi, sigma_h])
        h = pm.Deterministic("h", pt.concatenate([h0[None], h_seq]))
        sigma_obs = pt.exp(h / 2.0)[row_time]
        if student_t:
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
            pm.StudentT("y_obs", nu=nu, mu=mean, sigma=sigma_obs, observed=y)
        else:
            pm.Normal("y_obs", mu=mean, sigma=sigma_obs, observed=y)
    return model


def build_gnar_constant(X, y, names, N, minn=MINN):
    """GNAR with a single constant observation variance (no stochastic volatility)."""
    import pymc as pm
    import pytensor.tensor as pt

    mu0, sd0 = minnesota_prior(names, minn=minn)
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=mu0, sigma=sd0, shape=len(names))
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mean = pt.dot(X, beta)
        pm.Normal("y_obs", mu=mean, sigma=sigma, observed=y)
    return model


def mcmc_diagnostics(idata, var_names=None):
    """Return compact convergence and energy diagnostics for a fitted PyMC model."""
    import arviz as az

    divergences = (int(idata.sample_stats["diverging"].sum())
                   if "diverging" in idata.sample_stats else 0)

    min_bfmi = np.nan
    if "energy" in idata.sample_stats:
        energy = np.asarray(idata.sample_stats["energy"].values, dtype=float)
        if energy.ndim == 2 and energy.shape[1] > 1:
            numerator = np.mean(np.diff(energy, axis=1) ** 2, axis=1)
            denominator = np.var(energy, axis=1, ddof=1)
            bfmi = np.divide(
                numerator, denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
            finite = bfmi[np.isfinite(bfmi)]
            if len(finite):
                min_bfmi = float(finite.min())

    try:
        summary = az.summary(idata, var_names=var_names, round_to=None)
    except Exception:
        return dict(
            divergences=divergences, max_rhat=np.nan,
            min_ess_bulk=np.nan, min_ess_tail=np.nan, min_bfmi=min_bfmi,
        )

    def extreme(column, fn):
        if column not in summary:
            return np.nan
        values = summary[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(fn(values)) if len(values) else np.nan

    return dict(
        divergences=divergences,
        max_rhat=extreme("r_hat", np.max),
        min_ess_bulk=extreme("ess_bulk", np.min),
        min_ess_tail=extreme("ess_tail", np.min),
        min_bfmi=min_bfmi,
    )


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------
def full_system_radius(a1, a2, b1, b2, W):
    """Spectral radius of the full VAR(2) companion for a global-alpha GNAR(2).

    The stage-1 matrix is constructed exactly as in the fitted GNAR, so supplied
    edge weights are row-normalised before the companion matrix is formed.
    """
    W1 = compute_stage_weights(W, 1)[0]
    N = W1.shape[0]

    A1 = a1 * np.eye(N) + b1 * W1
    A2 = a2 * np.eye(N) + b2 * W1
    companion = np.block([[A1, A2], [np.eye(N), np.zeros((N, N))]])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _dense_companion(lag_matrices):
    """Construct the dense companion matrix for ordered VAR lag matrices."""
    n = lag_matrices[0].shape[0]
    p = len(lag_matrices)
    companion = np.zeros((n * p, n * p), dtype=float)
    companion[:n, :] = np.hstack(lag_matrices)
    if p > 1:
        companion[n:, :-n] = np.eye(n * (p - 1))
    return companion


def companion_radius(lag_matrices, method="auto", v0=None, return_vector=False):
    """Return the spectral radius of a finite-order VAR companion matrix."""
    lag_matrices = [np.asarray(A, dtype=float) for A in lag_matrices]
    if not lag_matrices:
        raise ValueError("At least one lag matrix is required.")

    n = lag_matrices[0].shape[0]
    if any(A.shape != (n, n) for A in lag_matrices):
        raise ValueError("All lag matrices must be square with common dimension.")

    p = len(lag_matrices)
    dim = n * p
    if method not in {"auto", "dense", "sparse"}:
        raise ValueError("method must be 'auto', 'dense' or 'sparse'.")
    if method == "auto":
        method = "dense" if dim <= 250 else "sparse"

    if method == "dense":
        companion = _dense_companion(lag_matrices)
        if return_vector:
            values, vectors = np.linalg.eig(companion)
            index = int(np.argmax(np.abs(values)))
            vector = vectors[:, index]
            vector = np.real(vector) if np.linalg.norm(np.real(vector)) >= np.linalg.norm(np.imag(vector)) \
                else np.imag(vector)
            vector = np.asarray(vector, dtype=float)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            return float(abs(values[index])), vector
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    from scipy import sparse
    from scipy.sparse.linalg import ArpackNoConvergence, eigs

    top = sparse.hstack([sparse.csr_matrix(A) for A in lag_matrices], format="csr")
    if p == 1:
        companion = top
    else:
        lower = sparse.hstack([
            sparse.eye(n * (p - 1), format="csr"),
            sparse.csr_matrix((n * (p - 1), n)),
        ], format="csr")
        companion = sparse.vstack([top, lower], format="csr")

    if v0 is not None:
        v0 = np.asarray(v0, dtype=float).ravel()
        if v0.shape != (dim,):
            raise ValueError(f"v0 must have shape ({dim},).")
        starts = [v0]
    else:
        grid = np.arange(1, dim + 1, dtype=float)
        starts = [
            np.ones(dim, dtype=float),
            np.sin(grid),
            np.cos(np.sqrt(2.0) * grid),
        ]
        starts = [x / np.linalg.norm(x) for x in starts]

    k = min(6, max(1, dim - 2))
    ncv = min(dim, max(80, 4 * k + 20))
    best = None
    best_vector = None

    for start_vector in starts:
        try:
            values, vectors = eigs(
                companion, k=k, which="LM", return_eigenvectors=True,
                tol=1e-10, maxiter=max(10000, 20 * dim),
                ncv=ncv, v0=start_vector,
            )
        except ArpackNoConvergence as exc:
            values = exc.eigenvalues
            vectors = exc.eigenvectors
            if values is None or vectors is None or len(values) == 0:
                continue

        for index, value in enumerate(values):
            vector = vectors[:, index]
            residual = np.linalg.norm(companion @ vector - value * vector)
            scale = max(np.linalg.norm(vector), 1e-15)
            if residual / scale > 1e-7:
                continue
            radius = float(abs(value))
            if best is None or radius > best:
                best = radius
                real_vector = (
                    np.real(vector)
                    if np.linalg.norm(np.real(vector)) >= np.linalg.norm(np.imag(vector))
                    else np.imag(vector)
                )
                real_vector = np.asarray(real_vector, dtype=float)
                norm = np.linalg.norm(real_vector)
                best_vector = real_vector / norm if norm > 0 else None

    if best is None:
        return companion_radius(
            lag_matrices, method="dense", return_vector=return_vector
        )
    return (best, best_vector) if return_vector else best


def global_gnar_radius(alpha, beta, W):
    """Return the exact spectral radius of a global stage-1 GNAR."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if alpha.ndim != 1 or beta.ndim != 1 or alpha.shape != beta.shape:
        raise ValueError("alpha and beta must be one-dimensional arrays of equal length.")
    if len(alpha) == 0:
        raise ValueError("At least one lag coefficient is required.")

    W1 = compute_stage_weights(W, 1)[0]
    radius = 0.0
    for eigenvalue in np.linalg.eigvals(W1):
        coefficients = alpha.astype(complex) + eigenvalue * beta
        roots = np.roots(np.r_[1.0 + 0j, -coefficients])
        radius = max(radius, float(np.max(np.abs(roots))))
    return radius


def gnar_lag_matrices(beta, names, W):
    """Construct ordered lag matrices from global GNAR coefficients and column names."""
    beta = np.asarray(beta, dtype=float)
    names = list(names)
    if beta.ndim != 1 or len(beta) != len(names):
        raise ValueError("beta and names must describe the same coefficient vector.")

    p = max(_extract_lag(name) for name in names)
    stages = [
        int(name.rsplit("stage", 1)[1])
        for name in names if name.startswith("net_")
    ]
    max_stage = max(stages, default=0)
    stage_weights = compute_stage_weights(W, max_stage)
    n = _as_square_array(W).shape[0]

    alpha = np.zeros(p)
    network = np.zeros((p, max_stage))
    own_seen = np.zeros(p, dtype=bool)

    for value, name in zip(beta, names):
        lag = _extract_lag(name) - 1
        if name.startswith("own_"):
            alpha[lag] = value
            own_seen[lag] = True
        elif name.startswith("net_"):
            stage = int(name.rsplit("stage", 1)[1]) - 1
            network[lag, stage] = value
        else:
            raise ValueError(f"Unrecognised design column {name!r}.")

    if not own_seen.all():
        raise ValueError("Each lag must contain one domestic coefficient.")

    eye = np.eye(n)
    lag_matrices = []
    for lag in range(p):
        A = alpha[lag] * eye
        for stage in range(max_stage):
            A = A + network[lag, stage] * stage_weights[stage]
        lag_matrices.append(A)
    return lag_matrices


def gnar_radius(beta, names, W, *, method="auto", v0=None, return_vector=False):
    """Return the spectral radius implied by one global GNAR coefficient vector."""
    beta = np.asarray(beta, dtype=float)
    names = list(names)
    if beta.ndim != 1 or len(beta) != len(names):
        raise ValueError("beta and names must describe the same coefficient vector.")

    p = max(_extract_lag(name) for name in names)
    alpha = np.zeros(p)
    network_terms = []
    for value, name in zip(beta, names):
        lag = _extract_lag(name) - 1
        if name.startswith("own_"):
            alpha[lag] = value
        elif name.startswith("net_"):
            stage = int(name.rsplit("stage", 1)[1])
            network_terms.append((lag, stage, value))
        else:
            raise ValueError(f"Unrecognised design column {name!r}.")

    max_stage = max((stage for _, stage, _ in network_terms), default=0)
    if max_stage == 0:
        radius = float(np.max(np.abs(np.roots(np.r_[1.0, -alpha]))))
        return (radius, None) if return_vector else radius

    if max_stage == 1:
        stage1 = np.zeros(p)
        for lag, stage, value in network_terms:
            if stage != 1:
                raise ValueError("Stage indexing is inconsistent.")
            stage1[lag] = value
        radius = global_gnar_radius(alpha, stage1, W)
        return (radius, None) if return_vector else radius

    return companion_radius(
        gnar_lag_matrices(beta, names, W),
        method=method, v0=v0, return_vector=return_vector,
    )


def posterior_gnar_radii(beta_draws, names, W):
    """Return spectral radii for every retained draw of a global GNAR."""
    draws = np.asarray(beta_draws, dtype=float)
    names = list(names)
    if draws.ndim == 2:
        draws = draws[None, ...]
    if draws.ndim != 3 or draws.shape[-1] != len(names):
        raise ValueError("beta_draws must have shape (chain, draw, coefficient).")

    max_stage = max(
        (int(name.rsplit("stage", 1)[1]) for name in names if name.startswith("net_")),
        default=0,
    )
    radii = np.empty(draws.shape[:2], dtype=float)

    if max_stage <= 1:
        for chain in range(draws.shape[0]):
            for draw in range(draws.shape[1]):
                radii[chain, draw] = gnar_radius(draws[chain, draw], names, W)
        return radii.ravel()

    dim = _as_square_array(W).shape[0] * max(_extract_lag(name) for name in names)
    initial = np.ones(dim, dtype=float)
    initial /= np.linalg.norm(initial)

    for chain in range(draws.shape[0]):
        v0 = None
        for draw in range(draws.shape[1]):
            # Periodic deterministic restarts reduce dependence on a single
            # Arnoldi start while retaining warm starts between nearby draws.
            if draw % 100 == 0:
                v0 = None
            radius, vector = gnar_radius(
                draws[chain, draw], names, W,
                method="sparse", v0=v0, return_vector=True,
            )
            radii[chain, draw] = radius
            v0 = initial if vector is None else vector
    return radii.ravel()


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
def _post_means(idata):
    """Posterior means of all model parameters as NumPy arrays."""
    post = idata.posterior
    return {name: post[name].mean(dim=("chain", "draw")).values for name in post.data_vars}


def _prepare_forecast_inputs(Y_hist, Y_future, W, p, stages, mode):
    """Validate inputs and precompute stage weights shared across forecast steps."""
    Y_hist = np.asarray(Y_hist, dtype=float)
    Y_future = None if Y_future is None else np.asarray(Y_future, dtype=float)
    _validate_gnar_spec(p, stages)
    mode = _canonical_mode(mode)
    stage_weights = compute_stage_weights(W, max(stages) if mode != "ar_only" else 0)
    return Y_hist, Y_future, mode, stage_weights


def _conditional_mean(Xr, beta):
    """Conditional mean for a design block, handling shared or per-country beta."""
    # A 2-D beta is the local-alpha case (one coefficient vector per country).
    return np.einsum("ik,ik->i", Xr, beta) if np.ndim(beta) == 2 else Xr @ beta


def _systematic_resample(rng, weights):
    """Systematic resampling of particle indices from a weight vector."""
    # One uniform jitter, then equally spaced pointers into the weight CDF.
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _forecast_online_core(idata, Y_hist, Y_future, W, p, stages, mode, n_particles, seed):
    """One-step online forecasts with plug-in coefficients and a filtered SV path.

    Coefficients are fixed at their posterior means; only the log-variance is
    filtered forward with a particle filter, reweighting on each realised residual.
    Returns (means, variances) each shaped (n_steps, N).
    """
    rng = np.random.default_rng(seed)

    # Plug in posterior-mean coefficients and volatility parameters.
    pm_ = _post_means(idata)
    beta = pm_["beta"]
    m_h, phi, sigma_h = float(pm_["m_h"]), float(pm_["phi"]), float(pm_["sigma_h"])
    h_last = float(pm_["h"][-1])
    Y_hist, Y_future, mode, stage_weights = _prepare_forecast_inputs(Y_hist, Y_future, W, p, stages, mode)
    N, hist = Y_hist.shape[1], list(Y_hist)

    # Seed the particle cloud at the last filtered log-variance; the AR(1)
    # transition below advances it exactly once per forecast step.
    particles = np.full(n_particles, h_last, dtype=float)
    weights = np.full(n_particles, 1.0 / n_particles)
    means, variances = [], []

    for step, y_real in enumerate(Y_future):
        # Rebuild the conditional mean from the current (growing) history.
        H = np.asarray(hist)
        Xr = build_forecast_features(H, W, p=p, stages=stages, mode=mode, stage_weights=stage_weights)
        mean_vec = _conditional_mean(Xr, beta)

        # Advance every particle one AR(1) step and form the predictive variance
        # as the weighted particle average of exp(h).
        particles = m_h + phi * (particles - m_h) + sigma_h * rng.standard_normal(n_particles)
        sig2 = np.exp(particles)
        pred_var = float(np.average(sig2, weights=weights))
        means.append(mean_vec)
        variances.append(np.full(N, pred_var))

        # Reweight on the realised residual (shifted for numerical stability).
        resid = y_real - mean_vec
        loglik = -0.5 * (N * np.log(2 * np.pi * sig2) + float(np.sum(resid**2)) / sig2)
        loglik -= loglik.max()
        new_weights = weights * np.exp(loglik)

        # Normalise and fail loudly if the particle weights degenerate numerically.
        total = new_weights.sum()
        if total <= 0 or not np.isfinite(total):
            raise FloatingPointError("Particle-filter weights degenerated during online forecasting.")
        weights = new_weights / total

        # Systematic-resample the particles, then reset to uniform weights.
        particles = particles[_systematic_resample(rng, weights)]
        weights.fill(1.0 / n_particles)
        hist.append(y_real)
    return np.asarray(means), np.asarray(variances)


def forecast_online(idata, Y_hist, Y_future, W, p, stages,
                    mode="global_gnar", n_particles=2000, seed=42):
    """One-step online forecasts with plug-in coefficients and filtered volatility."""
    return _forecast_online_core(idata, Y_hist, Y_future, W, p, stages, mode, n_particles, seed)


def forecast_online_local_alpha(idata, Y_hist, Y_future, W, p, stages,
                                mode="local_alpha_gnar", n_particles=2000, seed=42):
    """Online forecaster for the local-alpha model, where beta is N x K."""
    return _forecast_online_core(idata, Y_hist, Y_future, W, p, stages, mode, n_particles, seed)


def forecast_blind(idata, Y_hist, n_steps, W, p, stages, mode="global_gnar"):
    """Recursive multi-step forecasts with no intermediate observations.

    Each step feeds its own point forecast back in as history and propagates the
    log-variance analytically, so intervals widen with the horizon.
    """
    pm_ = _post_means(idata)
    beta = pm_["beta"]
    m_h, phi, sigma_h = float(pm_["m_h"]), float(pm_["phi"]), float(pm_["sigma_h"])
    h_last = float(pm_["h"][-1])
    Y_hist, _, mode, stage_weights = _prepare_forecast_inputs(Y_hist, None, W, p, stages, mode)
    N, hist = Y_hist.shape[1], list(Y_hist)

    # Track the log-variance mean and variance so intervals grow with the horizon.
    means, variances, h_mean, h_var = [], [], h_last, 0.0
    for _ in range(n_steps):
        H = np.asarray(hist)
        Xr = build_forecast_features(H, W, p=p, stages=stages, mode=mode, stage_weights=stage_weights)
        mean_vec = _conditional_mean(Xr, beta)

        # Propagate the AR(1) log-variance moments one step forward.
        h_mean = m_h + phi * (h_mean - m_h)
        h_var = phi**2 * h_var + sigma_h**2
        means.append(mean_vec)
        variances.append(np.full(N, float(np.exp(h_mean + 0.5 * h_var))))

        # Feed the point forecast back in as pseudo-history.
        hist.append(mean_vec)
    return np.asarray(means), np.asarray(variances)


def forecast_noupdate(idata, Y_hist, Y_future, W, p, stages, mode="global_gnar"):
    """One-step forecasts on realised history without online volatility reweighting."""
    pm_ = _post_means(idata)
    beta = pm_["beta"]
    m_h, phi, sigma_h = float(pm_["m_h"]), float(pm_["phi"]), float(pm_["sigma_h"])
    h_last = float(pm_["h"][-1])
    Y_hist, Y_future, mode, stage_weights = _prepare_forecast_inputs(Y_hist, Y_future, W, p, stages, mode)
    N, hist = Y_hist.shape[1], list(Y_hist)

    # Realised history is fed in each step, but the volatility is only propagated,
    # never reweighted on the residual.
    means, variances, h_mean, h_var = [], [], h_last, 0.0
    for y_real in Y_future:
        H = np.asarray(hist)
        Xr = build_forecast_features(H, W, p=p, stages=stages, mode=mode, stage_weights=stage_weights)
        mean_vec = _conditional_mean(Xr, beta)
        h_mean = m_h + phi * (h_mean - m_h)
        h_var = phi**2 * h_var + sigma_h**2
        means.append(mean_vec)
        variances.append(np.full(N, float(np.exp(h_mean + 0.5 * h_var))))
        hist.append(y_real)
    return np.asarray(means), np.asarray(variances)


def forecast_constant(idata, Y_hist, Y_future, W, p, stages, mode="global_gnar"):
    """One-step forecasts with a constant observation variance."""
    pm_ = _post_means(idata)
    beta, sigma = pm_["beta"], float(pm_["sigma"])
    Y_hist, Y_future, mode, stage_weights = _prepare_forecast_inputs(Y_hist, Y_future, W, p, stages, mode)
    N, hist = Y_hist.shape[1], list(Y_hist)
    means, variances = [], []
    for y_real in Y_future:
        H = np.asarray(hist)
        Xr = build_forecast_features(H, W, p=p, stages=stages, mode=mode, stage_weights=stage_weights)
        means.append(_conditional_mean(Xr, beta))
        variances.append(np.full(N, sigma**2))
        hist.append(y_real)
    return np.asarray(means), np.asarray(variances)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def crps_gaussian(y, mu, sigma):
    """Closed-form elementwise CRPS for a Gaussian predictive."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _crisis_mask(dates):
    """Boolean mask selecting the crisis window from a date index."""
    d = pd.to_datetime(dates)
    return (d >= pd.Timestamp(CRISIS_START)) & (d <= pd.Timestamp(CRISIS_END))


def _post2022_mask(dates):
    """Boolean mask selecting the post-2022 surge period from a date index."""
    d = pd.to_datetime(dates)
    return np.asarray(d >= pd.Timestamp(POST2022_START))


def score_forecasts(y_true, mean, var, dates=None):
    """RMSE, MAE, CRPS, 95% coverage and mean interval width for a forecast block.

    If dates are supplied, the same metrics are also computed on the calm and
    crisis sub-windows and on the periods either side of the start of 2022.
    Inputs are shaped (n_steps, N).
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)
    if y_true.ndim != 2 or mean.shape != y_true.shape or var.shape != y_true.shape:
        raise ValueError("y_true, mean and var must have identical (n_steps, N) shapes.")
    if dates is not None and len(dates) != y_true.shape[0]:
        raise ValueError("dates must have one entry per forecast step.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError("Forecast scoring inputs contain non-finite values.")
    if np.any(var <= 0):
        raise ValueError("Forecast variances must be strictly positive.")
    sd = np.sqrt(var)

    def _block(yt, mn, s):
        # Point-error metrics.
        err = yt - mn
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        # Distributional metric and 95% interval coverage/width.
        crps = float(np.mean(crps_gaussian(yt, mn, s)))
        lo, hi = mn - 1.96 * s, mn + 1.96 * s
        cov = float(np.mean((yt >= lo) & (yt <= hi)))
        width = float(np.mean(hi - lo))
        return dict(RMSE=rmse, MAE=mae, CRPS=crps, Coverage_95=cov, Width=width)

    # Whole-window metrics, then the calm/crisis and pre/post-2022 splits.
    out = _block(y_true, mean, sd)
    if dates is not None:
        cm = _crisis_mask(dates)
        if cm.any():
            out["crisis"] = _block(y_true[cm], mean[cm], sd[cm])
        if (~cm).any():
            out["calm"] = _block(y_true[~cm], mean[~cm], sd[~cm])
        # The surge split answers a different question from the COVID split, so the two
        # partitions are reported side by side rather than nested.
        sm = _post2022_mask(dates)
        if sm.any():
            out["post2022"] = _block(y_true[sm], mean[sm], sd[sm])
        if (~sm).any():
            out["pre2022"] = _block(y_true[~sm], mean[~sm], sd[~sm])
    return out


def crps_series(y_true, mean, var):
    """Country-averaged CRPS per time step, giving one loss per time point.

    This per-step series is the correct unit for a Diebold-Mariano test.
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)
    if y_true.ndim != 2 or mean.shape != y_true.shape or var.shape != y_true.shape:
        raise ValueError("y_true, mean and var must have identical (n_steps, N) shapes.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError("CRPS inputs contain non-finite values.")
    if np.any(var <= 0):
        raise ValueError("Forecast variances must be strictly positive.")
    c = crps_gaussian(y_true, mean, np.sqrt(var))
    return c.mean(axis=1)


def dm_test(loss_a, loss_b, h_dm=1):
    """Diebold-Mariano test with a HAC variance and the Harvey-Leybourne-Newbold
    small-sample correction.

    Operates on two loss series. A positive statistic means model A has the higher
    loss. Returns (statistic, two-sided p-value) against a t reference with n-1
    degrees of freedom.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    dbar = d.mean()
    z = d - dbar

    # HAC (Newey-West) long-run variance with a bandwidth of h_dm-1 lags.
    gamma0 = np.sum(z**2) / n
    lrv = gamma0
    for k in range(1, h_dm):
        gamma = np.sum(z[k:] * z[:-k]) / n
        lrv += 2 * (1 - k / h_dm) * gamma

    var = lrv / n
    if var <= 0:
        return np.nan, np.nan

    # Standardise, then apply the Harvey-Leybourne-Newbold finite-sample correction.
    stat = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h_dm + h_dm * (h_dm - 1) / n) / n)
    stat = stat * corr
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


def stars(p):
    """Significance stars for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def run_config(*, p=None, stages=None, **specification):
    """Return explicit model and prior metadata for a saved result."""
    out = dict(
        run_version=RUN_VERSION,
        rw_centre=MINN["rw_centre"],
        lambda1=MINN["lambda1"],
        lambda2=MINN["lambda2"],
        lambda3=MINN["lambda3"],
        split=SPLIT_DATE,
    )
    if p is not None:
        out["p"] = int(p)
    if stages is not None:
        out["stages"] = [int(s) for s in stages]
    out.update(specification)
    return out


def save_forecasts(name, y_true, mean, var, dates=None, config=None):
    """Persist a forecast bundle to outputs/ as a compressed .npz."""
    if config is None:
        raise ValueError("Forecast bundles require an explicit config.")

    path = OUTPUTS / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    var = np.asarray(var)
    if y_true.ndim != 2 or mean.shape != y_true.shape or var.shape != y_true.shape:
        raise ValueError(f"{name}: y_true, mean and var must have identical 2-D shapes.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError(f"{name}: forecast bundle contains non-finite values.")
    if np.any(var <= 0):
        raise ValueError(f"{name}: forecast variances must be strictly positive.")
    if dates is not None and len(dates) != y_true.shape[0]:
        raise ValueError(f"{name}: dates do not match the forecast length.")

    kw = dict(y_true=y_true, mean=mean, var=var)
    if dates is not None:
        kw["dates"] = np.asarray(pd.to_datetime(dates).astype("datetime64[ns]"))
    kw["config"] = np.asarray(json.dumps(config))
    np.savez_compressed(path, **kw)
    return path


def load_forecasts(name, expect_config=None):
    """Load a forecast bundle and optionally compare its explicit config."""
    path = OUTPUTS / f"{name}.npz"
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "dates" in out:
        out["dates"] = pd.to_datetime(out["dates"])

    if "config" in out:
        try:
            out["config"] = json.loads(str(np.asarray(out["config"]).item()))
            if expect_config is not None and out["config"] != expect_config:
                warnings.warn(
                    f"{name}.npz config {out['config']} != expected {expect_config}."
                )
        except Exception:
            pass
    elif expect_config is not None:
        warnings.warn(f"{name}.npz has no config stamp.")
    return out


def save_result(name, obj):
    """Persist a JSON-serialisable result dict to results/."""
    path = RESULTS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_result(name):
    """Load a result dict from results/."""
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


def _json_default(o):
    """Fallback JSON encoder for NumPy and pandas scalar/array types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not serialisable: {type(o)}")


# ==========================================================================
# Conjugate Bayesian GNAR
# ==========================================================================
class BayesianGNAR:
    """Conjugate Bayesian GNAR(p, [s]) with a Normal-Inverse-Gamma prior.

    Conjugacy means the coefficient and error-variance posteriors, the predictive
    distribution, and the model evidence are all closed-form, so no sampling is
    needed. This makes the model cheap enough for the selection sweeps and the
    simulation grids that call it thousands of times. prior_type sets the prior;
    "minnesota" applies the structured, random-walk-centred shrinkage of the report.
    """

    def __init__(self, p, s, local_alpha=False, prior_type="weakly_informative",
                 lambda1=0.2, lambda2=0.5, lambda3=1.0, rw_centre=1.0):
        self.p = p
        self.s = s
        assert len(s) == p
        self.local_alpha = local_alpha
        self.prior_type = prior_type
        # Minnesota hyperparameters (ignored by the zero-centred priors).
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.rw_centre = rw_centre
        self.fitted = False

    def _build_design(self, Y, stage_weights):
        """Build the stacked design X and response y, recording per-column metadata.

        self.col_info stores, for each column, whether it is an own-lag or network
        term and at which lag/stage, so the Minnesota prior can place a different
        mean and variance on each coefficient. The ordering matches the feature
        construction exactly: own-lag block first, then network block.
        """
        T, N = Y.shape
        self.n_nodes = N

        # Precompute the stage-weighted neighbour panels once per stage.
        Z_stage = []
        for r_idx in range(max(self.s)):
            if r_idx < len(stage_weights):
                Wr = stage_weights[r_idx]
                Z = Y @ Wr.T
            else:
                Z = np.zeros_like(Y)
            Z_stage.append(Z)

        # Record the column layout once; it is identical for every observation row.
        # Own-lag block first (per country under local-alpha), then the network block.
        col_info = []
        if self.local_alpha:
            for j in range(self.p):
                for node in range(N):
                    col_info.append({"type": "ar", "lag": j + 1,
                                     "node": node, "stage": None})
        else:
            for j in range(self.p):
                col_info.append({"type": "ar", "lag": j + 1,
                                 "node": None, "stage": None})
        for j in range(self.p):
            for r in range(self.s[j]):
                col_info.append({"type": "net", "lag": j + 1,
                                 "node": None, "stage": r + 1})
        self.col_info = col_info

        rows_X = []
        rows_y = []
        rows_info = []
        t_start = self.p

        # Build one design row per (time, country), skipping any with missing lags.
        for t in range(t_start, T):
            for i in range(N):
                if np.isnan(Y[t, i]):
                    continue
                features = []

                # Own-lag block: a per-country one-hot under local-alpha, else scalars.
                if self.local_alpha:
                    for j in range(self.p):
                        ar_block = np.zeros(N)
                        val = Y[t - j - 1, i]
                        if np.isnan(val):
                            break
                        ar_block[i] = val
                        features.extend(ar_block)
                    if len(features) != self.p * N:
                        continue
                else:
                    skip = False
                    for j in range(self.p):
                        val = Y[t - j - 1, i]
                        if np.isnan(val):
                            skip = True
                            break
                        features.append(val)
                    if skip:
                        continue

                # Network block: stage-weighted neighbour values at each lag.
                skip = False
                for j in range(self.p):
                    for r in range(self.s[j]):
                        val = Z_stage[r][t - j - 1, i]
                        if np.isnan(val):
                            skip = True
                            break
                        features.append(val)
                    if skip:
                        break
                if skip:
                    continue

                # A complete row: store the features, target and (country, time) tag.
                rows_X.append(features)
                rows_y.append(Y[t, i])
                rows_info.append((i, t))

        return np.array(rows_X), np.array(rows_y), rows_info

    def _set_prior(self, k):
        """Set the prior hyperparameters for the chosen prior_type.

        The zero-centred priors ("weakly_informative", "diffuse", "shrinkage")
        differ only in how wide the coefficient prior is; "minnesota" uses the
        structured mean and precision built in _set_minnesota_prior. All remain
        Normal-Inverse-Gamma, so the closed-form posterior is unchanged.
        """
        if self.prior_type == "weakly_informative":
            self.mu_0 = np.zeros(k)
            self.Lambda_0 = np.eye(k) * 0.01   # weak precision, wide prior
            self.a_0 = 3.0
            self.b_0 = 1.0
        elif self.prior_type == "diffuse":
            self.mu_0 = np.zeros(k)
            self.Lambda_0 = np.eye(k) * 1e-6   # near-zero precision, very wide
            self.a_0 = 0.01
            self.b_0 = 0.01
        elif self.prior_type == "shrinkage":
            self.mu_0 = np.zeros(k)
            self.Lambda_0 = np.eye(k) * 1.0    # tighter prior, stronger shrinkage
            self.a_0 = 3.0
            self.b_0 = 1.0
        elif self.prior_type == "minnesota":
            self._set_minnesota_prior(k)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def _set_minnesota_prior(self, k):
        """Structured Litterman-Minnesota prior in Normal-Inverse-Gamma form.

        The own lag-1 coefficients are centred at rw_centre and everything else at
        zero. Variances decay with lag, and network columns are shrunk harder than
        own-lag columns by lambda2. The cross-series scale term is omitted because
        the series are comparably scaled and the network regressors row-normalised.
        Requires self.col_info, so _build_design must run first.
        """
        assert hasattr(self, "col_info") and len(self.col_info) == k, \
            "col_info missing or wrong length; _build_design must run first."

        mu_0 = np.zeros(k)
        prior_var = np.zeros(k)

        # Per-column mean and variance, keyed on own-vs-network and lag.
        for idx, info in enumerate(self.col_info):
            j = info["lag"]
            decay = j ** self.lambda3
            if info["type"] == "ar":
                if j == 1:
                    mu_0[idx] = self.rw_centre
                prior_var[idx] = (self.lambda1 / decay) ** 2
            else:
                prior_var[idx] = (self.lambda1 * self.lambda2 / decay) ** 2

        # Convert variances to a diagonal precision, guarding any zero variance.
        prior_var = np.where(prior_var > 0, prior_var, 1e-12)
        self.mu_0 = mu_0
        self.Lambda_0 = np.diag(1.0 / prior_var)
        self.a_0 = 3.0
        self.b_0 = 1.0

    def fit(self, Y, stage_weights):
        """Compute the closed-form Normal-Inverse-Gamma posterior."""
        self.stage_weights = stage_weights
        X, y, info = self._build_design(Y, stage_weights)
        self.X_train = X
        self.y_train = y
        self.train_info = info

        n_obs, k = X.shape
        self._set_prior(k)

        # Posterior precision, its inverse, and posterior mean.
        self.Lambda_n = X.T @ X + self.Lambda_0
        self.Lambda_n_inv = np.linalg.inv(self.Lambda_n)
        self.mu_n = self.Lambda_n_inv @ (X.T @ y + self.Lambda_0 @ self.mu_0)

        # Posterior shape and scale for the error variance. The mu_0 term is active
        # under the Minnesota prior; the positivity check below catches a sign slip.
        self.a_n = self.a_0 + n_obs
        self.b_n = (self.b_0
                    + y @ y
                    + self.mu_0 @ self.Lambda_0 @ self.mu_0
                    - self.mu_n @ self.Lambda_n @ self.mu_n)

        if not np.isfinite(self.b_n) or self.b_n <= 0:
            raise FloatingPointError(
                f"Invalid posterior scale b_n={self.b_n!r} for prior_type={self.prior_type!r}."
            )

        self.sigma2_post_mean = self.b_n / (self.a_n - 2) if self.a_n > 2 else self.b_n / self.a_n

        # The marginal coefficient posterior is multivariate-t with a_n df.
        self.theta_post_mean = self.mu_n
        self.theta_post_scale = (self.b_n / self.a_n) * self.Lambda_n_inv

        # Training R-squared, for diagnostics only.
        y_hat = X @ self.mu_n
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot

        self.n_obs = n_obs
        self.k = k
        self.fitted = True
        return self

    def fit_design(self, X, y, names, n_nodes):
        """Fit the conjugate global-alpha model to an externally constructed design."""
        if self.local_alpha:
            raise ValueError("fit_design supports global-alpha models only.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y):
            raise ValueError("X and y have incompatible shapes.")
        if len(names) != X.shape[1]:
            raise ValueError("names must match the columns of X.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Conjugate design contains non-finite values.")

        self.n_nodes = int(n_nodes)
        self.names = list(names)
        self.col_info = []
        for name in names:
            lag = _extract_lag(name)
            if name.startswith("own_"):
                self.col_info.append({"type": "ar", "lag": lag,
                                      "node": None, "stage": None})
            elif name.startswith("net_"):
                stage = int(name.rsplit("stage", 1)[1])
                self.col_info.append({"type": "net", "lag": lag,
                                      "node": None, "stage": stage})
            else:
                raise ValueError(f"Unrecognised design column {name!r}.")

        n_obs, k = X.shape
        self._set_prior(k)
        self.Lambda_n = X.T @ X + self.Lambda_0
        self.Lambda_n_inv = np.linalg.inv(self.Lambda_n)
        self.mu_n = self.Lambda_n_inv @ (X.T @ y + self.Lambda_0 @ self.mu_0)
        self.a_n = self.a_0 + n_obs
        self.b_n = (self.b_0 + y @ y + self.mu_0 @ self.Lambda_0 @ self.mu_0
                    - self.mu_n @ self.Lambda_n @ self.mu_n)
        if not np.isfinite(self.b_n) or self.b_n <= 0:
            raise FloatingPointError(f"Invalid posterior scale b_n={self.b_n!r}.")

        self.sigma2_post_mean = self.b_n / (self.a_n - 2) if self.a_n > 2 else self.b_n / self.a_n
        self.theta_post_mean = self.mu_n
        self.theta_post_scale = (self.b_n / self.a_n) * self.Lambda_n_inv
        self.X_train = X
        self.y_train = y
        self.train_info = None
        self.n_obs = n_obs
        self.k = k

        y_hat = X @ self.mu_n
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        self.fitted = True
        return self

    def predict_from_design(self, X):
        """Return Student-t predictive means and variances for supplied design rows."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if self.a_n <= 2:
            raise ValueError("Predictive variance is undefined for a_n <= 2.")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.k:
            raise ValueError("Predictive design has the wrong number of columns.")
        means = X @ self.mu_n
        scale2 = (self.b_n / self.a_n) * (
            1.0 + np.einsum("ij,jk,ik->i", X, self.Lambda_n_inv, X)
        )
        variances = scale2 * self.a_n / (self.a_n - 2)
        return means, variances

    def log_marginal_likelihood(self):
        """Log model evidence under the Normal-Inverse-Gamma model, for averaging."""
        n = self.n_obs
        sign_0, logdet_0 = np.linalg.slogdet(self.Lambda_0)
        sign_n, logdet_n = np.linalg.slogdet(self.Lambda_n)
        if sign_0 <= 0 or sign_n <= 0:
            raise FloatingPointError("Conjugate precision matrix is not positive definite.")

        log_ml = (
            - n / 2 * np.log(np.pi)
            + 0.5 * logdet_0
            - 0.5 * logdet_n
            + (self.a_0 / 2) * np.log(self.b_0)
            - (self.a_n / 2) * np.log(self.b_n)
            + gammaln(self.a_n / 2)
            - gammaln(self.a_0 / 2)
        )
        return log_ml

    def predict_one_step(self, Y_full, t):
        """Posterior predictive mean and variance for all countries at time t.

        Uses data up to t-1. The predictive is Student-t; the returned variance is
        that of the t distribution.
        """
        N = Y_full.shape[1]

        # Stage-weighted neighbour panels, as in _build_design.
        Z_stage = []
        for r_idx in range(max(self.s)):
            if r_idx < len(self.stage_weights):
                Wr = self.stage_weights[r_idx]
                Z = Y_full @ Wr.T
            else:
                Z = np.zeros_like(Y_full)
            Z_stage.append(Z)

        means = np.full(N, np.nan)
        variances = np.full(N, np.nan)

        # Assemble each country's feature row exactly as at training time.
        for i in range(N):
            features = []
            if self.local_alpha:
                for j in range(self.p):
                    ar_block = np.zeros(N)
                    val = Y_full[t - j - 1, i]
                    if np.isnan(val):
                        break
                    ar_block[i] = val
                    features.extend(ar_block)
                if len(features) != self.p * N:
                    continue
            else:
                skip = False
                for j in range(self.p):
                    val = Y_full[t - j - 1, i]
                    if np.isnan(val):
                        skip = True
                        break
                    features.append(val)
                if skip:
                    continue

            skip = False
            for j in range(self.p):
                for r in range(self.s[j]):
                    val = Z_stage[r][t - j - 1, i]
                    if np.isnan(val):
                        skip = True
                        break
                    features.append(val)
                if skip:
                    break
            if skip:
                continue

            # Student-t predictive mean and variance for this country.
            x = np.array(features)
            means[i] = x @ self.mu_n
            pred_var = (self.b_n / self.a_n) * (1 + x @ self.Lambda_n_inv @ x)
            variances[i] = pred_var * self.a_n / (self.a_n - 2)

        return means, variances

    def forecast_test(self, Y_full, test_start, n_test):
        """One-step-ahead point and interval forecasts across the test window."""
        fc_means = np.full((n_test, self.n_nodes), np.nan)
        fc_vars = np.full((n_test, self.n_nodes), np.nan)
        for i in range(n_test):
            t = test_start + i
            if t < self.p:
                continue
            m, v = self.predict_one_step(Y_full, t)
            fc_means[i, :] = m
            fc_vars[i, :] = v
        return fc_means, fc_vars

    def summary(self):
        """One-line summary string of the fitted model."""
        n_ar = self.p * self.n_nodes if self.local_alpha else self.p
        n_net = sum(self.s)
        n_total = n_ar + n_net
        alpha_type = "local" if self.local_alpha else "global"
        return (f"Bayesian GNAR({self.p}, {self.s}) [{alpha_type}-alpha, {self.prior_type}] | "
                f"params: {n_total} | R2: {self.r_squared:.5f} | "
                f"sigma2_post: {self.sigma2_post_mean:.5f} | "
                f"log ML: {self.log_marginal_likelihood():.5g}")


# ==========================================================================
# RaGNAR-style random-network search for the conjugate Bayesian GNAR.
# The stage weights are neighbour averages (row-normalised), the correct builder
# for the unweighted random graphs generated here.
# ==========================================================================
def erdos_renyi_graph(N, pi, rng):
    """Undirected Erdos-Renyi 0/1 adjacency on N nodes with edge probability pi."""
    # Draw the upper triangle, symmetrise, and clear the diagonal.
    U = rng.random((N, N))
    A = np.triu((U < pi).astype(float), 1)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    return A


def ragnar_stage_weights(A, max_stage=2):
    """Row-normalised r-stage neighbour weights from a 0/1 adjacency.

    Stage 1 is the row-normalised adjacency, so each neighbour term is an average
    over immediate neighbours. Stage 2 covers nodes reachable in exactly two hops.
    Isolated nodes keep all-zero rows and contribute nothing.
    """
    A = (np.asarray(A) > 0).astype(float)
    N = A.shape[0]
    np.fill_diagonal(A, 0.0)

    def rn(M):
        rs = M.sum(1, keepdims=True)
        rs[rs == 0] = 1.0
        return M / rs

    # Stage 1: direct neighbours.
    stages = [rn(A.copy())]

    # Stage 2: nodes exactly two hops away (reachable via A@A but not already adjacent).
    if max_stage >= 2:
        reached = (np.eye(N) + A) > 0
        two = ((A @ A) > 0) & ~reached
        np.fill_diagonal(two, False)
        stages.append(rn(two.astype(float)))
    return stages


def uniform_factor_weights(N, max_stage=2):
    """Stage-1 weights equal to the uniform average over all other nodes.

    Feeding this to BayesianGNAR yields an AR-plus-common-factor model, the
    baseline that isolates genuine spillover from factor-proxying.
    """
    W = (np.ones((N, N)) - np.eye(N)) / (N - 1)
    stages = [W]
    if max_stage >= 2:
        stages.append(np.zeros((N, N)))
    return stages


def panel_crps(Y_true, means, variances):
    """Mean CRPS over an (n_test, N) forecast block, ignoring NaNs."""
    sd = np.sqrt(np.maximum(variances, 1e-12))
    return float(np.nanmean(crps_gaussian(Y_true, means, sd)))


def _fit_forecast_crps(Y, test_start, block_start, n_block, sw, p, s, prior_kwargs,
                       BayesianGNARClass):
    """Fit on Y[:test_start], forecast a block, and return (means, vars, panel CRPS)."""
    la = prior_kwargs.get("local_alpha", False)
    rest = {k: v for k, v in prior_kwargs.items() if k != "local_alpha"}
    m = BayesianGNARClass(p=p, s=s, local_alpha=la, **rest)
    m.fit(Y[:test_start], sw)
    fm, fv = m.forecast_test(Y, block_start, n_block)
    c = panel_crps(Y[block_start:block_start + n_block], fm, fv)
    return fm, fv, c


def ragnar_search(Y_full, test_start, n_val, n_eval, *, p, s, prior_kwargs,
                  n_graphs, pi, top_n, n_reps, BayesianGNARClass,
                  base_seed=0, max_stage=2, verbose=True):
    """Single-split RaGNAR random-network search on the conjugate Bayesian GNAR.

    Each replication generates n_graphs Erdos-Renyi graphs, ranks them by panel
    CRPS on a validation block, and averages the top_n graphs' forecasts on a
    disjoint evaluation block. Selection and evaluation windows never overlap.
    Reports the averaged evaluation CRPS against two baselines on the same block:
    AR-only and AR-plus-common-factor.
    """
    Y = np.asarray(Y_full, dtype=float)
    N = Y.shape[1]

    # Validation block immediately follows training; evaluation block follows that,
    # so graph selection never sees the data it is finally scored on.
    val_start = test_start
    eval_start = test_start + n_val

    # Two fixed baselines evaluated once on the evaluation block.
    _, _, ar_crps = _fit_forecast_crps(
        Y, test_start, eval_start, n_eval,
        ragnar_stage_weights(np.zeros((N, N)), max_stage),
        p, [0] * p, prior_kwargs, BayesianGNARClass)
    _, _, factor_crps = _fit_forecast_crps(
        Y, test_start, eval_start, n_eval,
        uniform_factor_weights(N, max_stage),
        p, s, prior_kwargs, BayesianGNARClass)

    reps = []
    for r in range(n_reps):
        rng = np.random.default_rng(base_seed + r)

        # Score every candidate graph on the validation block.
        scored = []
        for _ in range(n_graphs):
            A = erdos_renyi_graph(N, pi, rng)
            sw = ragnar_stage_weights(A, max_stage)
            try:
                _, _, val = _fit_forecast_crps(Y, test_start, val_start, n_val,
                                               sw, p, s, prior_kwargs, BayesianGNARClass)
            except Exception:
                val = np.inf
            scored.append((val, A))

        # Keep the top graphs by validation CRPS, rejecting failed candidates.
        scored.sort(key=lambda t: t[0])
        finite = [(loss, A) for loss, A in scored if np.isfinite(loss)]
        if len(finite) < top_n:
            raise RuntimeError(f"Only {len(finite)} finite candidate graphs; need {top_n}.")
        top = finite[:top_n]

        # Average the top graphs' forecasts, adding the between-graph spread to the
        # variance so the ensemble predictive is not overconfident.
        ems, evs = [], []
        for _, A in top:
            sw = ragnar_stage_weights(A, max_stage)
            em, ev, _ = _fit_forecast_crps(Y, test_start, eval_start, n_eval,
                                           sw, p, s, prior_kwargs, BayesianGNARClass)
            if not np.all(np.isfinite(em)) or not np.all(np.isfinite(ev)):
                raise RuntimeError("A selected RaGNAR graph produced non-finite forecasts.")
            ems.append(em); evs.append(ev)
        avg_m = np.mean(ems, axis=0)
        avg_v = np.mean(evs, axis=0) + np.mean([(e - avg_m) ** 2 for e in ems], axis=0)
        top_crps = panel_crps(Y[eval_start:eval_start + n_eval], avg_m, avg_v)

        reps.append(dict(top_crps=top_crps, best_val=top[0][0]))
        if verbose:
            print(f"rep {r:2d}: top-{top_n} eval CRPS={top_crps:.5f}  "
                  f"(AR-only {ar_crps:.5f}, AR+factor {factor_crps:.5f})")

    # Aggregate the replications and report improvements over each baseline.
    ts = np.array([x["top_crps"] for x in reps])
    return dict(
        ar_crps=ar_crps, factor_crps=factor_crps,
        top_crps_mean=float(ts.mean()), top_crps_std=float(ts.std()),
        top_crps_all=ts.tolist(),
        improvement_vs_ar=float(ar_crps - ts.mean()),
        improvement_vs_factor=float(factor_crps - ts.mean()),
        reps=reps)
