"""
Distribution comparison plots, following R's ``fitdistrplus``.

Each plotting function mirrors one of the package's graphics:

===========================  =====================================
This module                  fitdistrplus
===========================  =====================================
``cullen_and_frey_plot``     ``descdist(..., graph = TRUE)``
``descdist``                 ``descdist(..., graph = FALSE)``
``quantile_comparison_plot`` ``qqcomp`` + ``denscomp`` + ``ppcomp`` + ``cdfcomp``
===========================  =====================================

The plotting positions, parameter estimates, axis limits and theoretical
overlays follow the R implementations; where a choice had to be made, R's
behaviour was taken as the reference.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .distributions import (
    DISTRIBUTION_SPECS,
    SUPPORTED_DISTRIBUTIONS,
    density,
    fit_distribution,
    is_discrete,
    ppoints,
    resolve_distribution,
)

__all__ = [
    "SUPPORTED_DISTRIBUTIONS",
    "cullen_and_frey_plot",
    "descdist",
    "quantile_comparison_plot",
]

# R uses palette colours 2:(nft+1), i.e. red, green, blue, cyan, magenta, yellow,
# grey, then it recycles.  These are the plotly equivalents of that sequence.
DISTRIBUTION_COLORS = [
    "#DF536B",  # red
    "#61D04F",  # green
    "#2297E6",  # blue
    "#28E2E5",  # cyan
    "#CD0BBC",  # magenta
    "#F5C710",  # yellow
    "#9E9E9E",  # grey
    "#FF7F0E",
    "#8C564B",
    "#7F7F7F",
]

# R's descdist places these theoretical distributions at fixed (skewness^2,
# kurtosis) points.  Marker symbols approximate R's pch 8, 2, 7 and 3.
_THEORETICAL_POINTS = {
    "normal": (0.0, 3.0, "asterisk"),
    "uniform": (0.0, 9.0 / 5.0, "triangle-up-open"),
    "exponential": (4.0, 9.0, "square-open"),
    "logistic": (0.0, 4.2, "cross-thin"),
}

# R sweeps its theoretical curves over exp(seq(-100, 100, 0.1)).
_LOG_GRID = np.arange(-100.0, 100.0 + 1e-9, 0.1)


# ---------------------------------------------------------------------------
# Descriptive statistics -- descdist
# ---------------------------------------------------------------------------

def _moment(data: np.ndarray, k: int) -> float:
    """R's internal ``moment``: the k-th central moment divided by n."""
    return float(np.sum((data - np.mean(data)) ** k) / len(data))


def _skewness(data: np.ndarray, method: str) -> float:
    """Sample or unbiased (Fisher 1930) skewness, as ``descdist`` computes it."""
    sd = np.sqrt(_moment(data, 2))
    gamma1 = _moment(data, 3) / sd ** 3
    if method == "sample":
        return float(gamma1)
    n = len(data)
    return float(np.sqrt(n * (n - 1)) * gamma1 / (n - 2))


def _kurtosis(data: np.ndarray, method: str) -> float:
    """Sample or unbiased (Fisher 1930) kurtosis -- not excess kurtosis."""
    var = _moment(data, 2)
    gamma2 = _moment(data, 4) / var ** 2
    if method == "sample":
        return float(gamma2)
    n = len(data)
    return float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * gamma2 - 3 * (n - 1)) + 3)


def descdist(
    data: Union[pd.Series, np.ndarray, list],
    method: str = "unbiased",
) -> Dict[str, Any]:
    """Descriptive statistics of an empirical distribution.

    The non-graphical half of R's ``descdist``: it returns the same seven
    summaries, with skewness and kurtosis computed by the same estimators.

    Parameters
    ----------
    data : array-like
        Input data.  At least four values are required, as in R.
    method : {'unbiased', 'sample'}, default='unbiased'
        ``'unbiased'`` uses the Fisher (1930) corrections, R's default.

    Returns
    -------
    dict
        ``min``, ``max``, ``median``, ``mean``, ``sd``, ``skewness``,
        ``kurtosis`` and ``method``.  ``kurtosis`` is not excess kurtosis: a
        normal distribution gives 3.
    """
    if method not in ("unbiased", "sample"):
        raise ValueError("method must be 'unbiased' or 'sample'")
    clean = _validate_and_prepare_data(data, min_points=4)

    sd = float(np.std(clean, ddof=1)) if method == "unbiased" else float(np.sqrt(_moment(clean, 2)))
    return {
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "median": float(np.median(clean)),
        "mean": float(np.mean(clean)),
        "sd": sd,
        "skewness": _skewness(clean, method),
        "kurtosis": _kurtosis(clean, method),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Cullen and Frey graph -- descdist(graph = TRUE)
# ---------------------------------------------------------------------------

def cullen_and_frey_plot(
    data: Union[pd.Series, np.ndarray, list],
    title: str = "Cullen and Frey graph",
    data_name: str = "Data",
    discrete: bool = False,
    method: str = "unbiased",
    n_bootstrap: int = 100,
    show_bootstrap: bool = True,
    show_theoretical: bool = True,
    seed: Optional[int] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Cullen and Frey graph, as drawn by R's ``descdist``.

    Sample squared skewness is plotted against sample kurtosis, with the
    kurtosis axis inverted, against the regions and curves occupied by common
    distribution families.

    Parameters
    ----------
    data : array-like
        Input data; at least four values, as in R.
    title : str, default='Cullen and Frey graph'
        Plot title.
    data_name : str, default='Data'
        Legend label for the observed point.
    discrete : bool, default=False
        Draw the discrete overlays (negative binomial region and Poisson line)
        instead of the continuous ones (beta region, gamma and lognormal lines),
        matching R's ``discrete`` argument.
    method : {'unbiased', 'sample'}, default='unbiased'
        Moment estimator, as in R.
    n_bootstrap : int, default=100
        Bootstrap resamples, R's ``boot``.  R requires at least 10.
    show_bootstrap : bool, default=True
        Whether to draw the bootstrap cloud.  R's equivalent is passing
        ``boot = n_bootstrap`` rather than ``boot = NULL``; as in R, the
        bootstrap also widens the axis limits to cover the resampled points.
    show_theoretical : bool, default=True
        Whether to draw the theoretical points, curves and regions.
    seed : int, optional
        Seed for the bootstrap resampling.  Uses an isolated generator, so the
        global numpy random state is left untouched.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure

    See Also
    --------
    descdist : the summary statistics R's ``descdist`` returns.
    """
    clean = _validate_and_prepare_data(data, min_points=4)
    n = len(clean)

    skewdata = _skewness(clean, method)
    kurtdata = _kurtosis(clean, method)
    skew_sq = skewdata ** 2

    # Axis limits: R derives them from the bootstrap sample when there is one.
    boot_skew_sq = boot_kurtosis = None
    if show_bootstrap:
        if n_bootstrap is None or n_bootstrap < 10:
            raise ValueError("n_bootstrap must be an integer of at least 10")
        rng = np.random.default_rng(seed)
        resamples = rng.choice(clean, size=(n_bootstrap, n), replace=True)
        boot_skew_sq = np.array([_skewness(s, method) ** 2 for s in resamples])
        boot_kurtosis = np.array([_kurtosis(s, method) for s in resamples])
        kurtmax = max(10.0, float(np.ceil(np.max(boot_kurtosis))))
        xmax = max(4.0, float(np.ceil(np.max(boot_skew_sq))))
    else:
        kurtmax = max(10.0, float(np.ceil(kurtdata)))
        xmax = max(4.0, float(np.ceil(skew_sq)))

    # R plots kurtmax - kurtosis on a 0..kurtmax-1 axis and relabels the ticks,
    # which is the same picture as plotting kurtosis on a reversed axis.
    ymin, ymax = kurtmax, 1.0

    fig = go.Figure()

    if show_theoretical:
        if discrete:
            _add_negbin_region(fig, xmax)
            _add_poisson_curve(fig, xmax)
        else:
            _add_beta_region(fig, xmax, kurtmax)
            _add_gamma_curve(fig, xmax)
            _add_lognormal_curve(fig, xmax)

    if show_bootstrap:
        fig.add_trace(go.Scatter(
            x=boot_skew_sq,
            y=boot_kurtosis,
            mode="markers",
            name="bootstrap",
            marker=dict(size=5, color="orange", symbol="circle-open"),
            hovertemplate=(
                "<b>Bootstrap sample</b><br>"
                "Skewness²: %{x:.3f}<br>"
                "Kurtosis: %{y:.3f}<extra></extra>"
            ),
        ))

    if show_theoretical:
        # R draws the normal for both the discrete and the continuous case,
        # and the other three only when discrete = FALSE.
        for label, (px, py, symbol) in _THEORETICAL_POINTS.items():
            if discrete and label != "normal":
                continue
            fig.add_trace(go.Scatter(
                x=[px], y=[py],
                mode="markers",
                name=label,
                marker=dict(size=11, color="black", symbol=symbol,
                            line=dict(width=2, color="black")),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Skewness²: %{x:.3f}<br>"
                    "Kurtosis: %{y:.3f}<extra></extra>"
                ),
            ))

    fig.add_trace(go.Scatter(
        x=[skew_sq], y=[kurtdata],
        mode="markers",
        name=f"{data_name} (observed)",
        marker=dict(size=13, color="red", symbol="circle",
                    line=dict(width=1, color="darkred")),
        hovertemplate=(
            f"<b>{data_name}</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="square of skewness", showgrid=True,
                   gridcolor="lightgray", range=[0, xmax]),
        yaxis=dict(title="kurtosis", showgrid=True,
                   gridcolor="lightgray", range=[ymin, ymax]),
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
    )
    return fig


def _add_beta_region(fig: go.Figure, xmax: float, kurtmax: float) -> None:
    """Filled region occupied by the beta family.

    R traces the two boundary curves at shape1 = exp(-100) and exp(100), sweeping
    shape2 over exp(seq(-100, 100, 0.1)), and fills the polygon between them.
    """
    q = np.exp(_LOG_GRID)
    s2_parts, kurt_parts = [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for p in (np.exp(-100.0), np.exp(100.0)):
            s2 = (4 * (q - p) ** 2 * (p + q + 1)) / ((p + q + 2) ** 2 * p * q)
            kurt = (3 * (p + q + 1) * (p * q * (p + q - 6) + 2 * (p + q) ** 2)
                    / (p * q * (p + q + 2) * (p + q + 3)))
            s2_parts.append(s2)
            kurt_parts.append(kurt)

    s2 = np.concatenate(s2_parts)
    kurt = np.concatenate(kurt_parts)
    finite = np.isfinite(s2) & np.isfinite(kurt)

    fig.add_trace(go.Scatter(
        x=s2[finite], y=kurt[finite],
        mode="lines",
        fill="toself",
        name="beta",
        line=dict(color="lightgray", width=0),
        fillcolor="rgba(211,211,211,0.6)",
        hoverinfo="skip",
    ))


def _add_gamma_curve(fig: go.Figure, xmax: float) -> None:
    """Gamma family: skewness² = 4/shape, kurtosis = 3 + 6/shape."""
    with np.errstate(over="ignore", divide="ignore"):
        shape = np.exp(_LOG_GRID)
        s2 = 4.0 / shape
        kurt = 3.0 + 6.0 / shape
    keep = np.isfinite(s2) & (s2 <= xmax)  # R: lines(s2[s2 <= xmax], ...)
    fig.add_trace(go.Scatter(
        x=s2[keep], y=kurt[keep],
        mode="lines",
        name="gamma",
        line=dict(color="black", width=1.5, dash="dash"),
        hovertemplate=("<b>gamma</b><br>Skewness²: %{x:.3f}<br>"
                       "Kurtosis: %{y:.3f}<extra></extra>"),
    ))


def _add_lognormal_curve(fig: go.Figure, xmax: float) -> None:
    """Lognormal family, parameterised by the log-scale standard deviation."""
    with np.errstate(over="ignore", invalid="ignore"):
        shape = np.exp(_LOG_GRID)
        es2 = np.exp(shape ** 2)
        s2 = (es2 + 2) ** 2 * (es2 - 1)
        kurt = es2 ** 4 + 2 * es2 ** 3 + 3 * es2 ** 2 - 3
    keep = np.isfinite(s2) & np.isfinite(kurt) & (s2 <= xmax)
    fig.add_trace(go.Scatter(
        x=s2[keep], y=kurt[keep],
        mode="lines",
        name="lognormal",
        line=dict(color="black", width=1.5, dash="dot"),
        hovertemplate=("<b>lognormal</b><br>Skewness²: %{x:.3f}<br>"
                       "Kurtosis: %{y:.3f}<extra></extra>"),
    ))


def _add_negbin_region(fig: go.Figure, xmax: float) -> None:
    """Filled region occupied by the negative binomial family (discrete case)."""
    s2_parts, kurt_parts = [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for p, grid in ((np.exp(-10.0), _LOG_GRID), (1 - np.exp(-10.0), _LOG_GRID[::-1])):
            r = np.exp(grid)
            s2 = (2 - p) ** 2 / (r * (1 - p))
            kurt = 3 + 6 / r + p ** 2 / (r * (1 - p))
            s2_parts.append(s2)
            kurt_parts.append(kurt)

    s2 = np.concatenate(s2_parts)
    kurt = np.concatenate(kurt_parts)
    finite = np.isfinite(s2) & np.isfinite(kurt)
    fig.add_trace(go.Scatter(
        x=s2[finite], y=kurt[finite],
        mode="lines",
        fill="toself",
        name="negative binomial",
        line=dict(color="lightgray", width=0),
        fillcolor="rgba(204,204,204,0.6)",
        hoverinfo="skip",
    ))


def _add_poisson_curve(fig: go.Figure, xmax: float) -> None:
    """Poisson family: skewness² = 1/lambda, kurtosis = 3 + 1/lambda."""
    with np.errstate(over="ignore", divide="ignore"):
        lam = np.exp(_LOG_GRID)
        s2 = 1.0 / lam
        kurt = 3.0 + 1.0 / lam
    keep = np.isfinite(s2) & (s2 <= xmax)
    fig.add_trace(go.Scatter(
        x=s2[keep], y=kurt[keep],
        mode="lines",
        name="Poisson",
        line=dict(color="black", width=1.5, dash="dash"),
        hovertemplate=("<b>Poisson</b><br>Skewness²: %{x:.3f}<br>"
                       "Kurtosis: %{y:.3f}<extra></extra>"),
    ))


# ---------------------------------------------------------------------------
# Comparison plots -- qqcomp / denscomp / ppcomp / cdfcomp
# ---------------------------------------------------------------------------

def quantile_comparison_plot(
    data: Union[pd.Series, np.ndarray, list],
    models: Union[str, List[str], object, List[object]] = "normal",
    title: str = "Q-Q plot",
    data_name: str = "Data",
    dist_params: Optional[Union[tuple, dict, List[tuple], List[dict]]] = None,
    include_histogram: bool = True,
    a_ppoints: float = 0.5,
    ynoise: bool = False,
    bins: Union[int, str, Sequence[float]] = "sturges",
    fitnbpts: int = 101,
    seed: Optional[int] = None,
) -> Union[go.Figure, Tuple[go.Figure, ...]]:
    """Compare empirical data against one or more theoretical distributions.

    Produces the four fitdistrplus comparison graphics: ``qqcomp``,
    ``denscomp``, ``ppcomp`` and ``cdfcomp``.  Parameters not supplied are
    estimated by maximum likelihood, as ``fitdist(..., method = 'mle')`` does,
    using R's parameterisation of each distribution.

    Parameters
    ----------
    data : array-like
        Input data as pandas Series, numpy array, or list.
    models : str, object, or list thereof, default='normal'
        Distribution name(s) (see ``SUPPORTED_DISTRIBUTIONS``), R's own names
        (``'lnorm'``, ``'exp'``, ...), or scipy.stats distribution objects.
    title : str, default='Q-Q plot'
        Title of the Q-Q plot.
    data_name : str, default='Data'
        Name for the empirical data series.
    dist_params : tuple, dict, list thereof, or None, default=None
        Full scipy parameter tuples.  ``None`` estimates them from the data.
        A single tuple applies to the first model only, as before.
    include_histogram : bool, default=True
        Return the density, P-P and CDF plots alongside the Q-Q plot.
    a_ppoints : float, default=0.5
        Offset for the plotting positions ``(i - a) / (n + 1 - 2a)``.  R's
        ``*comp`` functions default to 0.5.
    ynoise : bool, default=False
        Jitter the empirical values of the second and subsequent fits by
        ``U(-0.02, 0.02)``, as R does to separate overlapping series.  R
        defaults this to ``TRUE``; it defaults to ``False`` here because the
        series are separable interactively.  Hover always reports the
        un-jittered value.
    bins : int, str or sequence, default='sturges'
        Histogram binning for the density plot, passed to
        ``numpy.histogram_bin_edges``.  R's ``hist`` also defaults to Sturges.
    fitnbpts : int, default=101
        Number of points used to draw fitted curves, as R's ``fitnbpts``.
    seed : int, optional
        Seed for the ``ynoise`` jitter.

    Returns
    -------
    go.Figure, or (qq, density, pp, cdf) figures when ``include_histogram``.
    """
    empirical_data = _validate_and_prepare_data(data)

    model_list = _normalize_models_input(models)
    param_list = _normalize_params_input(dist_params, len(model_list))
    distributions = _setup_distributions(model_list, empirical_data, param_list)

    qq_fig = _create_multi_qq_plot(
        empirical_data, distributions, title, data_name,
        a_ppoints=a_ppoints, ynoise=ynoise, seed=seed,
    )
    if not include_histogram:
        return qq_fig

    hist_fig = _create_multi_histogram_plot(
        empirical_data, distributions,
        "Histogram and theoretical densities", data_name,
        bins=bins, fitnbpts=fitnbpts,
    )
    pp_fig = _create_multi_pp_plot(
        empirical_data, distributions, "P-P plot", data_name,
        a_ppoints=a_ppoints, ynoise=ynoise, seed=seed,
    )
    cdf_fig = _create_multi_cdf_plot(
        empirical_data, distributions,
        "Empirical and theoretical CDFs", data_name,
        a_ppoints=a_ppoints, fitnbpts=fitnbpts,
    )
    return qq_fig, hist_fig, pp_fig, cdf_fig


def _validate_and_prepare_data(
    data: Union[pd.Series, np.ndarray, list],
    min_points: int = 3,
) -> np.ndarray:
    """Validate input data and return it sorted, with NAs dropped."""
    if not isinstance(data, (pd.Series, np.ndarray, list, tuple)):
        raise TypeError("Data must be a pandas Series, numpy array, or list")

    series = pd.Series(data).dropna()
    if len(series) < min_points:
        raise ValueError(f"At least {min_points} non-NA data points are required")

    return np.sort(np.asarray(series.to_numpy(), dtype=float))


def _normalize_models_input(models) -> List[Union[str, object]]:
    """Normalize the models argument to a list."""
    if isinstance(models, (str, bytes)) or not isinstance(models, (list, tuple)):
        return [models]
    return list(models)


def _normalize_params_input(params, n_models: int) -> List[Optional[Union[tuple, dict]]]:
    """Normalize the parameters argument to one entry per model."""
    if params is None:
        return [None] * n_models
    if isinstance(params, (tuple, dict)):
        return [params] + [None] * (n_models - 1)
    if isinstance(params, list):
        return params + [None] * max(0, n_models - len(params))
    return [None] * n_models


def _setup_distributions(models, data, params_list) -> List[Tuple[object, tuple, str]]:
    """Resolve each model to a ``(distribution, parameters, label)`` triple."""
    return [
        _setup_single_distribution(model, data, params)
        for model, params in zip(models, params_list)
    ]


def _setup_single_distribution(model, data, dist_params) -> Tuple[object, tuple, str]:
    dist, spec = resolve_distribution(model)
    label = model if isinstance(model, str) else getattr(dist, "name", str(model))
    if spec is not None:
        label = spec.r_name

    if dist_params is None:
        # A fitted mixture carries its own parameters, so it needs none passed.
        if getattr(dist, "n_params", None) == 0:
            return dist, (), label
        if spec is None:
            raise ValueError(
                "Unregistered scipy distributions require explicit dist_params"
            )
        _, params = fit_distribution(model, data)
        return dist, params, label

    params = tuple(dist_params.values()) if isinstance(dist_params, dict) else tuple(dist_params)
    # A distribution may declare its own count; a fitted mixture carries its
    # parameters internally and so declares zero.
    n_expected = getattr(dist, "n_params", None)
    if n_expected is None:
        n_expected = len(dist.shapes.split(",")) + 2 if dist.shapes else 2
    if len(params) != n_expected:
        raise ValueError(
            f"{label} takes {n_expected} scipy parameters "
            f"({dist.shapes + ', ' if dist.shapes else ''}loc, scale), "
            f"got {len(params)}"
        )
    return dist, params, label


def _jitter(values: np.ndarray, seed: Optional[int]) -> np.ndarray:
    """R's ``ynoise``: add U(-0.02, 0.02) to separate overlapping series."""
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-0.02, 0.02, size=len(values))


def _create_multi_qq_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple, str]],
    title: str,
    data_name: str,
    a_ppoints: float = 0.5,
    ynoise: bool = False,
    seed: Optional[int] = None,
) -> go.Figure:
    """Q-Q plot of empirical against fitted quantiles -- R's ``qqcomp``."""
    fig = go.Figure()
    n = len(empirical_data)
    obsp = ppoints(n, a=a_ppoints)

    all_quantiles = []
    for i, (distribution, params, label) in enumerate(distributions):
        theoretical_quantiles = distribution.ppf(obsp, *params)
        if not np.all(np.isfinite(theoretical_quantiles)):
            warnings.warn(
                f"Non-finite theoretical quantiles for '{label}'; skipping it",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        all_quantiles.append(theoretical_quantiles)

        y = empirical_data
        if ynoise and i > 0:
            y = _jitter(empirical_data, None if seed is None else seed + i)

        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=y,
            customdata=empirical_data,
            mode="markers",
            name=label,
            marker=dict(size=6, opacity=0.7,
                        color=DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Theoretical: %{x:.3f}<br>"
                "Empirical: %{customdata:.3f}<extra></extra>"
            ),
        ))

    if not all_quantiles:
        raise ValueError("No distribution produced finite theoretical quantiles")

    # R draws abline(0, 1) across the whole panel.
    stacked = np.concatenate(all_quantiles)
    lo = float(min(stacked.min(), empirical_data.min()))
    hi = float(max(stacked.max(), empirical_data.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="y = x",
        line=dict(dash="dash", color="black", width=1.5),
        hoverinfo="skip",
    ))

    _finish(
        fig, title,
        xaxis=dict(title="Theoretical quantiles", showgrid=True, gridcolor="lightgray",
                   range=[float(stacked.min()), float(stacked.max())]),
        yaxis=dict(title="Empirical quantiles", showgrid=True, gridcolor="lightgray",
                   range=[float(empirical_data.min()), float(empirical_data.max())]),
    )
    return fig


def _create_multi_pp_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple, str]],
    title: str,
    data_name: str,
    a_ppoints: float = 0.5,
    ynoise: bool = False,
    seed: Optional[int] = None,
) -> go.Figure:
    """P-P plot of empirical against fitted probabilities -- R's ``ppcomp``."""
    fig = go.Figure()
    n = len(empirical_data)

    # R uses the same plotting positions here as in qqcomp, not (1:n)/n.
    obsp = ppoints(n, a=a_ppoints)

    for i, (distribution, params, label) in enumerate(distributions):
        theoretical_probs = distribution.cdf(empirical_data, *params)

        y = obsp
        if ynoise and i > 0:
            y = _jitter(obsp, None if seed is None else seed + i)

        fig.add_trace(go.Scatter(
            x=theoretical_probs,
            y=y,
            customdata=obsp,
            mode="markers",
            name=label,
            marker=dict(size=6, opacity=0.7,
                        color=DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Theoretical probability: %{x:.3f}<br>"
                "Empirical probability: %{customdata:.3f}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="y = x",
        line=dict(dash="dash", color="black", width=1.5),
        hoverinfo="skip",
    ))

    _finish(
        fig, title,
        xaxis=dict(title="Theoretical probabilities", showgrid=True,
                   gridcolor="lightgray", range=[0, 1]),
        yaxis=dict(title="Empirical probabilities", showgrid=True,
                   gridcolor="lightgray", range=[0, 1]),
    )
    return fig


def _create_multi_cdf_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple, str]],
    title: str,
    data_name: str,
    a_ppoints: float = 0.5,
    fitnbpts: int = 101,
) -> go.Figure:
    """Empirical and fitted CDFs -- R's ``cdfcomp``."""
    fig = go.Figure()
    n = len(empirical_data)

    # R plots the empirical CDF at ppoints(n, a = 0.5) for continuous data,
    # with horizontal steps and points (horizontals = TRUE, verticals = FALSE).
    # For discrete data cdfcomp falls back to (1:n)/n, because ppoints would
    # place the steps off the integers where the mass actually sits.
    discrete = any(is_discrete(d) for d, _, _ in distributions)
    obsp = np.arange(1, n + 1) / n if discrete else ppoints(n, a=a_ppoints)

    fig.add_trace(go.Scatter(
        x=empirical_data,
        y=obsp,
        mode="markers+lines",
        name=f"{data_name} (empirical)",
        line=dict(shape="hv", color="black", width=1.5),
        marker=dict(size=4, opacity=0.7, color="black"),
        hovertemplate=(
            f"<b>{data_name}</b><br>"
            "Value: %{x:.3f}<br>"
            "CDF: %{y:.3f}<extra></extra>"
        ),
    ))

    sfin = _fit_grid(empirical_data, fitnbpts, discrete)
    for i, (distribution, params, label) in enumerate(distributions):
        fig.add_trace(go.Scatter(
            x=sfin,
            y=distribution.cdf(sfin, *params),
            mode="lines",
            line_shape="hv" if discrete else "linear",
            name=label,
            line=dict(color=DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)], width=2),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Value: %{x:.3f}<br>"
                "CDF: %{y:.3f}<extra></extra>"
            ),
        ))

    _finish(
        fig, title,
        xaxis=dict(title="data", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="CDF", showgrid=True, gridcolor="lightgray", range=[0, 1]),
    )
    return fig


def _create_multi_histogram_plot(
    data: np.ndarray,
    distributions: List[Tuple[object, tuple, str]],
    title: str,
    name: str,
    bins: Union[int, str, Sequence[float]] = "sturges",
    fitnbpts: int = 101,
    demp: bool = False,
) -> go.Figure:
    """Histogram with fitted densities -- R's ``denscomp``.

    ``bins`` defaults to Sturges' rule, which is also R's ``hist`` default.
    For a discrete fit the empirical bars become relative frequencies at each
    observed count and the fitted curves become probability masses, drawn as
    markers joined by thin lines rather than as a continuous density.
    """
    fig = go.Figure()
    discrete = any(is_discrete(d) for d, _, _ in distributions)

    if discrete:
        # One bar per integer, holding the proportion of the sample at it.
        values = np.arange(int(np.min(data)), int(np.max(data)) + 1)
        counts = np.array([np.sum(data == v) for v in values]) / len(data)
        centres, widths = values.astype(float), np.full(len(values), 0.85)
        y_title = "Probability"
        hover = "Value: %{x:.0f}<br>Proportion: %{y:.4f}<extra></extra>"
    else:
        edges = np.histogram_bin_edges(data, bins=bins)
        counts, edges = np.histogram(data, bins=edges, density=True)
        centres, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        y_title = "Density"
        hover = "Bin centre: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>"

    fig.add_trace(go.Bar(
        x=centres,
        y=counts,
        width=widths,
        name=f"{name} (empirical)",
        marker=dict(color="gray", opacity=0.7,
                    line=dict(width=0.5, color="white")),
        hovertemplate=hover,
    ))

    sfin = _fit_grid(data, fitnbpts, discrete)
    for i, (distribution, params, label) in enumerate(distributions):
        fig.add_trace(go.Scatter(
            x=sfin,
            y=density(distribution, sfin, params),
            mode="markers+lines" if discrete else "lines",
            name=label,
            marker=dict(size=7) if discrete else None,
            line=dict(color=DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)],
                      width=1 if discrete else 2),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Value: %{x:.3f}<br>"
                f"{'Probability' if discrete else 'Density'}: %{{y:.4f}}<extra></extra>"
            ),
        ))

    if demp:  # R's demp = TRUE overlays the empirical density
        kde = stats.gaussian_kde(data)
        fig.add_trace(go.Scatter(
            x=sfin, y=kde(sfin),
            mode="lines",
            name=f"{name} (empirical density)",
            line=dict(color="black", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    _finish(
        fig, title,
        xaxis=dict(title="data", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="lightgray"),
        legend_x=0.98, legend_xanchor="right",
    )
    fig.update_layout(bargap=0 if not discrete else 0.15)
    return fig


def _fit_grid(data: np.ndarray, fitnbpts: int, discrete: bool) -> np.ndarray:
    """Points at which to evaluate a fitted distribution.

    Continuous fits get an evenly spaced grid; discrete fits are evaluated on
    the integers, where all their mass lies.  R does the same, rounding its
    ``seq`` to whole numbers when ``discrete`` is set.
    """
    low, high = float(np.min(data)), float(np.max(data))
    if discrete:
        return np.arange(int(np.floor(low)), int(np.ceil(high)) + 1, dtype=float)
    return np.linspace(low, high, fitnbpts)


def _finish(
    fig: go.Figure,
    title: str,
    xaxis: dict,
    yaxis: dict,
    legend_x: float = 0.02,
    legend_xanchor: str = "left",
    width: int = 700,
    height: int = 500,
) -> None:
    """Apply the shared layout to a comparison figure."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=xaxis,
        yaxis=yaxis,
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(x=legend_x, y=0.98, xanchor=legend_xanchor, yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
    )
