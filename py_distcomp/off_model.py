"""
Off-model fraction analysis.

This implements the method of

    Rushton, C.E., Tate, J.E. and Shepherd, S.P. (2021) "A novel method for
    comparing passenger car fleets and identifying high-chance gross emitting
    vehicles using kerbside remote sensing data", Science of the Total
    Environment, 750, 142088. https://doi.org/10.1016/j.scitotenv.2020.142088

which extends the fitdistrplus workflow rather than reproducing it.  The idea:
most of a population follows a single distribution -- a Gumbel, in the paper --
but a small high-valued subset does not.  That subset is found by cutting the
data at successively lower percentiles, refitting, and measuring how well the
Q-Q relationship follows the 1:1 line.  The percentile that maximises the fit is
the *off-model percentile* ``P_off``; the remaining ``100 - P_off`` per cent is
the *off-model fraction*, interpreted in the paper as candidate gross emitters.

Fitting the retained data and the off-model tail separately describes the
population as a superposition of two distributions.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .distributions import fit_distribution, ppoints, resolve_distribution
from .gofstat import FitResult, _r_estimate

__all__ = ["qq_r_squared", "off_model_fraction", "OffModelResult"]


def _clean(data) -> np.ndarray:
    series = pd.Series(data).dropna()
    if len(series) < 3:
        raise ValueError("At least 3 non-NA data points are required")
    return np.sort(np.asarray(series.to_numpy(), dtype=float))


def qq_r_squared(
    data: Union[np.ndarray, pd.Series, list],
    model: Union[str, object] = "gumbel",
    dist_params: Optional[Sequence[float]] = None,
    method: str = "identity",
    a_ppoints: float = 0.5,
) -> float:
    """R² of the relationship between empirical and theoretical quantiles.

    The goodness-of-fit measure used in section 3.2 of Rushton et al. (2021).

    Parameters
    ----------
    data : array-like
        Input data.
    model : str or scipy distribution, default='gumbel'
        Distribution to compare against.  Fitted by maximum likelihood unless
        ``dist_params`` is given.
    dist_params : sequence of float, optional
        Full scipy parameter tuple, bypassing the fit.
    method : {'identity', 'pearson'}, default='identity'
        ``'identity'`` measures deviation from the 1:1 line,
        ``1 - Σ(y - x)² / Σ(y - ȳ)²``, so location or scale error is penalised.
        ``'pearson'`` is the squared correlation of the two quantile sets, which
        only measures how straight the Q-Q relationship is.

        The paper reports "the R² value of the relationship between the
        empirical and theoretical quantiles" without giving a formula, but
        describes fit throughout as agreement with the 1:1 line, so
        ``'identity'`` is the default.  In practice the two pick the same
        off-model percentile; ``'identity'`` simply falls further when the fit
        is poor.
    a_ppoints : float, default=0.5
        Offset for the plotting positions, matching the Q-Q plots.

    Returns
    -------
    float
        R².  With ``method='identity'`` this can be negative, meaning the fitted
        quantiles track the data worse than a horizontal line at its mean.
    """
    if method not in ("identity", "pearson"):
        raise ValueError("method must be 'identity' or 'pearson'")

    sorted_data = _clean(data)
    dist, _ = resolve_distribution(model)
    params = tuple(dist_params) if dist_params is not None else fit_distribution(model, sorted_data)[1]

    theoretical = dist.ppf(ppoints(len(sorted_data), a=a_ppoints), *params)
    if not np.all(np.isfinite(theoretical)):
        return float("nan")

    if method == "pearson":
        if np.ptp(theoretical) == 0 or np.ptp(sorted_data) == 0:
            return float("nan")
        return float(np.corrcoef(theoretical, sorted_data)[0, 1] ** 2)

    ss_res = float(np.sum((sorted_data - theoretical) ** 2))
    ss_tot = float(np.sum((sorted_data - np.mean(sorted_data)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


class OffModelResult:
    """Outcome of an off-model fraction sweep.

    Attributes
    ----------
    model : str
        Distribution that was fitted.
    percentile : float
        ``P_off``, the percentile cut that maximised R².  100 means no cut was
        needed and the whole sample follows the distribution.
    fraction : float
        ``100 - percentile``: the percentage of observations judged off-model.
    r_squared : float
        R² achieved at ``percentile``.
    threshold : float
        Data value at the cut.  Observations above it are the off-model set.
    n_total, n_retained, n_off_model : int
        Sample sizes.
    fit : FitResult
        Fit to the retained data -- the parameters the paper reports.
    tail_fit : FitResult or None
        Second-stage fit to the off-model observations, giving the superposition
        model.  ``None`` when no cut was made or the tail is too small.
    curve : pandas.DataFrame
        One row per candidate percentile: ``percentile``, ``n``, ``threshold``,
        ``r_squared`` and the fitted parameters, in R's parameterisation.  This
        is the data behind Figure 5 of the paper.
    method : str
        Which R² definition was used.
    """

    def __init__(self, model, percentile, r_squared, threshold, data,
                 fit, tail_fit, curve, method):
        self.model = model
        self.percentile = float(percentile)
        self.fraction = float(100.0 - percentile)
        self.r_squared = float(r_squared)
        self.threshold = float(threshold)
        self.data = data
        self.n_total = len(data)
        self.n_off_model = int(np.sum(data > threshold))
        self.n_retained = self.n_total - self.n_off_model
        self.fit = fit
        self.tail_fit = tail_fit
        self.curve = curve
        self.method = method

    @property
    def off_model_values(self) -> np.ndarray:
        """The observations above the cut -- the candidate gross emitters."""
        return self.data[self.data > self.threshold]

    def summary(self) -> pd.Series:
        """One-row summary, in the style of Table 1 of the paper."""
        row = {
            "model": self.model,
            "n": self.n_total,
            "off_model_percentile": self.percentile,
            "off_model_fraction": self.fraction,
            "r_squared": self.r_squared,
            "threshold": self.threshold,
            "n_off_model": self.n_off_model,
        }
        row.update(self.fit.estimate)
        return pd.Series(row)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OffModelResult({self.model}: P_off={self.percentile:g}, "
            f"fraction={self.fraction:g}%, R²={self.r_squared:.4f}, "
            f"n_off_model={self.n_off_model})"
        )


def off_model_fraction(
    data: Union[np.ndarray, pd.Series, list],
    model: Union[str, object] = "gumbel",
    percentiles: Optional[Iterable[float]] = None,
    method: str = "identity",
    min_points: int = 10,
    fit_tail: bool = True,
    a_ppoints: float = 0.5,
) -> OffModelResult:
    """Find the off-model fraction of a sample.

    Cuts the data at each candidate percentile, refits ``model`` to what
    remains, and records the Q-Q R².  The percentile with the highest R² is
    taken as ``P_off``, and ``100 - P_off`` is the off-model fraction --
    section 3.2 of Rushton et al. (2021).

    Parameters
    ----------
    data : array-like
        Input data.
    model : str or scipy distribution, default='gumbel'
        Distribution to fit.  The paper uses the Gumbel, whose location is the
        modal value and whose scale describes the spread, so that both
        parameters are directly comparable between population subsets.
    percentiles : iterable of float, optional
        Candidate cuts.  Defaults to every integer percentile from 1 to 100,
        where 100 means no cut.  The paper sweeps integer percentiles.
    method : {'identity', 'pearson'}, default='identity'
        R² definition; see :func:`qq_r_squared`.
    min_points : int, default=10
        Skip cuts that would leave fewer than this many observations.
    fit_tail : bool, default=True
        Also fit ``model`` to the off-model observations, which is the paper's
        second iteration and gives the superposition description.
    a_ppoints : float, default=0.5
        Offset for the plotting positions.

    Returns
    -------
    OffModelResult

    Examples
    --------
    >>> result = off_model_fraction(emission_ratios, 'gumbel')
    >>> result.percentile, result.fraction
    (95.0, 5.0)
    >>> result.fit.estimate
    {'loc': 11.2, 'scale': 8.3}
    """
    sorted_data = _clean(data)
    if percentiles is None:
        percentiles = range(1, 101)
    candidates = sorted({float(p) for p in percentiles})
    if not candidates:
        raise ValueError("percentiles must contain at least one value")
    if any(p <= 0 or p > 100 for p in candidates):
        raise ValueError("percentiles must lie in (0, 100]")

    dist, spec = resolve_distribution(model)
    model_name = model if isinstance(model, str) else getattr(dist, "name", str(model))
    r_name = spec.r_name if spec is not None else getattr(dist, "name", model_name)

    rows: List[Dict[str, object]] = []
    for pct in candidates:
        threshold = float(np.percentile(sorted_data, pct))
        retained = sorted_data[sorted_data <= threshold]
        if len(retained) < min_points:
            continue

        try:
            _, params = fit_distribution(model, retained)
            r2 = qq_r_squared(retained, model, dist_params=params,
                              method=method, a_ppoints=a_ppoints)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if not np.isfinite(r2):
            continue

        row: Dict[str, object] = {
            "percentile": pct,
            "n": len(retained),
            "threshold": threshold,
            "r_squared": r2,
        }
        row.update(_r_estimate(r_name, params, spec))
        rows.append(row)

    if not rows:
        raise ValueError(
            "No candidate percentile produced a usable fit; the sample may be "
            f"too small (n = {len(sorted_data)}) for min_points = {min_points}"
        )

    curve = pd.DataFrame(rows)

    # "The highest percentile, maximal R^2 value was chosen as the best model":
    # take the maximum, and the highest percentile among any ties.
    best_r2 = curve["r_squared"].max()
    best = curve[curve["r_squared"] == best_r2]["percentile"].max()
    threshold = float(np.percentile(sorted_data, best))
    retained = sorted_data[sorted_data <= threshold]

    fit = _make_fit(model, model_name, retained)

    tail_fit = None
    if fit_tail:
        tail = sorted_data[sorted_data > threshold]
        if len(tail) >= min_points:
            try:
                tail_fit = _make_fit(model, model_name, tail)
            except (ValueError, RuntimeError, FloatingPointError):
                tail_fit = None

    return OffModelResult(
        model=model_name,
        percentile=best,
        r_squared=best_r2,
        threshold=threshold,
        data=sorted_data,
        fit=fit,
        tail_fit=tail_fit,
        curve=curve,
        method=method,
    )


def _make_fit(model, model_name: str, subset: np.ndarray) -> FitResult:
    dist, spec = resolve_distribution(model)
    _, params = fit_distribution(model, subset)
    return FitResult(model_name, dist, params, subset, spec)
