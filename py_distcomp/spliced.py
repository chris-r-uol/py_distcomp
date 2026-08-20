"""
Spliced (composite) distributions -- one density below a threshold, another above.

A mixture says the tail is a *second population* sitting on top of the first:
both components have mass everywhere, and every observation belongs to each with
some probability.  A spliced model says something different -- that there is one
population whose tail is simply heavier than the body would imply.  Below the
threshold θ the lower family governs, above it the upper family does, and each
piece is renormalised so the whole remains a proper density::

              ⎧  w · f₁(x) / F₁(θ)                x ≤ θ
    f(x)  =   ⎨
              ⎩  (1−w) · f₂(x) / (1 − F₂(θ))      x > θ

This is the principled successor to the percentile cut of Rushton et al. (2021).
That method searches percentiles for the cut that maximises a Q-Q R²; this one
estimates θ by likelihood alongside both sides, and hands it back as an estimate
with a profile-likelihood confidence interval.

By default the two pieces are required to *meet* at θ, which is what makes the
result a single density rather than two stitched fragments.  That constraint
determines ``w`` rather than leaving it free::

    w = b / (a + b),    a = f₁(θ)/F₁(θ),   b = f₂(θ)/(1 − F₂(θ))

so a continuous splice of two two-parameter families estimates five quantities:
θ and two parameters per side.  Pass ``continuous=False`` to free ``w`` instead
and allow a jump at the join.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .distributions import r_estimate, resolve_distribution, scipy_params
from .estimation import _TRANSFORMS, _from_unconstrained, _to_unconstrained

__all__ = ["SplicedDistribution", "SplicedResult", "fit_spliced"]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# The fitted splice, exposed through the scipy distribution interface
# ---------------------------------------------------------------------------

class SplicedDistribution:
    """A frozen spliced distribution that behaves like a scipy one.

    ``pdf``, ``cdf``, ``ppf``, ``logpdf``, ``sf`` and ``rvs`` accept and ignore
    trailing positional parameters, so the object drops into anything expecting
    ``dist.method(x, *params)`` -- the comparison plots and
    :func:`~py_distcomp.gofstat`.

    Parameters
    ----------
    lower, upper : (distribution, params)
        The families governing below and above the threshold, with their full
        scipy parameter tuples.
    threshold : float
        Where one takes over from the other.
    weight : float
        Probability mass below the threshold.
    """

    shapes = None
    n_params = 0

    def __init__(self, lower, upper, threshold: float, weight: float):
        self.lower_dist, self.lower_params = lower[0], tuple(lower[1])
        self.upper_dist, self.upper_params = upper[0], tuple(upper[1])
        self.threshold = float(threshold)
        if not 0 < weight < 1:
            raise ValueError("weight must lie strictly between 0 and 1")
        self.weight = float(weight)

        self._f_lo = float(self.lower_dist.cdf(self.threshold, *self.lower_params))
        self._f_hi = float(self.upper_dist.cdf(self.threshold, *self.upper_params))
        if not 0 < self._f_lo <= 1:
            raise ValueError(
                "the lower family puts no mass below the threshold, so it "
                "cannot be renormalised there"
            )
        if not 0 <= self._f_hi < 1:
            raise ValueError(
                "the upper family puts no mass above the threshold, so it "
                "cannot be renormalised there"
            )
        self.name = (f"{getattr(self.lower_dist, 'name', '?')}"
                     f"|{getattr(self.upper_dist, 'name', '?')}")

    def pdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        below = x <= self.threshold
        out = np.zeros(x.shape, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.any(below):
                out[below] = (self.weight
                              * self.lower_dist.pdf(x[below], *self.lower_params)
                              / self._f_lo)
            if np.any(~below):
                out[~below] = ((1 - self.weight)
                               * self.upper_dist.pdf(x[~below], *self.upper_params)
                               / (1 - self._f_hi))
        return out if out.shape else float(out)

    def logpdf(self, x, *_) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(self.pdf(x), _TINY))

    def cdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        below = x <= self.threshold
        out = np.empty(x.shape, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.any(below):
                out[below] = (self.weight
                              * self.lower_dist.cdf(x[below], *self.lower_params)
                              / self._f_lo)
            if np.any(~below):
                upper = self.upper_dist.cdf(x[~below], *self.upper_params)
                out[~below] = self.weight + (1 - self.weight) * (
                    (upper - self._f_hi) / (1 - self._f_hi)
                )
        return np.clip(out, 0.0, 1.0)

    def sf(self, x, *_) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def ppf(self, q, *_) -> np.ndarray:
        """Quantile function, inverted piecewise and exactly.

        Unlike a mixture, a splice needs no numerical inversion: each piece is a
        rescaled parent, so its own quantile function does the work.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.full(q.shape, np.nan)
        out[q <= 0] = -np.inf
        out[q >= 1] = np.inf

        low = (q > 0) & (q <= self.weight)
        high = (q > self.weight) & (q < 1)
        if np.any(low):
            out[low] = self.lower_dist.ppf(q[low] * self._f_lo / self.weight,
                                           *self.lower_params)
        if np.any(high):
            scaled = self._f_hi + (q[high] - self.weight) * (1 - self._f_hi) / (1 - self.weight)
            out[high] = self.upper_dist.ppf(scaled, *self.upper_params)
        return out if out.size > 1 else out.item()

    def rvs(self, size=1, random_state=None, *_) -> np.ndarray:
        """Draw by inverse transform, which is exact here."""
        rng = np.random.default_rng(random_state)
        return np.asarray(self.ppf(rng.uniform(size=int(size))))

    @property
    def jump(self) -> float:
        """Relative size of the discontinuity at the threshold.

        Zero for a continuous splice.  For one fitted with ``continuous=False``
        this says how large a step the density takes at the join, as a fraction
        of the density there -- a large value is a sign the two families do not
        belong together.
        """
        left = self.weight * self.lower_dist.pdf(self.threshold, *self.lower_params) / self._f_lo
        right = ((1 - self.weight)
                 * self.upper_dist.pdf(self.threshold, *self.upper_params)
                 / (1 - self._f_hi))
        scale = max(abs(left), abs(right))
        return 0.0 if scale == 0 else float(abs(left - right) / scale)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SplicedDistribution({self.name} at {self.threshold:.4g}, "
                f"w={self.weight:.4g})")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _continuity_weight(lower, upper, threshold: float) -> Optional[float]:
    """The weight that makes the two pieces meet at the threshold."""
    lo_dist, lo_params = lower
    hi_dist, hi_params = upper
    with np.errstate(divide="ignore", invalid="ignore"):
        f_lo = float(lo_dist.cdf(threshold, *lo_params))
        f_hi = float(hi_dist.cdf(threshold, *hi_params))
        if not (0 < f_lo <= 1) or not (0 <= f_hi < 1):
            return None
        a = float(lo_dist.pdf(threshold, *lo_params)) / f_lo
        b = float(hi_dist.pdf(threshold, *hi_params)) / (1 - f_hi)
    if not np.isfinite(a) or not np.isfinite(b) or a + b <= 0:
        return None
    return float(b / (a + b))


def _cover_threshold(dist, spec, values, threshold: float) -> List[float]:
    """Move a family's support down so it reaches the threshold.

    Some families begin where a parameter says they do -- a Pareto starts at its
    scale -- and fitting one to the exceedances puts that start at the smallest
    of them, which is above the threshold.  The density at the join is then zero
    and the continuity weight collapses, so the whole splice is rejected before
    the optimiser has seen it.

    Pulling the support down to the threshold is the standard composite
    formulation: a Pareto tail spliced at θ begins at θ.
    """
    values = [float(v) for v in values]
    try:
        low = float(dist.support(*scipy_params(spec.r_name, values, spec))[0])
    except (ValueError, TypeError, IndexError):
        return values
    if not np.isfinite(low) or low <= threshold:
        return values

    # The registry lists a scale last wherever the support depends on one, so
    # shrinking it in proportion lands the support just under the threshold.
    if values[-1] > 0 and low > 0:
        values[-1] *= (threshold / low) * (1 - 1e-9)
    return values


def _spliced_loglik(data, lower, upper, threshold, weight) -> float:
    """Log-likelihood of a splice, with both pieces renormalised."""
    try:
        dist = SplicedDistribution(lower, upper, threshold, weight)
    except ValueError:
        return -np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        density = dist.pdf(data)
    if not np.all(np.isfinite(density)) or np.any(density <= 0):
        return -np.inf
    return float(np.sum(np.log(density)))


def _fit_at_threshold(data, lower_model, upper_model, threshold, continuous,
                      start_lower, start_upper, maxiter):
    """Best fit for a fixed threshold, optimising both sides together.

    The sides cannot be fitted independently under the continuity constraint:
    the weight it implies depends on both, so it couples them.
    """
    lo_dist, lo_spec = resolve_distribution(lower_model)
    hi_dist, hi_spec = resolve_distribution(upper_model)
    lo_kinds = _TRANSFORMS.get(lo_spec.r_name, ("free",) * len(start_lower))
    hi_kinds = _TRANSFORMS.get(hi_spec.r_name, ("free",) * len(start_upper))
    n_lo = len(start_lower)

    below = data <= threshold
    n_below = int(np.sum(below))

    def unpack(z):
        lo_values = _from_unconstrained(z[:n_lo], lo_kinds)
        hi_values = _from_unconstrained(z[n_lo:n_lo + len(start_upper)], hi_kinds)
        lower = (lo_dist, scipy_params(lo_spec.r_name, lo_values, lo_spec))
        upper = (hi_dist, scipy_params(hi_spec.r_name, hi_values, hi_spec))
        if continuous:
            w = _continuity_weight(lower, upper, threshold)
        else:
            w = 1.0 / (1.0 + np.exp(-np.clip(z[-1], -700, 700)))
        return lower, upper, w

    def negll(z):
        try:
            lower, upper, w = unpack(z)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return np.inf
        if w is None or not 0 < w < 1:
            return np.inf
        value = _spliced_loglik(data, lower, upper, threshold, w)
        return np.inf if not np.isfinite(value) else -value

    z0 = np.concatenate([
        _to_unconstrained(start_lower, lo_kinds),
        _to_unconstrained(start_upper, hi_kinds),
    ])
    if not continuous:
        share = np.clip(n_below / max(len(data), 1), 1e-3, 1 - 1e-3)
        z0 = np.append(z0, np.log(share / (1 - share)))

    if not np.isfinite(negll(z0)):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = optimize.minimize(
            negll, z0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-11},
        )
    if not np.isfinite(outcome.fun):
        return None

    lower, upper, w = unpack(outcome.x)
    return lower, upper, float(w), -float(outcome.fun)


class SplicedResult:
    """A fitted spliced distribution.

    Duck-types :class:`~py_distcomp.FitResult`, so it goes straight into
    :func:`~py_distcomp.gofstat` and the comparison plots beside single fits and
    mixtures.

    Attributes
    ----------
    dist : SplicedDistribution
    threshold : float
        The estimated join, θ.
    weight : float
        Mass below the threshold.  Determined by continuity unless the fit was
        made with ``continuous=False``.
    lower_estimate, upper_estimate : dict
        Each side's parameters, in R's parameterisation.
    profile : pandas.DataFrame
        Log-likelihood at every candidate threshold, which is what
        :meth:`threshold_ci` reads its interval off.
    continuous : bool
    loglik, aic, bic : float
    """

    def __init__(self, dist, lower_model, upper_model, data, loglik,
                 n_free_params, profile, continuous):
        self.dist = dist
        self.params: tuple = ()
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.name = f"{lower_model}|{upper_model}"
        self.r_name = self.name
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        self.threshold = dist.threshold
        self.weight = dist.weight
        self.continuous = bool(continuous)
        self.loglik = float(loglik)
        self.n_free_params = int(n_free_params)
        self.aic = -2.0 * self.loglik + 2.0 * self.n_free_params
        self.bic = -2.0 * self.loglik + np.log(self.n) * self.n_free_params
        self.profile = profile
        self.discrete = False

        _, lo_spec = resolve_distribution(lower_model)
        _, hi_spec = resolve_distribution(upper_model)
        self.lower_estimate = r_estimate(lo_spec.r_name, dist.lower_params, lo_spec)
        self.upper_estimate = r_estimate(hi_spec.r_name, dist.upper_params, hi_spec)
        self.estimate = self._build_estimate()

    def _build_estimate(self) -> Dict[str, float]:
        out = {"threshold": self.threshold, "weight": self.weight}
        for key, value in self.lower_estimate.items():
            out[f"{key}_lower"] = value
        for key, value in self.upper_estimate.items():
            out[f"{key}_upper"] = value
        return out

    @property
    def n_below(self) -> int:
        """Observations at or below the threshold."""
        return int(np.sum(self.data <= self.threshold))

    @property
    def n_above(self) -> int:
        return self.n - self.n_below

    @property
    def tail_values(self) -> np.ndarray:
        """The observations the upper family governs."""
        return self.data[self.data > self.threshold]

    def threshold_ci(self, level: float = 0.95) -> Tuple[float, float]:
        """Profile-likelihood confidence interval for the threshold.

        The set of thresholds whose profile log-likelihood sits within
        ``chi2(1, level) / 2`` of the maximum.  Unlike a Wald interval this
        needs no symmetry assumption, which matters: the threshold's profile is
        usually lopsided, being bounded by the data on one side.

        Returns ``(nan, nan)`` if the profile never falls far enough within the
        candidate range, which means the data does not pin the threshold down.
        """
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between 0 and 1")
        cutoff = self.loglik - stats.chi2.ppf(level, 1) / 2.0
        inside = self.profile[self.profile["loglik"] >= cutoff]["threshold"]
        if inside.empty:
            return (float("nan"), float("nan"))

        lo, hi = float(inside.min()), float(inside.max())
        edges = self.profile["threshold"]
        # An interval that runs to the edge of the search is not an interval.
        if lo <= edges.min() or hi >= edges.max():
            warnings.warn(
                "The profile-likelihood interval for the threshold reaches the "
                "edge of the candidate range; widen 'thresholds' to bound it",
                RuntimeWarning,
                stacklevel=2,
            )
        return (lo, hi)

    def summary(self) -> pd.Series:
        row = {
            "model": self.name,
            "n": self.n,
            "continuous": self.continuous,
            "threshold": self.threshold,
            "weight": self.weight,
            "n_above": self.n_above,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
        }
        row.update(self.estimate)
        return pd.Series(row)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SplicedResult({self.name} at θ={self.threshold:.4g}, "
                f"w={self.weight:.3f}, {self.n_above} above; "
                f"loglik={self.loglik:.1f}, aic={self.aic:.1f})")


def fit_spliced(
    data: Union[np.ndarray, pd.Series, list],
    lower: Union[str, object] = "gumbel",
    upper: Union[str, object] = "pareto",
    thresholds: Optional[Sequence[float]] = None,
    continuous: bool = True,
    min_points: int = 20,
    maxiter: int = 3000,
) -> SplicedResult:
    """Fit a spliced distribution: one family below a threshold, another above.

    The threshold is profiled over rather than optimised smoothly, because the
    likelihood changes shape whenever θ crosses an observation and is not
    differentiable there.  Profiling also produces the interval for θ for free.

    Parameters
    ----------
    data : array-like
        Input data.
    lower, upper : str or scipy distribution
        Families governing below and above the threshold.  Defaults pair a
        Gumbel body with a Pareto tail, the usual shape for an emissions or
        loss distribution.
    thresholds : sequence of float, optional
        Candidate thresholds.  Defaults to the 50th to 99th percentiles of the
        data, which keeps enough observations on both sides to estimate with.
    continuous : bool, default=True
        Require the two pieces to meet at the threshold, which determines the
        weight and makes the result a single density.  ``False`` frees the
        weight and permits a jump -- see :attr:`SplicedDistribution.jump`.
    min_points : int, default=20
        Minimum observations on each side of a candidate threshold.
    maxiter : int, default=3000
        Iterations allowed per threshold.

    Returns
    -------
    SplicedResult

    Examples
    --------
    >>> fit = fit_spliced(emissions, 'gumbel', 'pareto')
    >>> fit.threshold, fit.threshold_ci()
    (46.2, (38.1, 58.4))
    >>> fit.weight                      # mass below the join
    0.962

    Compare it against the alternatives on the same footing:

    >>> gofstat([fit_distributions(x, 'gumbel')[0], mixture, fit])
    """
    clean = np.sort(np.asarray(pd.Series(data).dropna().to_numpy(), dtype=float))
    if len(clean) < 4 * min_points:
        raise ValueError(
            f"Splicing needs at least {4 * min_points} observations to leave "
            f"{min_points} on each side of a range of candidate thresholds; "
            f"got {len(clean)}"
        )

    for role, model in (("lower", lower), ("upper", upper)):
        _, spec = resolve_distribution(model)
        if spec is None:
            raise ValueError(f"the {role} family must be a registered distribution")
        if spec.discrete:
            raise ValueError(
                f"the {role} family is discrete; splicing is defined here for "
                "continuous distributions, whose CDF is invertible at the join"
            )

    if thresholds is None:
        thresholds = np.quantile(clean, np.linspace(0.50, 0.99, 40))
    candidates = np.unique(np.asarray(thresholds, dtype=float))
    candidates = [
        t for t in candidates
        if np.sum(clean <= t) >= min_points and np.sum(clean > t) >= min_points
    ]
    if not candidates:
        raise ValueError(
            f"No candidate threshold leaves {min_points} observations on both "
            "sides; lower 'min_points' or supply 'thresholds' yourself"
        )

    from .distributions import fit_distribution

    rows, best = [], None
    for threshold in candidates:
        below, above = clean[clean <= threshold], clean[clean > threshold]
        # Untruncated fits to each side are only starting values; the optimiser
        # then works with the truncated, renormalised likelihood.
        try:
            _, lo_params = fit_distribution(lower, below)
            _, hi_params = fit_distribution(upper, above)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        _, lo_spec = resolve_distribution(lower)
        _, hi_spec = resolve_distribution(upper)
        start_lower = list(r_estimate(lo_spec.r_name, lo_params, lo_spec).values())
        start_upper = _cover_threshold(
            hi_dist_for_start := resolve_distribution(upper)[0], hi_spec,
            list(r_estimate(hi_spec.r_name, hi_params, hi_spec).values()),
            float(threshold),
        )

        outcome = _fit_at_threshold(clean, lower, upper, threshold, continuous,
                                    start_lower, start_upper, maxiter)
        if outcome is None:
            continue
        lo, hi, weight, loglik = outcome
        rows.append({"threshold": float(threshold), "loglik": loglik,
                     "weight": weight, "n_above": int(len(above))})
        if best is None or loglik > best[3]:
            best = (lo, hi, weight, loglik, float(threshold))

    if best is None:
        raise ValueError(
            "No candidate threshold produced a usable fit; try different "
            "families, or a narrower range of thresholds"
        )

    lo, hi, weight, loglik, threshold = best
    dist = SplicedDistribution(lo, hi, threshold, weight)

    # Free parameters: the threshold, both sides, and the weight unless
    # continuity has already determined it.
    _, lo_spec = resolve_distribution(lower)
    _, hi_spec = resolve_distribution(upper)
    n_free = 1 + lo_spec.n_free_params + hi_spec.n_free_params + (0 if continuous else 1)

    lower_name = lower if isinstance(lower, str) else getattr(lo[0], "name", "lower")
    upper_name = upper if isinstance(upper, str) else getattr(hi[0], "name", "upper")

    return SplicedResult(dist, lower_name, upper_name, clean, loglik, n_free,
                         pd.DataFrame(rows), continuous)
