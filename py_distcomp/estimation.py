"""
Estimation methods other than maximum likelihood.

R's ``fitdist`` takes a ``method`` argument and dispatches to one of several
estimators.  This module ports three of them:

=================  ==============  ==================================================
``method=``        fitdistrplus    minimises
=================  ==============  ==================================================
``'mme'``          ``mmedist``     the distance between theoretical and sample moments
``'qme'``          ``qmedist``     the distance between theoretical and sample quantiles
``'mge'``          ``mgedist``     a goodness-of-fit statistic directly
=================  ==============  ==================================================

They matter when maximum likelihood is awkward: ``'mme'`` has closed forms for
ten distributions and so cannot fail to converge, ``'qme'`` targets the part of
the distribution you actually care about, and ``'mge'`` optimises the statistic
a reader will judge the fit by.  All three report parameters in R's
parameterisation, like every other estimate in this package.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize

from .distributions import (
    DistributionSpec,
    fit_distribution,
    r_estimate,
    resolve_distribution,
    scipy_params,
)

__all__ = [
    "ESTIMATION_METHODS",
    "GOF_STATISTICS",
    "fit_by_method",
    "moment_match",
    "quantile_match",
    "maximum_goodness_of_fit",
]

#: Methods ``fit_distributions`` accepts, matching R's ``fitdist(method = ...)``.
ESTIMATION_METHODS = ("mle", "mme", "qme", "mge")

#: Statistics ``method='mge'`` can minimise, as R's ``mgedist(gof = ...)``.
#: The one-sided and squared variants weight the tails differently: ``R``/``L``
#: for the right and left tail, ``2`` for the squared versions that weight the
#: extremes harder still.
GOF_STATISTICS = ("CvM", "KS", "AD", "ADR", "ADL", "AD2R", "AD2L", "AD2")

_BIG = float(np.finfo(np.float64).max) ** 0.5


# ---------------------------------------------------------------------------
# Optimising in R's parameterisation
# ---------------------------------------------------------------------------
#
# Every estimate this package reports is in R's parameterisation, so the search
# runs there too.  Each parameter is mapped to an unconstrained scale first: a
# log for the strictly positive ones, a logit for a probability, and the
# identity for a location that may legitimately be negative.

_TRANSFORMS: Dict[str, Tuple[str, ...]] = {
    "norm": ("free", "log"),
    "lnorm": ("free", "log"),
    "weibull": ("log", "log"),
    "gamma": ("log", "log"),
    "exp": ("log",),
    "unif": ("free", "free"),
    "logis": ("free", "log"),
    "beta": ("log", "log"),
    "cauchy": ("free", "log"),
    "gumbel": ("free", "log"),
    "laplace": ("free", "log"),
    "pareto": ("log", "log"),
    "rayleigh": ("log",),
    "chisq": ("log",),
    "t": ("log",),
    "f": ("log", "log"),
    "pois": ("log",),
    "nbinom": ("log", "log"),
    "geom": ("logit",),
}


def _to_unconstrained(values: Sequence[float], kinds: Sequence[str]) -> np.ndarray:
    out = []
    for value, kind in zip(values, kinds):
        if kind == "log":
            out.append(np.log(max(float(value), 1e-12)))
        elif kind == "logit":
            p = float(np.clip(value, 1e-9, 1 - 1e-9))
            out.append(np.log(p / (1 - p)))
        else:
            out.append(float(value))
    return np.asarray(out)


def _from_unconstrained(z: Sequence[float], kinds: Sequence[str]) -> np.ndarray:
    out = []
    for value, kind in zip(z, kinds):
        if kind == "log":
            out.append(np.exp(np.clip(value, -700, 700)))
        elif kind == "logit":
            out.append(1.0 / (1.0 + np.exp(-np.clip(value, -700, 700))))
        else:
            out.append(float(value))
    return np.asarray(out)


def _optimise(
    spec: DistributionSpec,
    objective: Callable[[np.ndarray], float],
    start: Sequence[float],
    maxiter: int = 4000,
) -> np.ndarray:
    """Minimise ``objective`` over R's parameters, from ``start``.

    ``objective`` receives the R parameter vector.  Nelder-Mead is used because
    none of these objectives has a usable gradient -- the quantile-matching and
    goodness-of-fit ones involve sorting and clipping.
    """
    kinds = _TRANSFORMS.get(spec.r_name, ("free",) * len(start))

    def wrapped(z):
        values = _from_unconstrained(z, kinds)
        try:
            result = objective(values)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return _BIG
        return _BIG if not np.isfinite(result) else float(result)

    z0 = _to_unconstrained(start, kinds)
    if not np.isfinite(wrapped(z0)):
        raise ValueError(
            f"The starting values for the {spec.r_name} distribution give a "
            "non-finite objective; supply your own via 'start'"
        )

    outcome = optimize.minimize(
        wrapped, z0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-12},
    )
    if not np.isfinite(outcome.fun):
        raise ValueError(f"Estimation did not converge for the {spec.r_name} distribution")
    return _from_unconstrained(outcome.x, kinds)


def _starting_values(model, data: np.ndarray, spec: DistributionSpec) -> np.ndarray:
    """Sensible starting parameters, in R's parameterisation.

    Moment matching first, since it is closed-form and cannot fail to converge;
    maximum likelihood as a fallback for the distributions it does not cover.
    """
    try:
        closed = _closed_form_moments(spec, data)
        if closed is not None and np.all(np.isfinite(closed)):
            return closed
    except ValueError:
        pass
    _, params = fit_distribution(model, data)
    return np.asarray(list(r_estimate(spec.r_name, params, spec).values()), dtype=float)


def _prepare(model, data, method: str) -> Tuple[object, DistributionSpec, np.ndarray]:
    dist, spec = resolve_distribution(model)
    if spec is None:
        raise ValueError(
            f"method='{method}' needs a registered distribution, so that the "
            "estimate can be reported in R's parameterisation"
        )
    clean = np.asarray(data, dtype=float)
    if spec.support is not None:
        low, high = spec.support
        if clean.min() < low or clean.max() > high:
            raise ValueError(
                f"The {spec.r_name} distribution is defined on [{low}, {high}]; "
                f"the data range [{clean.min():.4g}, {clean.max():.4g}] falls outside it"
            )
    return dist, spec, clean


# ---------------------------------------------------------------------------
# Moment matching -- mmedist
# ---------------------------------------------------------------------------

def _closed_form_moments(spec: DistributionSpec, data: np.ndarray) -> Optional[np.ndarray]:
    """R's closed-form moment estimators, for the ten distributions that have them.

    ``m`` and ``v`` are the sample mean and the *population* variance, which is
    what ``mmedist`` uses: ``v <- (n - 1)/n * var(data)``.
    """
    m = float(np.mean(data))
    v = float(np.var(data, ddof=0))
    name = spec.r_name

    if name == "norm":
        return np.array([m, np.sqrt(v)])
    if name == "lnorm":
        if np.any(data <= 0):
            raise ValueError("values must be positive to fit a lognormal distribution")
        sd2 = np.log(1 + v / m ** 2)
        return np.array([np.log(m) - sd2 / 2, np.sqrt(sd2)])
    if name == "pois":
        return np.array([m])
    if name == "exp":
        return np.array([1.0 / m])
    if name == "gamma":
        return np.array([m ** 2 / v, m / v])
    if name == "nbinom":
        if v <= m:
            raise ValueError(
                "moment matching needs the variance to exceed the mean for a "
                "negative binomial; this sample is not over-dispersed"
            )
        return np.array([m ** 2 / (v - m), m])
    if name == "geom":
        if m <= 0:
            raise ValueError("moment matching needs a positive mean for a geometric")
        return np.array([1.0 / (1.0 + m)])
    if name == "beta":
        aux = m * (1 - m) / v - 1
        return np.array([m * aux, (1 - m) * aux])
    if name == "unif":
        return np.array([m - np.sqrt(3 * v), m + np.sqrt(3 * v)])
    if name == "logis":
        return np.array([m, np.sqrt(3 * v) / np.pi])
    return None


def moment_match(
    model: Union[str, object],
    data: np.ndarray,
    order: Optional[Sequence[int]] = None,
) -> tuple:
    """Fit by matching moments -- R's ``mmedist``.

    Ten distributions have closed-form estimators, which R uses directly and so
    does this; for the rest the raw moments of order ``1 .. npar`` are matched
    numerically.

    Parameters
    ----------
    model : str or scipy distribution
        Distribution to fit.
    data : array-like
        Input data.
    order : sequence of int, optional
        Moment orders to match.  Defaults to ``1 .. npar``.  Ignored where a
        closed form applies, as in R.

    Returns
    -------
    tuple
        The full scipy parameter tuple.
    """
    dist, spec, clean = _prepare(model, data, "mme")

    if order is None:
        closed = _closed_form_moments(spec, clean)
        if closed is not None:
            return scipy_params(spec.r_name, closed, spec)

    npar = spec.n_free_params
    orders = np.asarray(order if order is not None else range(1, npar + 1), dtype=int)
    if len(orders) < npar:
        raise ValueError(
            f"{len(orders)} moment(s) given for {npar} parameters; matching needs "
            "at least as many moments as parameters"
        )
    empirical = np.array([np.mean(clean ** k) for k in orders])

    def objective(values):
        params = scipy_params(spec.r_name, values, spec)
        with np.errstate(over="ignore", invalid="ignore"):
            theoretical = np.array([dist.moment(int(k), *params) for k in orders])
        if not np.all(np.isfinite(theoretical)):
            return np.inf
        # Relative, so that a fourth moment does not swamp a first.
        return float(np.sum(((theoretical - empirical) / np.maximum(np.abs(empirical), 1e-12)) ** 2))

    best = _optimise(spec, objective, _starting_values(model, clean, spec))
    return scipy_params(spec.r_name, best, spec)


# ---------------------------------------------------------------------------
# Quantile matching -- qmedist
# ---------------------------------------------------------------------------

def quantile_match(
    model: Union[str, object],
    data: np.ndarray,
    probs: Optional[Sequence[float]] = None,
) -> tuple:
    """Fit by matching quantiles -- R's ``qmedist``.

    Minimises the mean squared difference between the fitted and the sample
    quantiles at ``probs``.  Useful when the fit only has to be right over part
    of the range: match at 0.9 and 0.99 and the tail is what gets fitted.

    Parameters
    ----------
    model : str or scipy distribution
        Distribution to fit.
    data : array-like
        Input data.
    probs : sequence of float, optional
        Probabilities to match at.  R requires as many as there are parameters
        and has no default; this defaults to ``i / (npar + 1)``, which spreads
        them evenly.  Empirical quantiles use linear interpolation, R's
        ``type = 7``.

    Returns
    -------
    tuple
        The full scipy parameter tuple.
    """
    dist, spec, clean = _prepare(model, data, "qme")
    npar = spec.n_free_params

    if probs is None:
        probs = np.arange(1, npar + 1) / (npar + 1)
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    if np.any((probs <= 0) | (probs >= 1)):
        raise ValueError("probs must lie strictly between 0 and 1")
    if len(probs) < npar:
        raise ValueError(
            f"{len(probs)} probability/probabilities given for {npar} parameters; "
            "quantile matching needs at least as many as there are parameters"
        )

    empirical = np.quantile(clean, probs, method="linear")

    def objective(values):
        params = scipy_params(spec.r_name, values, spec)
        with np.errstate(over="ignore", invalid="ignore"):
            theoretical = dist.ppf(probs, *params)
        if not np.all(np.isfinite(theoretical)):
            return np.inf
        return float(np.mean((empirical - theoretical) ** 2))

    best = _optimise(spec, objective, _starting_values(model, clean, spec))
    return scipy_params(spec.r_name, best, spec)


# ---------------------------------------------------------------------------
# Maximum goodness-of-fit -- mgedist
# ---------------------------------------------------------------------------

def _gof_objective(gof: str, theop: np.ndarray, n: int) -> float:
    """The statistic ``mgedist`` minimises, for a given fitted CDF at the data.

    Transcribed from ``mgedist``; the ``idx`` masking guards the logs, which go
    infinite whenever a fitted probability reaches 0 or 1.
    """
    i = np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if gof == "CvM":
            return 1.0 / (12 * n ** 2) + float(np.mean((theop - (2 * i - 1) / (2 * n)) ** 2))

        if gof == "KS":
            upper, lower = i / n, (i - 1) / n
            return float(np.max(np.maximum(np.abs(theop - upper), np.abs(theop - lower))))

        if gof == "AD":
            term = np.log(theop * (1 - theop[::-1])) * (2 * i - 1)
            ok = np.isfinite(term)
            return -ok.sum() / n - float(np.mean(term[ok])) / n if ok.any() else np.inf

        if gof == "ADR":
            term = np.log(1 - theop[::-1]) * (2 * i - 1)
            ok = np.isfinite(term)
            if not ok.any():
                return np.inf
            return ok.sum() / 2 / n - 2 * float(np.sum(theop[ok])) / n - float(np.mean(term[ok])) / n

        if gof == "ADL":
            term = (2 * i - 1) * np.log(theop)
            ok = np.isfinite(term)
            if not ok.any():
                return np.inf
            return -3 * ok.sum() / 2 / n + 2 * float(np.sum(theop[ok])) / n - float(np.mean(term[ok])) / n

        if gof == "AD2R":
            logpi = np.log(1 - theop)
            i1pi2 = (2 * i - 1) / (1 - theop[::-1])
            ok = np.isfinite(logpi) & np.isfinite(i1pi2)
            if not ok.any():
                return np.inf
            return 2 * float(np.sum(logpi[ok])) / n + float(np.mean(i1pi2[ok])) / n

        if gof == "AD2L":
            logpi = np.log(theop)
            i1pi = (2 * i - 1) / theop
            ok = np.isfinite(logpi) & np.isfinite(i1pi)
            if not ok.any():
                return np.inf
            return 2 * float(np.sum(logpi[ok])) / n + float(np.mean(i1pi[ok])) / n

        if gof == "AD2":
            logpi = np.log(theop * (1 - theop))
            i1pi = (2 * i - 1) / theop
            i1pi2 = (2 * i - 1) / (1 - theop[::-1])
            ok = np.isfinite(logpi) & np.isfinite(i1pi) & np.isfinite(i1pi2)
            if not ok.any():
                return np.inf
            return 2 * float(np.sum(logpi[ok])) / n + float(np.mean(i1pi[ok] + i1pi2[ok])) / n

    raise ValueError(f"Unknown goodness-of-fit statistic: '{gof}'")


def maximum_goodness_of_fit(
    model: Union[str, object],
    data: np.ndarray,
    gof: str = "CvM",
) -> tuple:
    """Fit by minimising a goodness-of-fit statistic -- R's ``mgedist``.

    Optimises the statistic the fit will be judged by, rather than the
    likelihood.  ``'ADR'`` and ``'AD2R'`` weight the right tail, which is the
    reason to reach for this when the tail is what matters.

    Parameters
    ----------
    model : str or scipy distribution
        Distribution to fit.
    data : array-like
        Input data.
    gof : str, default='CvM'
        One of :data:`GOF_STATISTICS`.  ``'CvM'`` is Cramer-von Mises, R's
        default; ``'KS'`` Kolmogorov-Smirnov; ``'AD'`` Anderson-Darling, with
        ``R``/``L`` for the right and left tail and ``2`` for the squared
        variants that weight the extremes harder still.

    Returns
    -------
    tuple
        The full scipy parameter tuple.
    """
    dist, spec, clean = _prepare(model, data, "mge")
    if gof not in GOF_STATISTICS:
        raise ValueError(
            f"gof must be one of {', '.join(GOF_STATISTICS)}; got '{gof}'"
        )
    if spec.discrete:
        raise ValueError(
            "maximum goodness-of-fit estimation needs a continuous distribution: "
            "its statistics compare an empirical step function against a smooth one"
        )

    sdata = np.sort(clean)
    n = len(sdata)

    def objective(values):
        params = scipy_params(spec.r_name, values, spec)
        with np.errstate(over="ignore", invalid="ignore"):
            theop = dist.cdf(sdata, *params)
        if not np.all(np.isfinite(theop)):
            return np.inf
        return _gof_objective(gof, theop, n)

    best = _optimise(spec, objective, _starting_values(model, clean, spec))
    return scipy_params(spec.r_name, best, spec)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def fit_by_method(
    model: Union[str, object],
    data: np.ndarray,
    method: str = "mle",
    **kwargs,
) -> Tuple[object, tuple]:
    """Fit ``model`` by the named method -- R's ``fitdist(..., method = ...)``.

    Parameters
    ----------
    model : str or scipy distribution
        Distribution to fit.
    data : array-like
        Input data.
    method : {'mle', 'mme', 'qme', 'mge'}, default='mle'
        Maximum likelihood, moment matching, quantile matching, or maximum
        goodness-of-fit.
    **kwargs
        Passed to the estimator: ``order`` for ``'mme'``, ``probs`` for
        ``'qme'``, ``gof`` for ``'mge'``.

    Returns
    -------
    (distribution, params)
        The scipy distribution and its full parameter tuple.
    """
    if method not in ESTIMATION_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(ESTIMATION_METHODS)}; got '{method}'"
        )
    dist, _ = resolve_distribution(model)

    if method == "mle":
        if kwargs:
            raise TypeError(
                f"method='mle' takes no extra arguments; got {', '.join(kwargs)}"
            )
        return fit_distribution(model, data)
    if method == "mme":
        return dist, moment_match(model, data, **kwargs)
    if method == "qme":
        return dist, quantile_match(model, data, **kwargs)
    return dist, maximum_goodness_of_fit(model, data, **kwargs)
