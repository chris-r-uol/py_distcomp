"""
Finite mixture fitting -- a population as a superposition of distributions.

The off-model method of Rushton et al. (2021) splits a population with a hard
percentile cut: everything below the cut is fitted as one distribution, and
everything above it as another.  That works, but the cut is chosen by a search
over percentiles rather than by the likelihood, an observation belongs wholly
to one component or the other, and the two fits never inform each other.

This module fits the same superposition properly:

    f(x) = w₁ f₁(x; θ₁) + w₂ f₂(x; θ₂)

by expectation-maximisation, so the weights, both components' parameters and the
assignment of observations are all estimated jointly.  Nothing is discarded, and
every observation gets a *probability* of belonging to each component rather
than a hard label -- which is what "high-chance gross emitting vehicle" asks
for.  The components need not be from the same family.

The hard cut remains useful for initialising the EM, and that is what
``init='off_model'`` does.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

from .distributions import density, fit_distribution, is_discrete, log_density, resolve_distribution
from .gofstat import _r_estimate
from .off_model import off_model_fraction

__all__ = ["MixtureDistribution", "MixtureResult", "fit_mixture"]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# The fitted mixture, exposed through the scipy distribution interface
# ---------------------------------------------------------------------------

class MixtureDistribution:
    """A frozen mixture that behaves like a scipy distribution.

    ``pdf``, ``cdf``, ``ppf``, ``logpdf`` and ``rvs`` accept and ignore trailing
    positional parameters, so the object can be handed to anything in this
    package that expects ``dist.method(x, *params)`` -- the Q-Q, P-P, CDF and
    density comparison plots, and :func:`~py_distcomp.gofstat`.

    Parameters
    ----------
    components : sequence of (distribution, params)
        The component distributions and their full scipy parameter tuples.
    weights : sequence of float
        Mixing weights; normalised to sum to one.
    """

    # Declares to the comparison plots that this takes no external parameters.
    shapes = None
    n_params = 0

    def __init__(self, components: Sequence[Tuple[object, tuple]], weights: Sequence[float]):
        self.components = [(dist, tuple(params)) for dist, params in components]
        w = np.asarray(weights, dtype=float)
        if len(w) != len(self.components):
            raise ValueError("weights must have one entry per component")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = w / total
        self.name = "+".join(getattr(d, "name", "?") for d, _ in self.components)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def _stack(self, method: str, x: np.ndarray) -> np.ndarray:
        """(n_components, n) array of a per-component method applied to x."""
        if method == "pdf":
            # density() picks pmf for a discrete component.
            return np.vstack([density(d, x, p) for d, p in self.components])
        return np.vstack([
            getattr(dist, method)(x, *params) for dist, params in self.components
        ])

    def pdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.tensordot(self.weights, self._stack("pdf", x), axes=(0, 0))

    def logpdf(self, x, *_) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(self.pdf(x), _TINY))

    def cdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.tensordot(self.weights, self._stack("cdf", x), axes=(0, 0))

    def sf(self, x, *_) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def ppf(self, q, *_, tol: float = 1e-10, maxiter: int = 200) -> np.ndarray:
        """Quantile function, by bisection on the mixture CDF.

        A mixture has no closed-form quantile function, so this inverts the CDF
        numerically.  The bracket is taken from the component quantile functions,
        which necessarily straddle the mixture's.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.full(q.shape, np.nan)

        valid = (q > 0) & (q < 1)
        out[q == 0] = -np.inf
        out[q == 1] = np.inf
        if not np.any(valid):
            return out if out.size > 1 else out.item()

        qv = q[valid]
        eps = min(np.min(qv), 1 - np.max(qv)) / 10
        eps = max(min(eps, 1e-10), 1e-300)
        lo = np.min([dist.ppf(eps, *p) for dist, p in self.components])
        hi = np.max([dist.ppf(1 - eps, *p) for dist, p in self.components])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("Component quantiles are unbounded; cannot invert the mixture CDF")

        lower = np.full(qv.shape, float(lo))
        upper = np.full(qv.shape, float(hi))
        for _ in range(maxiter):
            mid = 0.5 * (lower + upper)
            too_low = self.cdf(mid) < qv
            lower = np.where(too_low, mid, lower)
            upper = np.where(too_low, upper, mid)
            if np.all(upper - lower < tol * np.maximum(1.0, np.abs(mid))):
                break

        out[valid] = 0.5 * (lower + upper)
        return out if out.size > 1 else out.item()

    def rvs(self, size=1, random_state=None, *_) -> np.ndarray:
        """Draw from the mixture: pick a component, then draw from it."""
        rng = np.random.default_rng(random_state)
        size = int(size)
        which = rng.choice(self.n_components, size=size, p=self.weights)
        out = np.empty(size)
        for k, (dist, params) in enumerate(self.components):
            mask = which == k
            n = int(mask.sum())
            if n:
                out[mask] = dist.rvs(*params, size=n, random_state=rng)
        return out

    def responsibilities(self, x) -> np.ndarray:
        """Posterior probability that each observation came from each component.

        Returns an ``(n, n_components)`` array whose rows sum to one.
        """
        x = np.asarray(x, dtype=float)
        weighted = self._stack("pdf", x) * self.weights[:, None]
        total = weighted.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(total > 0, weighted / np.maximum(total, _TINY), np.nan)
        # Where every component has zero density, fall back to the prior.
        degenerate = total <= 0
        if np.any(degenerate):
            out[:, degenerate] = self.weights[:, None]
        return out.T

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = ", ".join(
            f"{w:.3f}·{getattr(d, 'name', '?')}" for w, (d, _) in zip(self.weights, self.components)
        )
        return f"MixtureDistribution({parts})"


# ---------------------------------------------------------------------------
# Weighted maximum likelihood -- the M step
# ---------------------------------------------------------------------------

def _free_parameter_layout(dist, spec) -> Tuple[List[int], List[bool], int]:
    """Which scipy parameters are estimated, and which must stay positive.

    Returns ``(indices, positive, n_total)``.  Shape parameters and the scale
    are strictly positive for every distribution in the registry, so they are
    optimised on a log scale; the location is unconstrained.
    """
    n_shapes = len(dist.shapes.split(",")) if dist.shapes else 0
    discrete = spec.discrete if spec is not None else is_discrete(dist)
    n_total = n_shapes + (1 if discrete else 2)
    loc_i = n_shapes
    scale_i = -1 if discrete else n_shapes + 1

    fixed = spec.fixed if spec is not None else {}
    indices, positive = [], []
    for i in range(n_total):
        if i == loc_i and "floc" in fixed:
            continue
        if i == scale_i and "fscale" in fixed:
            continue
        indices.append(i)
        positive.append(i != loc_i)
    return indices, positive, n_total


def _weighted_mle(dist, spec, data: np.ndarray, weights: np.ndarray, start: tuple) -> tuple:
    """Maximise the weighted log-likelihood, holding pinned parameters fixed.

    scipy's ``.fit`` takes no observation weights, so the M step optimises the
    weighted negative log-likelihood directly.  Positive parameters are
    optimised on a log scale so the search stays in the valid region.
    """
    indices, positive, _ = _free_parameter_layout(dist, spec)
    params = list(start)
    if not indices:
        return tuple(params)

    def unpack(z):
        out = list(params)
        for value, i, pos in zip(z, indices, positive):
            out[i] = np.exp(value) if pos else value
        return tuple(out)

    def negll(z):
        trial = unpack(z)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            logpdf = log_density(dist, data, trial)
        if not np.all(np.isfinite(logpdf)):
            return np.inf
        return -float(np.sum(weights * logpdf))

    z0 = np.array([
        np.log(max(params[i], 1e-12)) if pos else params[i]
        for i, pos in zip(indices, positive)
    ])
    if not np.isfinite(negll(z0)):
        return tuple(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize(negll, z0, method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})

    return unpack(result.x) if result.success or np.isfinite(result.fun) else tuple(params)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

class MixtureResult:
    """A fitted mixture.

    Duck-types :class:`~py_distcomp.FitResult`, so it can be passed straight to
    :func:`~py_distcomp.gofstat` alongside single-distribution fits and compared
    on AIC and BIC.

    Attributes
    ----------
    dist : MixtureDistribution
        The fitted mixture, usable anywhere a scipy distribution is.
    params : tuple
        Empty; the mixture carries its own parameters.
    weights : np.ndarray
        Mixing weights, in the order the components were given.
    components : list of (distribution, params)
        Component distributions and their full scipy parameter tuples.
    estimate : dict
        Flat mapping of every estimated quantity, in R's parameterisation,
        suffixed by component number.
    loglik, aic, bic : float
        ``aic = -2 loglik + 2 npar``, ``bic = -2 loglik + log(n) npar``, with
        ``npar`` counting both components' free parameters plus ``K - 1``
        weights.
    converged : bool
    n_iter : int
        EM iterations used.
    """

    def __init__(self, dist, model_names, data, loglik, n_free_params,
                 converged, n_iter, history):
        self.dist = dist
        self.params: tuple = ()
        self.model_names = list(model_names)
        self.name = " + ".join(self.model_names)
        self.r_name = self.name
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        self.loglik = float(loglik)
        self.n_free_params = int(n_free_params)
        self.aic = -2.0 * self.loglik + 2.0 * self.n_free_params
        self.bic = -2.0 * self.loglik + np.log(self.n) * self.n_free_params
        self.converged = bool(converged)
        self.n_iter = int(n_iter)
        self.history = list(history)
        self.estimate = self._build_estimate()

    @property
    def weights(self) -> np.ndarray:
        return self.dist.weights

    @property
    def components(self):
        return self.dist.components

    @property
    def n_components(self) -> int:
        return self.dist.n_components

    def _build_estimate(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, (name, (dist, params)) in enumerate(zip(self.model_names, self.dist.components), 1):
            _, spec = resolve_distribution(name)
            r_name = spec.r_name if spec is not None else getattr(dist, "name", name)
            out[f"weight{k}"] = float(self.dist.weights[k - 1])
            for key, value in _r_estimate(r_name, params, spec).items():
                out[f"{key}{k}"] = value
        return out

    def responsibilities(self, x=None) -> np.ndarray:
        """Posterior component probabilities, ``(n, n_components)``.

        Defaults to the data the mixture was fitted to.
        """
        return self.dist.responsibilities(self.data if x is None else x)

    def component_probability(self, x=None, component: int = -1) -> np.ndarray:
        """Probability that each observation belongs to one component.

        With the default ``component=-1`` this is the probability of belonging
        to the last component -- the high-valued one when components are
        supplied in ascending order, so the per-observation analogue of the
        paper's off-model classification.
        """
        return self.responsibilities(x)[:, component]

    def classify(self, x=None, threshold: float = 0.5, component: int = -1) -> np.ndarray:
        """Boolean flag for observations whose component probability exceeds a threshold."""
        return self.component_probability(x, component) >= threshold

    def expected_counts(self) -> np.ndarray:
        """Expected number of observations from each component."""
        return self.responsibilities().sum(axis=0)

    def summary(self) -> pd.Series:
        row = {
            "model": self.name,
            "n": self.n,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "n_free_params": self.n_free_params,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }
        row.update(self.estimate)
        return pd.Series(row)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = ", ".join(
            f"{w:.3f}·{n}" for w, n in zip(self.dist.weights, self.model_names)
        )
        return (f"MixtureResult({parts}; loglik={self.loglik:.1f}, "
                f"aic={self.aic:.1f}, converged={self.converged})")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _clean(data) -> np.ndarray:
    series = pd.Series(data).dropna()
    if len(series) < 4:
        raise ValueError("At least 4 non-NA data points are required")
    return np.sort(np.asarray(series.to_numpy(), dtype=float))


def _split_starts(data: np.ndarray, models, init, min_points: int) -> List[List[np.ndarray]]:
    """Candidate partitions of the data used to seed the EM."""
    n_comp = len(models)
    candidates: List[List[np.ndarray]] = []

    if init in ("off_model", "auto") and n_comp == 2:
        try:
            result = off_model_fraction(data, models[0], min_points=min_points)
            if 0 < result.n_off_model < len(data):
                candidates.append([data[data <= result.threshold],
                                   data[data > result.threshold]])
        except (ValueError, RuntimeError):
            pass

    if init in ("quantile", "auto", "off_model"):
        # Equal-mass split, plus a couple of tail-heavy alternatives for the
        # asymmetric case the off-model method targets.
        fractions = [np.linspace(0, 1, n_comp + 1)[1:-1]]
        if n_comp == 2:
            fractions += [np.array([0.9]), np.array([0.75])]
        for cuts in fractions:
            edges = [-np.inf] + [float(np.quantile(data, c)) for c in cuts] + [np.inf]
            parts = [data[(data > lo) & (data <= hi)] for lo, hi in zip(edges[:-1], edges[1:])]
            if all(len(p) >= min_points for p in parts):
                candidates.append(parts)

    if not candidates:
        raise ValueError(
            f"Could not build a starting partition with at least {min_points} "
            f"points per component from n = {len(data)}"
        )
    return candidates


def fit_mixture(
    data: Union[np.ndarray, pd.Series, list],
    models: Union[str, Sequence] = ("gumbel", "gumbel"),
    init: str = "auto",
    max_iter: int = 500,
    tol: float = 1e-8,
    min_points: int = 5,
    min_weight: float = 1e-4,
) -> MixtureResult:
    """Fit a superposition of distributions by expectation-maximisation.

    Estimates the mixing weights and every component's parameters jointly, so
    each observation contributes to both components in proportion to how likely
    it is to have come from each -- unlike a hard percentile cut, where an
    observation belongs entirely to one side.

    Parameters
    ----------
    data : array-like
        Input data.
    models : sequence, default=('gumbel', 'gumbel')
        One distribution name or scipy object per component.  Families may
        differ, e.g. ``('gumbel', 'normal')``.  Two components describe the
        bulk-plus-outlier structure of Rushton et al. (2021).
    init : {'auto', 'off_model', 'quantile'} or MixtureResult, default='auto'
        How to seed the EM.  ``'off_model'`` uses the percentile cut of
        :func:`~py_distcomp.off_model_fraction`, the paper's own split.
        ``'quantile'`` uses equal-mass and tail-heavy splits.  ``'auto'`` tries
        both and keeps whichever converges to the higher likelihood.  EM finds
        a local optimum, so the starting point matters.

        Passing an already fitted mixture warm-starts from its parameters and
        runs a single EM pass, which is much faster.  This is how
        :func:`~py_distcomp.bootdist` refits a mixture on each resample.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-8
        Convergence tolerance on the relative change in log-likelihood.
    min_points : int, default=5
        Minimum observations per component in the starting partition.
    min_weight : float, default=1e-4
        Floor on the mixing weights; a component that collapses below this is
        reported as fitted but the fit should be treated with suspicion.

    Returns
    -------
    MixtureResult

    Notes
    -----
    Components are returned in the order given.  When seeded from a split they
    come out in ascending order of location, so the last component is the
    high-valued one and ``component_probability()`` gives the probability of
    belonging to it.

    Examples
    --------
    >>> mix = fit_mixture(emission_ratios, ('gumbel', 'gumbel'))
    >>> mix.weights
    array([0.951, 0.049])
    >>> mix.estimate['loc1'], mix.estimate['loc2']
    (11.2, 84.9)
    >>> mix.component_probability()[-5:]   # chance of being off-model
    array([0.97, 0.98, 0.99, 0.99, 1.0])

    Compare against a single distribution on the same footing:

    >>> gofstat([fit_distributions(x, 'gumbel')[0], mix])[['aic', 'bic']]
    """
    clean = _clean(data)
    if isinstance(models, (str, bytes)) or not isinstance(models, (list, tuple)):
        models = [models]
    models = list(models)
    if len(models) < 2:
        raise ValueError("A mixture needs at least two components")

    warm_start = _as_warm_start(init)
    if warm_start is None and init not in ("auto", "off_model", "quantile"):
        raise ValueError(
            "init must be 'auto', 'off_model', 'quantile', or a fitted mixture "
            "to start from"
        )

    specs = [resolve_distribution(m) for m in models]
    n_free = sum(
        len(_free_parameter_layout(dist, spec)[0]) for dist, spec in specs
    ) + (len(models) - 1)

    if warm_start is not None:
        if warm_start.n_components != len(models):
            raise ValueError(
                f"The mixture to start from has {warm_start.n_components} "
                f"components but {len(models)} were requested"
            )
        starts = [(list(warm_start.components), np.array(warm_start.weights))]
    else:
        starts = [
            _initial_from_partition(models, partition, len(clean), min_weight)
            for partition in _split_starts(clean, models, init, min_points)
        ]

    best = None
    for components, weights in starts:
        try:
            candidate = _run_em(clean, models, specs, components, weights,
                                max_iter, tol, min_weight, n_free)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if candidate is not None and (best is None or candidate.loglik > best.loglik):
            best = candidate

    if best is None:
        raise ValueError(
            "Expectation-maximisation failed from every starting partition; "
            "try different component families or a different init"
        )
    if np.any(best.weights < min_weight):
        warnings.warn(
            "A mixture component collapsed to a negligible weight; the data may "
            "not support this many components",
            RuntimeWarning,
            stacklevel=2,
        )
    return best


def _as_warm_start(init):
    """Return the mixture to start from, when ``init`` names one."""
    if isinstance(init, MixtureDistribution):
        return init
    dist = getattr(init, "dist", None)
    return dist if isinstance(dist, MixtureDistribution) else None


def _initial_from_partition(models, partition, n, min_weight):
    """Fit each component to its slice of a starting partition."""
    components: List[Tuple[object, tuple]] = []
    weights = []
    for model, part in zip(models, partition):
        dist, params = fit_distribution(model, part)
        components.append((dist, params))
        weights.append(max(len(part) / n, min_weight))
    return components, np.asarray(weights) / np.sum(weights)


def _run_em(data, models, specs, components, weights, max_iter, tol, min_weight, n_free):
    """One EM run from given starting parameters."""
    components = list(components)
    weights = np.asarray(weights, dtype=float)

    history: List[float] = []
    previous = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # E step: responsibilities under the current parameters.
        with np.errstate(under="ignore"):
            densities = np.vstack([density(d, data, p) for d, p in components])
        weighted = densities * weights[:, None]
        total = weighted.sum(axis=0)
        if not np.all(np.isfinite(total)) or np.all(total <= 0):
            return None
        loglik = float(np.sum(np.log(np.maximum(total, _TINY))))
        history.append(loglik)

        resp = weighted / np.maximum(total, _TINY)

        if previous > -np.inf:
            change = abs(loglik - previous) / max(1.0, abs(previous))
            if change < tol:
                converged = True
                break
        previous = loglik

        # M step: weighted MLE for each component, and the new weights.
        weights = np.maximum(resp.mean(axis=1), min_weight)
        weights = weights / weights.sum()
        components = [
            (dist, _weighted_mle(dist, spec, data, resp[k], params))
            for k, ((dist, spec), (_, params)) in enumerate(zip(specs, components))
        ]

    mixture = MixtureDistribution(components, weights)
    final_loglik = float(np.sum(mixture.logpdf(data)))

    model_names = [m if isinstance(m, str) else getattr(d, "name", str(m))
                   for m, (d, _) in zip(models, components)]

    return MixtureResult(mixture, model_names, data, final_loglik,
                         n_free, converged, iteration, history)
