"""
Distribution registry and maximum-likelihood fitting.

The parameterisations here follow R's ``fitdistrplus``, which in turn uses the
d/p/q functions of base R (and of ``actuar``/``VGAM`` for a few extras).  Several
of those R densities have no location or scale argument at all -- ``dlnorm`` has
only ``meanlog``/``sdlog``, ``dgamma`` only ``shape``/``rate``, ``dchisq`` only
``df`` -- whereas every ``scipy.stats`` continuous distribution carries a ``loc``
and a ``scale``.  Fitting the scipy versions with those extra parameters free
would silently estimate a different (larger) model than ``fitdist`` does, so the
registry pins them to the values R implies.

``ppoints`` is also defined here because ``qqcomp``, ``ppcomp`` and ``cdfcomp``
all place their empirical points with it.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

__all__ = [
    "DistributionSpec",
    "SUPPORTED_DISTRIBUTIONS",
    "DISTRIBUTION_SPECS",
    "ppoints",
    "resolve_distribution",
    "fit_distribution",
    "loglik",
    "aic_bic",
    "r_estimate",
    "scipy_params",
    "observed_information",
]


class DistributionSpec:
    """A scipy distribution together with the constraints R's version implies.

    Parameters
    ----------
    dist : scipy.stats rv_continuous
        The scipy distribution object.
    r_name : str
        Name of the equivalent R distribution, as ``fitdist`` would spell it.
    fixed : dict
        Keyword arguments forced on ``dist.fit`` (e.g. ``{'floc': 0}``) so that
        the estimated model matches R's parameterisation.
    r_params : tuple of str
        Names of the parameters R estimates, in R's order.  Used for reporting
        and for counting free parameters in AIC/BIC.
    support : tuple of (float, float) or None
        Range the data must lie in, when R's version is restricted (beta only).
    """

    def __init__(self, dist, r_name, fixed=None, r_params=(), support=None):
        self.dist = dist
        self.r_name = r_name
        self.fixed = dict(fixed or {})
        self.r_params = tuple(r_params)
        self.support = support

    @property
    def n_free_params(self) -> int:
        """Number of parameters actually estimated -- R's ``length(estimate)``."""
        return len(self.r_params)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DistributionSpec({self.r_name!r}, fixed={self.fixed!r})"


# Keys are the names this package exposes; ``r_name`` records what R calls the
# same distribution.  ``fixed`` encodes the loc/scale parameters that R's density
# does not have and that therefore must not be estimated.
DISTRIBUTION_SPECS: Dict[str, DistributionSpec] = {
    # --- distributions fitdistrplus documents and exercises directly ---------
    "normal": DistributionSpec(
        stats.norm, "norm", {}, ("mean", "sd")
    ),
    "lognormal": DistributionSpec(
        stats.lognorm, "lnorm", {"floc": 0}, ("meanlog", "sdlog")
    ),
    "weibull": DistributionSpec(
        stats.weibull_min, "weibull", {"floc": 0}, ("shape", "scale")
    ),
    "gamma": DistributionSpec(
        stats.gamma, "gamma", {"floc": 0}, ("shape", "rate")
    ),
    "exponential": DistributionSpec(
        stats.expon, "exp", {"floc": 0}, ("rate",)
    ),
    "uniform": DistributionSpec(
        stats.uniform, "unif", {}, ("min", "max")
    ),
    "logistic": DistributionSpec(
        stats.logistic, "logis", {}, ("location", "scale")
    ),
    "beta": DistributionSpec(
        stats.beta, "beta", {"floc": 0, "fscale": 1}, ("shape1", "shape2"),
        support=(0.0, 1.0),
    ),
    "cauchy": DistributionSpec(
        stats.cauchy, "cauchy", {}, ("location", "scale")
    ),
    # --- available in R via actuar / VGAM / evd, same parameterisation ------
    "gumbel": DistributionSpec(
        stats.gumbel_r, "gumbel", {}, ("loc", "scale")
    ),
    "laplace": DistributionSpec(
        stats.laplace, "laplace", {}, ("location", "scale")
    ),
    "pareto": DistributionSpec(
        stats.pareto, "pareto", {"floc": 0}, ("shape", "scale")
    ),
    "rayleigh": DistributionSpec(
        stats.rayleigh, "rayleigh", {"floc": 0}, ("scale",)
    ),
    "chi2": DistributionSpec(
        stats.chi2, "chisq", {"floc": 0, "fscale": 1}, ("df",)
    ),
    "student_t": DistributionSpec(
        stats.t, "t", {"floc": 0, "fscale": 1}, ("df",)
    ),
    "f": DistributionSpec(
        stats.f, "f", {"floc": 0, "fscale": 1}, ("df1", "df2")
    ),
}

# Backwards-compatible view: name -> scipy distribution object.
SUPPORTED_DISTRIBUTIONS: Dict[str, object] = {
    name: spec.dist for name, spec in DISTRIBUTION_SPECS.items()
}

# Accept R's own spellings as aliases, so code ported from R reads the same.
_R_ALIASES: Dict[str, str] = {
    spec.r_name: name for name, spec in DISTRIBUTION_SPECS.items()
}
_R_ALIASES.update({"norm": "normal", "unif": "uniform", "expon": "exponential"})


def ppoints(n: int, a: Optional[float] = None) -> np.ndarray:
    """R's ``ppoints``: plotting positions ``(i - a) / (n + 1 - 2a)``.

    Parameters
    ----------
    n : int
        Number of points.
    a : float, optional
        Offset.  R's default is ``3/8`` for ``n <= 10`` and ``1/2`` otherwise;
        every ``*comp`` plot in fitdistrplus overrides it to ``0.5`` via the
        ``a.ppoints`` argument, which is what the plotting code here passes.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if a is None:
        a = 3.0 / 8.0 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def resolve_distribution(model: Union[str, object]) -> Tuple[object, Optional[DistributionSpec]]:
    """Turn a model specification into a ``(scipy distribution, spec)`` pair.

    ``spec`` is ``None`` for a user-supplied scipy distribution object that is
    not in the registry, in which case no R parameterisation is implied.
    """
    if isinstance(model, str):
        key = model.lower().strip()
        key = _R_ALIASES.get(key, key)
        if key not in DISTRIBUTION_SPECS:
            supported = ", ".join(sorted(DISTRIBUTION_SPECS))
            raise ValueError(
                f"Unsupported distribution name: '{model}'. "
                f"Supported: {supported}"
            )
        spec = DISTRIBUTION_SPECS[key]
        return spec.dist, spec

    if hasattr(model, "ppf") and hasattr(model, "cdf") and hasattr(model, "pdf"):
        # A scipy distribution passed directly.  Look it up so a registered
        # distribution keeps its R parameterisation even when passed as an object.
        name = getattr(model, "name", None)
        for spec in DISTRIBUTION_SPECS.values():
            if getattr(spec.dist, "name", None) == name:
                return model, spec
        return model, None

    raise ValueError(
        "Model must be a distribution name (e.g. 'normal') or a "
        "scipy.stats distribution object"
    )


def fit_distribution(
    model: Union[str, object],
    data: np.ndarray,
) -> Tuple[object, tuple]:
    """Fit ``model`` to ``data`` by maximum likelihood, as ``fitdist(..., 'mle')``.

    Returns the scipy distribution and the full scipy parameter tuple
    (shapes + loc + scale), with loc/scale pinned wherever R's version of the
    distribution has no such parameter.
    """
    dist, spec = resolve_distribution(model)
    data = np.asarray(data, dtype=float)

    if spec is None:
        # Unregistered scipy distribution: nothing to pin, fit everything.
        return dist, tuple(dist.fit(data))

    if spec.support is not None:
        low, high = spec.support
        if data.min() < low or data.max() > high:
            raise ValueError(
                f"The {spec.r_name} distribution is defined on "
                f"[{low}, {high}]; the data range "
                f"[{data.min():.4g}, {data.max():.4g}] falls outside it. "
                "fitdistrplus raises the same error."
            )

    if spec.r_name == "norm":
        # R's mledist optimises the normal log-likelihood numerically and lands
        # on the closed-form MLE, whose sd divides by n rather than n - 1.
        return dist, (float(np.mean(data)), float(np.std(data, ddof=0)))

    return dist, tuple(dist.fit(data, **spec.fixed))


def loglik(dist: object, params: Sequence[float], data: np.ndarray) -> float:
    """Log-likelihood of ``data`` under ``dist(*params)``."""
    logpdf = dist.logpdf(np.asarray(data, dtype=float), *params)
    return float(np.sum(logpdf))


def aic_bic(
    dist: object,
    params: Sequence[float],
    data: np.ndarray,
    n_free_params: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Return ``(loglik, aic, bic)`` using R's definitions.

    ``aic = -2 * loglik + 2 * npar`` and ``bic = -2 * loglik + log(n) * npar``,
    where ``npar`` is the number of parameters R would actually estimate.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    ll = loglik(dist, params, data)
    if n_free_params is None:
        n_free_params = len(params)
    aic = -2.0 * ll + 2.0 * n_free_params
    bic = -2.0 * ll + np.log(n) * n_free_params
    return ll, float(aic), float(bic)


# ---------------------------------------------------------------------------
# Translating between scipy's parameterisation and R's
# ---------------------------------------------------------------------------
#
# These two functions are inverses.  Keeping them side by side matters: the
# standard errors ``fitdist`` reports are those of *R's* parameters, so the
# log-likelihood has to be differentiated with respect to those rather than
# scipy's.  For the gamma, for instance, R estimates a rate where scipy carries
# a scale, and the two have different standard errors.

def r_estimate(r_name: str, params: Sequence[float], spec=None) -> Dict[str, float]:
    """Translate a scipy parameter tuple into the parameters R reports."""
    p = tuple(float(x) for x in params)

    if r_name == "norm":
        return {"mean": p[0], "sd": p[1]}
    if r_name == "lnorm":
        # scipy: s = sdlog, scale = exp(meanlog)
        return {"meanlog": float(np.log(p[2])), "sdlog": p[0]}
    if r_name == "weibull":
        return {"shape": p[0], "scale": p[2]}
    if r_name == "gamma":
        return {"shape": p[0], "rate": 1.0 / p[2]}
    if r_name == "exp":
        return {"rate": 1.0 / p[1]}
    if r_name == "unif":
        return {"min": p[0], "max": p[0] + p[1]}
    if r_name == "pareto":
        return {"shape": p[0], "scale": p[2]}
    if r_name == "rayleigh":
        return {"scale": p[1]}
    if r_name in ("chisq", "t"):
        return {"df": p[0]}
    if r_name == "f":
        return {"df1": p[0], "df2": p[1]}
    if r_name == "beta":
        return {"shape1": p[0], "shape2": p[1]}

    # logis, laplace, cauchy, gumbel: (location, scale)
    if spec is not None and len(spec.r_params) == len(p):
        return dict(zip(spec.r_params, p))
    return {f"par{i + 1}": v for i, v in enumerate(p)}


def scipy_params(r_name: str, values: Sequence[float], spec=None) -> tuple:
    """Translate R's parameter vector back into a full scipy parameter tuple.

    The inverse of :func:`r_estimate`.  ``values`` must be in the order
    :func:`r_estimate` returns them.
    """
    v = [float(x) for x in values]

    if r_name == "norm":
        return (v[0], v[1])
    if r_name == "lnorm":
        return (v[1], 0.0, float(np.exp(v[0])))
    if r_name == "weibull":
        return (v[0], 0.0, v[1])
    if r_name == "gamma":
        return (v[0], 0.0, 1.0 / v[1])
    if r_name == "exp":
        return (0.0, 1.0 / v[0])
    if r_name == "unif":
        return (v[0], v[1] - v[0])
    if r_name == "pareto":
        return (v[0], 0.0, v[1])
    if r_name == "rayleigh":
        return (0.0, v[0])
    if r_name in ("chisq", "t"):
        return (v[0], 0.0, 1.0)
    if r_name == "f":
        return (v[0], v[1], 0.0, 1.0)
    if r_name == "beta":
        return (v[0], v[1], 0.0, 1.0)

    # logis, laplace, cauchy, gumbel: (location, scale)
    return tuple(v)


def _hessian(func, x: np.ndarray, rel_step: Optional[float] = None) -> np.ndarray:
    """Symmetric numerical Hessian of ``func`` at ``x`` by central differences.

    The step is scaled to each parameter's magnitude.  ``eps ** 0.25`` is the
    usual compromise for second derivatives: large enough that the four function
    values differ in more than rounding noise, small enough that the truncation
    error stays negligible.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if rel_step is None:
        rel_step = np.finfo(float).eps ** 0.25
    h = rel_step * np.maximum(np.abs(x), 1.0)

    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = h[i]
            ej = np.zeros(n)
            ej[j] = h[j]
            value = (
                func(x + ei + ej) - func(x + ei - ej)
                - func(x - ei + ej) + func(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
            out[i, j] = out[j, i] = value
    return out


def observed_information(
    dist,
    r_name: str,
    r_values: Sequence[float],
    data: np.ndarray,
    spec=None,
) -> Optional[np.ndarray]:
    """Variance-covariance matrix of the estimates, in R's parameterisation.

    The inverse of the observed information -- the Hessian of the summed
    negative log-likelihood at the maximum.  This is what ``fitdist`` reports
    standard errors from; R writes it as ``solve(hessian) / n`` because its
    optimiser works with the *mean* negative log-likelihood, which comes to the
    same thing.

    Returns ``None`` when the Hessian is singular or not finite, as R does.
    """
    data = np.asarray(data, dtype=float)
    values = np.asarray([float(x) for x in r_values])

    def negll(theta):
        try:
            params = scipy_params(r_name, theta, spec)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                logpdf = dist.logpdf(data, *params)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return np.inf
        if not np.all(np.isfinite(logpdf)):
            return np.inf
        return -float(np.sum(logpdf))

    if not np.isfinite(negll(values)):
        return None

    hessian = _hessian(negll, values)
    if not np.all(np.isfinite(hessian)):
        return None

    # R checks the rank before inverting and returns NULL when it is deficient.
    if np.linalg.matrix_rank(hessian) != hessian.shape[0]:
        return None
    try:
        vcov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None

    return vcov if np.all(np.isfinite(vcov)) else None
