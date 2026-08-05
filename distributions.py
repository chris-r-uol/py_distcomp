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
