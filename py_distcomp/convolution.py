"""
Measurement-error models -- fitting the distribution behind noisy observations.

Some quantities cannot be observed directly.  A remote sensing device measures a
concentration ratio that is physically non-negative, but propagates enough error
that individual readings come back negative; discarding those would bias the
distribution high, because the error works both ways.  What is observed is not
the quantity of interest but::

    Y = X + ε

with ``X`` the true value and ``ε`` symmetric instrument error.  The density of
``Y`` is the convolution of the two, so fitting a family directly to ``Y``
estimates the *smeared* distribution: its spread is the true spread inflated by
the error, and its shape has to accommodate values the true quantity can never
take.

:func:`fit_convolved` fits ``X`` instead.  The likelihood is evaluated on the
convolved density, so the estimate describes the quantity of interest while the
observations keep the noise they actually have -- and ``X`` may then come from a
family that could not have been fitted to ``Y`` at all, a gamma or a lognormal
being undefined on the negative readings.

The error scale can be supplied from calibration, or estimated alongside
everything else: a symmetric error and a skewed signal are separable.  Fixing it
wrongly matters more than it looks -- see :attr:`ConvolvedResult.inflation`.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

from .distributions import r_estimate, resolve_distribution, scipy_params
from .estimation import _TRANSFORMS, _from_unconstrained, _to_unconstrained

__all__ = [
    "ConvolvedDistribution",
    "ConvolvedResult",
    "fit_convolved",
    "ERROR_FAMILIES",
]

#: Symmetric, zero-centred families available for the error term.  Each is
#: parameterised by a single scale; for the normal that scale is its standard
#: deviation, for the others it is not.
ERROR_FAMILIES = {
    "normal": stats.norm,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
    "cauchy": stats.cauchy,
}

_TINY = 1e-300


def _grid(data: np.ndarray, scale: float, n: int) -> np.ndarray:
    """A grid wide enough that the convolution loses no mass off either end.

    The length is forced odd: with an even one the kernel's centre falls between
    two cells and ``mode='same'`` shifts the result by half a step, which is a
    3e-4 error where an odd grid is exact to 3e-17.
    """
    n = int(n) | 1
    pad = 8.0 * max(scale, 1e-12)
    lo = float(np.min(data)) - pad
    hi = float(np.max(data)) + pad
    return np.linspace(lo, hi, n)


def _convolve(true_dist, true_params, error_dist, scale, grid) -> np.ndarray:
    """Density of ``X + ε`` on a grid.

    scipy returns zero outside a distribution's support, so a family bounded
    below -- a gamma, say -- simply contributes nothing left of it, and the
    convolution carries that boundary into the observed density as a smooth
    shoulder rather than a hard edge.
    """
    step = grid[1] - grid[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        signal_density = true_dist.pdf(grid, *true_params)
        offsets = (np.arange(len(grid)) - len(grid) // 2) * step
        kernel = error_dist.pdf(offsets, 0.0, scale)
        out = signal.fftconvolve(signal_density, kernel, mode="same") * step
    return np.maximum(out, 0.0)


class ConvolvedDistribution:
    """The distribution of ``X + ε``, exposed through the scipy interface.

    ``pdf``, ``cdf``, ``ppf``, ``logpdf``, ``sf`` and ``rvs`` describe the
    *observed* quantity, so the object drops into the comparison plots and
    :func:`~py_distcomp.gofstat` beside anything else.  The distribution of
    interest is :attr:`true_dist` with :attr:`true_params`.

    Parameters
    ----------
    true_dist, true_params
        The family and parameters of the unobserved quantity.
    error_dist, scale
        The error family and its scale.
    grid : np.ndarray
        Where the convolution is evaluated; everything else interpolates on it.
    """

    shapes = None
    n_params = 0

    def __init__(self, true_dist, true_params, error_dist, scale, grid):
        self.true_dist = true_dist
        self.true_params = tuple(true_params)
        self.error_dist = error_dist
        self.scale = float(scale)
        self.grid = np.asarray(grid, dtype=float)

        self._pdf = _convolve(true_dist, self.true_params, error_dist,
                              self.scale, self.grid)
        step = self.grid[1] - self.grid[0]
        total = float(np.sum(self._pdf) * step)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("the convolved density is not a usable density")
        self._pdf = self._pdf / total          # guard against grid truncation
        self._cdf = np.clip(np.concatenate([[0.0], np.cumsum(
            (self._pdf[1:] + self._pdf[:-1]) / 2 * step)]), 0.0, 1.0)
        self.name = (f"{getattr(true_dist, 'name', '?')}"
                     f"+{getattr(error_dist, 'name', '?')}")

    def pdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.grid, self._pdf, left=0.0, right=0.0)

    def logpdf(self, x, *_) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(self.pdf(x), _TINY))

    def cdf(self, x, *_) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.grid, self._cdf, left=0.0, right=1.0)

    def sf(self, x, *_) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def ppf(self, q, *_) -> np.ndarray:
        """Quantile function, by inverting the tabulated CDF."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.full(q.shape, np.nan)
        out[q <= 0] = -np.inf
        out[q >= 1] = np.inf
        inside = (q > 0) & (q < 1)
        if np.any(inside):
            # np.interp needs an increasing x; ties in the flat tails would
            # otherwise pick an arbitrary one of them.
            keep = np.concatenate([[True], np.diff(self._cdf) > 0])
            out[inside] = np.interp(q[inside], self._cdf[keep], self.grid[keep])
        return out if out.size > 1 else out.item()

    def rvs(self, size=1, random_state=None, *_) -> np.ndarray:
        """Draw a true value and add an error to it, as the instrument does."""
        rng = np.random.default_rng(random_state)
        size = int(size)
        signal_draw = self.true_dist.rvs(*self.true_params, size=size, random_state=rng)
        noise = self.error_dist.rvs(0.0, self.scale, size=size, random_state=rng)
        return np.asarray(signal_draw + noise)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ConvolvedDistribution({self.name}, scale={self.scale:.4g})"


class ConvolvedResult:
    """A fitted measurement-error model.

    Duck-types :class:`~py_distcomp.FitResult`, so it goes into
    :func:`~py_distcomp.gofstat` and the comparison plots beside ordinary fits.
    Its ``estimate`` describes the *true* quantity, which is the point.

    Attributes
    ----------
    true_estimate : dict
        Parameters of the unobserved quantity, in R's parameterisation.
    scale : float
        The error scale used, whether supplied or estimated.
    scale_estimated : bool
    dist : ConvolvedDistribution
    """

    def __init__(self, dist, true_model, error_model, data, loglik,
                 n_free_params, scale_estimated):
        self.dist = dist
        self.params: tuple = ()
        self.true_model = true_model
        self.error_model = error_model
        self.name = f"{true_model}+{error_model}"
        self.r_name = self.name
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        self.scale = dist.scale
        self.scale_estimated = bool(scale_estimated)
        self.loglik = float(loglik)
        self.n_free_params = int(n_free_params)
        self.aic = -2.0 * self.loglik + 2.0 * self.n_free_params
        self.bic = -2.0 * self.loglik + np.log(self.n) * self.n_free_params
        self.discrete = False

        _, spec = resolve_distribution(true_model)
        self.true_estimate = r_estimate(spec.r_name, dist.true_params, spec)
        self.estimate = dict(self.true_estimate)
        self.estimate["error_scale"] = self.scale

    @property
    def true_sd(self) -> float:
        """Standard deviation of the unobserved quantity."""
        with np.errstate(invalid="ignore", over="ignore"):
            return float(self.dist.true_dist.std(*self.dist.true_params))

    @property
    def observed_sd(self) -> float:
        """Standard deviation the model implies for the observations."""
        grid, density = self.dist.grid, self.dist._pdf
        mean = float(np.trapezoid(grid * density, grid))
        var = float(np.trapezoid((grid - mean) ** 2 * density, grid))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def inflation(self) -> float:
        """How much wider the observations are than the quantity behind them.

        ``observed_sd / true_sd - 1``.  This is the bias a family fitted
        directly to the observations carries in its spread, which is usually the
        parameter being compared between groups.
        """
        true = self.true_sd
        return float("nan") if not true else self.observed_sd / true - 1.0

    def summary(self) -> pd.Series:
        row = {
            "model": self.name,
            "n": self.n,
            "error_scale": self.scale,
            "error_estimated": self.scale_estimated,
            "true_sd": self.true_sd,
            "observed_sd": self.observed_sd,
            "inflation": self.inflation,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
        }
        row.update(self.true_estimate)
        return pd.Series(row)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pretty = ", ".join(f"{k}={v:.4g}" for k, v in self.true_estimate.items())
        how = "estimated" if self.scale_estimated else "supplied"
        return (f"ConvolvedResult({self.true_model}: {pretty}; "
                f"error scale {self.scale:.4g} ({how}), aic={self.aic:.1f})")


def fit_convolved(
    data: Union[np.ndarray, pd.Series, list],
    true_family: Union[str, object] = "gamma",
    error: str = "normal",
    scale: Optional[float] = None,
    grid_points: int = 8193,
    maxiter: int = 4000,
) -> ConvolvedResult:
    """Fit the distribution behind noisy observations of ``Y = X + ε``.

    Parameters
    ----------
    data : array-like
        The observations, noise and all.  Values the true quantity could never
        take -- negative readings of a non-negative quantity -- belong here:
        dropping them truncates the error and biases the result.
    true_family : str or scipy distribution, default='gamma'
        Family for the unobserved quantity.  It need not be able to describe the
        observations; that is the point.  A gamma cannot be fitted to data
        containing negatives, but it can perfectly well be the truth behind it.
    error : {'normal', 'laplace', 'logistic', 'cauchy'}, default='normal'
        Family for the error, taken as symmetric and centred on zero.
    scale : float, optional
        The error's scale, from calibration.  For a normal error this is its
        standard deviation.  ``None`` estimates it alongside everything else,
        which works because a symmetric error and a skewed signal are separable.
    grid_points : int, default=8193
        Points the convolution is evaluated on.  Forced odd: an even grid puts
        the kernel's centre between two cells and costs three orders of accuracy.
    maxiter : int, default=4000
        Optimiser iterations.

    Returns
    -------
    ConvolvedResult

    Notes
    -----
    A supplied ``scale`` that is wrong distorts the estimated *spread* far more
    than the estimated mean, and the spread is usually the quantity being
    compared between groups.  When calibration is not solid, estimating it is
    the safer choice.

    Examples
    --------
    >>> fit = fit_convolved(readings, 'gamma', error='normal')
    >>> fit.true_estimate            # the emission distribution itself
    {'shape': 2.09, 'rate': 0.171}
    >>> fit.inflation                # how much noise widened the observations
    0.19
    """
    clean = np.asarray(pd.Series(data).dropna().to_numpy(), dtype=float)
    if len(clean) < 10:
        raise ValueError("At least 10 observations are required")
    if error not in ERROR_FAMILIES:
        raise ValueError(
            f"error must be one of {', '.join(ERROR_FAMILIES)}; got '{error}'"
        )
    error_dist = ERROR_FAMILIES[error]

    true_dist, spec = resolve_distribution(true_family)
    if spec is None:
        raise ValueError("true_family must be a registered distribution")
    if spec.discrete:
        raise ValueError(
            "true_family is discrete; a convolution with a continuous error is "
            "defined here for continuous families only"
        )

    estimate_scale = scale is None
    if not estimate_scale and scale <= 0:
        raise ValueError("scale must be positive")

    kinds = _TRANSFORMS.get(spec.r_name, ("free",) * spec.n_free_params)
    spread = float(np.std(clean))
    start_scale = float(scale) if not estimate_scale else max(spread / 3.0, 1e-6)

    # Starting values: fit the family to whatever part of the data it can take,
    # which is enough to get the optimiser into the right region.
    from .distributions import fit_distribution

    usable = clean[clean > 0] if float(true_dist.support(*_probe(spec))[0]) >= 0 else clean
    try:
        _, raw = fit_distribution(true_family, usable if len(usable) > 5 else clean)
        start_values = list(r_estimate(spec.r_name, raw, spec).values())
    except (ValueError, RuntimeError, FloatingPointError):
        start_values = [max(spread, 1.0)] * spec.n_free_params

    def unpack(z):
        values = _from_unconstrained(z[:len(start_values)], kinds)
        params = scipy_params(spec.r_name, values, spec)
        s = float(np.exp(np.clip(z[-1], -700, 700))) if estimate_scale else float(scale)
        return params, s

    def negll(z):
        try:
            params, s = unpack(z)
            if not np.isfinite(s) or s <= 0:
                return np.inf
            grid = _grid(clean, s, grid_points)
            density = _convolve(true_dist, params, error_dist, s, grid)
            step = grid[1] - grid[0]
            total = float(np.sum(density) * step)
            if not np.isfinite(total) or total <= 0:
                return np.inf
            at_data = np.interp(clean, grid, density / total, left=0.0, right=0.0)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return np.inf
        if np.any(at_data <= 0) or not np.all(np.isfinite(at_data)):
            return np.inf
        return -float(np.sum(np.log(at_data)))

    z0 = _to_unconstrained(start_values, kinds)
    if estimate_scale:
        z0 = np.append(z0, np.log(start_scale))
    else:
        z0 = np.append(z0, 0.0)          # ignored, keeps the layout uniform

    if not np.isfinite(negll(z0)):
        raise ValueError(
            f"Could not start a {spec.r_name} convolution from these data; try "
            "another family for the true quantity, or supply 'scale'"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = optimize.minimize(
            negll, z0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-10},
        )
    if not np.isfinite(outcome.fun):
        raise ValueError("The convolution fit did not converge")

    params, fitted_scale = unpack(outcome.x)
    dist = ConvolvedDistribution(true_dist, params, error_dist, fitted_scale,
                                 _grid(clean, fitted_scale, grid_points))
    loglik = float(np.sum(dist.logpdf(clean)))
    n_free = spec.n_free_params + (1 if estimate_scale else 0)

    name = true_family if isinstance(true_family, str) else getattr(
        true_dist, "name", "true")
    return ConvolvedResult(dist, name, error, clean, loglik, n_free, estimate_scale)


def _probe(spec) -> tuple:
    """A harmless parameter tuple, just to ask a family where its support starts."""
    return scipy_params(spec.r_name, [1.0] * spec.n_free_params, spec)
