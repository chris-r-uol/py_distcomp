"""
Bootstrap uncertainty on fitted parameters, ported from ``fitdistrplus::bootdist``.

The standard errors on :class:`~py_distcomp.FitResult` come from the observed
information, which assumes the log-likelihood is well approximated by a
quadratic at its maximum.  That is an asymptotic argument: for a small sample,
or for a parameter whose sampling distribution is skewed -- a shape, or a scale
near zero -- the symmetric Wald interval it implies can be poor, and can even
extend below zero for a parameter that cannot be negative.

The bootstrap makes no such assumption.  Resample, refit, and read the interval
off the percentiles of the refits.

It also covers a case the Hessian cannot reach at all: a mixture, whose weights
and component parameters come out of expectation-maximisation rather than a
single optimisation, and whose upper component is often weakly identified.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .distributions import fit_distribution, r_estimate, resolve_distribution
from .gofstat import FitResult
from .mixture import MixtureResult, fit_mixture

__all__ = ["bootdist", "BootdistResult", "qq_confidence_band"]


class BootdistResult:
    """Bootstrapped parameter estimates.

    Attributes
    ----------
    estimates : pandas.DataFrame
        One row per successful resample, one column per parameter, in R's
        parameterisation.
    ci : pandas.DataFrame
        Median and percentile confidence limits per parameter, as R's
        ``bootdist`` reports them.
    fit : FitResult or MixtureResult
        The fit that was resampled.
    method : {'param', 'nonparam'}
    niter : int
        Resamples requested.
    n_converged, n_failed : int
        Refits that succeeded and that did not.
    conf_level : float
    """

    def __init__(self, estimates, fit, method, niter, n_failed, conf_level):
        self.estimates = estimates
        self.fit = fit
        self.method = method
        self.niter = int(niter)
        self.n_converged = len(estimates)
        self.n_failed = int(n_failed)
        self.conf_level = float(conf_level)
        self.ci = self._percentiles(estimates, conf_level)

    @staticmethod
    def _percentiles(estimates: pd.DataFrame, conf_level: float) -> pd.DataFrame:
        alpha = (1.0 - conf_level) / 2.0
        lower_pct, upper_pct = 100 * alpha, 100 * (1 - alpha)
        return pd.DataFrame(
            {
                "median": estimates.median(),
                f"{lower_pct:g}%": estimates.quantile(alpha),
                f"{upper_pct:g}%": estimates.quantile(1 - alpha),
            }
        )

    def quantile_ci(self, p: Union[float, Sequence[float]]) -> pd.DataFrame:
        """Confidence intervals on the fitted distribution's own quantiles.

        R's ``quantile.bootdist``: evaluate the quantile function at ``p`` for
        every resample's parameters, then take percentiles across resamples.
        Useful when the quantity of interest is a percentile of the fitted
        distribution rather than a parameter.
        """
        probs = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError("p must lie strictly between 0 and 1")

        rows = []
        for _, values in self.estimates.iterrows():
            rows.append(self._quantiles_for(values.to_dict(), probs))
        draws = np.vstack(rows)

        alpha = (1.0 - self.conf_level) / 2.0
        return pd.DataFrame(
            {
                "median": np.median(draws, axis=0),
                f"{100 * alpha:g}%": np.quantile(draws, alpha, axis=0),
                f"{100 * (1 - alpha):g}%": np.quantile(draws, 1 - alpha, axis=0),
            },
            index=pd.Index(probs, name="p"),
        )

    def _quantiles_for(self, values: Dict[str, float], probs: np.ndarray) -> np.ndarray:
        fit = self.fit
        if isinstance(fit, MixtureResult):
            # Rebuild the mixture this resample implies and invert its CDF.
            from .mixture import MixtureDistribution

            components, weights = [], []
            for k, ((dist, _), name) in enumerate(
                zip(fit.components, fit.model_names), start=1
            ):
                _, spec = resolve_distribution(name)
                r_name = spec.r_name if spec is not None else name
                # Look the parameters up by exact name; matching on a numeric
                # suffix would confuse component 1 with component 11.
                params = [values[f"{p}{k}"] for p in spec.r_params]
                components.append((dist, _scipy_from_r(r_name, params, spec)))
                weights.append(values[f"weight{k}"])
            return np.asarray(MixtureDistribution(components, weights).ppf(probs))

        params = _scipy_from_r(fit.r_name, list(values.values()), fit.spec)
        return np.asarray(fit.dist.ppf(probs, *params))

    def summary(self) -> pd.DataFrame:
        """The parameter estimates beside their bootstrap intervals."""
        estimate = pd.Series(self.fit.estimate, name="estimate")
        return pd.concat([estimate, self.ci], axis=1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BootdistResult({self.fit.name}, method={self.method!r}, "
            f"{self.n_converged}/{self.niter} refits, "
            f"{self.conf_level:.0%} CI)"
        )


def _scipy_from_r(r_name, values, spec):
    from .distributions import scipy_params

    return scipy_params(r_name, values, spec)


def bootdist(
    fit: Union[FitResult, MixtureResult],
    niter: int = 1001,
    bootmethod: str = "param",
    conf_level: float = 0.95,
    seed: Optional[int] = None,
    silent: bool = True,
) -> BootdistResult:
    """Bootstrap the uncertainty of a fit, as R's ``bootdist``.

    Parameters
    ----------
    fit : FitResult or MixtureResult
        The fit to resample, from :func:`~py_distcomp.fit_distributions` or
        :func:`~py_distcomp.fit_mixture`.
    niter : int, default=1001
        Number of resamples.  R's default, and R requires at least 10.
    bootmethod : {'param', 'nonparam'}, default='param'
        ``'param'`` draws each resample from the *fitted* distribution, so the
        interval reflects sampling variability under the model being assumed.
        ``'nonparam'`` resamples the observed data with replacement, which makes
        no such assumption and is the safer choice when the fit is imperfect.
    conf_level : float, default=0.95
        Coverage of the percentile interval.  R reports 2.5% and 97.5% limits,
        which is this default.
    seed : int, optional
        Seed for resampling.  Uses an isolated generator, so the global numpy
        random state is left untouched.
    silent : bool, default=True
        Suppress the warning about failed refits.

    Returns
    -------
    BootdistResult

    Notes
    -----
    Refits that fail are dropped, and ``n_failed`` records how many.  A large
    count means the model is struggling on resampled data and the interval
    should not be trusted.

    For a mixture the refit is warm-started from the original fit, which is both
    far faster and keeps the component ordering stable -- without it, components
    could swap between resamples and the percentiles would be meaningless.

    Examples
    --------
    >>> fit = fit_distributions(serving, 'weibull')[0]
    >>> boot = bootdist(fit, niter=1001, seed=1)
    >>> boot.summary()
           estimate     median      2.5%     97.5%
    shape     2.186      2.189     1.994     2.393
    scale    83.348     83.339    78.586    88.331

    Intervals on the fitted distribution's own quantiles:

    >>> boot.quantile_ci([0.5, 0.95])
    """
    if niter < 10:
        raise ValueError("niter must be an integer of at least 10")
    if bootmethod not in ("param", "nonparam"):
        raise ValueError("bootmethod must be 'param' or 'nonparam'")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    data = np.asarray(fit.data, dtype=float)
    n = len(data)

    if bootmethod == "param":
        draw = lambda: _rvs(fit, n, rng)  # noqa: E731
    else:
        draw = lambda: rng.choice(data, size=n, replace=True)  # noqa: E731

    rows: List[Dict[str, float]] = []
    n_failed = 0
    for _ in range(niter):
        sample = draw()
        try:
            rows.append(_refit(fit, sample))
        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            n_failed += 1

    if not rows:
        raise ValueError(
            f"Every one of the {niter} bootstrap refits failed; the model may "
            "not be estimable on resampled data"
        )
    if n_failed and not silent:
        warnings.warn(
            f"{n_failed} of {niter} bootstrap refits failed and were dropped",
            RuntimeWarning,
            stacklevel=2,
        )

    estimates = pd.DataFrame(rows, columns=list(fit.estimate))
    return BootdistResult(estimates, fit, bootmethod, niter, n_failed, conf_level)


def _rvs(fit, n: int, rng) -> np.ndarray:
    """Draw a sample of size ``n`` from the fitted distribution."""
    if isinstance(fit, MixtureResult):
        return fit.dist.rvs(size=n, random_state=rng)
    return fit.dist.rvs(*fit.params, size=n, random_state=rng)


def _refit(fit, sample: np.ndarray) -> Dict[str, float]:
    """Refit the same model to a resample, returning R's parameters."""
    if isinstance(fit, MixtureResult):
        # Warm-started, so the components keep their original ordering.
        refitted = fit_mixture(sample, fit.model_names, init=fit)
        return dict(refitted.estimate)

    # Refit by whichever estimator produced the original: bootstrapping a
    # maximum-goodness-of-fit fit with maximum likelihood would describe the
    # sampling behaviour of a different estimator entirely.
    from .estimation import fit_by_method

    _, params = fit_by_method(
        fit.model, sample, getattr(fit, "method", "mle"),
        **getattr(fit, "method_kwargs", {}),
    )
    return r_estimate(fit.r_name, params, fit.spec)


# ---------------------------------------------------------------------------
# Q-Q confidence bands
# ---------------------------------------------------------------------------

def qq_confidence_band(
    fit: Union[FitResult, MixtureResult],
    level: float = 0.95,
    niter: int = 1000,
    kind: str = "pointwise",
    refit: bool = True,
    a_ppoints: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """A confidence band for a Q-Q plot, by parametric bootstrap.

    A Q-Q plot shows where the data departs from the fitted distribution, but
    not whether the departure is larger than sampling noise would produce
    anyway.  This simulates samples from the fitted distribution and reads the
    spread of their order statistics, giving a band the real data can be judged
    against: points outside it are departures worth taking seriously.

    Parameters
    ----------
    fit : FitResult or MixtureResult
        The fit to build the band around.
    level : float, default=0.95
        Coverage of the band.
    niter : int, default=1000
        Simulated samples.
    kind : {'pointwise', 'simultaneous'}, default='pointwise'
        ``'pointwise'`` gives each order statistic its own interval, so *each*
        point has the stated coverage.  Across ``n`` points a few will fall
        outside by chance even under a perfect fit, so a handful of excursions
        means nothing on its own.  ``'simultaneous'`` widens the band so the
        *whole* curve stays inside with the stated probability, which is what
        you want for "does this data depart from the model at all?".
    refit : bool, default=True
        Re-estimate the parameters on each simulated sample.  This matters:
        the real points are plotted against quantiles fitted to the real data,
        so the fit has already absorbed part of the discrepancy, and the null
        distribution has to absorb the same part to be comparable.  On data
        genuinely from the fitted family this gives 4.9% of points outside a
        nominal 5% pointwise band.

        Turning it off treats the fitted parameters as if they were known to be
        correct.  That is faster, but the band comes out far too wide -- 0.2%
        of points outside on the same check -- so real departures can hide
        inside it.
    a_ppoints : float, default=0.5
        Offset for the plotting positions, matching the Q-Q plot.
    seed : int, optional
        Seed for the simulation.  Uses an isolated generator.

    Returns
    -------
    pandas.DataFrame
        ``theoretical``, ``observed``, ``lower`` and ``upper``, one row per
        order statistic, sorted by ``theoretical``.

    Examples
    --------
    >>> band = qq_confidence_band(fit, kind='simultaneous')
    >>> outside = (band.observed < band.lower) | (band.observed > band.upper)
    >>> outside.sum()
    0
    """
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    if niter < 10:
        raise ValueError("niter must be an integer of at least 10")
    if kind not in ("pointwise", "simultaneous"):
        raise ValueError("kind must be 'pointwise' or 'simultaneous'")

    from .distributions import ppoints

    rng = np.random.default_rng(seed)
    data = np.sort(np.asarray(fit.data, dtype=float))
    n = len(data)
    probs = ppoints(n, a=a_ppoints)

    theoretical = _ppf(fit, fit.params, probs)

    # Each replicate contributes the gap between its own order statistics and
    # the quantiles its own refit predicts, which is exactly the discrepancy a
    # Q-Q plot displays.
    deviations = np.empty((niter, n))
    kept = 0
    for _ in range(niter):
        sample = np.sort(_rvs(fit, n, rng))
        reference = theoretical
        if refit:
            try:
                reference = _refit_quantiles(fit, sample, probs)
            except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
                continue
        deviations[kept] = sample - reference
        kept += 1

    if kept < 10:
        raise ValueError(
            f"Only {kept} of {niter} simulated samples could be refitted; the "
            "band cannot be built. Try refit=False."
        )
    deviations = deviations[:kept]

    alpha = 1.0 - level
    if kind == "pointwise":
        lower = theoretical + np.quantile(deviations, alpha / 2, axis=0)
        upper = theoretical + np.quantile(deviations, 1 - alpha / 2, axis=0)
    else:
        # Scale each position by its own spread, take the level-quantile of the
        # worst standardised excursion, and widen every position by that much.
        centre = np.median(deviations, axis=0)
        spread = np.std(deviations, axis=0)
        spread = np.where(spread > 0, spread, np.finfo(float).tiny)
        worst = np.max(np.abs(deviations - centre) / spread, axis=1)
        critical = float(np.quantile(worst, level))
        lower = theoretical + centre - critical * spread
        upper = theoretical + centre + critical * spread

    return pd.DataFrame(
        {
            "theoretical": theoretical,
            "observed": data,
            "lower": lower,
            "upper": upper,
        }
    )


def _ppf(fit, params, probs: np.ndarray) -> np.ndarray:
    """Quantile function of a fit, whether a single distribution or a mixture."""
    if isinstance(fit, MixtureResult):
        return np.asarray(fit.dist.ppf(probs))
    return np.asarray(fit.dist.ppf(probs, *params))


def _refit_quantiles(fit, sample: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Refit on a simulated sample and return its predicted quantiles."""
    if isinstance(fit, MixtureResult):
        refitted = fit_mixture(sample, fit.model_names, init=fit)
        return np.asarray(refitted.dist.ppf(probs))

    from .estimation import fit_by_method

    dist, params = fit_by_method(
        fit.model, sample, getattr(fit, "method", "mle"),
        **getattr(fit, "method_kwargs", {}),
    )
    return np.asarray(dist.ppf(probs, *params))
