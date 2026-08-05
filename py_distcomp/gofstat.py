"""
Goodness-of-fit statistics, ported from ``fitdistrplus::gofstat``.

The statistics and the decision rules are those of ``computegofstatKSCvMAD`` and
``computegofstatChi2`` in the R package, including the modified statistics and
critical values that come from the Stephens tables.  The test columns carry the
same strings R prints: ``"rejected"``, ``"not rejected"`` or ``"not computed"``.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

#: What R prints when a test has no tabulated critical values for the fitted
#: distribution, or when the sample is too small to apply them.
NOT_COMPUTED = "not computed"

from .distributions import aic_bic, fit_distribution, resolve_distribution

__all__ = ["fit_distributions", "gofstat", "FitResult"]


# Stephens' critical values for the Anderson-Darling and Cramer-von Mises tests
# of the gamma distribution, indexed by the estimated shape.  R interpolates
# linearly and clamps to the right-hand value beyond shape = 20.
_GAMMA_SHAPES = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20], dtype=float)
_GAMMA_AD_CRIT = np.array(
    [0.786, 0.768, 0.762, 0.759, 0.758, 0.757, 0.755, 0.754, 0.754, 0.754, 0.753]
)
_GAMMA_CVM_CRIT = np.array(
    [0.136, 0.131, 0.129, 0.128, 0.128, 0.128, 0.127, 0.127, 0.127, 0.127, 0.126]
)


class FitResult:
    """The parts of an R ``fitdist`` object this package needs.

    Attributes
    ----------
    name : str
        The name this package uses for the distribution.
    r_name : str
        What R calls the same distribution.
    dist : scipy.stats rv_continuous
        The fitted scipy distribution.
    params : tuple
        Full scipy parameter tuple (shapes, loc, scale).
    estimate : dict
        The parameters R would report, in R's parameterisation.
    n : int
        Sample size.
    loglik, aic, bic : float
        As defined in ``fitdist``: ``aic = -2 loglik + 2 npar``,
        ``bic = -2 loglik + log(n) npar``.
    """

    def __init__(self, name, dist, params, data, spec=None):
        self.name = name
        self.dist = dist
        self.params = tuple(params)
        self.spec = spec
        self.r_name = spec.r_name if spec is not None else getattr(dist, "name", name)
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        npar = spec.n_free_params if spec is not None else len(self.params)
        self.n_free_params = npar
        self.loglik, self.aic, self.bic = aic_bic(dist, self.params, self.data, npar)
        self.estimate = _r_estimate(self.r_name, self.params, spec)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pretty = ", ".join(f"{k}={v:.6g}" for k, v in self.estimate.items())
        return f"FitResult({self.name}: {pretty})"


def _r_estimate(r_name: str, params: Sequence[float], spec) -> Dict[str, float]:
    """Translate a scipy parameter tuple into the parameters R reports.

    scipy and R disagree on several parameterisations -- scipy's lognormal is
    ``(s, loc, scale)`` where R's is ``(meanlog, sdlog)``, and scipy's gamma and
    exponential use a scale where R uses a rate.
    """
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
    if r_name == "chisq":
        return {"df": p[0]}
    if r_name == "t":
        return {"df": p[0]}
    if r_name == "f":
        return {"df1": p[0], "df2": p[1]}
    if r_name == "beta":
        return {"shape1": p[0], "shape2": p[1]}

    # logis, laplace, cauchy, gumbel: (location, scale)
    if spec is not None and len(spec.r_params) == len(p):
        return dict(zip(spec.r_params, p))
    return {f"par{i + 1}": v for i, v in enumerate(p)}


def fit_distributions(
    data: Union[np.ndarray, pd.Series, list],
    models: Union[str, object, Sequence] = "normal",
) -> List[FitResult]:
    """Fit one or more distributions by maximum likelihood.

    This is the equivalent of calling ``fitdist(data, distname)`` once per
    distribution and collecting the results in a list, which is what
    ``gofstat`` and the ``*comp`` plots consume in R.
    """
    clean = _clean(data)
    if isinstance(models, (str, bytes)) or not isinstance(models, (list, tuple)):
        models = [models]

    results = []
    for model in models:
        dist, spec = resolve_distribution(model)
        _, params = fit_distribution(model, clean)
        name = model if isinstance(model, str) else getattr(dist, "name", str(model))
        results.append(FitResult(name, dist, params, clean, spec))
    return results


def _clean(data) -> np.ndarray:
    series = pd.Series(data).dropna()
    if len(series) < 3:
        raise ValueError("At least 3 non-NA data points are required")
    return np.asarray(series.to_numpy(), dtype=float)


def gofstat(
    fits: Union[FitResult, Sequence[FitResult]],
    chisqbreaks: Optional[Sequence[float]] = None,
    meancount: Optional[int] = None,
) -> pd.DataFrame:
    """Goodness-of-fit statistics for one or more fits, as ``gofstat``.

    Parameters
    ----------
    fits : FitResult, MixtureResult, or a sequence of them
        Fits obtained from :func:`fit_distributions` or
        :func:`~py_distcomp.fit_mixture`.  All must share a dataset, so a
        mixture can be compared against single distributions on AIC and BIC.
    chisqbreaks : sequence of float, optional
        Cell boundaries for the chi-squared statistic.  If omitted they are
        derived from the data exactly as R does, targeting ``meancount``
        observations per cell.
    meancount : int, optional
        Target number of observations per chi-squared cell.  R's default is
        ``round(n / (4n)^(2/5))``.

    Returns
    -------
    pandas.DataFrame
        One row per fit with the Kolmogorov-Smirnov, Cramer-von Mises and
        Anderson-Darling statistics and their test decisions, the chi-squared
        statistic with its degrees of freedom and p-value, and AIC/BIC.
        Test columns hold ``'rejected'``, ``'not rejected'`` or
        ``'not computed'``, as R prints them.  Only the exponential, gamma,
        Weibull and logistic distributions have tabulated critical values in
        fitdistrplus, so the other fits report ``'not computed'``.
    """
    # A single fit, or anything that duck-types one (a MixtureResult).
    if not isinstance(fits, (list, tuple)):
        fits = [fits]
    fits = list(fits)
    if not fits:
        raise ValueError("At least one fit is required")

    odata = fits[0].data
    sdata = np.sort(odata)
    for fit in fits[1:]:
        # Compared on sorted values: every statistic below uses the order
        # statistics, and fits reaching here may have stored the sample sorted.
        if len(fit.data) != len(sdata) or not np.allclose(np.sort(fit.data), sdata):
            raise ValueError("All compared fits must have been obtained with the same dataset")

    n = len(sdata)

    if len(np.unique(sdata)) != n:
        warnings.warn(
            "Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling "
            "statistics may not be correct with ties",
            RuntimeWarning,
            stacklevel=2,
        )

    breaks = (
        np.asarray(chisqbreaks, dtype=float)
        if chisqbreaks is not None
        else _default_chisqbreaks(sdata, n, meancount)
    )

    rows = []
    for fit in fits:
        row = {"distribution": fit.name, "r_name": fit.r_name}
        row.update(_ks_cvm_ad(sdata, n, fit))
        row.update(_chisq(sdata, n, fit, breaks))
        row["aic"] = fit.aic
        row["bic"] = fit.bic
        row["loglik"] = fit.loglik
        rows.append(row)

    return pd.DataFrame(rows).set_index("distribution")


def _default_chisqbreaks(sdata: np.ndarray, n: int, meancount: Optional[int]):
    """R's automatic chi-squared cell boundaries.

    Peels off ``meancount`` observations at a time from the bottom of the sorted
    sample and records the boundary, until fewer than ``1.5 * meancount`` remain.
    """
    if meancount is None:
        meancount = int(np.round(n / ((4 * n) ** (2.0 / 5.0))))
    if meancount < 1:
        return None

    remaining = sdata
    breaks: List[float] = []
    while len(remaining) > np.ceil(1.5 * meancount):
        limit = remaining[meancount - 1]  # R's sdata[meancount], 1-indexed
        remaining = remaining[remaining > limit]
        breaks.append(float(limit))
    return np.asarray(breaks) if breaks else None


def _ecdf_at(sdata: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``ecdf(sdata)(x)`` -- the proportion of sdata that is <= each x."""
    return np.searchsorted(sdata, x, side="right") / len(sdata)


def _ks_cvm_ad(sdata: np.ndarray, n: int, fit: FitResult) -> Dict[str, object]:
    """Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling statistics."""
    theop = fit.dist.cdf(sdata, *fit.params)
    obspu = np.arange(1, n + 1) / n
    obspl = np.arange(0, n) / n

    ks = float(np.max(np.maximum(np.abs(theop - obspu), np.abs(theop - obspl))))
    dmod = ks * (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))
    kstest = _decide(dmod > 1.358) if n >= 30 else NOT_COMPUTED

    i = np.arange(1, n + 1)
    with np.errstate(divide="ignore"):
        ad = float(-n - np.mean((2 * i - 1) * (np.log(theop) + np.log(1 - theop[::-1]))))
    cvm = float(1.0 / (12 * n) + np.sum((theop - (2 * i - 1) / (2 * n)) ** 2))

    return {
        "ks": ks,
        "ks_test": kstest,
        "cvm": cvm,
        "cvm_test": _cvm_test(cvm, n, fit),
        "ad": ad,
        "ad_test": _ad_test(ad, n, fit),
    }


def _decide(rejected: bool) -> str:
    return "rejected" if rejected else "not rejected"


def _gamma_critical(shape: float, table: np.ndarray) -> Optional[float]:
    """Interpolate a Stephens critical value, matching R's ``approxfun``.

    R sets ``yright`` but leaves ``yleft`` unset, so a shape below 1 yields NA
    and the test is not computed.
    """
    if shape < _GAMMA_SHAPES[0]:
        return None
    if shape > _GAMMA_SHAPES[-1]:
        return float(table[-1])
    return float(np.interp(shape, _GAMMA_SHAPES, table))


def _ad_test(ad: float, n: int, fit: FitResult) -> str:
    if n < 5:
        return NOT_COMPUTED
    name = fit.r_name
    if name == "exp":
        return _decide(ad * (1 + 0.6 / n) > 1.321)
    if name == "gamma":
        crit = _gamma_critical(fit.estimate["shape"], _GAMMA_AD_CRIT)
        return NOT_COMPUTED if crit is None else _decide(ad > crit)
    if name == "weibull":
        return _decide(ad * (1 + 0.2 / np.sqrt(n)) > 0.757)
    if name == "logis":
        return _decide(ad * (1 + 0.25 / n) > 0.66)
    return NOT_COMPUTED


def _cvm_test(cvm: float, n: int, fit: FitResult) -> str:
    if n < 5:
        return NOT_COMPUTED
    name = fit.r_name
    if name == "exp":
        return _decide(cvm * (1 + 0.16 / n) > 0.222)
    if name == "gamma":
        crit = _gamma_critical(fit.estimate["shape"], _GAMMA_CVM_CRIT)
        return NOT_COMPUTED if crit is None else _decide(cvm > crit)
    if name == "weibull":
        return _decide(cvm * (1 + 0.2 / np.sqrt(n)) > 0.124)
    if name == "logis":
        return _decide((n * cvm - 0.08) / (n - 1) > 0.098)
    return NOT_COMPUTED


def _chisq(sdata, n, fit: FitResult, breaks) -> Dict[str, object]:
    """Chi-squared statistic over the cells defined by ``breaks``."""
    if breaks is None or len(breaks) == 0:
        return {"chisq": np.nan, "chisq_df": np.nan, "chisq_pvalue": np.nan}

    breaks = np.asarray(breaks, dtype=float)
    pbreaks = fit.dist.cdf(breaks, *fit.params)
    fobsbreaks = _ecdf_at(sdata, breaks)

    punder = np.concatenate(([0.0], pbreaks[:-1]))
    fobsunder = np.concatenate(([0.0], fobsbreaks[:-1]))

    if pbreaks[-1] == 1 and fobsbreaks[-1] == 1:
        p = pbreaks - punder
        fobs = fobsbreaks - fobsunder
    else:
        p = np.concatenate((pbreaks - punder, [1 - pbreaks[-1]]))
        fobs = np.concatenate((fobsbreaks - fobsunder, [1 - fobsbreaks[-1]]))

    obscounts = np.round(fobs * n)
    theocounts = p * n
    with np.errstate(divide="ignore", invalid="ignore"):
        chisq = float(np.sum((obscounts - theocounts) ** 2 / theocounts))
    df = len(obscounts) - 1 - fit.n_free_params
    pvalue = float(stats.chi2.sf(chisq, df)) if df > 0 else np.nan

    return {"chisq": chisq, "chisq_df": df, "chisq_pvalue": pvalue}
