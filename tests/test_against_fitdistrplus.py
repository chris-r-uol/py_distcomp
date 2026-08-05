"""
Checks that this package reproduces R's fitdistrplus.

Two kinds of reference are used:

* ``groundbeef``, the serving-size dataset shipped with fitdistrplus, together
  with the numbers printed in the package vignette.  ``groundbeef.npy`` holds
  the 254 values extracted from the package's own ``groundbeef.rda``.
* Closed-form results, where the quantity R computes numerically has an exact
  expression (the normal and lognormal MLEs, for instance), or where scipy
  implements the same textbook statistic independently.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.distributions import fit_distribution, ppoints  # noqa: E402
from py_distcomp.empirical_plots import empirical_cdf_plot, empirical_density_plot  # noqa: E402
from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.quantile_multi_comparison import (  # noqa: E402
    cullen_and_frey_plot,
    descdist,
    quantile_comparison_plot,
)


@pytest.fixture(scope="module")
def groundbeef():
    """The 254 serving sizes from fitdistrplus's groundbeef dataset."""
    return np.load(pathlib.Path(__file__).parent / "groundbeef.npy")


# ---------------------------------------------------------------------------
# ppoints
# ---------------------------------------------------------------------------

def test_ppoints_matches_r_formula():
    for n, a in [(5, 0.5), (7, 3 / 8), (100, 0.5), (1000, 0.5)]:
        expected = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
        assert np.allclose(ppoints(n, a), expected)


def test_ppoints_default_switches_at_ten():
    # R: ppoints(n, a = if (n <= 10) 3/8 else 1/2)
    assert np.allclose(ppoints(10), ppoints(10, 3 / 8))
    assert np.allclose(ppoints(11), ppoints(11, 0.5))


def test_ppoints_a_half_is_the_midpoint_rule():
    n = 50
    assert np.allclose(ppoints(n, 0.5), (np.arange(1, n + 1) - 0.5) / n)


# ---------------------------------------------------------------------------
# descdist -- reference values printed in the vignette
# ---------------------------------------------------------------------------

def test_descdist_matches_vignette(groundbeef):
    # descdist(groundbeef$serving) prints:
    #   min: 10  max: 200 / median: 79 / mean: 73.65
    #   estimated sd: 35.88 / skewness: 0.7353 / kurtosis: 3.551
    d = descdist(groundbeef)
    assert d["min"] == 10
    assert d["max"] == 200
    assert d["median"] == 79
    assert d["mean"] == pytest.approx(73.65, abs=5e-3)
    assert d["sd"] == pytest.approx(35.88, abs=5e-3)
    assert d["skewness"] == pytest.approx(0.7353, abs=5e-5)
    assert d["kurtosis"] == pytest.approx(3.551, abs=5e-4)


def test_descdist_unbiased_matches_scipy_bias_corrected(groundbeef):
    # R's Fisher (1930) corrections are what scipy calls bias=False; R's
    # kurtosis is not excess kurtosis, hence the + 3.
    d = descdist(groundbeef, method="unbiased")
    assert d["skewness"] == pytest.approx(stats.skew(groundbeef, bias=False))
    assert d["kurtosis"] == pytest.approx(stats.kurtosis(groundbeef, bias=False) + 3)


def test_descdist_sample_method_uses_uncorrected_moments(groundbeef):
    d = descdist(groundbeef, method="sample")
    assert d["skewness"] == pytest.approx(stats.skew(groundbeef, bias=True))
    assert d["kurtosis"] == pytest.approx(stats.kurtosis(groundbeef, bias=True) + 3)
    assert d["sd"] == pytest.approx(np.std(groundbeef, ddof=0))


def test_descdist_requires_four_values():
    # R: "data must be a numeric vector containing at least four values"
    with pytest.raises(ValueError):
        descdist([1.0, 2.0, 3.0])


def test_descdist_rejects_unknown_method(groundbeef):
    with pytest.raises(ValueError):
        descdist(groundbeef, method="moments")


# ---------------------------------------------------------------------------
# Maximum likelihood fitting -- fitdist(..., method = "mle")
# ---------------------------------------------------------------------------

def test_weibull_fit_matches_vignette(groundbeef):
    # fitdist(groundbeef$serving, "weibull"): shape 2.186, scale 83.348
    fit = fit_distributions(groundbeef, "weibull")[0]
    assert fit.estimate["shape"] == pytest.approx(2.186, abs=1e-3)
    assert fit.estimate["scale"] == pytest.approx(83.348, abs=1e-2)
    assert fit.loglik == pytest.approx(-1255, abs=0.5)
    assert fit.aic == pytest.approx(2514, abs=0.5)
    assert fit.bic == pytest.approx(2522, abs=0.5)


def test_normal_mle_divides_by_n_not_n_minus_one(groundbeef):
    # R's mledist maximises the likelihood, so the sd is the MLE.
    _, params = fit_distribution("normal", groundbeef)
    assert params[0] == pytest.approx(np.mean(groundbeef))
    assert params[1] == pytest.approx(np.std(groundbeef, ddof=0))
    assert params[1] != pytest.approx(np.std(groundbeef, ddof=1))


def test_lognormal_fit_matches_closed_form_mle(groundbeef):
    # R reports meanlog/sdlog, whose MLEs are the mean and (1/n) sd of log(x).
    fit = fit_distributions(groundbeef, "lognormal")[0]
    logx = np.log(groundbeef)
    assert fit.estimate["meanlog"] == pytest.approx(np.mean(logx), rel=1e-6)
    assert fit.estimate["sdlog"] == pytest.approx(np.std(logx, ddof=0), rel=1e-6)
    # R's dlnorm has no location, so scipy's loc must stay pinned at zero.
    assert fit.params[1] == 0


def test_exponential_fit_rate_is_reciprocal_mean(groundbeef):
    fit = fit_distributions(groundbeef, "exponential")[0]
    assert fit.estimate["rate"] == pytest.approx(1 / np.mean(groundbeef), rel=1e-6)
    assert fit.params[0] == 0  # R's dexp has no location


def test_gamma_and_weibull_keep_loc_pinned(groundbeef):
    for name in ("gamma", "weibull"):
        _, params = fit_distribution(name, groundbeef)
        assert params[1] == 0, f"{name} must not estimate a location"


def test_free_parameter_counts_match_r(groundbeef):
    # R's npar is length(estimate); it drives AIC and BIC.
    expected = {"normal": 2, "lognormal": 2, "weibull": 2, "gamma": 2,
                "exponential": 1, "chi2": 1}
    for name, npar in expected.items():
        fit = fit_distributions(groundbeef, name)[0]
        assert fit.n_free_params == npar, name
        assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * npar)
        assert fit.bic == pytest.approx(-2 * fit.loglik + np.log(fit.n) * npar)


def test_r_distribution_names_are_accepted(groundbeef):
    for r_name, own_name in [("lnorm", "lognormal"), ("exp", "exponential"),
                             ("norm", "normal"), ("unif", "uniform")]:
        a = fit_distributions(groundbeef, r_name)[0]
        b = fit_distributions(groundbeef, own_name)[0]
        assert a.params == pytest.approx(b.params)


def test_beta_rejects_data_outside_unit_interval(groundbeef):
    # R's dbeta lives on [0, 1] and fitdist errors out the same way.
    with pytest.raises(ValueError, match="outside"):
        fit_distributions(groundbeef, "beta")


def test_beta_fits_on_the_unit_interval():
    rng = np.random.default_rng(0)
    x = rng.beta(2, 5, size=500)
    fit = fit_distributions(x, "beta")[0]
    assert fit.estimate["shape1"] == pytest.approx(2, rel=0.2)
    assert fit.estimate["shape2"] == pytest.approx(5, rel=0.2)
    assert fit.params[2] == 0 and fit.params[3] == 1  # loc, scale pinned


# ---------------------------------------------------------------------------
# gofstat
# ---------------------------------------------------------------------------

def test_gofstat_matches_vignette(groundbeef):
    # gofstat(list(fitW, fitg, fitln)) in the vignette prints:
    #   KS   0.1396646 0.1280459 0.1493265
    #   CvM  0.6840994 0.6936073 0.8277206
    #   AD   3.5736460 3.5672984 4.5432209
    fits = fit_distributions(groundbeef, ["weibull", "gamma", "lognormal"])
    with pytest.warns(RuntimeWarning, match="ties"):
        res = gofstat(fits)

    assert res.loc["weibull", "ks"] == pytest.approx(0.1396646, abs=1e-4)
    assert res.loc["gamma", "ks"] == pytest.approx(0.1280459, abs=1e-4)
    assert res.loc["lognormal", "ks"] == pytest.approx(0.1493265, abs=1e-4)

    assert res.loc["weibull", "cvm"] == pytest.approx(0.6840994, abs=1e-3)
    assert res.loc["gamma", "cvm"] == pytest.approx(0.6936073, abs=1e-3)
    assert res.loc["lognormal", "cvm"] == pytest.approx(0.8277206, abs=1e-3)

    assert res.loc["weibull", "ad"] == pytest.approx(3.5736460, abs=1e-2)
    assert res.loc["gamma", "ad"] == pytest.approx(3.5672984, abs=1e-2)
    assert res.loc["lognormal", "ad"] == pytest.approx(4.5432209, abs=1e-2)


def test_ks_statistic_matches_scipy(groundbeef):
    """R's max(|F - i/n|, |F - (i-1)/n|) is the two-sided KS statistic."""
    fit = fit_distributions(groundbeef, "weibull")[0]
    with pytest.warns(RuntimeWarning):
        res = gofstat(fit)
    expected = stats.kstest(groundbeef, fit.dist.cdf, args=fit.params).statistic
    assert res.loc["weibull", "ks"] == pytest.approx(expected)


def test_cvm_statistic_matches_scipy():
    """R's 1/(12n) + sum((F - (2i-1)/2n)^2) is the standard CvM statistic."""
    rng = np.random.default_rng(7)
    x = rng.normal(3, 2, size=300)
    fit = fit_distributions(x, "normal")[0]
    res = gofstat(fit)
    expected = stats.cramervonmises(x, fit.dist.cdf, args=fit.params).statistic
    assert res.loc["normal", "cvm"] == pytest.approx(expected)


def test_ad_statistic_matches_textbook_formula():
    rng = np.random.default_rng(11)
    x = np.sort(rng.gamma(3, 2, size=400))
    fit = fit_distributions(x, "gamma")[0]
    res = gofstat(fit)

    n = len(x)
    f = fit.dist.cdf(x, *fit.params)
    i = np.arange(1, n + 1)
    expected = -n - np.sum((2 * i - 1) * (np.log(f) + np.log(1 - f[::-1]))) / n
    assert res.loc["gamma", "ad"] == pytest.approx(expected)


def test_ks_test_not_computed_below_thirty_observations():
    rng = np.random.default_rng(3)
    x = rng.normal(size=25)
    res = gofstat(fit_distributions(x, "normal"))
    assert res.loc["normal", "ks_test"] == "not computed"


def test_only_tabulated_distributions_get_test_decisions():
    """R has critical values for exp, gamma, weibull and logis only."""
    rng = np.random.default_rng(5)
    x = rng.exponential(2.0, size=200)
    res = gofstat(fit_distributions(x, ["exponential", "normal"]))
    assert res.loc["exponential", "ad_test"] in ("rejected", "not rejected")
    assert res.loc["exponential", "cvm_test"] in ("rejected", "not rejected")
    assert res.loc["normal", "ad_test"] == "not computed"
    assert res.loc["normal", "cvm_test"] == "not computed"


def test_exponential_data_is_not_rejected_as_exponential():
    rng = np.random.default_rng(2024)
    x = rng.exponential(3.0, size=500)
    res = gofstat(fit_distributions(x, "exponential"))
    assert res.loc["exponential", "ks_test"] == "not rejected"
    assert res.loc["exponential", "ad_test"] == "not rejected"
    assert res.loc["exponential", "cvm_test"] == "not rejected"


def test_normal_data_is_rejected_as_exponential():
    rng = np.random.default_rng(2025)
    x = rng.normal(50, 5, size=500)
    res = gofstat(fit_distributions(x, "exponential"))
    assert res.loc["exponential", "ks_test"] == "rejected"


def test_chisq_degrees_of_freedom_account_for_estimated_parameters(groundbeef):
    with pytest.warns(RuntimeWarning):
        res = gofstat(fit_distributions(groundbeef, ["weibull", "exponential"]))
    # df = cells - 1 - npar, and exponential estimates one fewer parameter.
    assert res.loc["exponential", "chisq_df"] == res.loc["weibull", "chisq_df"] + 1


def test_gofstat_rejects_mismatched_datasets():
    a = fit_distributions(np.arange(1.0, 51.0), "normal")
    b = fit_distributions(np.arange(2.0, 52.0), "normal")
    with pytest.raises(ValueError, match="same dataset"):
        gofstat(a + b)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def test_qq_plot_uses_ppoints_positions(groundbeef):
    fig = quantile_comparison_plot(groundbeef, "weibull", include_histogram=False)
    fit = fit_distributions(groundbeef, "weibull")[0]
    expected = fit.dist.ppf(ppoints(len(groundbeef), 0.5), *fit.params)
    assert np.allclose(fig.data[0].x, expected)
    assert np.allclose(fig.data[0].y, np.sort(groundbeef))


def test_pp_plot_uses_ppoints_not_i_over_n(groundbeef):
    """R's ppcomp puts the empirical probabilities at ppoints(n, a = 0.5)."""
    _, _, pp_fig, _ = quantile_comparison_plot(groundbeef, "weibull")
    n = len(groundbeef)
    assert np.allclose(pp_fig.data[0].y, ppoints(n, 0.5))
    assert not np.allclose(pp_fig.data[0].y, np.arange(1, n + 1) / n)


def test_cdf_plot_uses_ppoints_not_i_over_n(groundbeef):
    """R's cdfcomp does the same for continuous data."""
    _, _, _, cdf_fig = quantile_comparison_plot(groundbeef, "weibull")
    n = len(groundbeef)
    assert np.allclose(cdf_fig.data[0].y, ppoints(n, 0.5))


def test_identity_line_spans_the_whole_panel(groundbeef):
    """The y = x line is abline(0, 1), not a copy of one fit's quantiles."""
    fig = quantile_comparison_plot(
        groundbeef, ["weibull", "gamma", "lognormal"], include_histogram=False
    )
    line = fig.data[-1]
    assert line.name == "y = x"
    assert len(line.x) == 2
    assert np.allclose(line.x, line.y)
    # It must cover every plotted series, whichever fit is drawn last.
    for trace in fig.data[:-1]:
        assert line.x[0] <= min(trace.x) and line.x[1] >= max(trace.x)


def test_all_four_figures_are_returned(groundbeef):
    figs = quantile_comparison_plot(groundbeef, ["weibull", "gamma"])
    assert len(figs) == 4
    qq, dens, pp, cdf = figs
    # One trace per fit, plus the identity line or the empirical series.
    assert len(qq.data) == 3
    assert len(dens.data) == 3
    assert len(pp.data) == 3
    assert len(cdf.data) == 3


def test_ynoise_jitters_only_later_series_and_keeps_hover_exact(groundbeef):
    fig = quantile_comparison_plot(
        groundbeef, ["weibull", "gamma"], include_histogram=False,
        ynoise=True, seed=1,
    )
    sorted_data = np.sort(groundbeef)
    assert np.allclose(fig.data[0].y, sorted_data)          # first fit untouched
    assert not np.allclose(fig.data[1].y, sorted_data)      # second jittered
    assert np.max(np.abs(fig.data[1].y - sorted_data)) <= 0.02
    assert np.allclose(fig.data[1].customdata, sorted_data)  # hover unaffected


def test_explicit_params_must_match_the_scipy_signature(groundbeef):
    with pytest.raises(ValueError, match="scipy parameters"):
        quantile_comparison_plot(groundbeef, "weibull", dist_params=(2.0, 83.0))
    # shape, loc, scale is the full weibull_min signature
    quantile_comparison_plot(
        groundbeef, "weibull", dist_params=(2.0, 0.0, 83.0), include_histogram=False
    )


def test_density_plot_defaults_to_sturges_bins(groundbeef):
    """R's hist() also defaults to Sturges' rule."""
    _, dens, _, _ = quantile_comparison_plot(groundbeef, "weibull")
    expected = len(np.histogram_bin_edges(groundbeef, bins="sturges")) - 1
    assert len(dens.data[0].x) == expected


# ---------------------------------------------------------------------------
# Cullen and Frey graph
# ---------------------------------------------------------------------------

def test_cullen_frey_axis_limits_follow_r(groundbeef):
    """R: xmax = max(4, ceiling(skew^2)), kurtmax = max(10, ceiling(kurt))."""
    fig = cullen_and_frey_plot(groundbeef, show_bootstrap=False)
    d = descdist(groundbeef)
    assert fig.layout.xaxis.range == (0, max(4, np.ceil(d["skewness"] ** 2)))
    # The kurtosis axis is inverted, running from kurtmax down to 1.
    assert fig.layout.yaxis.range == (max(10, np.ceil(d["kurtosis"])), 1)


def test_cullen_frey_bootstrap_widens_the_limits(groundbeef):
    plain = cullen_and_frey_plot(groundbeef, show_bootstrap=False)
    booted = cullen_and_frey_plot(groundbeef, n_bootstrap=200, seed=1)
    assert booted.layout.xaxis.range[1] >= plain.layout.xaxis.range[1]
    assert booted.layout.yaxis.range[0] >= plain.layout.yaxis.range[0]


def test_cullen_frey_observed_point_is_skewness_squared_vs_kurtosis(groundbeef):
    fig = cullen_and_frey_plot(groundbeef, show_bootstrap=False)
    observed = [t for t in fig.data if "observed" in (t.name or "")][0]
    d = descdist(groundbeef)
    assert observed.x[0] == pytest.approx(d["skewness"] ** 2)
    assert observed.y[0] == pytest.approx(d["kurtosis"])


def test_cullen_frey_bootstrap_does_not_disturb_global_rng(groundbeef):
    np.random.seed(12345)
    before = np.random.random()
    np.random.seed(12345)
    cullen_and_frey_plot(groundbeef, n_bootstrap=50, seed=99)
    assert np.random.random() == before


def test_cullen_frey_bootstrap_is_reproducible(groundbeef):
    a = cullen_and_frey_plot(groundbeef, n_bootstrap=50, seed=42)
    b = cullen_and_frey_plot(groundbeef, n_bootstrap=50, seed=42)
    boot_a = [t for t in a.data if t.name == "bootstrap"][0]
    boot_b = [t for t in b.data if t.name == "bootstrap"][0]
    assert np.allclose(boot_a.x, boot_b.x)


def test_cullen_frey_theoretical_points_match_r():
    """R marks normal (0, 3), uniform (0, 9/5), exponential (4, 9), logistic (0, 4.2)."""
    rng = np.random.default_rng(0)
    fig = cullen_and_frey_plot(rng.normal(size=200), show_bootstrap=False)
    points = {t.name: (t.x[0], t.y[0]) for t in fig.data
              if t.name in {"normal", "uniform", "exponential", "logistic"}}
    assert points == {
        "normal": (0.0, 3.0),
        "uniform": (0.0, 1.8),
        "exponential": (4.0, 9.0),
        "logistic": (0.0, 4.2),
    }


def test_cullen_frey_gamma_curve_passes_through_the_exponential_point():
    """Gamma with shape 1 is the exponential, so (4, 9) lies on the curve."""
    rng = np.random.default_rng(0)
    fig = cullen_and_frey_plot(rng.exponential(size=300), show_bootstrap=False)
    gamma = [t for t in fig.data if t.name == "gamma"][0]
    i = int(np.argmin(np.abs(np.asarray(gamma.x) - 4.0)))
    assert gamma.x[i] == pytest.approx(4.0, abs=1e-3)
    assert gamma.y[i] == pytest.approx(9.0, abs=1e-3)


def test_cullen_frey_curves_are_finite():
    """R's exp(seq(-100, 100, 0.1)) sweep must not leak infinities into plotly."""
    rng = np.random.default_rng(0)
    fig = cullen_and_frey_plot(rng.lognormal(size=300), show_bootstrap=False)
    for trace in fig.data:
        assert np.all(np.isfinite(np.asarray(trace.x, dtype=float))), trace.name
        assert np.all(np.isfinite(np.asarray(trace.y, dtype=float))), trace.name


def test_cullen_frey_discrete_switches_the_overlays():
    rng = np.random.default_rng(0)
    x = rng.poisson(4.0, size=300).astype(float)
    names = {t.name for t in cullen_and_frey_plot(x, discrete=True, show_bootstrap=False).data}
    assert "Poisson" in names and "negative binomial" in names
    assert "gamma" not in names and "beta" not in names
    # R draws only the normal point in the discrete case.
    assert "uniform" not in names and "exponential" not in names


def test_cullen_frey_rejects_tiny_bootstrap(groundbeef):
    # R: "boot must be NULL or a integer above 10"
    with pytest.raises(ValueError):
        cullen_and_frey_plot(groundbeef, n_bootstrap=5)


# ---------------------------------------------------------------------------
# Empirical plots
# ---------------------------------------------------------------------------

def test_empirical_cdf_uses_ppoints(groundbeef):
    """R's plotdist draws the empirical CDF at ppoints(n)."""
    fig = empirical_cdf_plot(groundbeef, show_percentiles=False)
    assert np.allclose(fig.data[0].y, ppoints(len(groundbeef)))


def test_empirical_density_bins_default_to_sturges(groundbeef):
    fig = empirical_density_plot(groundbeef)
    expected = len(np.histogram_bin_edges(groundbeef, bins="sturges")) - 1
    assert len(fig.data[0].x) == expected


def test_empirical_density_histogram_integrates_to_one(groundbeef):
    fig = empirical_density_plot(groundbeef)
    bar = fig.data[0]
    assert np.sum(np.asarray(bar.y) * np.asarray(bar.width)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_nan_values_are_dropped():
    x = np.concatenate([np.arange(1.0, 51.0), [np.nan, np.nan]])
    assert descdist(x)["max"] == 50
    fig = quantile_comparison_plot(x, "normal", include_histogram=False)
    assert len(fig.data[0].x) == 50


def test_unsupported_distribution_name_is_reported():
    with pytest.raises(ValueError, match="Unsupported distribution name"):
        quantile_comparison_plot(np.arange(1.0, 51.0), "wibble")


def test_scipy_distribution_objects_are_accepted(groundbeef):
    """A registered scipy object keeps R's parameterisation."""
    fig = quantile_comparison_plot(groundbeef, stats.weibull_min, include_histogram=False)
    reference = quantile_comparison_plot(groundbeef, "weibull", include_histogram=False)
    assert np.allclose(fig.data[0].x, reference.data[0].x)
