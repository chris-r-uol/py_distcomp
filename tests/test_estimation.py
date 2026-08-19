"""
Checks for the estimation methods other than maximum likelihood, and for the
Q-Q confidence bands.

The moment estimators have closed forms, which R uses and which are checked
here directly.  Quantile matching is checked by the property that defines it --
the fitted quantiles hit the sample quantiles it was asked to match.  The
goodness-of-fit estimators are checked by the property that defines them: each
beats maximum likelihood on the statistic it was asked to minimise.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.bootdist import bootdist, qq_confidence_band  # noqa: E402
from py_distcomp.estimation import (  # noqa: E402
    ESTIMATION_METHODS,
    GOF_STATISTICS,
    fit_by_method,
    maximum_goodness_of_fit,
    moment_match,
    quantile_match,
)
from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.quantile_multi_comparison import quantile_comparison_plot  # noqa: E402


@pytest.fixture(scope="module")
def groundbeef():
    return np.load(pathlib.Path(__file__).parent / "groundbeef.npy")


# ---------------------------------------------------------------------------
# Moment matching -- mmedist
# ---------------------------------------------------------------------------

def test_normal_moment_estimates_are_the_sample_moments(groundbeef):
    """R: estimate <- c(mean = m, sd = sqrt(v)), with v the population variance."""
    fit = fit_distributions(groundbeef, "normal", method="mme")[0]
    assert fit.estimate["mean"] == pytest.approx(np.mean(groundbeef))
    assert fit.estimate["sd"] == pytest.approx(np.std(groundbeef, ddof=0))


def test_moment_matching_uses_the_population_variance(groundbeef):
    """mmedist uses v <- (n - 1)/n * var(data), not the sample variance."""
    fit = fit_distributions(groundbeef, "normal", method="mme")[0]
    assert fit.estimate["sd"] != pytest.approx(np.std(groundbeef, ddof=1))


def test_gamma_moment_estimates_match_the_closed_form(groundbeef):
    # R: shape <- m^2/v, rate <- m/v
    m, v = np.mean(groundbeef), np.var(groundbeef, ddof=0)
    fit = fit_distributions(groundbeef, "gamma", method="mme")[0]
    assert fit.estimate["shape"] == pytest.approx(m ** 2 / v)
    assert fit.estimate["rate"] == pytest.approx(m / v)


def test_lognormal_moment_estimates_match_the_closed_form(groundbeef):
    # R: sd2 <- log(1 + v/m^2); meanlog <- log(m) - sd2/2; sdlog <- sqrt(sd2)
    m, v = np.mean(groundbeef), np.var(groundbeef, ddof=0)
    sd2 = np.log(1 + v / m ** 2)
    fit = fit_distributions(groundbeef, "lognormal", method="mme")[0]
    assert fit.estimate["meanlog"] == pytest.approx(np.log(m) - sd2 / 2)
    assert fit.estimate["sdlog"] == pytest.approx(np.sqrt(sd2))


def test_exponential_and_poisson_moment_estimates(groundbeef):
    m = np.mean(groundbeef)
    assert fit_distributions(groundbeef, "exponential", method="mme")[0].estimate[
        "rate"] == pytest.approx(1 / m)
    counts = np.round(groundbeef / 10)
    assert fit_distributions(counts, "poisson", method="mme")[0].estimate[
        "lambda"] == pytest.approx(np.mean(counts))


def test_uniform_and_logistic_moment_estimates(groundbeef):
    m, v = np.mean(groundbeef), np.var(groundbeef, ddof=0)
    unif = fit_distributions(groundbeef, "uniform", method="mme")[0].estimate
    assert unif["min"] == pytest.approx(m - np.sqrt(3 * v))
    assert unif["max"] == pytest.approx(m + np.sqrt(3 * v))
    logis = fit_distributions(groundbeef, "logistic", method="mme")[0].estimate
    assert logis["location"] == pytest.approx(m)
    assert logis["scale"] == pytest.approx(np.sqrt(3 * v) / np.pi)


def test_moment_matching_reproduces_the_first_moment(groundbeef):
    """Whatever else it does, the fitted mean should equal the sample mean."""
    for name in ("normal", "gamma", "lognormal", "exponential"):
        fit = fit_distributions(groundbeef, name, method="mme")[0]
        assert fit.dist.mean(*fit.params) == pytest.approx(np.mean(groundbeef), rel=1e-6)


def test_numerical_moment_matching_for_a_distribution_without_a_closed_form(groundbeef):
    """The Weibull has no closed form in R, so the moments are matched numerically."""
    fit = fit_distributions(groundbeef, "weibull", method="mme")[0]
    assert fit.dist.mean(*fit.params) == pytest.approx(np.mean(groundbeef), rel=0.02)
    assert fit.dist.var(*fit.params) == pytest.approx(np.var(groundbeef, ddof=0), rel=0.1)


def test_moment_matching_rejects_an_under_dispersed_negative_binomial():
    equidispersed = np.random.default_rng(0).poisson(3.0, 500).astype(float)
    with pytest.raises(ValueError, match="over-dispersed"):
        moment_match("negative_binomial", equidispersed)


def test_moment_matching_needs_enough_moments(groundbeef):
    with pytest.raises(ValueError, match="at least as many moments"):
        moment_match("weibull", groundbeef, order=[1])


# ---------------------------------------------------------------------------
# Quantile matching -- qmedist
# ---------------------------------------------------------------------------

def test_quantile_matching_hits_the_quantiles_it_was_given(groundbeef):
    """The defining property: the fitted quantiles land on the sample ones."""
    probs = [0.25, 0.75]
    fit = fit_distributions(groundbeef, "weibull", method="qme", probs=probs)[0]
    fitted = fit.dist.ppf(probs, *fit.params)
    empirical = np.quantile(groundbeef, probs, method="linear")
    assert np.allclose(fitted, empirical, rtol=1e-3)


def test_quantile_matching_can_target_the_tail(groundbeef):
    """Matching high quantiles fits the tail better than the middle."""
    tail = [0.90, 0.99]
    fit = fit_distributions(groundbeef, "weibull", method="qme", probs=tail)[0]
    fitted = fit.dist.ppf(tail, *fit.params)
    assert np.allclose(fitted, np.quantile(groundbeef, tail, method="linear"), rtol=1e-3)


def test_quantile_matching_defaults_to_evenly_spaced_probabilities(groundbeef):
    """Two parameters, so 1/3 and 2/3 by default."""
    default = fit_distributions(groundbeef, "weibull", method="qme")[0]
    explicit = fit_distributions(
        groundbeef, "weibull", method="qme", probs=[1 / 3, 2 / 3]
    )[0]
    assert np.allclose(default.params, explicit.params)


def test_quantile_matching_validates_its_probabilities(groundbeef):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        quantile_match("weibull", groundbeef, probs=[0.0, 0.5])
    with pytest.raises(ValueError, match="at least as many"):
        quantile_match("weibull", groundbeef, probs=[0.5])


# ---------------------------------------------------------------------------
# Maximum goodness-of-fit -- mgedist
# ---------------------------------------------------------------------------

def test_each_gof_estimator_beats_mle_on_its_own_statistic(groundbeef):
    """The property that defines the method, checked for KS, CvM and AD."""
    mle = fit_distributions(groundbeef, "weibull")[0]
    with pytest.warns(RuntimeWarning):
        mle_stats = gofstat(mle)

    for gof, column in [("KS", "ks"), ("CvM", "cvm"), ("AD", "ad")]:
        fit = fit_distributions(groundbeef, "weibull", method="mge", gof=gof)[0]
        with pytest.warns(RuntimeWarning):
            got = gofstat(fit)
        assert got.loc["weibull", column] <= mle_stats.loc["weibull", column] + 1e-9, gof


def test_mle_still_beats_them_on_the_likelihood(groundbeef):
    """The converse: nothing beats maximum likelihood at maximising it."""
    mle = fit_distributions(groundbeef, "weibull")[0]
    for method in ("mme", "qme", "mge"):
        other = fit_distributions(groundbeef, "weibull", method=method)[0]
        assert other.loglik <= mle.loglik + 1e-6, method


def test_every_gof_statistic_produces_a_fit(groundbeef):
    for gof in GOF_STATISTICS:
        fit = fit_distributions(groundbeef, "weibull", method="mge", gof=gof)[0]
        assert np.all(np.isfinite(fit.params))
        assert fit.estimate["shape"] > 0 and fit.estimate["scale"] > 0


def test_tail_weighted_variants_differ_from_the_symmetric_one(groundbeef):
    """ADR weights the right tail and ADL the left, so they should disagree."""
    right = fit_distributions(groundbeef, "weibull", method="mge", gof="AD2R")[0]
    left = fit_distributions(groundbeef, "weibull", method="mge", gof="AD2L")[0]
    assert right.estimate["shape"] != pytest.approx(left.estimate["shape"], rel=0.05)


def test_unknown_gof_is_rejected(groundbeef):
    with pytest.raises(ValueError, match="gof must be one of"):
        maximum_goodness_of_fit("weibull", groundbeef, gof="chisq")


def test_gof_estimation_refuses_discrete_distributions():
    counts = np.random.default_rng(0).poisson(3.0, 300).astype(float)
    with pytest.raises(ValueError, match="continuous"):
        maximum_goodness_of_fit("poisson", counts)


# ---------------------------------------------------------------------------
# Dispatch and integration
# ---------------------------------------------------------------------------

def test_all_methods_are_reachable_and_recorded(groundbeef):
    for method in ESTIMATION_METHODS:
        fit = fit_distributions(groundbeef, "weibull", method=method)[0]
        assert fit.method == method
        assert np.all(np.isfinite(fit.params))


def test_unknown_method_is_rejected(groundbeef):
    with pytest.raises(ValueError, match="method must be one of"):
        fit_distributions(groundbeef, "weibull", method="bayes")


def test_mle_takes_no_extra_arguments(groundbeef):
    with pytest.raises(TypeError, match="no extra arguments"):
        fit_by_method("weibull", groundbeef, "mle", gof="KS")


def test_methods_agree_broadly_on_a_well_behaved_sample():
    """On a large clean sample every estimator should land in the same place."""
    x = stats.weibull_min.rvs(2.0, 0, 50, size=4000, random_state=np.random.default_rng(0))
    estimates = {
        m: fit_distributions(x, "weibull", method=m)[0].estimate
        for m in ESTIMATION_METHODS
    }
    for method, estimate in estimates.items():
        assert estimate["shape"] == pytest.approx(2.0, rel=0.15), method
        assert estimate["scale"] == pytest.approx(50.0, rel=0.15), method


def test_bootstrap_refits_with_the_same_method(groundbeef):
    """Bootstrapping an mge fit with mle would describe a different estimator."""
    fit = fit_distributions(groundbeef, "weibull", method="mge", gof="ADR")[0]
    boot = bootdist(fit, niter=120, seed=1)
    assert boot.n_failed == 0
    # The resamples should centre on the mge estimate, not the mle one.
    mle = fit_distributions(groundbeef, "weibull")[0]
    assert abs(boot.ci.loc["shape", "median"] - fit.estimate["shape"]) < abs(
        boot.ci.loc["shape", "median"] - mle.estimate["shape"]
    )


def test_estimation_needs_a_registered_distribution(groundbeef):
    with pytest.raises(ValueError, match="registered distribution"):
        moment_match(stats.rice, groundbeef)


# ---------------------------------------------------------------------------
# Q-Q confidence bands
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weibull_sample():
    return stats.weibull_min.rvs(2.2, 0, 83, size=200,
                                 random_state=np.random.default_rng(7))


def test_band_has_a_row_per_observation(weibull_sample):
    fit = fit_distributions(weibull_sample, "weibull")[0]
    band = qq_confidence_band(fit, niter=200, seed=1)
    assert len(band) == len(weibull_sample)
    assert list(band.columns) == ["theoretical", "observed", "lower", "upper"]
    assert np.all(band.lower <= band.upper)
    assert np.allclose(band.observed, np.sort(weibull_sample))


def test_pointwise_band_is_calibrated(weibull_sample):
    """On data truly from the fitted family, ~5% of points fall outside a 95% band."""
    rng = np.random.default_rng(11)
    outside = []
    for _ in range(12):
        x = stats.weibull_min.rvs(2.2, 0, 83, size=150, random_state=rng)
        fit = fit_distributions(x, "weibull")[0]
        band = qq_confidence_band(fit, kind="pointwise", niter=250,
                                  seed=int(rng.integers(1e6)))
        outside.append(((band.observed < band.lower) | (band.observed > band.upper)).mean())
    assert np.mean(outside) == pytest.approx(0.05, abs=0.04)


def test_simultaneous_band_is_wider_than_the_pointwise_one(weibull_sample):
    fit = fit_distributions(weibull_sample, "weibull")[0]
    point = qq_confidence_band(fit, kind="pointwise", niter=250, seed=1)
    sim = qq_confidence_band(fit, kind="simultaneous", niter=250, seed=1)
    assert np.mean(sim.upper - sim.lower) > np.mean(point.upper - point.lower)


def test_not_refitting_gives_a_conservative_band(weibull_sample):
    """Treating the estimated parameters as known makes the band too wide."""
    fit = fit_distributions(weibull_sample, "weibull")[0]
    refit = qq_confidence_band(fit, niter=250, refit=True, seed=1)
    fixed = qq_confidence_band(fit, niter=250, refit=False, seed=1)
    assert np.mean(fixed.upper - fixed.lower) > np.mean(refit.upper - refit.lower)


def test_a_wrong_model_breaks_out_of_the_band():
    """Lognormal data fitted with a Weibull should not stay inside."""
    x = stats.lognorm.rvs(0.9, 0, 60, size=250, random_state=np.random.default_rng(3))
    fit = fit_distributions(x, "weibull")[0]
    band = qq_confidence_band(fit, kind="simultaneous", niter=250, seed=1)
    assert ((band.observed < band.lower) | (band.observed > band.upper)).any()


def test_band_widens_with_the_level(weibull_sample):
    fit = fit_distributions(weibull_sample, "weibull")[0]
    narrow = qq_confidence_band(fit, level=0.80, niter=250, seed=1)
    wide = qq_confidence_band(fit, level=0.99, niter=250, seed=1)
    assert np.mean(wide.upper - wide.lower) > np.mean(narrow.upper - narrow.lower)


def test_band_validates_its_arguments(weibull_sample):
    fit = fit_distributions(weibull_sample, "weibull")[0]
    with pytest.raises(ValueError, match="between 0 and 1"):
        qq_confidence_band(fit, level=1.5)
    with pytest.raises(ValueError, match="at least 10"):
        qq_confidence_band(fit, niter=5)
    with pytest.raises(ValueError, match="pointwise"):
        qq_confidence_band(fit, kind="envelope")


def test_band_does_not_disturb_the_global_rng(weibull_sample):
    fit = fit_distributions(weibull_sample, "weibull")[0]
    np.random.seed(4321)
    before = np.random.random()
    np.random.seed(4321)
    qq_confidence_band(fit, niter=50, seed=1)
    assert np.random.random() == before


def test_qq_plot_draws_the_band(weibull_sample):
    fig = quantile_comparison_plot(
        weibull_sample, "weibull", include_histogram=False,
        confidence_band=0.95, band_niter=200, seed=1,
    )
    names = [t.name for t in fig.data]
    assert any("band" in (n or "") for n in names)
    # The band is drawn first so the points sit on top of it.
    assert "band" in fig.data[0].name
    assert fig.data[0].fill == "toself"


def test_qq_plot_has_no_band_by_default(weibull_sample):
    fig = quantile_comparison_plot(weibull_sample, "weibull", include_histogram=False)
    assert not any("band" in (t.name or "") for t in fig.data)


def test_band_covers_only_the_first_model(weibull_sample):
    fig = quantile_comparison_plot(
        weibull_sample, ["weibull", "gamma"], include_histogram=False,
        confidence_band=0.95, band_niter=200, seed=1,
    )
    assert sum("band" in (t.name or "") for t in fig.data) == 1
