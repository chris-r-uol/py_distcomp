"""
Checks for standard errors and the bootstrap.

The standard errors are validated three ways: against the closed-form maximum
likelihood standard errors, which several distributions have exactly; against
the "Std. Error" column printed in the fitdistrplus vignette; and against the
bootstrap, which reaches the same answer by an entirely different route.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.bootdist import bootdist  # noqa: E402
from py_distcomp.bootdist_plots import bootdist_plot, confint_plot  # noqa: E402
from py_distcomp.distributions import (  # noqa: E402
    DISTRIBUTION_SPECS,
    fit_distribution,
    r_estimate,
    scipy_params,
)
from py_distcomp.gofstat import fit_distributions  # noqa: E402
from py_distcomp.mixture import fit_mixture  # noqa: E402


@pytest.fixture(scope="module")
def groundbeef():
    return np.load(pathlib.Path(__file__).parent / "groundbeef.npy")


@pytest.fixture(scope="module")
def weibull_fit(groundbeef):
    return fit_distributions(groundbeef, "weibull")[0]


# ---------------------------------------------------------------------------
# The parameterisation round trip the standard errors depend on
# ---------------------------------------------------------------------------

def test_scipy_params_inverts_r_estimate(groundbeef):
    """SEs are of R's parameters, so the two mappings must agree exactly."""
    positive = np.abs(groundbeef)
    counts = np.round(positive / 10.0)  # whole numbers, for the discrete families
    for name, spec in DISTRIBUTION_SPECS.items():
        if spec.support is not None:
            continue  # beta needs data on [0, 1]
        _, params = fit_distribution(name, counts if spec.discrete else positive)
        values = list(r_estimate(spec.r_name, params, spec).values())
        assert np.allclose(scipy_params(spec.r_name, values, spec), params), name


def test_gamma_rate_is_not_scipys_scale(groundbeef):
    """The mapping is not the identity, which is the whole reason it exists."""
    fit = fit_distributions(groundbeef, "gamma")[0]
    assert fit.estimate["rate"] == pytest.approx(1 / fit.params[2])
    assert fit.estimate["rate"] != pytest.approx(fit.params[2])


# ---------------------------------------------------------------------------
# Standard errors against closed forms
# ---------------------------------------------------------------------------

def test_normal_standard_errors_match_closed_form(groundbeef):
    # SE(mean) = sd/sqrt(n), SE(sd) = sd/sqrt(2n) for the normal MLE.
    fit = fit_distributions(groundbeef, "normal")[0]
    n, sd = fit.n, fit.estimate["sd"]
    errors = fit.std_error
    assert errors["mean"] == pytest.approx(sd / np.sqrt(n), rel=1e-5)
    assert errors["sd"] == pytest.approx(sd / np.sqrt(2 * n), rel=1e-5)


def test_lognormal_standard_errors_match_closed_form(groundbeef):
    # On the log scale the lognormal MLE is just the normal one.
    fit = fit_distributions(groundbeef, "lognormal")[0]
    n, sdlog = fit.n, fit.estimate["sdlog"]
    errors = fit.std_error
    assert errors["meanlog"] == pytest.approx(sdlog / np.sqrt(n), rel=1e-5)
    assert errors["sdlog"] == pytest.approx(sdlog / np.sqrt(2 * n), rel=1e-5)


def test_exponential_standard_error_matches_closed_form(groundbeef):
    # SE(rate) = rate / sqrt(n).
    fit = fit_distributions(groundbeef, "exponential")[0]
    assert fit.std_error["rate"] == pytest.approx(
        fit.estimate["rate"] / np.sqrt(fit.n), rel=1e-4
    )


def test_weibull_standard_errors_match_the_vignette(weibull_fit):
    # fitdist(groundbeef$serving, "weibull") prints:
    #   shape 2.186 (Std. Error 0.1046) / scale 83.348 (Std. Error 2.5269)
    errors = weibull_fit.std_error
    assert errors["shape"] == pytest.approx(0.1046, abs=5e-4)
    assert errors["scale"] == pytest.approx(2.5269, abs=5e-3)


def test_standard_errors_shrink_with_sample_size():
    """They should fall as 1/sqrt(n)."""
    rng = np.random.default_rng(0)
    small = fit_distributions(rng.gumbel(10, 3, 250), "gumbel")[0]
    large = fit_distributions(rng.gumbel(10, 3, 4000), "gumbel")[0]
    ratio = small.std_error["loc"] / large.std_error["loc"]
    assert ratio == pytest.approx(np.sqrt(4000 / 250), rel=0.3)


# ---------------------------------------------------------------------------
# vcov, correlation, confint
# ---------------------------------------------------------------------------

def test_vcov_is_symmetric_and_matches_the_standard_errors(weibull_fit):
    vcov = weibull_fit.vcov
    assert list(vcov.index) == list(weibull_fit.estimate)
    assert np.allclose(vcov.to_numpy(), vcov.to_numpy().T)
    assert np.allclose(
        np.sqrt(np.diag(vcov.to_numpy())),
        [weibull_fit.std_error[k] for k in weibull_fit.estimate],
    )


def test_correlation_has_unit_diagonal_and_valid_range(weibull_fit):
    correlation = weibull_fit.correlation.to_numpy()
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.all(np.abs(correlation) <= 1 + 1e-9)


def test_confint_brackets_the_estimate(weibull_fit):
    interval = weibull_fit.confint()
    for name, estimate in weibull_fit.estimate.items():
        assert interval.loc[name, "lower"] < estimate < interval.loc[name, "upper"]


def test_confint_widens_with_the_level(weibull_fit):
    narrow = weibull_fit.confint(0.90)
    wide = weibull_fit.confint(0.99)
    for name in weibull_fit.estimate:
        assert (wide.loc[name, "upper"] - wide.loc[name, "lower"]) > (
            narrow.loc[name, "upper"] - narrow.loc[name, "lower"]
        )


def test_confint_rejects_an_impossible_level(weibull_fit):
    for level in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            weibull_fit.confint(level)


def test_summary_reports_estimates_beside_standard_errors(weibull_fit):
    summary = weibull_fit.summary()
    assert list(summary.columns) == ["estimate", "std_error"]
    assert summary.loc["shape", "estimate"] == pytest.approx(2.186, abs=1e-3)


def test_single_parameter_fit_has_a_standard_error():
    """A 1x1 Hessian must still invert."""
    rng = np.random.default_rng(3)
    fit = fit_distributions(rng.chisquare(4, 500), "chi2")[0]
    assert np.isfinite(fit.std_error["df"])
    assert fit.std_error["df"] > 0


def test_uniform_has_no_usable_hessian():
    """Its likelihood is flat inside the support, so the information is singular.

    R returns NULL for the variance-covariance matrix in the same situation.
    """
    rng = np.random.default_rng(4)
    fit = fit_distributions(rng.uniform(0, 10, 500), "uniform")[0]
    assert fit.vcov is None
    assert fit.correlation is None
    assert all(np.isnan(v) for v in fit.std_error.values())


# ---------------------------------------------------------------------------
# bootdist
# ---------------------------------------------------------------------------

def test_bootstrap_interval_agrees_with_the_wald_interval(weibull_fit):
    """Two different routes to the same uncertainty, on a well-behaved fit."""
    boot = bootdist(weibull_fit, niter=600, seed=1)
    wald = weibull_fit.confint(0.95)
    for name in weibull_fit.estimate:
        lower, upper = boot.ci.loc[name].iloc[1], boot.ci.loc[name].iloc[2]
        assert lower == pytest.approx(wald.loc[name, "lower"], rel=0.1)
        assert upper == pytest.approx(wald.loc[name, "upper"], rel=0.1)


def test_bootstrap_median_is_close_to_the_estimate(weibull_fit):
    boot = bootdist(weibull_fit, niter=400, seed=2)
    for name, estimate in weibull_fit.estimate.items():
        assert boot.ci.loc[name, "median"] == pytest.approx(estimate, rel=0.05)


def test_bootstrap_interval_contains_the_estimate(weibull_fit):
    boot = bootdist(weibull_fit, niter=400, seed=3)
    for name, estimate in weibull_fit.estimate.items():
        assert boot.ci.loc[name].iloc[1] < estimate < boot.ci.loc[name].iloc[2]


def test_nonparametric_bootstrap_also_works(weibull_fit):
    boot = bootdist(weibull_fit, niter=400, bootmethod="nonparam", seed=4)
    assert boot.method == "nonparam"
    assert boot.n_converged > 0
    for name, estimate in weibull_fit.estimate.items():
        assert boot.ci.loc[name].iloc[1] < estimate < boot.ci.loc[name].iloc[2]


def test_bootstrap_covers_the_truth_at_about_the_nominal_rate():
    """Fit many samples from a known distribution; ~95% of intervals should cover."""
    rng = np.random.default_rng(11)
    covered = 0
    trials = 40
    for _ in range(trials):
        sample = stats.gumbel_r.rvs(10.0, 3.0, size=300, random_state=rng)
        boot = bootdist(fit_distributions(sample, "gumbel")[0], niter=200,
                        seed=int(rng.integers(1e6)))
        limits = boot.ci.loc["loc"]
        covered += limits.iloc[1] <= 10.0 <= limits.iloc[2]
    # Binomial noise on 40 trials is wide; this catches gross miscalibration.
    assert covered >= 32, f"only {covered}/{trials} intervals covered the truth"


def test_conf_level_changes_the_interval_width(weibull_fit):
    narrow = bootdist(weibull_fit, niter=400, seed=5, conf_level=0.80)
    wide = bootdist(weibull_fit, niter=400, seed=5, conf_level=0.99)
    n_lo, n_hi = narrow.ci.loc["shape"].iloc[1], narrow.ci.loc["shape"].iloc[2]
    w_lo, w_hi = wide.ci.loc["shape"].iloc[1], wide.ci.loc["shape"].iloc[2]
    assert (w_hi - w_lo) > (n_hi - n_lo)
    # An 80% interval is bounded by the 10th and 90th percentiles.
    assert list(narrow.ci.columns) == ["median", "10%", "90%"]
    assert list(wide.ci.columns) == ["median", "0.5%", "99.5%"]


def test_bootstrap_is_reproducible(weibull_fit):
    a = bootdist(weibull_fit, niter=200, seed=42)
    b = bootdist(weibull_fit, niter=200, seed=42)
    assert np.allclose(a.estimates.to_numpy(), b.estimates.to_numpy())


def test_bootstrap_does_not_disturb_the_global_rng(weibull_fit):
    np.random.seed(999)
    before = np.random.random()
    np.random.seed(999)
    bootdist(weibull_fit, niter=100, seed=7)
    assert np.random.random() == before


def test_bootstrap_validates_its_arguments(weibull_fit):
    with pytest.raises(ValueError, match="at least 10"):
        bootdist(weibull_fit, niter=5)
    with pytest.raises(ValueError, match="param"):
        bootdist(weibull_fit, bootmethod="jackknife")
    with pytest.raises(ValueError, match="between 0 and 1"):
        bootdist(weibull_fit, conf_level=1.5)


def test_estimates_frame_has_one_row_per_successful_refit(weibull_fit):
    boot = bootdist(weibull_fit, niter=150, seed=8)
    assert len(boot.estimates) == boot.n_converged
    assert boot.n_converged + boot.n_failed == boot.niter
    assert list(boot.estimates.columns) == list(weibull_fit.estimate)


def test_summary_puts_the_estimate_beside_the_interval(weibull_fit):
    summary = bootdist(weibull_fit, niter=150, seed=9).summary()
    assert "estimate" in summary.columns and "median" in summary.columns
    assert summary.loc["shape", "estimate"] == pytest.approx(2.186, abs=1e-3)


def test_quantile_ci_brackets_the_fitted_quantile(weibull_fit):
    boot = bootdist(weibull_fit, niter=300, seed=10)
    intervals = boot.quantile_ci([0.5, 0.95])
    for p in (0.5, 0.95):
        fitted = weibull_fit.dist.ppf(p, *weibull_fit.params)
        assert intervals.loc[p].iloc[1] < fitted < intervals.loc[p].iloc[2]
    # A higher quantile sits above a lower one.
    assert intervals.loc[0.95, "median"] > intervals.loc[0.5, "median"]


def test_quantile_ci_rejects_probabilities_outside_the_open_interval(weibull_fit):
    boot = bootdist(weibull_fit, niter=100, seed=11)
    for p in (0.0, 1.0, 1.2):
        with pytest.raises(ValueError, match="between 0 and 1"):
            boot.quantile_ci(p)


# ---------------------------------------------------------------------------
# Bootstrapping a mixture -- the case the Hessian cannot reach
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contaminated_mixture():
    rng = np.random.default_rng(0)
    data = np.concatenate([
        stats.gumbel_r.rvs(11.2, 8.3, size=475, random_state=rng),
        stats.gumbel_r.rvs(70, 25, size=25, random_state=rng),
    ])
    return fit_mixture(data, ("gumbel", "gumbel"))


def test_mixture_bootstrap_gives_an_interval_on_the_weight(contaminated_mixture):
    boot = bootdist(contaminated_mixture, niter=60, seed=1)
    limits = boot.ci.loc["weight2"]
    assert limits.iloc[1] < contaminated_mixture.weights[1] < limits.iloc[2]
    assert 0 <= limits.iloc[1] < limits.iloc[2] <= 1


def test_mixture_bootstrap_shows_the_tail_is_less_identified(contaminated_mixture):
    """The documented caveat, now quantified rather than asserted.

    The bulk component's location is pinned down far more tightly than the
    upper component's, relative to the size of each.
    """
    boot = bootdist(contaminated_mixture, niter=60, seed=2)
    relative_width = {}
    for name in ("loc1", "loc2"):
        limits = boot.ci.loc[name]
        relative_width[name] = (limits.iloc[2] - limits.iloc[1]) / abs(
            contaminated_mixture.estimate[name]
        )
    assert relative_width["loc2"] > relative_width["loc1"]


def test_mixture_bootstrap_keeps_component_order(contaminated_mixture):
    """Warm-starting stops components swapping between resamples.

    Without it the percentiles would mix the two components together.
    """
    boot = bootdist(contaminated_mixture, niter=60, seed=3)
    assert np.all(boot.estimates["loc1"] < boot.estimates["loc2"])


def test_mixture_quantile_ci_works(contaminated_mixture):
    boot = bootdist(contaminated_mixture, niter=40, seed=4)
    intervals = boot.quantile_ci([0.5, 0.99])
    assert intervals.loc[0.99, "median"] > intervals.loc[0.5, "median"]
    assert np.all(np.isfinite(intervals.to_numpy()))


def test_warm_started_mixture_reaches_the_same_optimum(contaminated_mixture):
    warm = fit_mixture(contaminated_mixture.data, ("gumbel", "gumbel"),
                       init=contaminated_mixture)
    assert warm.loglik == pytest.approx(contaminated_mixture.loglik, rel=1e-6)


def test_warm_start_rejects_a_component_count_mismatch(contaminated_mixture):
    with pytest.raises(ValueError, match="components"):
        fit_mixture(contaminated_mixture.data, ("gumbel", "gumbel", "gumbel"),
                    init=contaminated_mixture)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_bootdist_plot_pairs_every_parameter(weibull_fit):
    fig = bootdist_plot(bootdist(weibull_fit, niter=150, seed=1))
    # One pair for two parameters: a scatter plus the original estimate.
    assert len(fig.data) == 2


def test_bootdist_plot_uses_a_histogram_for_one_parameter():
    rng = np.random.default_rng(5)
    fit = fit_distributions(rng.exponential(2.0, 400), "exponential")[0]
    fig = bootdist_plot(bootdist(fit, niter=150, seed=1))
    assert fig.data[0].type == "histogram"


def test_bootdist_plot_can_select_parameters(contaminated_mixture):
    boot = bootdist(contaminated_mixture, niter=40, seed=1)
    fig = bootdist_plot(boot, parameters=["weight2", "loc2"])
    assert len(fig.data) == 2
    with pytest.raises(ValueError, match="Unknown parameter"):
        bootdist_plot(boot, parameters=["nonesuch"])


def test_confint_plot_draws_one_row_per_fit():
    rng = np.random.default_rng(6)
    fits = {
        label: fit_distributions(stats.gumbel_r.rvs(loc, 3, size=300,
                                                    random_state=rng), "gumbel")[0]
        for label, loc in [("E3", 8.0), ("E4", 6.5), ("E6", 5.0)]
    }
    fig = confint_plot(fits, parameter="loc")
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == ["E3", "E4", "E6"]
    assert len(fig.data[0].x) == 3


def test_confint_plot_accepts_bootstrap_results(weibull_fit):
    boot = bootdist(weibull_fit, niter=150, seed=1)
    fig = confint_plot({"groundbeef": boot}, parameter="shape")
    assert "percentile" in fig.layout.title.text
    assert fig.data[0].x[0] == pytest.approx(2.186, abs=1e-3)


def test_confint_plot_reports_an_unknown_parameter(weibull_fit):
    with pytest.raises(ValueError, match="not estimated"):
        confint_plot({"a": weibull_fit}, parameter="lambda")


def test_confint_plot_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="At least one fit"):
        confint_plot({})
