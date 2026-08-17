"""
Checks for the discrete distributions.

scipy differs from R in three ways that all had to be handled, and each has a
test here: scipy's discrete distributions have no ``fit`` method, they expose
``logpmf`` rather than ``logpdf``, and scipy's geometric starts at 1 where R's
``dgeom`` starts at 0.

The estimators are checked against their closed forms, which the Poisson and
geometric have exactly, and against the parameters used to simulate the data.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.bootdist import bootdist  # noqa: E402
from py_distcomp.distributions import (  # noqa: E402
    DISCRETE_DISTRIBUTIONS,
    DISTRIBUTION_SPECS,
    density,
    fit_distribution,
    is_discrete,
    log_density,
    r_estimate,
    scipy_params,
)
from py_distcomp.empirical_plots import empirical_density_plot  # noqa: E402
from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.mixture import fit_mixture  # noqa: E402
from py_distcomp.quantile_multi_comparison import quantile_comparison_plot  # noqa: E402


@pytest.fixture(scope="module")
def poisson_counts():
    return np.random.default_rng(0).poisson(3.0, 800).astype(float)


@pytest.fixture(scope="module")
def nbinom_counts():
    # size 4, mu 6 in R's parameterisation
    rng = np.random.default_rng(1)
    return rng.negative_binomial(4.0, 4.0 / (4.0 + 6.0), 2000).astype(float)


@pytest.fixture(scope="module")
def geometric_counts():
    # numpy's geometric counts trials from 1; R's dgeom counts failures from 0
    rng = np.random.default_rng(2)
    return (rng.geometric(0.3, 2000) - 1).astype(float)


# ---------------------------------------------------------------------------
# Registry and support alignment
# ---------------------------------------------------------------------------

def test_the_three_discrete_families_are_registered():
    assert set(DISCRETE_DISTRIBUTIONS) == {
        "poisson", "negative_binomial", "geometric"
    }
    for name in DISCRETE_DISTRIBUTIONS:
        assert DISTRIBUTION_SPECS[name].discrete
        assert is_discrete(DISTRIBUTION_SPECS[name].dist)


def test_continuous_distributions_are_not_flagged_discrete():
    for name in ("normal", "weibull", "gamma"):
        assert not DISTRIBUTION_SPECS[name].discrete
        assert not is_discrete(DISTRIBUTION_SPECS[name].dist)


def test_geometric_is_shifted_to_match_r():
    """R's dgeom is supported on 0, 1, 2, ...; scipy's geom starts at 1."""
    _, params = fit_distribution("geometric", np.array([0.0, 1.0, 2.0, 3.0, 0.0]))
    assert params[1] == -1  # loc
    prob = params[0]
    # R: dgeom(0:2, prob) = prob * (1 - prob) ** (0:2)
    expected = prob * (1 - prob) ** np.arange(3)
    assert np.allclose(density(stats.geom, np.arange(3), params), expected)


def test_discrete_specs_have_one_fewer_scipy_parameter():
    """A discrete distribution carries a loc but no scale."""
    assert DISTRIBUTION_SPECS["poisson"].n_scipy_params == 2       # mu, loc
    assert DISTRIBUTION_SPECS["negative_binomial"].n_scipy_params == 3  # n, p, loc
    assert DISTRIBUTION_SPECS["normal"].n_scipy_params == 2        # loc, scale


def test_log_density_picks_the_right_function():
    """scipy names it logpmf on a discrete distribution, logpdf on continuous."""
    assert not hasattr(stats.poisson, "logpdf")
    assert np.allclose(
        log_density(stats.poisson, [0, 1, 2], (3.0, 0.0)),
        stats.poisson.logpmf([0, 1, 2], 3.0),
    )
    assert np.allclose(
        log_density(stats.norm, [0.0, 1.0], (0.0, 1.0)),
        stats.norm.logpdf([0.0, 1.0], 0.0, 1.0),
    )


def test_r_names_are_accepted_as_aliases(poisson_counts):
    for r_name, own in [("pois", "poisson"), ("nbinom", "negative_binomial"),
                        ("geom", "geometric")]:
        a = fit_distributions(poisson_counts, r_name)[0]
        b = fit_distributions(poisson_counts, own)[0]
        assert a.params == pytest.approx(b.params)


def test_parameter_round_trip(poisson_counts, nbinom_counts, geometric_counts):
    for name, data in [("poisson", poisson_counts),
                       ("negative_binomial", nbinom_counts),
                       ("geometric", geometric_counts)]:
        spec = DISTRIBUTION_SPECS[name]
        _, params = fit_distribution(name, data)
        values = list(r_estimate(spec.r_name, params, spec).values())
        assert np.allclose(scipy_params(spec.r_name, values, spec), params), name


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def test_poisson_lambda_is_the_sample_mean(poisson_counts):
    fit = fit_distributions(poisson_counts, "poisson")[0]
    assert fit.estimate["lambda"] == pytest.approx(np.mean(poisson_counts))


def test_geometric_prob_matches_its_closed_form(geometric_counts):
    """For R's parameterisation the mean is (1 - p)/p, so p = 1/(1 + mean)."""
    fit = fit_distributions(geometric_counts, "geometric")[0]
    expected = 1.0 / (1.0 + np.mean(geometric_counts))
    assert fit.estimate["prob"] == pytest.approx(expected)
    assert fit.estimate["prob"] == pytest.approx(0.3, abs=0.02)


def test_negative_binomial_recovers_its_parameters(nbinom_counts):
    fit = fit_distributions(nbinom_counts, "negative_binomial")[0]
    assert fit.estimate["mu"] == pytest.approx(np.mean(nbinom_counts))
    assert fit.estimate["mu"] == pytest.approx(6.0, rel=0.1)
    assert fit.estimate["size"] == pytest.approx(4.0, rel=0.25)


def test_negative_binomial_beats_poisson_on_overdispersed_data(nbinom_counts):
    fits = fit_distributions(nbinom_counts, ["poisson", "negative_binomial"])
    poisson, nbinom = fits
    assert nbinom.aic < poisson.aic
    assert nbinom.loglik > poisson.loglik


def test_poisson_wins_on_equidispersed_data(poisson_counts):
    """Its extra parameter has to pay for itself, and here it cannot."""
    fits = fit_distributions(poisson_counts, ["poisson", "negative_binomial"])
    assert fits[0].aic < fits[1].aic


def test_free_parameter_counts_match_r(poisson_counts):
    expected = {"poisson": 1, "negative_binomial": 2, "geometric": 1}
    for name, npar in expected.items():
        fit = fit_distributions(poisson_counts, name)[0]
        assert fit.n_free_params == npar, name
        assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * npar)


def test_non_integer_data_is_rejected():
    with pytest.raises(ValueError, match="non-negative integers"):
        fit_distributions(np.array([1.0, 2.5, 3.0]), "poisson")


def test_negative_data_is_rejected():
    with pytest.raises(ValueError, match="non-negative integers"):
        fit_distributions(np.array([-1.0, 2.0, 3.0]), "poisson")


def test_all_zero_data_is_rejected_for_the_negative_binomial():
    with pytest.raises(ValueError, match="all-zero"):
        fit_distributions(np.zeros(50), "negative_binomial")


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------

def test_poisson_standard_error_matches_its_closed_form(poisson_counts):
    # Var(lambda-hat) = lambda / n.
    fit = fit_distributions(poisson_counts, "poisson")[0]
    expected = np.sqrt(fit.estimate["lambda"] / fit.n)
    assert fit.std_error["lambda"] == pytest.approx(expected, rel=1e-4)


def test_geometric_standard_error_matches_its_closed_form(geometric_counts):
    # Var(p-hat) = p^2 (1 - p) / n for R's parameterisation.
    fit = fit_distributions(geometric_counts, "geometric")[0]
    p = fit.estimate["prob"]
    expected = np.sqrt(p ** 2 * (1 - p) / fit.n)
    assert fit.std_error["prob"] == pytest.approx(expected, rel=1e-3)


def test_confint_works_for_a_discrete_fit(poisson_counts):
    fit = fit_distributions(poisson_counts, "poisson")[0]
    interval = fit.confint()
    assert (interval.loc["lambda", "lower"]
            < fit.estimate["lambda"]
            < interval.loc["lambda", "upper"])


# ---------------------------------------------------------------------------
# gofstat
# ---------------------------------------------------------------------------

def test_gofstat_reports_only_chi_squared_for_discrete_fits(poisson_counts):
    """R computes KS, Cramer-von Mises and Anderson-Darling only for continuous fits."""
    table = gofstat(fit_distributions(poisson_counts, ["poisson", "geometric"]))
    for name in ("poisson", "geometric"):
        assert np.isnan(table.loc[name, "ks"])
        assert np.isnan(table.loc[name, "cvm"])
        assert np.isnan(table.loc[name, "ad"])
        assert table.loc[name, "ks_test"] == "not computed"
        assert np.isfinite(table.loc[name, "chisq"])


def test_gofstat_chi_squared_prefers_the_right_family(poisson_counts):
    table = gofstat(fit_distributions(poisson_counts, ["poisson", "geometric"]))
    assert table.loc["poisson", "chisq"] < table.loc["geometric", "chisq"]
    assert table.loc["poisson", "aic"] < table.loc["geometric", "aic"]


def test_gofstat_does_not_warn_about_ties_for_counts(poisson_counts):
    """Repeated values are the norm for counts, not a problem to warn about."""
    with warnings_as_errors():
        gofstat(fit_distributions(poisson_counts, "poisson"))


class warnings_as_errors:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_density_plot_puts_mass_on_the_integers(nbinom_counts):
    _, dens, _, _ = quantile_comparison_plot(
        nbinom_counts, ["poisson", "negative_binomial"]
    )
    for trace in dens.data[1:]:
        x = np.asarray(trace.x, dtype=float)
        assert np.allclose(x, np.round(x)), "fitted mass must sit on integers"
    # The empirical bars are relative frequencies summing to one.
    assert np.sum(dens.data[0].y) == pytest.approx(1.0)


def test_density_plot_masses_match_the_fitted_pmf(poisson_counts):
    _, dens, _, _ = quantile_comparison_plot(poisson_counts, "poisson")
    fit = fit_distributions(poisson_counts, "poisson")[0]
    x = np.asarray(dens.data[1].x)
    assert np.allclose(dens.data[1].y, stats.poisson.pmf(x, *fit.params))


def test_cdf_plot_uses_i_over_n_for_counts(poisson_counts):
    """R's cdfcomp drops ppoints when the fit is discrete."""
    _, _, _, cdf = quantile_comparison_plot(poisson_counts, "poisson")
    n = len(poisson_counts)
    assert np.allclose(cdf.data[0].y, np.arange(1, n + 1) / n)


def test_cdf_plot_draws_the_fitted_curve_as_a_step(poisson_counts):
    _, _, _, cdf = quantile_comparison_plot(poisson_counts, "poisson")
    assert cdf.data[1].line.shape == "hv"


def test_continuous_plots_are_unaffected():
    """The discrete branch must not change how continuous data is drawn."""
    rng = np.random.default_rng(3)
    x = rng.gamma(3.0, 2.0, 400)
    _, dens, _, cdf = quantile_comparison_plot(x, "gamma")
    from py_distcomp.distributions import ppoints
    assert np.allclose(cdf.data[0].y, ppoints(len(x), 0.5))
    assert cdf.data[1].line.shape in (None, "linear")
    assert len(dens.data[1].x) == 101  # the continuous fit grid


def test_empirical_density_plot_has_a_discrete_mode(poisson_counts):
    fig = empirical_density_plot(poisson_counts, discrete=True)
    # One bar trace, and no KDE curve, which assumes a continuous distribution.
    assert len(fig.data) == 1
    assert np.sum(fig.data[0].y) == pytest.approx(1.0)

    continuous = empirical_density_plot(poisson_counts)
    assert any("KDE" in (t.name or "") for t in continuous.data)


# ---------------------------------------------------------------------------
# Bootstrap and mixtures
# ---------------------------------------------------------------------------

def test_bootdist_works_on_a_discrete_fit(poisson_counts):
    fit = fit_distributions(poisson_counts, "poisson")[0]
    boot = bootdist(fit, niter=300, seed=1)
    limits = boot.ci.loc["lambda"]
    assert limits.iloc[1] < fit.estimate["lambda"] < limits.iloc[2]
    # The parametric bootstrap must draw whole numbers.
    sample = fit.dist.rvs(*fit.params, size=50, random_state=0)
    assert np.allclose(sample, np.round(sample))


def test_nonparametric_bootstrap_works_on_counts(poisson_counts):
    fit = fit_distributions(poisson_counts, "poisson")[0]
    boot = bootdist(fit, niter=200, bootmethod="nonparam", seed=2)
    assert boot.n_failed == 0


def test_poisson_mixture_recovers_two_regimes():
    """Over-dispersed counts as a superposition of two Poisson processes."""
    rng = np.random.default_rng(4)
    data = np.concatenate([rng.poisson(2.0, 700), rng.poisson(12.0, 300)]).astype(float)
    mix = fit_mixture(data, ("poisson", "poisson"))

    assert mix.converged
    assert np.all(np.diff(np.array(mix.history)) > -1e-6)
    lambdas = sorted([mix.estimate["lambda1"], mix.estimate["lambda2"]])
    assert lambdas[0] == pytest.approx(2.0, rel=0.2)
    assert lambdas[1] == pytest.approx(12.0, rel=0.2)
    assert sorted(mix.weights)[1] == pytest.approx(0.7, abs=0.06)


def test_poisson_mixture_beats_a_single_poisson():
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.poisson(2.0, 700), rng.poisson(12.0, 300)]).astype(float)
    single = fit_distributions(data, "poisson")[0]
    mix = fit_mixture(data, ("poisson", "poisson"))
    assert mix.aic < single.aic


def test_discrete_mixture_free_parameter_count():
    """Two Poissons: one parameter each, plus one free weight."""
    rng = np.random.default_rng(6)
    data = np.concatenate([rng.poisson(2.0, 400), rng.poisson(9.0, 400)]).astype(float)
    assert fit_mixture(data, ("poisson", "poisson")).n_free_params == 3
