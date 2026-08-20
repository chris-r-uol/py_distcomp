"""
Checks for measurement-error models.

The motivating case: a remote sensing device measures a physically non-negative
concentration ratio but propagates enough error that individual readings come
back negative.  Discarding those biases the distribution high, because the error
works both ways -- so they are kept, and the model has to account for them.

The tests below check that the convolution is computed correctly (against a case
with an exact analytic answer), that it recovers a known truth, and that it does
what fitting a family directly to the observations cannot.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy import signal, stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.convolution import (  # noqa: E402
    ConvolvedDistribution,
    _convolve,
    _grid,
    fit_convolved,
)
from py_distcomp.convolution_plots import convolved_density_plot  # noqa: E402
from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.quantile_multi_comparison import quantile_comparison_plot  # noqa: E402

TRUE_SHAPE, TRUE_SCALE, ERROR_SD = 2.0, 6.0, 6.0


def observations(n=4000, seed=0):
    """Non-negative truth, symmetric error, so some readings come back negative."""
    rng = np.random.default_rng(seed)
    truth = stats.gamma.rvs(TRUE_SHAPE, scale=TRUE_SCALE, size=n, random_state=rng)
    return truth + rng.normal(0, ERROR_SD, n)


@pytest.fixture(scope="module")
def data():
    return observations()


@pytest.fixture(scope="module")
def fit(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_convolved(data, "gamma", error="normal")


# ---------------------------------------------------------------------------
# The convolution itself
# ---------------------------------------------------------------------------

def test_convolution_matches_an_exact_analytic_case():
    """Normal ⊛ normal is normal with the variances added."""
    grid = _grid(np.array([-40.0, 60.0]), 4.0, 8193)
    got = _convolve(stats.norm, (5.0, 3.0), stats.norm, 4.0, grid)
    exact = stats.norm.pdf(grid, 5.0, np.hypot(3.0, 4.0))
    assert np.max(np.abs(got - exact)) < 1e-12


def test_an_odd_grid_is_required_for_that_accuracy():
    """An even grid puts the kernel centre between cells and costs accuracy.

    This is why the grid length is forced odd rather than left to the caller.
    """
    assert len(_grid(np.array([0.0, 1.0]), 1.0, 4096)) % 2 == 1
    assert len(_grid(np.array([0.0, 1.0]), 1.0, 8193)) % 2 == 1


def test_the_convolved_density_integrates_to_one():
    dist = ConvolvedDistribution(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 6.0,
                                 _grid(np.array([-40.0, 90.0]), 6.0, 8193))
    assert np.trapezoid(dist.pdf(dist.grid), dist.grid) == pytest.approx(1.0, abs=1e-6)


def test_ppf_inverts_cdf():
    dist = ConvolvedDistribution(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 6.0,
                                 _grid(np.array([-40.0, 90.0]), 6.0, 8193))
    q = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(dist.cdf(dist.ppf(q)), q, atol=1e-6)


def test_cdf_is_monotone_and_bounded():
    dist = ConvolvedDistribution(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 6.0,
                                 _grid(np.array([-40.0, 90.0]), 6.0, 8193))
    values = dist.cdf(dist.grid)
    assert np.all(np.diff(values) >= -1e-12)
    assert values[0] >= 0 and values[-1] <= 1


def test_rvs_draws_a_truth_and_adds_an_error():
    dist = ConvolvedDistribution(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 6.0,
                                 _grid(np.array([-40.0, 90.0]), 6.0, 8193))
    sample = dist.rvs(30000, random_state=1)
    # The variances add, and some draws land below zero even though the truth
    # never does.
    assert sample.std() == pytest.approx(np.hypot(np.sqrt(2) * 6, 6), rel=0.05)
    assert np.mean(sample < 0) > 0.02


def test_the_convolution_widens_the_distribution():
    grid = _grid(np.array([-60.0, 120.0]), 6.0, 8193)
    narrow = _convolve(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 1e-6, grid)
    wide = _convolve(stats.gamma, (2.0, 0.0, 6.0), stats.norm, 6.0, grid)
    assert wide.max() < narrow.max()          # smeared flatter
    assert np.trapezoid(wide, grid) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_it_recovers_the_true_parameters(fit):
    assert fit.true_estimate["shape"] == pytest.approx(TRUE_SHAPE, rel=0.15)
    assert fit.true_estimate["rate"] == pytest.approx(1 / TRUE_SCALE, rel=0.15)


def test_it_recovers_the_error_scale(fit):
    """A symmetric error and a skewed signal are separable, so this is estimable."""
    assert fit.scale == pytest.approx(ERROR_SD, rel=0.15)
    assert fit.scale_estimated


def test_it_recovers_the_true_spread(fit):
    truth = np.sqrt(TRUE_SHAPE) * TRUE_SCALE
    assert fit.true_sd == pytest.approx(truth, rel=0.1)
    assert fit.observed_sd > fit.true_sd          # the error can only widen


def test_inflation_reports_the_smearing(fit):
    assert fit.inflation > 0
    assert fit.inflation == pytest.approx(fit.observed_sd / fit.true_sd - 1)


def test_a_supplied_scale_is_used_exactly(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_convolved(data, "gamma", error="normal", scale=ERROR_SD)
    assert fit.scale == ERROR_SD
    assert not fit.scale_estimated
    assert fit.true_estimate["shape"] == pytest.approx(TRUE_SHAPE, rel=0.15)


def test_free_parameters_are_counted_honestly(data, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = fit_convolved(data, "gamma", error="normal", scale=ERROR_SD)
    assert fit.n_free_params == 3        # shape, rate, error scale
    assert fixed.n_free_params == 2      # the scale is not estimated
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * 3)


# ---------------------------------------------------------------------------
# What it does that a direct fit cannot
# ---------------------------------------------------------------------------

def test_it_fits_a_family_the_observations_forbid():
    """A lognormal cannot be fitted to data containing negatives -- but it can
    perfectly well be the truth behind it."""
    rng = np.random.default_rng(0)
    truth = stats.lognorm.rvs(0.7, 0, 10.0, size=3000, random_state=rng)
    noisy = truth + rng.normal(0, 5.0, 3000)
    assert np.any(noisy < 0)

    with pytest.raises(Exception):
        fit_distributions(noisy, "lognormal")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_convolved(noisy, "lognormal", error="normal")
    assert fit.true_estimate["meanlog"] == pytest.approx(np.log(10.0), rel=0.1)
    assert fit.true_estimate["sdlog"] == pytest.approx(0.7, rel=0.2)


def test_discarding_the_negatives_biases_the_spread_high(data):
    """The reason the negative readings are kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        honest = fit_convolved(data, "gamma", error="normal")
        naive = fit_distributions(data[data > 0], "gamma")[0]
    truth = np.sqrt(TRUE_SHAPE) * TRUE_SCALE
    naive_sd = naive.dist.std(*naive.params)
    assert abs(honest.true_sd - truth) < abs(naive_sd - truth)


def test_it_beats_direct_fits_on_its_own_data(data, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = gofstat(fit_distributions(data, ["gumbel", "normal"]) + [fit])
    assert table["aic"].idxmin() == fit.name
    assert table.loc[fit.name, "ks"] < table.loc["normal", "ks"]


# ---------------------------------------------------------------------------
# Integration and guards
# ---------------------------------------------------------------------------

def test_it_plugs_into_the_comparison_plots(data, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figures = quantile_comparison_plot(data, ["gumbel", fit.dist])
    assert len(figures) == 4
    assert np.all(np.isfinite(np.asarray(figures[0].data[1].x, dtype=float)))


@pytest.mark.parametrize("error", ["normal", "laplace", "logistic"])
def test_every_error_family_fits(data, error):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_convolved(data, "gamma", error=error)
    assert np.isfinite(fit.loglik)
    assert fit.scale > 0


def test_an_unknown_error_family_is_refused(data):
    with pytest.raises(ValueError, match="error must be one of"):
        fit_convolved(data, "gamma", error="student")


def test_a_discrete_true_family_is_refused(data):
    with pytest.raises(ValueError, match="discrete"):
        fit_convolved(data, "poisson")


def test_a_negative_scale_is_refused(data):
    with pytest.raises(ValueError, match="positive"):
        fit_convolved(data, "gamma", scale=-1.0)


def test_too_little_data_is_refused():
    with pytest.raises(ValueError, match="At least 10"):
        fit_convolved(np.array([1.0, 2.0, 3.0]), "gamma")


def test_summary_reports_the_recovered_truth(fit):
    summary = fit.summary()
    for key in ("shape", "rate", "error_scale", "true_sd", "observed_sd",
                "inflation", "aic"):
        assert key in summary.index


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def test_the_plot_shows_both_densities(fit):
    figure = convolved_density_plot(fit)
    names = [t.name or "" for t in figure.data]
    assert any("with error" in n for n in names)
    assert any("error removed" in n for n in names)


def test_the_plot_marks_the_impossible_region(fit):
    """Negative readings of a non-negative quantity, kept deliberately."""
    figure = convolved_density_plot(fit)
    assert any(s.type == "rect" for s in figure.layout.shapes)


def test_the_true_density_can_be_hidden(fit):
    figure = convolved_density_plot(fit, show_true=False)
    assert not any("error removed" in (t.name or "") for t in figure.data)
