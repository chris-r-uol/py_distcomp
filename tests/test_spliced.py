"""
Checks for spliced (composite) distributions.

A splice is a different claim from a mixture.  A mixture says the tail is a
second population sitting on top of the first; a splice says there is one
population whose tail is heavier than its body implies.  The most important test
here is that the two are *distinguishable*: data generated each way is correctly
identified by AIC, so the question can be settled from the data rather than
assumed.

The density itself is checked the way any distribution should be -- that it
integrates to one, that its quantile function inverts its CDF, and that the two
pieces meet where they are supposed to.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.mixture import fit_mixture  # noqa: E402
from py_distcomp.quantile_multi_comparison import quantile_comparison_plot  # noqa: E402
from py_distcomp.spliced import (  # noqa: E402
    SplicedDistribution,
    _continuity_weight,
    fit_spliced,
)
from py_distcomp.spliced_plots import (  # noqa: E402
    spliced_density_plot,
    threshold_profile_plot,
)

LOWER = (stats.gumbel_r, (11.0, 8.0))
UPPER = (stats.pareto, (2.2, 0.0, 30.0))
THRESHOLD = 45.0


def continuous_splice():
    weight = _continuity_weight(LOWER, UPPER, THRESHOLD)
    return SplicedDistribution(LOWER, UPPER, THRESHOLD, weight)


@pytest.fixture(scope="module")
def spliced_sample():
    return continuous_splice().rvs(size=3000, random_state=0)


@pytest.fixture(scope="module")
def fit(spliced_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_spliced(spliced_sample, "gumbel", "pareto")


# ---------------------------------------------------------------------------
# The density is a real density
# ---------------------------------------------------------------------------

def test_it_integrates_to_one():
    total, _ = integrate.quad(continuous_splice().pdf, -200, 20000, limit=500)
    assert total == pytest.approx(1.0, abs=1e-4)


def test_ppf_inverts_cdf_exactly():
    """No numerical inversion is needed -- each piece is a rescaled parent."""
    dist = continuous_splice()
    q = np.array([1e-4, 0.05, 0.5, 0.9, 0.95, 0.99, 0.9999])
    assert np.allclose(dist.cdf(dist.ppf(q)), q, atol=1e-12)


def test_cdf_is_monotone_and_bounded():
    dist = continuous_splice()
    grid = np.linspace(-50, 5000, 4000)
    cdf = dist.cdf(grid)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] >= 0 and cdf[-1] <= 1


def test_the_weight_is_the_mass_below_the_threshold():
    dist = continuous_splice()
    assert dist.cdf(dist.threshold) == pytest.approx(dist.weight)


def test_continuity_makes_the_pieces_meet():
    """The whole point of the constraint: no step at the join."""
    assert continuous_splice().jump == pytest.approx(0.0, abs=1e-9)


def test_an_arbitrary_weight_leaves_a_jump():
    """Which is what continuity is there to remove."""
    assert SplicedDistribution(LOWER, UPPER, THRESHOLD, 0.80).jump > 0.1


def test_rvs_reproduces_the_distribution():
    dist = continuous_splice()
    sample = dist.rvs(size=40000, random_state=1)
    for q in (0.25, 0.5, 0.9, 0.99):
        assert np.quantile(sample, q) == pytest.approx(dist.ppf(q), rel=0.06), q


def test_ppf_handles_the_endpoints():
    dist = continuous_splice()
    assert dist.ppf(0.0) == -np.inf
    assert dist.ppf(1.0) == np.inf


def test_a_weight_outside_the_unit_interval_is_rejected():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            SplicedDistribution(LOWER, UPPER, THRESHOLD, bad)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_it_recovers_a_known_threshold(fit):
    assert fit.threshold == pytest.approx(THRESHOLD, rel=0.15)
    lower, upper = fit.threshold_ci()
    assert lower <= THRESHOLD <= upper


def test_it_recovers_the_known_weight(fit):
    truth = _continuity_weight(LOWER, UPPER, THRESHOLD)
    assert fit.weight == pytest.approx(truth, abs=0.03)


def test_it_recovers_the_body(fit):
    assert fit.lower_estimate["loc"] == pytest.approx(11.0, abs=1.0)
    assert fit.lower_estimate["scale"] == pytest.approx(8.0, abs=1.0)


def test_it_recovers_the_tail_index(fit):
    """Only the Pareto shape is identifiable once truncated above the join.

    A Pareto truncated above θ is a Pareto starting at θ with the same shape, so
    the scale carries no separate information -- but the shape, which is what
    governs how heavy the tail is, does.
    """
    assert fit.upper_estimate["shape"] == pytest.approx(2.2, rel=0.2)


def test_the_fitted_splice_is_continuous(fit):
    assert fit.dist.jump == pytest.approx(0.0, abs=1e-6)
    assert fit.continuous


def test_counts_either_side_add_up(fit):
    assert fit.n_below + fit.n_above == fit.n
    assert len(fit.tail_values) == fit.n_above
    assert np.all(fit.tail_values > fit.threshold)


def test_free_parameters_are_counted_honestly(fit):
    """Threshold plus two per side; the weight is not free under continuity."""
    assert fit.n_free_params == 5
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * 5)


def test_freeing_the_weight_costs_a_parameter(spliced_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = fit_spliced(spliced_sample, "gumbel", "pareto", continuous=False)
    assert loose.n_free_params == 6
    assert not loose.continuous
    # A free weight can only fit at least as well as a constrained one.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = fit_spliced(spliced_sample, "gumbel", "pareto")
    assert loose.loglik >= tight.loglik - 1e-6


# ---------------------------------------------------------------------------
# The question this model exists to answer
# ---------------------------------------------------------------------------

def test_splices_and_mixtures_are_distinguishable(spliced_sample):
    """Two populations, or one with a heavy tail?  AIC can tell them apart.

    This is the claim the whole construction rests on: fit both models to data
    generated each way, and the right one wins each time.
    """
    rng = np.random.default_rng(0)
    mixture_sample = np.concatenate([
        stats.gumbel_r.rvs(11.2, 8.3, size=950, random_state=rng),
        stats.gumbel_r.rvs(70.0, 25.0, size=50, random_state=rng),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sample, expected in [(mixture_sample, "gumbel + gumbel"),
                                 (spliced_sample, "gumbel|pareto")]:
            candidates = [
                fit_distributions(sample, "gumbel")[0],
                fit_mixture(sample, ("gumbel", "gumbel")),
                fit_spliced(sample, "gumbel", "pareto"),
            ]
            table = gofstat(candidates)
            assert table["aic"].idxmin() == expected


def test_a_splice_beats_a_single_distribution_on_its_own_data(fit, spliced_sample):
    single = fit_distributions(spliced_sample, "gumbel")[0]
    assert fit.loglik > single.loglik
    assert fit.aic < single.aic


# ---------------------------------------------------------------------------
# Profile and interval
# ---------------------------------------------------------------------------

def test_the_profile_covers_every_candidate(fit):
    assert set(fit.profile.columns) == {"threshold", "loglik", "weight", "n_above"}
    assert len(fit.profile) > 10
    assert fit.loglik == pytest.approx(fit.profile["loglik"].max())
    assert fit.threshold == pytest.approx(
        fit.profile.loc[fit.profile["loglik"].idxmax(), "threshold"]
    )


def test_the_interval_brackets_the_estimate(fit):
    lower, upper = fit.threshold_ci()
    assert lower <= fit.threshold <= upper


def test_the_interval_widens_with_the_level(fit):
    narrow = fit.threshold_ci(0.50)
    wide = fit.threshold_ci(0.99)
    assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])


def test_the_interval_rejects_an_impossible_level(fit):
    with pytest.raises(ValueError, match="between 0 and 1"):
        fit.threshold_ci(1.5)


def test_summary_reports_both_sides(fit):
    summary = fit.summary()
    for key in ("threshold", "weight", "loc_lower", "scale_lower",
                "shape_upper", "scale_upper", "aic", "continuous"):
        assert key in summary.index


# ---------------------------------------------------------------------------
# Integration and guards
# ---------------------------------------------------------------------------

def test_it_plugs_into_gofstat_and_the_plots(fit, spliced_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = gofstat([fit_distributions(spliced_sample, "gumbel")[0], fit])
        assert "gumbel|pareto" in table.index
        assert np.isfinite(table.loc["gumbel|pareto", "ks"])

        figures = quantile_comparison_plot(spliced_sample, ["gumbel", fit.dist])
    assert len(figures) == 4
    assert np.all(np.isfinite(np.asarray(figures[0].data[1].x, dtype=float)))


def test_custom_thresholds_are_honoured(spliced_sample):
    grid = [30.0, 40.0, 50.0, 60.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_spliced(spliced_sample, "gumbel", "pareto", thresholds=grid)
    assert set(result.profile["threshold"]) <= set(grid)
    assert result.threshold in grid


def test_a_discrete_family_is_refused():
    data = np.random.default_rng(0).gamma(3, 2, 500)
    with pytest.raises(ValueError, match="discrete"):
        fit_spliced(data, "poisson", "pareto")


def test_too_little_data_is_refused():
    with pytest.raises(ValueError, match="at least"):
        fit_spliced(np.random.default_rng(0).gamma(3, 2, 40), "gumbel", "pareto")


def test_impossible_thresholds_are_refused(spliced_sample):
    with pytest.raises(ValueError, match="No candidate threshold"):
        fit_spliced(spliced_sample, "gumbel", "pareto",
                    thresholds=[float(spliced_sample.max()) + 1])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_density_plot_draws_both_pieces(fit):
    figure = spliced_density_plot(fit)
    names = [t.name or "" for t in figure.data]
    assert any("below" in n for n in names)
    assert any("above" in n for n in names)
    assert any(s.type == "line" and s.x0 == pytest.approx(fit.threshold)
               for s in figure.layout.shapes)


def test_density_plot_can_use_a_log_axis(fit):
    assert spliced_density_plot(fit, log_y=True).layout.yaxis.type == "log"


def test_profile_plot_shows_the_interval(fit):
    figure = threshold_profile_plot(fit)
    assert len(figure.data) == 1
    assert np.allclose(figure.data[0].x, fit.profile.sort_values("threshold")["threshold"])
    kinds = [s.type for s in figure.layout.shapes]
    assert "rect" in kinds and "line" in kinds
