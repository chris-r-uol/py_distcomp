"""
Checks for mixture fitting.

Expectation-maximisation is verified three ways: against the closed-form
behaviour it must obey (the log-likelihood increases every iteration), against
scikit-learn's ``GaussianMixture`` where the model coincides, and against
samples drawn from a known mixture.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.gofstat import fit_distributions, gofstat  # noqa: E402
from py_distcomp.mixture import MixtureDistribution, fit_mixture  # noqa: E402
from py_distcomp.mixture_plots import component_probability_plot, mixture_density_plot  # noqa: E402
from py_distcomp.off_model import off_model_fraction  # noqa: E402
from py_distcomp.quantile_multi_comparison import quantile_comparison_plot  # noqa: E402

# numpy renamed trapz to trapezoid in 2.0; the suite supports both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def two_gumbels(n1=700, n2=300, seed=1, loc1=10, scale1=3, loc2=80, scale2=8):
    """A well-separated two-component Gumbel mixture with known parameters."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        stats.gumbel_r.rvs(loc1, scale1, size=n1, random_state=rng),
        stats.gumbel_r.rvs(loc2, scale2, size=n2, random_state=rng),
    ])


def contaminated(seed=0):
    """The paper's structure: a large bulk with a small high-valued tail."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        stats.gumbel_r.rvs(11.2, 8.3, size=950, random_state=rng),
        stats.gumbel_r.rvs(70, 25, size=50, random_state=rng),
    ])


@pytest.fixture(scope="module")
def normal_mixture():
    rng = np.random.default_rng(2)
    return np.concatenate([rng.normal(0, 1, 600), rng.normal(6, 2, 400)])


# ---------------------------------------------------------------------------
# The EM algorithm itself
# ---------------------------------------------------------------------------

def test_loglikelihood_increases_every_iteration(normal_mixture):
    """The defining property of EM; a decrease means the M step is wrong."""
    history = np.array(fit_mixture(normal_mixture, ("normal", "normal")).history)
    assert len(history) > 1
    assert np.all(np.diff(history) > -1e-6)


def test_matches_sklearn_gaussian_mixture(normal_mixture):
    sklearn_mixture = pytest.importorskip("sklearn.mixture")
    reference = sklearn_mixture.GaussianMixture(
        2, covariance_type="full", tol=1e-10, max_iter=1000, random_state=0
    ).fit(normal_mixture.reshape(-1, 1))

    got = fit_mixture(normal_mixture, ("normal", "normal"))

    assert got.loglik == pytest.approx(
        reference.score(normal_mixture.reshape(-1, 1)) * len(normal_mixture), abs=1e-3
    )
    assert np.allclose(np.sort(got.weights), np.sort(reference.weights_), atol=1e-3)
    assert np.allclose(
        sorted([got.estimate["mean1"], got.estimate["mean2"]]),
        np.sort(reference.means_.ravel()), atol=1e-3,
    )
    assert np.allclose(
        sorted([got.estimate["sd1"], got.estimate["sd2"]]),
        np.sort(np.sqrt(reference.covariances_.ravel())), atol=1e-3,
    )


def test_recovers_known_mixture_parameters():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    assert result.converged
    assert result.weights[0] == pytest.approx(0.7, abs=0.05)
    assert result.estimate["loc1"] == pytest.approx(10, abs=1.0)
    assert result.estimate["scale1"] == pytest.approx(3, abs=1.0)
    assert result.estimate["loc2"] == pytest.approx(80, abs=2.0)
    assert result.estimate["scale2"] == pytest.approx(8, abs=2.0)


def test_recovers_weights_of_an_asymmetric_mixture():
    result = fit_mixture(two_gumbels(n1=900, n2=100), ("gumbel", "gumbel"))
    assert result.weights[1] == pytest.approx(0.1, abs=0.03)


def test_components_may_come_from_different_families():
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(0, 1, 700),
                           stats.gumbel_r.rvs(20, 4, size=300, random_state=rng)])
    result = fit_mixture(data, ("normal", "gumbel"))
    assert result.model_names == ["normal", "gumbel"]
    assert result.estimate["mean1"] == pytest.approx(0, abs=0.5)
    assert result.estimate["loc2"] == pytest.approx(20, abs=1.5)


def test_pinned_parameters_stay_pinned():
    """R's dlnorm has no location, and the M step must not invent one."""
    rng = np.random.default_rng(6)
    data = np.concatenate([rng.lognormal(0, 0.4, 700), rng.lognormal(2.5, 0.3, 300)])
    result = fit_mixture(data, ("lognormal", "lognormal"))
    for _, params in result.components:
        assert params[1] == 0


def test_more_components_are_supported():
    rng = np.random.default_rng(7)
    data = np.concatenate([rng.normal(0, 1, 400), rng.normal(10, 1, 400),
                           rng.normal(20, 1, 400)])
    result = fit_mixture(data, ("normal", "normal", "normal"))
    assert result.n_components == 3
    means = sorted(result.estimate[f"mean{k}"] for k in (1, 2, 3))
    assert np.allclose(means, [0, 10, 20], atol=1.0)


def test_single_component_is_rejected():
    with pytest.raises(ValueError, match="at least two components"):
        fit_mixture(two_gumbels(), ("gumbel",))


def test_invalid_init_is_rejected():
    with pytest.raises(ValueError, match="init must be"):
        fit_mixture(two_gumbels(), ("gumbel", "gumbel"), init="kmeans")


def test_both_initialisations_reach_the_same_optimum():
    data = contaminated()
    a = fit_mixture(data, ("gumbel", "gumbel"), init="off_model")
    b = fit_mixture(data, ("gumbel", "gumbel"), init="quantile")
    assert a.loglik == pytest.approx(b.loglik, rel=1e-3)


def test_free_parameter_count_drives_aic():
    """Two Gumbels: 2 parameters each, plus one free weight."""
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    assert result.n_free_params == 5
    assert result.aic == pytest.approx(-2 * result.loglik + 2 * 5)
    assert result.bic == pytest.approx(-2 * result.loglik + np.log(result.n) * 5)


def test_exponential_component_counts_one_parameter():
    """R's dexp has only a rate, so the mixture has 1 + 2 + 1 = 4."""
    rng = np.random.default_rng(8)
    data = np.concatenate([rng.exponential(2, 700), rng.normal(30, 2, 300)])
    assert fit_mixture(data, ("exponential", "normal")).n_free_params == 4


# ---------------------------------------------------------------------------
# The mixture distribution object
# ---------------------------------------------------------------------------

def test_mixture_pdf_integrates_to_one():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    grid = np.linspace(-50, 200, 20001)
    assert _trapezoid(result.dist.pdf(grid), grid) == pytest.approx(1.0, abs=1e-3)


def test_mixture_pdf_is_the_weighted_sum_of_components():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    grid = np.linspace(-10, 120, 500)
    expected = sum(
        w * dist.pdf(grid, *params)
        for w, (dist, params) in zip(result.weights, result.components)
    )
    assert np.allclose(result.dist.pdf(grid), expected)


def test_mixture_ppf_inverts_its_cdf():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    q = np.array([0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999])
    assert np.allclose(result.dist.cdf(result.dist.ppf(q)), q, atol=1e-6)


def test_mixture_cdf_is_monotone_and_bounded():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    grid = np.linspace(-50, 200, 1000)
    cdf = result.dist.cdf(grid)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] >= 0 and cdf[-1] <= 1


def test_mixture_ppf_handles_the_endpoints():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    assert result.dist.ppf(0.0) == -np.inf
    assert result.dist.ppf(1.0) == np.inf


def test_mixture_rvs_reproduces_its_own_parameters():
    """A large draw should have the mixture's mean, w1·m1 + w2·m2."""
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    sample = result.dist.rvs(size=20000, random_state=0)
    expected = sum(
        w * dist.mean(*params)
        for w, (dist, params) in zip(result.weights, result.components)
    )
    assert np.mean(sample) == pytest.approx(expected, rel=0.05)


def test_mixture_distribution_validates_its_weights():
    dist = stats.norm
    with pytest.raises(ValueError, match="one entry per component"):
        MixtureDistribution([(dist, (0, 1)), (dist, (5, 1))], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        MixtureDistribution([(dist, (0, 1)), (dist, (5, 1))], [-0.5, 1.5])
    with pytest.raises(ValueError, match="positive"):
        MixtureDistribution([(dist, (0, 1)), (dist, (5, 1))], [0.0, 0.0])


def test_mixture_weights_are_normalised():
    dist = stats.norm
    mixture = MixtureDistribution([(dist, (0, 1)), (dist, (5, 1))], [2.0, 2.0])
    assert np.allclose(mixture.weights, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Responsibilities -- the per-observation probabilities
# ---------------------------------------------------------------------------

def test_responsibilities_sum_to_one():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    assert np.allclose(result.responsibilities().sum(axis=1), 1.0)


def test_component_probability_rises_with_value():
    """Higher observations are more likely to belong to the upper component."""
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    grid = np.linspace(0.0, 100.0, 50)
    prob = result.dist.responsibilities(grid)[:, -1]
    # Non-decreasing: far below the upper component the probability underflows
    # to exactly zero, so consecutive points can tie.
    assert np.all(np.diff(prob) >= 0)
    assert prob[0] < 0.01 and prob[-1] > 0.99
    assert np.any((prob > 0.05) & (prob < 0.95))  # and it does actually vary


def test_expected_counts_match_the_sample_size():
    result = fit_mixture(two_gumbels(n1=700, n2=300), ("gumbel", "gumbel"))
    counts = result.expected_counts()
    assert counts.sum() == pytest.approx(result.n)
    assert counts[1] == pytest.approx(300, abs=40)


def test_classify_agrees_with_the_probability_threshold():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    flags = result.classify(threshold=0.5)
    assert np.array_equal(flags, result.component_probability() >= 0.5)


def test_summary_reports_weights_and_parameters():
    summary = fit_mixture(two_gumbels(), ("gumbel", "gumbel")).summary()
    for key in ["weight1", "loc1", "scale1", "weight2", "loc2", "scale2",
                "loglik", "aic", "bic", "converged"]:
        assert key in summary.index


# ---------------------------------------------------------------------------
# Fitting the paper's structure, and comparison against the hard cut
# ---------------------------------------------------------------------------

def test_mixture_beats_a_single_distribution_on_contaminated_data():
    data = contaminated()
    single = fit_distributions(data, "gumbel")[0]
    mixture = fit_mixture(data, ("gumbel", "gumbel"))
    assert mixture.loglik > single.loglik
    assert mixture.aic < single.aic
    assert mixture.bic < single.bic


def test_mixture_agrees_with_the_hard_cut_on_the_bulk():
    """Both methods should describe the same bulk population."""
    data = contaminated()
    hard = off_model_fraction(data, "gumbel")
    mixture = fit_mixture(data, ("gumbel", "gumbel"))
    assert mixture.estimate["loc1"] == pytest.approx(hard.fit.estimate["loc"], rel=0.2)
    assert mixture.estimate["scale1"] == pytest.approx(hard.fit.estimate["scale"], rel=0.3)


def test_mixture_weight_is_close_to_the_off_model_fraction():
    data = contaminated()
    hard = off_model_fraction(data, "gumbel")
    mixture = fit_mixture(data, ("gumbel", "gumbel"))
    assert mixture.weights[1] * 100 == pytest.approx(hard.fraction, abs=3)


def test_mixture_can_be_compared_in_gofstat():
    data = contaminated()
    fits = fit_distributions(data, ["gumbel", "normal"])
    mixture = fit_mixture(data, ("gumbel", "gumbel"))
    table = gofstat(fits + [mixture])

    assert "gumbel + gumbel" in table.index
    assert table.loc["gumbel + gumbel", "ks"] < table.loc["gumbel", "ks"]
    assert table.loc["gumbel + gumbel", "ad"] < table.loc["gumbel", "ad"]
    # No tabulated critical values exist for a mixture.
    assert table.loc["gumbel + gumbel", "ad_test"] == "not computed"


def test_mixture_plugs_into_the_comparison_plots():
    data = contaminated()
    mixture = fit_mixture(data, ("gumbel", "gumbel"))
    figs = quantile_comparison_plot(data, ["gumbel", mixture.dist])
    assert len(figs) == 4
    qq = figs[0]
    # One series per model plus the identity line.
    assert len(qq.data) == 3
    assert np.all(np.isfinite(np.asarray(qq.data[1].x, dtype=float)))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_density_plot_shows_components_and_their_sum():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    fig = mixture_density_plot(result)
    names = [t.name for t in fig.data]
    assert sum(n.startswith("Component") for n in names if n) == 2
    assert "Mixture" in names

    traces = {t.name: np.asarray(t.y) for t in fig.data if t.name}
    components = [v for k, v in traces.items() if k.startswith("Component")]
    assert np.allclose(sum(components), traces["Mixture"])


def test_density_plot_can_hide_the_components():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    fig = mixture_density_plot(result, show_components=False)
    assert not any((t.name or "").startswith("Component") for t in fig.data)


def test_probability_plot_is_monotone_for_a_two_component_fit():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    fig = component_probability_plot(result)
    curve = np.asarray(fig.data[0].y)
    assert np.all(np.diff(curve) >= -1e-9)
    assert curve[0] < 0.05 and curve[-1] > 0.95


def test_probability_plot_marks_the_threshold():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    fig = component_probability_plot(result, threshold=0.8)
    assert any(s.type == "line" and s.y0 == pytest.approx(0.8)
               for s in fig.layout.shapes)


def test_probability_plot_threshold_can_be_omitted():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    assert not component_probability_plot(result, threshold=None).layout.shapes


def test_probability_plot_can_target_the_lower_component():
    result = fit_mixture(two_gumbels(), ("gumbel", "gumbel"))
    fig = component_probability_plot(result, component=0)
    curve = np.asarray(fig.data[0].y)
    assert curve[0] > 0.95 and curve[-1] < 0.05
