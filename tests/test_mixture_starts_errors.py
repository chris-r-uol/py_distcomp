"""
Checks for multi-start EM and for mixture standard errors.

EM finds a local optimum, so the useful question is not only which fit is best
but how many independent starts agreed on it.  Beyond two components the
deterministic strategy offers a single starting point, which is why restarts are
on by default.

The standard errors are validated the way the single-fit ones were: against the
bootstrap, which reaches the same quantity by an entirely different route, and
against a closed form where one exists.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.bootdist import bootdist  # noqa: E402
from py_distcomp.mixture import _split_starts, fit_mixture  # noqa: E402


def two_normals(seed=0, n1=700, n2=300):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, n1), rng.normal(6, 1.5, n2)])


@pytest.fixture(scope="module")
def fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_mixture(two_normals(), ("normal", "normal"))


# ---------------------------------------------------------------------------
# 1.2 -- multi-start
# ---------------------------------------------------------------------------

def test_restarts_add_starting_points(fit):
    assert fit.n_starts > fit.n_starts_converged - 1  # sanity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = fit_mixture(two_normals(), ("normal", "normal"), n_start=0)
    assert fit.n_starts > bare.n_starts


def test_beyond_two_components_restarts_are_the_only_diversity():
    """The reason restarts default to on rather than off."""
    data = np.random.default_rng(0).normal(0, 1, 600)
    assert len(_split_starts(data, ("normal",) * 2, "auto", 5, n_start=0, rng=None)) > 1
    assert len(_split_starts(data, ("normal",) * 3, "auto", 5, n_start=0, rng=None)) == 1
    rng = np.random.default_rng(1)
    assert len(_split_starts(data, ("normal",) * 3, "auto", 5, n_start=4, rng=rng)) == 5


def test_agreement_across_starts_is_reported(fit):
    assert 1 <= fit.n_starts_at_best <= fit.n_starts_converged
    assert fit.n_starts_converged <= fit.n_starts
    assert len(fit.start_logliks) == fit.n_starts_converged
    # start_logliks is sorted best first and the best one is the fit returned.
    assert fit.start_logliks == sorted(fit.start_logliks, reverse=True)
    assert fit.loglik == pytest.approx(max(fit.start_logliks))


def test_well_separated_data_gets_unanimous_agreement():
    rng = np.random.default_rng(3)
    data = np.concatenate([rng.normal(0, 1, 300), rng.normal(10, 1, 300)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_mixture(data, ("normal", "normal"))
    assert result.n_starts_at_best == result.n_starts_converged


def test_restarts_never_return_a_worse_optimum():
    """More starts can only improve the maximum, never degrade it."""
    for seed in range(4):
        data = two_normals(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bare = fit_mixture(data, ("normal", "normal"), n_start=0)
            more = fit_mixture(data, ("normal", "normal"), n_start=6)
        assert more.loglik >= bare.loglik - 1e-6, seed


def test_fits_are_reproducible_by_default():
    """The seed is fixed by default, so a fit does not move between runs."""
    a = fit_mixture(two_normals(), ("normal", "normal"))
    b = fit_mixture(two_normals(), ("normal", "normal"))
    assert a.loglik == b.loglik
    assert np.allclose(a.weights, b.weights)


def test_a_different_seed_gives_different_starting_points():
    data = two_normals()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fit_mixture(data, ("normal", "normal"), seed=1)
        b = fit_mixture(data, ("normal", "normal"), seed=2)
    # Different starts, but on separable data they should still agree.
    assert a.loglik == pytest.approx(b.loglik, rel=1e-6)


def test_restarts_do_not_disturb_the_global_rng():
    np.random.seed(4242)
    before = np.random.random()
    np.random.seed(4242)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_mixture(two_normals(), ("normal", "normal"))
    assert np.random.random() == before


def test_warm_start_skips_the_restarts(fit):
    """A warm start is one deliberate pass, not a search."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = fit_mixture(fit.data, ("normal", "normal"), init=fit)
    assert warm.n_starts == 1


def test_kmeanspp_partition_covers_the_data():
    from py_distcomp.mixture import _kmeanspp_partition

    data = np.sort(two_normals())
    parts = _kmeanspp_partition(data, 3, np.random.default_rng(0))
    assert len(parts) == 3
    assert sum(len(p) for p in parts) == len(data)
    # Centres are sorted, so the parts come out in ascending order.
    means = [p.mean() for p in parts if len(p)]
    assert means == sorted(means)


# ---------------------------------------------------------------------------
# 1.3 -- standard errors
# ---------------------------------------------------------------------------

def test_standard_errors_agree_with_the_bootstrap(fit):
    """Two routes to the same uncertainty, on a well-separated fit."""
    hessian = fit.std_error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = bootdist(fit, niter=250, seed=1).estimates.std(ddof=1)
    for name in fit.estimate:
        assert hessian[name] == pytest.approx(boot[name], rel=0.25), name


def test_weight_standard_error_is_near_the_binomial_value(fit):
    """With well-separated components, classification is nearly certain."""
    w = fit.estimate["weight1"]
    assert fit.std_error["weight1"] == pytest.approx(
        np.sqrt(w * (1 - w) / fit.n), rel=0.15
    )


def test_the_weights_are_perfectly_anticorrelated(fit):
    """Two weights summing to one leaves exactly one free quantity."""
    assert fit.correlation.loc["weight1", "weight2"] == pytest.approx(-1.0, abs=1e-6)


def test_vcov_is_symmetric_and_labelled(fit):
    vcov = fit.vcov
    assert list(vcov.index) == list(fit.estimate)
    assert list(vcov.columns) == list(fit.estimate)
    assert np.allclose(vcov.to_numpy(), vcov.to_numpy().T)
    assert np.allclose(
        np.sqrt(np.diag(vcov.to_numpy())),
        [fit.std_error[k] for k in fit.estimate],
    )


def test_confint_brackets_every_estimate(fit):
    interval = fit.confint()
    for name, value in fit.estimate.items():
        assert interval.loc[name, "lower"] < value < interval.loc[name, "upper"], name


def test_confint_widens_with_the_level(fit):
    narrow, wide = fit.confint(0.80), fit.confint(0.99)
    for name in fit.estimate:
        assert (wide.loc[name, "upper"] - wide.loc[name, "lower"]) > (
            narrow.loc[name, "upper"] - narrow.loc[name, "lower"]
        )


def test_confint_rejects_an_impossible_level(fit):
    for level in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            fit.confint(level)


def test_standard_errors_shrink_with_sample_size():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = fit_mixture(two_normals(0, 350, 150), ("normal", "normal"))
        large = fit_mixture(two_normals(0, 2800, 1200), ("normal", "normal"))
    ratio = small.std_error["mean1"] / large.std_error["mean1"]
    assert ratio == pytest.approx(np.sqrt(8), rel=0.5)


def test_a_degenerate_fit_reports_no_uncertainty():
    """A collapsed component has no meaningful uncertainty to quote."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0, 1, 300), np.full(50, 7.0)])
    result = fit_mixture(data, ("normal", "normal"), on_degenerate="ignore")
    assert result.degenerate
    assert result.vcov is None
    assert result.correlation is None
    assert all(np.isnan(v) for v in result.std_error.values())


def test_three_component_standard_errors_are_finite():
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(m, 1, 300) for m in (0, 7, 14)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_mixture(data, ("normal",) * 3)
    errors = result.std_error
    assert len(errors) == 9  # three weights plus two parameters each
    assert all(np.isfinite(v) and v > 0 for v in errors.values())


def test_standard_errors_work_for_a_discrete_mixture():
    rng = np.random.default_rng(6)
    data = np.concatenate([rng.poisson(2, 700), rng.poisson(12, 300)]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_mixture(data, ("poisson", "poisson"))
    errors = result.std_error
    assert all(np.isfinite(v) and v > 0 for v in errors.values())
    assert set(errors) == {"weight1", "lambda1", "weight2", "lambda2"}


def test_vcov_is_computed_once(fit):
    assert fit.vcov is fit.vcov
