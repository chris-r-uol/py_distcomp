"""
Checks for degenerate mixture components.

A mixture likelihood is frequently unbounded: a component that shrinks onto a
few points drives it to infinity, so the highest-likelihood fit can be one that
describes nothing at all.  Before this guard existed, four near-identical values
among 300 standard normal draws produced a component with ``sd = 0.002``,
reported as converged, unwarned, and beating the honest one-component fit on AIC
by 145.

There are two independent routes to it and a test for each: too few observations
to estimate a component, and a collapse onto coincident observations.  The
false-positive tests matter as much -- a guard that fires on legitimate fits
would be worse than no guard.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.gofstat import fit_distributions  # noqa: E402
from py_distcomp.mixture import (  # noqa: E402
    MIN_EFFECTIVE_PER_PARAM,
    MIN_EFFECTIVE_TOTAL,
    WIDTH_COLLAPSE_RATIO,
    fit_mixture,
)


def starved():
    """Too few observations to estimate a second component."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0, 1, 300),
                           np.full(4, 7.0) + rng.normal(0, 1e-3, 4)])


def collapsed():
    """Enough observations, but all identical, so the component has no width."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0, 1, 300), np.full(50, 7.0)])


def healthy():
    rng = np.random.default_rng(1)
    return np.concatenate([rng.normal(0, 1, 400), rng.normal(6, 1.5, 400)])


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_too_few_observations_is_caught():
    with pytest.warns(RuntimeWarning, match="degenerate"):
        fit = fit_mixture(starved(), ("normal", "normal"))
    assert fit.degenerate
    failing = fit.diagnostics[fit.diagnostics["degenerate"]]
    assert len(failing) == 1
    assert "effective observations" in failing.iloc[0]["reason"]


def test_collapse_onto_tied_values_is_caught():
    """The support check cannot see this one: 50 observations, but zero width."""
    with pytest.warns(RuntimeWarning, match="degenerate"):
        fit = fit_mixture(collapsed(), ("normal", "normal"))
    assert fit.degenerate
    failing = fit.diagnostics[fit.diagnostics["degenerate"]]
    assert failing.iloc[0]["n_effective"] > MIN_EFFECTIVE_TOTAL  # well supported
    assert failing.iloc[0]["width"] < 1e-6                        # and yet a spike
    assert "collapsed" in failing.iloc[0]["reason"]


def test_the_two_criteria_are_independent():
    """Each case is caught by its own rule, not by both."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fit_mixture(starved(), ("normal", "normal")).diagnostics
        b = fit_mixture(collapsed(), ("normal", "normal")).diagnostics
    assert "effective observations" in a[a["degenerate"]].iloc[0]["reason"]
    assert "collapsed" not in a[a["degenerate"]].iloc[0]["reason"]
    assert "collapsed" in b[b["degenerate"]].iloc[0]["reason"]
    assert "effective observations" not in b[b["degenerate"]].iloc[0]["reason"]


def test_the_degenerate_fit_is_no_longer_silent():
    """The regression this guard exists for: converged, unwarned, winning on AIC."""
    data = starved()
    with pytest.warns(RuntimeWarning):
        fit = fit_mixture(data, ("normal", "normal"))
    single = fit_distributions(data, "normal")[0]

    # It still wins on AIC -- the point is that the user is now told not to
    # believe that, rather than the number being quietly altered.
    assert fit.aic < single.aic
    assert fit.degenerate
    assert "DEGENERATE" in repr(fit)
    assert fit.summary()["degenerate"]


# ---------------------------------------------------------------------------
# No false positives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,build,families", [
    ("balanced", lambda r: np.concatenate([r.normal(0, 1, 400), r.normal(6, 1.5, 400)]),
     ("normal", "normal")),
    ("5% component", lambda r: np.concatenate([r.normal(0, 1, 950), r.normal(9, 1, 50)]),
     ("normal", "normal")),
    ("2% component", lambda r: np.concatenate([r.normal(0, 1, 980), r.normal(9, 1, 20)]),
     ("normal", "normal")),
    ("component 100x narrower than the data",
     lambda r: np.concatenate([r.normal(0, 5, 500), r.normal(40, 0.05, 200)]),
     ("normal", "normal")),
    ("three components",
     lambda r: np.concatenate([r.normal(m, 1, 300) for m in (0, 6, 12)]),
     ("normal", "normal", "normal")),
    ("counts", lambda r: np.concatenate([r.poisson(2, 700), r.poisson(12, 300)]).astype(float),
     ("poisson", "poisson")),
])
def test_legitimate_mixtures_are_not_flagged(label, build, families):
    for seed in range(4):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit = fit_mixture(build(np.random.default_rng(seed)), families)
        assert not fit.degenerate, f"{label} wrongly flagged at seed {seed}"


def test_the_papers_own_structure_is_not_flagged():
    """A Gumbel bulk with a small high-valued tail is exactly the target case."""
    rng = np.random.default_rng(0)
    data = np.concatenate([
        stats.gumbel_r.rvs(11.2, 8.3, size=950, random_state=rng),
        stats.gumbel_r.rvs(70, 25, size=50, random_state=rng),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = fit_mixture(data, ("gumbel", "gumbel"))
    assert not fit.degenerate


# ---------------------------------------------------------------------------
# Selection: a supported fit beats a higher-likelihood collapsed one
# ---------------------------------------------------------------------------

def test_a_healthy_start_is_preferred_over_a_collapsed_one():
    """Three stray points that could attract a collapse, and do not."""
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(0, 1, 400), rng.normal(5, 1, 100), np.full(3, 20.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = fit_mixture(data, ("normal", "normal"))
    assert not fit.degenerate


# ---------------------------------------------------------------------------
# Diagnostics table and the on_degenerate switch
# ---------------------------------------------------------------------------

def test_diagnostics_has_a_row_per_component():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_mixture(healthy(), ("normal", "normal"))
    d = fit.diagnostics
    assert list(d.index) == [1, 2]
    for column in ("distribution", "weight", "n_effective", "n_free_params",
                   "n_required", "width", "degenerate", "reason"):
        assert column in d.columns
    # Effective sizes are a partition of the sample.
    assert d["n_effective"].sum() == pytest.approx(fit.n)
    assert d["weight"].sum() == pytest.approx(1.0)


def test_required_support_scales_with_the_parameter_count():
    """A two-parameter normal needs more support than a one-parameter Poisson."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normal = fit_mixture(healthy(), ("normal", "normal")).diagnostics
        rng = np.random.default_rng(2)
        counts = np.concatenate([rng.poisson(2, 700), rng.poisson(12, 300)]).astype(float)
        poisson = fit_mixture(counts, ("poisson", "poisson")).diagnostics
    assert normal.iloc[0]["n_required"] == max(
        MIN_EFFECTIVE_TOTAL, 2 * MIN_EFFECTIVE_PER_PARAM)
    assert poisson.iloc[0]["n_required"] == MIN_EFFECTIVE_TOTAL  # the floor binds
    assert normal.iloc[0]["n_required"] > poisson.iloc[0]["n_required"]


def test_on_degenerate_raise():
    with pytest.raises(ValueError, match="degenerate"):
        fit_mixture(collapsed(), ("normal", "normal"), on_degenerate="raise")


def test_on_degenerate_ignore_is_silent_but_still_flags():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = fit_mixture(collapsed(), ("normal", "normal"), on_degenerate="ignore")
    assert fit.degenerate, "the flag must survive even when the warning is suppressed"


def test_on_degenerate_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="on_degenerate must be"):
        fit_mixture(healthy(), ("normal", "normal"), on_degenerate="shrug")


def test_the_warning_names_the_offending_component():
    with pytest.warns(RuntimeWarning) as record:
        fit_mixture(starved(), ("normal", "normal"))
    message = str(record[0].message)
    assert "component 2" in message
    assert "fit fewer components" in message


def test_diagnostics_are_computed_once():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_mixture(healthy(), ("normal", "normal"))
    assert fit.diagnostics is fit.diagnostics


def test_width_collapse_ratio_is_far_below_any_real_component():
    """The threshold must not be reachable by a genuinely tight component."""
    assert WIDTH_COLLAPSE_RATIO <= 1e-5
