"""
Checks for the off-model fraction method of Rushton et al. (2021).

There is no published reference dataset for this method, so the tests build
samples with a known contamination fraction -- a Gumbel bulk plus a
high-valued Gumbel tail, the structure the paper describes for the petrol and
Euro 6 diesel fleets -- and check that the sweep recovers it.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from py_distcomp.distributions import ppoints  # noqa: E402
from py_distcomp.off_model import off_model_fraction, qq_r_squared  # noqa: E402
from py_distcomp.off_model_plots import (  # noqa: E402
    off_model_density_plot,
    percentile_cut_qq_plot,
    r_squared_sweep_plot,
)


def contaminated(n_bulk, n_tail, seed, loc=11.2, scale=8.3, tail_loc=70, tail_scale=25):
    """A Gumbel population with a known high-valued off-model fraction."""
    rng = np.random.default_rng(seed)
    bulk = stats.gumbel_r.rvs(loc, scale, size=n_bulk, random_state=rng)
    tail = stats.gumbel_r.rvs(tail_loc, tail_scale, size=n_tail, random_state=rng)
    return np.concatenate([bulk, tail])


@pytest.fixture(scope="module")
def clean_gumbel():
    """A pure Gumbel sample -- the paper's pre-Euro 6 diesel case."""
    return stats.gumbel_r.rvs(29.0, 22.8, size=1000, random_state=np.random.default_rng(3))


# ---------------------------------------------------------------------------
# qq_r_squared
# ---------------------------------------------------------------------------

def test_r_squared_is_high_for_a_correct_model(clean_gumbel):
    assert qq_r_squared(clean_gumbel, "gumbel") > 0.99


def test_r_squared_is_lower_for_a_wrong_model(clean_gumbel):
    """The Gumbel is skewed, so a normal fit tracks the 1:1 line less well."""
    assert qq_r_squared(clean_gumbel, "normal") < qq_r_squared(clean_gumbel, "gumbel")


def test_identity_r_squared_matches_its_definition(clean_gumbel):
    params = stats.gumbel_r.fit(clean_gumbel)
    sorted_data = np.sort(clean_gumbel)
    theoretical = stats.gumbel_r.ppf(ppoints(len(sorted_data), 0.5), *params)
    expected = 1 - (np.sum((sorted_data - theoretical) ** 2)
                    / np.sum((sorted_data - sorted_data.mean()) ** 2))
    assert qq_r_squared(clean_gumbel, "gumbel", dist_params=params) == pytest.approx(expected)


def test_pearson_r_squared_matches_squared_correlation(clean_gumbel):
    params = stats.gumbel_r.fit(clean_gumbel)
    sorted_data = np.sort(clean_gumbel)
    theoretical = stats.gumbel_r.ppf(ppoints(len(sorted_data), 0.5), *params)
    expected = np.corrcoef(theoretical, sorted_data)[0, 1] ** 2
    got = qq_r_squared(clean_gumbel, "gumbel", dist_params=params, method="pearson")
    assert got == pytest.approx(expected)


def test_identity_r_squared_penalises_a_scale_error(clean_gumbel):
    """Pearson only sees straightness; identity sees the 1:1 line."""
    wrong = (29.0, 22.8 * 3)
    assert qq_r_squared(clean_gumbel, "gumbel", dist_params=wrong) < 0.5
    assert qq_r_squared(clean_gumbel, "gumbel", dist_params=wrong, method="pearson") > 0.99


def test_r_squared_rejects_unknown_method(clean_gumbel):
    with pytest.raises(ValueError, match="identity"):
        qq_r_squared(clean_gumbel, "gumbel", method="spearman")


# ---------------------------------------------------------------------------
# off_model_fraction
# ---------------------------------------------------------------------------

def test_recovers_a_known_contamination_fraction():
    # 5% injected; the lowest of the injected tail overlaps the bulk and is
    # not separable, so the recovered fraction sits just under the truth.
    result = off_model_fraction(contaminated(950, 50, seed=0), "gumbel")
    assert result.fraction == pytest.approx(5, abs=2)
    assert result.r_squared > 0.99


def test_recovers_the_bulk_parameters_not_the_contaminated_ones():
    """The point of the cut: the reported parameters describe the bulk."""
    data = contaminated(950, 50, seed=0, loc=11.2, scale=8.3)
    result = off_model_fraction(data, "gumbel")
    assert result.fit.estimate["loc"] == pytest.approx(11.2, abs=1.5)
    assert result.fit.estimate["scale"] == pytest.approx(8.3, abs=1.5)

    # Fitting everything without a cut is visibly worse.
    assert qq_r_squared(data, "gumbel") < result.r_squared


def test_clean_sample_needs_no_cut(clean_gumbel):
    """R^2 peaks at 100 when the whole population follows the distribution."""
    result = off_model_fraction(clean_gumbel, "gumbel")
    assert result.percentile == 100
    assert result.fraction == 0
    assert result.n_off_model == 0
    assert result.tail_fit is None


def test_larger_contamination_gives_a_larger_fraction():
    small = off_model_fraction(contaminated(980, 20, seed=1), "gumbel")
    large = off_model_fraction(contaminated(880, 120, seed=1), "gumbel")
    assert large.fraction > small.fraction


def test_threshold_and_counts_are_consistent():
    data = contaminated(950, 50, seed=0)
    result = off_model_fraction(data, "gumbel")
    assert result.n_retained + result.n_off_model == result.n_total
    assert np.all(result.off_model_values > result.threshold)
    assert len(result.off_model_values) == result.n_off_model
    assert result.fraction == pytest.approx(100 - result.percentile)


def test_tail_fit_describes_the_off_model_population():
    data = contaminated(950, 50, seed=0, tail_loc=70, tail_scale=25)
    result = off_model_fraction(data, "gumbel")
    assert result.tail_fit is not None
    # The tail component sits well above the bulk, which is the whole point.
    assert result.tail_fit.estimate["loc"] > result.fit.estimate["loc"]


def test_curve_covers_every_candidate_percentile():
    data = contaminated(950, 50, seed=0)
    result = off_model_fraction(data, "gumbel")
    assert set(result.curve["percentile"]) == set(float(p) for p in range(1, 101))
    assert result.curve["n"].is_monotonic_increasing
    # The selected row is the maximum of the curve.
    assert result.r_squared == result.curve["r_squared"].max()


def test_curve_reports_parameters_in_r_parameterisation():
    result = off_model_fraction(contaminated(950, 50, seed=0), "gumbel")
    assert {"loc", "scale"} <= set(result.curve.columns)


def test_both_r_squared_methods_select_the_same_percentile():
    data = contaminated(950, 50, seed=0)
    a = off_model_fraction(data, "gumbel", method="identity")
    b = off_model_fraction(data, "gumbel", method="pearson")
    assert abs(a.percentile - b.percentile) <= 1


def test_works_with_other_distributions():
    rng = np.random.default_rng(4)
    data = np.concatenate([
        stats.weibull_min.rvs(2.0, scale=80, size=900, random_state=rng),
        stats.weibull_min.rvs(2.0, scale=400, size=100, random_state=rng),
    ])
    result = off_model_fraction(data, "weibull")
    assert result.percentile < 100
    assert result.fit.r_name == "weibull"


def test_custom_percentile_grid_is_honoured():
    data = contaminated(950, 50, seed=0)
    result = off_model_fraction(data, "gumbel", percentiles=[90, 95, 100])
    assert set(result.curve["percentile"]) == {90.0, 95.0, 100.0}
    assert result.percentile in {90.0, 95.0, 100.0}


def test_ties_resolve_to_the_highest_percentile():
    """R^2 is flat across these cuts, so the highest one wins."""
    data = np.arange(1.0, 1001.0)
    result = off_model_fraction(data, "uniform", percentiles=[100, 100])
    assert result.percentile == 100


def test_invalid_percentiles_are_rejected():
    data = contaminated(950, 50, seed=0)
    with pytest.raises(ValueError, match=r"\(0, 100\]"):
        off_model_fraction(data, "gumbel", percentiles=[0, 50])
    with pytest.raises(ValueError, match=r"\(0, 100\]"):
        off_model_fraction(data, "gumbel", percentiles=[50, 120])


def test_min_points_guards_tiny_subsets():
    data = contaminated(950, 50, seed=0)
    result = off_model_fraction(data, "gumbel", min_points=500)
    assert result.curve["n"].min() >= 500


def test_sample_too_small_for_min_points_raises():
    with pytest.raises(ValueError, match="too small"):
        off_model_fraction(np.arange(1.0, 11.0), "gumbel", min_points=50)


def test_fit_tail_can_be_disabled():
    data = contaminated(950, 50, seed=0)
    assert off_model_fraction(data, "gumbel", fit_tail=False).tail_fit is None


def test_summary_reports_the_paper_table_columns():
    result = off_model_fraction(contaminated(950, 50, seed=0), "gumbel")
    summary = result.summary()
    for key in ["n", "off_model_percentile", "off_model_fraction",
                "r_squared", "loc", "scale"]:
        assert key in summary.index
    assert summary["off_model_percentile"] == result.percentile


def test_nan_values_are_dropped():
    data = np.concatenate([contaminated(950, 50, seed=0), [np.nan, np.nan]])
    assert off_model_fraction(data, "gumbel").n_total == 1000


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_sweep_plot_draws_a_curve_per_result():
    results = {
        "E6 diesel": off_model_fraction(contaminated(950, 50, seed=0), "gumbel"),
        "E3 petrol": off_model_fraction(contaminated(880, 120, seed=1), "gumbel"),
    }
    fig = r_squared_sweep_plot(results)
    curves = [t for t in fig.data if t.mode == "lines+markers"]
    assert len(curves) == 2
    assert np.allclose(curves[0].x, results["E6 diesel"].curve["percentile"])
    # Each curve gets a marker at its selected percentile.
    assert len([t for t in fig.data if t.mode == "markers"]) == 2


def test_sweep_plot_accepts_a_single_result():
    fig = r_squared_sweep_plot(off_model_fraction(contaminated(950, 50, seed=0), "gumbel"))
    assert len([t for t in fig.data if t.mode == "lines+markers"]) == 1


def test_sweep_plot_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="At least one result"):
        r_squared_sweep_plot({})


def test_cut_qq_grid_has_one_panel_per_percentile():
    data = contaminated(950, 50, seed=0)
    percentiles = [99, 97, 95, 93]
    fig = percentile_cut_qq_plot(data, "gumbel", percentiles=percentiles, ncols=2)
    # Each panel contributes a scatter and its 1:1 line.
    assert len(fig.data) == 2 * len(percentiles)
    titles = [a.text for a in fig.layout.annotations]
    assert any("99th" in t for t in titles)
    assert any("93rd" in t for t in titles)


def test_cut_qq_grid_uses_ordinal_suffixes():
    fig = percentile_cut_qq_plot(contaminated(950, 50, seed=0), "gumbel",
                                 percentiles=[91, 92, 93, 100, 11], ncols=5)
    titles = " ".join(a.text for a in fig.layout.annotations)
    for expected in ["91st", "92nd", "93rd", "100th", "11th"]:
        assert expected in titles


def test_density_plot_shows_both_components_and_the_cut():
    result = off_model_fraction(contaminated(950, 50, seed=0), "gumbel")
    fig = off_model_density_plot(result, data_name="E6 diesel")
    names = [t.name for t in fig.data]
    assert any(n and n.startswith("On-model") for n in names)
    assert any(n and n.startswith("Off-model") for n in names)
    assert "Superposition" in names
    # The vertical line marking the cut.
    assert any(s.type == "line" and s.x0 == pytest.approx(result.threshold)
               for s in fig.layout.shapes)


def test_density_plot_components_sum_to_the_superposition():
    result = off_model_fraction(contaminated(950, 50, seed=0), "gumbel")
    fig = off_model_density_plot(result)
    traces = {t.name: np.asarray(t.y) for t in fig.data if t.name}
    on = traces[next(n for n in traces if n.startswith("On-model"))]
    off = traces[next(n for n in traces if n.startswith("Off-model"))]
    assert np.allclose(on + off, traces["Superposition"])


def test_density_plot_handles_a_clean_sample(clean_gumbel):
    """No tail fit, so no off-model curve to draw."""
    fig = off_model_density_plot(off_model_fraction(clean_gumbel, "gumbel"))
    names = [t.name for t in fig.data]
    assert not any(n and n.startswith("Off-model") for n in names)
