#!/usr/bin/env python3
"""
Regenerate every figure in the readme.

Run from the repository root::

    python docs/make_figures.py

Each figure is built from the same code the readme shows, on the same data, so
what is documented and what is pictured cannot drift apart.  Two datasets are
used throughout:

``groundbeef``
    The 254 serving sizes shipped with fitdistrplus, so the numbers in the
    readme are the ones the R vignette prints.
``fleet``
    A simulated emission-ratio sample with a Gumbel bulk and a small
    high-valued tail -- the structure Rushton et al. (2021) describe.
"""

import pathlib
import sys
import warnings

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))       # run from a checkout without installing

import py_distcomp as pdc  # noqa: E402

IMAGES = HERE / "images"
DATA = HERE.parent / "tests" / "groundbeef.npy"

WIDTH, HEIGHT, SCALE = 760, 460, 2


def save(fig, name, width=WIDTH, height=HEIGHT):
    fig.update_layout(width=width, height=height)
    path = IMAGES / f"{name}.png"
    fig.write_image(path, scale=SCALE)
    print(f"  {path.relative_to(HERE.parent)}")
    return path


def groundbeef():
    return np.load(DATA)


def fleet(n_bulk=950, n_tail=50, seed=0):
    """A Gumbel bulk with a small high-emitting tail."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        stats.gumbel_r.rvs(11.2, 8.3, size=n_bulk, random_state=rng),
        stats.gumbel_r.rvs(70.0, 25.0, size=n_tail, random_state=rng),
    ])


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    serving, emissions = groundbeef(), fleet()
    print("writing figures:")

    # --- the core workflow, on fitdistrplus's own data ----------------------
    save(pdc.cullen_and_frey_plot(serving, data_name="Serving size", seed=42),
         "cullen-frey", width=800, height=520)

    qq, dens, _, cdf = pdc.quantile_comparison_plot(
        serving, ["weibull", "gamma", "lognormal"], data_name="Serving size",
    )
    save(qq, "qq-comparison")
    save(dens, "density-comparison")
    save(cdf, "cdf-comparison")

    # --- uncertainty --------------------------------------------------------
    save(pdc.quantile_comparison_plot(
        serving, "weibull", include_histogram=False,
        confidence_band=0.95, band_niter=600, seed=1,
    ), "qq-band")

    fits = {
        label: pdc.bootdist(pdc.fit_distributions(subset, "gumbel")[0],
                            niter=400, seed=2)
        for label, subset in {
            "Euro 3 petrol": stats.gumbel_r.rvs(7.8, 8.3, size=1701,
                                                random_state=np.random.default_rng(1)),
            "Euro 4 petrol": stats.gumbel_r.rvs(6.7, 7.2, size=3732,
                                                random_state=np.random.default_rng(2)),
            "Euro 6 petrol": stats.gumbel_r.rvs(5.5, 5.5, size=374,
                                                random_state=np.random.default_rng(3)),
            "Euro 6 diesel": stats.gumbel_r.rvs(11.2, 8.3, size=362,
                                                random_state=np.random.default_rng(4)),
        }.items()
    }
    save(pdc.confint_plot(fits, parameter="loc"), "confint")
    save(pdc.bootdist_plot(pdc.bootdist(
        pdc.fit_distributions(serving, "weibull")[0], niter=800, seed=1)),
        "bootdist", width=560, height=440)

    # --- off-model fraction -------------------------------------------------
    result = pdc.off_model_fraction(emissions, "gumbel")
    save(pdc.r_squared_sweep_plot({"Simulated fleet": result}), "r2-sweep", width=820)
    save(pdc.off_model_density_plot(result, data_name="Emission ratio"),
         "off-model-density")

    # --- mixtures -----------------------------------------------------------
    mixture = pdc.fit_mixture(emissions, ("gumbel", "gumbel"))
    save(pdc.mixture_density_plot(mixture, data_name="Emission ratio"),
         "mixture-density")
    save(pdc.component_probability_plot(mixture, data_name="Emission ratio"),
         "component-probability")

    # --- spliced ------------------------------------------------------------
    spliced = pdc.fit_spliced(emissions, "gumbel", "pareto")
    save(pdc.spliced_density_plot(spliced, data_name="Emission ratio", log_y=True),
         "spliced-density")
    save(pdc.threshold_profile_plot(spliced), "threshold-profile")

    # --- measurement error --------------------------------------------------
    # A non-negative truth read by an instrument that propagates symmetric
    # error, so some readings come back negative.
    noise_rng = np.random.default_rng(0)
    truth = stats.gamma.rvs(2.0, scale=6.0, size=4000, random_state=noise_rng)
    readings = truth + noise_rng.normal(0, 6.0, 4000)
    save(pdc.convolved_density_plot(
        pdc.fit_convolved(readings, "gamma", error="normal"),
        data_name="Instrument reading"), "convolved-density")

    # --- counts -------------------------------------------------------------
    counts = np.random.default_rng(1).negative_binomial(4, 0.4, 2000).astype(float)
    _, count_density, _, _ = pdc.quantile_comparison_plot(
        counts, ["poisson", "negative_binomial", "geometric"], data_name="Counts",
    )
    save(count_density, "counts-density")

    print(f"\n{len(list(IMAGES.glob('*.png')))} figures in {IMAGES.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
