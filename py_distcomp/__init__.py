"""
PyDistComp: Python Distribution Comparison Tool

A Python port of the distribution-fitting and comparison workflow provided by
R's fitdistrplus package (Delignette-Muller & Dutang), covering descdist /
Cullen and Frey graphs, mle fitting, the qqcomp / ppcomp / cdfcomp / denscomp
comparison plots, and gofstat goodness-of-fit statistics, for both continuous
and discrete distributions, fitted by maximum likelihood, moment matching,
quantile matching or maximum goodness-of-fit.

It also provides the off-model fraction extension of Rushton, Tate & Shepherd
(2021), which identifies the subset of a population that does not follow the
fitted distribution, and mixture fitting, which estimates the same
superposition jointly by expectation-maximisation.

Uncertainty on the estimates comes either from the observed information
(``FitResult.std_error``, as ``fitdist`` reports it) or from the bootstrap
(``bootdist``).
"""

from .distributions import (
    DISCRETE_DISTRIBUTIONS,
    DISTRIBUTION_SPECS,
    SUPPORTED_DISTRIBUTIONS,
    fit_distribution,
    ppoints,
)
from .bootdist import BootdistResult, bootdist, qq_confidence_band
from .bootdist_plots import bootdist_plot, confint_plot
from .empirical_plots import empirical_cdf_plot, empirical_density_plot
from .estimation import (
    ESTIMATION_METHODS,
    GOF_STATISTICS,
    fit_by_method,
    maximum_goodness_of_fit,
    moment_match,
    quantile_match,
)
from .gofstat import FitResult, fit_distributions, gofstat
from .mixture import (
    MIN_EFFECTIVE_PER_PARAM,
    MIN_EFFECTIVE_TOTAL,
    WIDTH_COLLAPSE_RATIO,
    MixtureDistribution,
    MixtureResult,
    fit_mixture,
)
from .mixture_plots import component_probability_plot, mixture_density_plot
from .off_model import OffModelResult, off_model_fraction, qq_r_squared
from .off_model_plots import (
    off_model_density_plot,
    percentile_cut_qq_plot,
    r_squared_sweep_plot,
)
from .spliced import SplicedDistribution, SplicedResult, fit_spliced
from .spliced_plots import spliced_density_plot, threshold_profile_plot
from .quantile_multi_comparison import (
    cullen_and_frey_plot,
    descdist,
    quantile_comparison_plot,
)

__version__ = "0.9.0"
__author__ = "Chris Rushton"
__email__ = "c.e.rushton@leeds.ac.uk"

__all__ = [
    "quantile_comparison_plot",
    "cullen_and_frey_plot",
    "descdist",
    "empirical_cdf_plot",
    "empirical_density_plot",
    "fit_distribution",
    "fit_distributions",
    "gofstat",
    "FitResult",
    "ppoints",
    "SUPPORTED_DISTRIBUTIONS",
    "DISTRIBUTION_SPECS",
    "DISCRETE_DISTRIBUTIONS",
    # Off-model fraction (Rushton et al., 2021)
    "qq_r_squared",
    "off_model_fraction",
    "OffModelResult",
    "r_squared_sweep_plot",
    "percentile_cut_qq_plot",
    "off_model_density_plot",
    # Mixture fitting
    "fit_mixture",
    "MixtureResult",
    "MixtureDistribution",
    "mixture_density_plot",
    "component_probability_plot",
    "MIN_EFFECTIVE_PER_PARAM",
    "MIN_EFFECTIVE_TOTAL",
    "WIDTH_COLLAPSE_RATIO",
    # Uncertainty
    "bootdist",
    "BootdistResult",
    "bootdist_plot",
    "confint_plot",
    "qq_confidence_band",
    # Estimation methods
    "fit_by_method",
    "moment_match",
    "quantile_match",
    "maximum_goodness_of_fit",
    "ESTIMATION_METHODS",
    "GOF_STATISTICS",
    # Spliced (composite) distributions
    "fit_spliced",
    "SplicedResult",
    "SplicedDistribution",
    "spliced_density_plot",
    "threshold_profile_plot",
]
