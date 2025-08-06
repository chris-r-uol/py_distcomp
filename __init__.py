"""
PyDistComp: Python Distribution Comparison Tool

A professional Python library for comprehensive statistical distribution comparison and visualization.
"""

from .quantile_multi_comparison import quantile_comparison_plot, cullen_and_frey_plot
from .empirical_plots import empirical_cdf_plot, empirical_density_plot

__version__ = "0.1.0"
__author__ = "Chris Russell"
__email__ = "your.email@leeds.ac.uk"

__all__ = [
    "quantile_comparison_plot", 
    "cullen_and_frey_plot",
    "empirical_cdf_plot", 
    "empirical_density_plot"
]
