#!/usr/bin/env python3
"""
Example usage of PyDistComp for distribution comparison analysis.

This script demonstrates basic and advanced usage patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from quantile_multi_comparison import quantile_comparison_plot


def basic_example():
    """Basic usage example with synthetic data."""
    print("Running basic example...")
    
    # Generate sample data from a normal distribution
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=1000)
    
    # Single distribution comparison
    qq_fig = quantile_comparison_plot(
        data=data,
        models='normal',
        title='Basic Q-Q Plot Example',
        data_name='Sample Data'
    )
    
    print("✓ Basic Q-Q plot created")
    return qq_fig


def multi_distribution_example():
    """Example comparing multiple distributions."""
    print("Running multi-distribution example...")
    
    # Generate sample data from a Weibull distribution
    np.random.seed(123)
    data = np.random.weibull(a=2, size=500) * 10
    
    # Compare against multiple distributions
    qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
        data=data,
        models=['normal', 'weibull', 'lognormal', 'gamma'],
        title='Multi-Distribution Comparison',
        data_name='Weibull Sample'
    )
    
    print("✓ Multi-distribution plots created")
    return qq_fig, hist_fig, pp_fig, cdf_fig


def custom_parameters_example():
    """Example with custom distribution parameters."""
    print("Running custom parameters example...")
    
    # Generate mixed data
    np.random.seed(456)
    data1 = np.random.normal(0, 1, 300)
    data2 = np.random.normal(3, 0.5, 200)
    data = np.concatenate([data1, data2])
    
    # Test against distributions with specific parameters
    models = ['normal', stats.norm]
    params = [
        (1.2, 1.5),    # Normal with mean=1.2, std=1.5
        (0, 1)         # Standard normal
    ]
    
    qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
        data=data,
        models=models,
        dist_params=params,
        title='Custom Parameters Example',
        data_name='Mixed Normal Data'
    )
    
    print("✓ Custom parameters plots created")
    return qq_fig, hist_fig, pp_fig, cdf_fig


def financial_data_example():
    """Example simulating financial returns analysis."""
    print("Running financial data example...")
    
    # Simulate daily returns (fat-tailed distribution)
    np.random.seed(789)
    returns = np.random.standard_t(df=4, size=1000) * 0.02
    
    # Compare against common financial distributions
    qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
        data=returns,
        models=['normal', 'student_t', 'laplace'],
        title='Financial Returns Distribution Analysis',
        data_name='Daily Returns'
    )
    
    print("✓ Financial data analysis plots created")
    return qq_fig, hist_fig, pp_fig, cdf_fig


def main():
    """Run all examples."""
    print("PyDistComp Example Usage")
    print("=" * 40)
    
    try:
        # Run examples
        basic_example()
        multi_distribution_example()
        custom_parameters_example()
        financial_data_example()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo view the plots, use .show() method on returned figures:")
        print("Example: qq_fig.show()")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
