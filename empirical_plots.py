from typing import Union, Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize_scalar
import streamlit as st


def empirical_cdf_plot(
    data: Union[np.ndarray, pd.Series, List[float]], 
    name: str = "Data",
    color: str = 'seagreen',
    width: int = 700,
    height: int = 500,
    show_percentiles: bool = True,
    percentile_lines: Optional[List[float]] = None,
    show_annotations: bool = True
) -> go.Figure:
    """
    Create an empirical CDF plot using Plotly Go.
    
    This function creates a visualization of the empirical cumulative distribution
    function, which shows the proportion of data points less than or equal to 
    each value.
    
    Args:
        data: Input data for the empirical CDF plot.
        name: Name for the data series in the legend.
        color: Color for the CDF line.
        width: Plot width in pixels.
        height: Plot height in pixels.
        show_percentiles: Whether to show common percentile lines (25th, 50th, 75th).
        percentile_lines: Custom percentiles to highlight (0-100).
        show_annotations: Whether to show annotations on percentile lines.
        
    Returns:
        Plotly Figure object containing the empirical CDF plot.
        
    Raises:
        ValueError: If data is empty or contains non-numeric values.
        TypeError: If data type is not supported.
    """
    # Input validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Convert to numpy array for consistent handling
    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Data must contain numeric values: {e}")
    
    # Remove NaN values
    data_clean = data_array[~np.isnan(data_array)]
    if len(data_clean) == 0:
        raise ValueError("No valid (non-NaN) data points found")
    
    # Sort data for CDF calculation
    sorted_data = np.sort(data_clean)
    n = len(sorted_data)
    
    # Calculate empirical CDF values
    # Using (i+1)/n to avoid CDF values of 0 (more common convention)
    cdf_values = np.arange(1, n + 1) / n
    
    # Create figure
    fig = go.Figure()
    
    # Add empirical CDF line
    fig.add_trace(go.Scatter(
        x=sorted_data,
        y=cdf_values,
        mode='lines',
        name=f'{name} (Empirical CDF)',
        line=dict(
            color=color,
            width=2,
            shape='hv'  # Step function appearance
        ),
        hovertemplate=(
            "Value: %{x:.3f}<br>"
            "Cumulative Probability: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add percentile lines if requested
    if show_percentiles or percentile_lines:
        percentiles_to_show = percentile_lines if percentile_lines else [25, 50, 75]
        
        for p in percentiles_to_show:
            if not 0 <= p <= 100:
                continue
                
            percentile_value = np.percentile(sorted_data, p)
            percentile_prob = p / 100
            
            # Vertical line at percentile value
            fig.add_vline(
                x=percentile_value,
                line=dict(
                    color='red',
                    width=1,
                    dash='dash'
                ),
                annotation=dict(
                    text=f"P{int(p)}: {percentile_value:.2f}",
                    textangle=90,
                    font=dict(size=10)
                ) if show_annotations else None
            )
            
            # Horizontal line at percentile probability
            fig.add_hline(
                y=percentile_prob,
                line=dict(
                    color='red',
                    width=1,
                    dash='dash'
                ),
                annotation=dict(
                    text=f"{p}%",
                    font=dict(size=10)
                ) if show_annotations else None
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Empirical Cumulative Distribution Function - {name}",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Cumulative Probability',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            range=[0, 1]
        ),
        template='plotly_white',
        height=height,
        width=width,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


    


def empirical_density_plot(
    data: Union[np.ndarray, pd.Series, List[float]], 
    name: str = "Data",
    bins: int = 75,
    kde_points: int = 1000,
    color_histogram: str = 'gray',
    color_density: str = 'seagreen',
    opacity_histogram: float = 0.7,
    width: int = 700,
    height: int = 500
) -> go.Figure:
    """
    Create an empirical density plot using Plotly Go.
    
    This function creates a visualization combining a histogram (showing empirical 
    probability density) with a smooth kernel density estimation curve.
    
    Args:
        data: Input data for the empirical density plot.
        name: Name for the data series in the legend.
        bins: Number of bins for the histogram.
        kde_points: Number of points for the KDE curve.
        color_histogram: Color for the histogram bars.
        color_density: Color for the density curve.
        opacity_histogram: Opacity level for histogram bars (0-1).
        width: Plot width in pixels.
        height: Plot height in pixels.
        
    Returns:
        Plotly Figure object containing the density plot.
        
    Raises:
        ValueError: If data is empty or contains non-numeric values.
        TypeError: If data type is not supported.
    """
    # Input validation
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Convert to numpy array for consistent handling
    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Data must contain numeric values: {e}")
    
    # Remove NaN values
    data_clean = data_array[~np.isnan(data_array)]
    if len(data_clean) == 0:
        raise ValueError("No valid (non-NaN) data points found")
    
    if len(data_clean) < 2:
        raise ValueError("At least 2 data points required for density estimation")
    
    # Calculate kernel density estimation
    try:
        density = stats.gaussian_kde(data_clean)
        data_min, data_max = np.min(data_clean), np.max(data_clean)
        
        # Add small padding to avoid edge effects
        data_range = data_max - data_min
        padding = data_range * 0.05
        x_values = np.linspace(
            data_min - padding, 
            data_max + padding, 
            kde_points
        )
        y_values = density(x_values)
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute kernel density estimation: {e}")
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram of the underlying data
    fig.add_trace(go.Histogram(
        x=data_clean,
        nbinsx=bins,
        histnorm='probability density',
        name=f'{name} (Histogram)',
        marker=dict(
            color=color_histogram,
            opacity=opacity_histogram,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate=(
            "Bin Center: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add empirical density line
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values, 
        mode='lines', 
        name=f'{name} (KDE)',
        line=dict(
            color=color_density,
            width=2,
            dash='solid'
        ),
        hovertemplate=(
            "Value: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        ),
        opacity=1.0
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Empirical Density Plot - {name}", 
            x=0.5, 
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Value', 
            showgrid=True, 
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Probability Density', 
            showgrid=True, 
            gridcolor='lightgray',
            zeroline=False
        ),
        template='plotly_white',
        height=height,
        width=width,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest',
        bargap=0.1,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig