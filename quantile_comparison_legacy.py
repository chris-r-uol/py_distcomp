from typing import Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize_scalar
import streamlit as st


def quantile_comparison_plot(
    data: Union[pd.Series, np.ndarray, list],
    model: Union[str, object] = 'normal',
    title: str = 'Q–Q Plot',
    name: str = 'Data',
    dist_params: Union[tuple, dict, None] = None,
    include_histogram: bool = True
) -> Union[go.Figure, Tuple[go.Figure, go.Figure], Tuple[go.Figure, go.Figure, go.Figure]]:
    """
    Create a quantile-quantile plot comparing empirical data against a theoretical distribution,
    **exactly** as in R's qqcomp(plotstyle='graphics').

    Returns either just the Q-Q plot, or (Q-Q plot, histogram) if include_histogram=True.
    """
    #st.write(data)
    # -- Data prep
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Data must be a pandas Series, numpy array, or list")
    series = pd.Series(data).dropna()
    if len(series) < 3:
        raise ValueError("At least 3 non-NA data points are required")
    sorted_data = np.sort(series.values)
    n = len(sorted_data)
    

    # -- choose plotting probabilities, matching use.ppoints=TRUE, a.ppoints=0.5 in R
    obsp = (np.arange(1, n+1) - 0.5) / n

    # -- distribution setup (only 'normal' supported here)
    if isinstance(model, str):
        if model.lower() == 'normal':
            dist = stats.norm
            if dist_params is None:
                mu, sigma = series.mean(), series.std(ddof=0)
            else:
                mu, sigma = dist_params
            params = (mu, sigma)
        elif model.lower() == 'lognormal':
            dist = stats.lognorm
            if dist_params is None:
                # Fit lognormal distribution (returns shape, loc, scale)
                params = stats.lognorm.fit(series, floc=0)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 3:
                    raise ValueError("Lognormal distribution requires exactly 3 parameters (shape, loc, scale)")
                params = tuple(dist_params)
        elif model.lower() == 'weibull':
            dist = stats.weibull_min
            if dist_params is None:
                # Fit Weibull distribution (returns shape, loc, scale)
                params = stats.weibull_min.fit(series, floc=0)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 3:
                    raise ValueError("Weibull distribution requires exactly 3 parameters (shape, loc, scale)")
                params = tuple(dist_params)
        elif model.lower() == 'gumbel':
            dist = stats.gumbel_r
            if dist_params is None:
                # Fit Gumbel distribution (returns loc, scale)
                params = stats.gumbel_r.fit(series)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 2:
                    raise ValueError("Gumbel distribution requires exactly 2 parameters (loc, scale)")
                params = tuple(dist_params)
        else:
            raise ValueError(f"Unsupported string model: '{model}'. Currently supported: 'normal', 'lognormal', 'weibull', 'gumbel'")
    
    elif hasattr(model, 'ppf') and hasattr(model, 'cdf'):
        # Custom scipy.stats distribution
        dist = model
        if dist_params is None:
            raise ValueError("Custom distributions require explicit dist_params")
        
        if isinstance(dist_params, dict):
            # Convert dict to tuple for scipy.stats compatibility
            params = tuple(dist_params.values())
        else:
            params = tuple(dist_params)
    
    else:
        raise ValueError("Model must be a string ('normal', 'lognormal', 'weibull', 'gumbel') or scipy.stats distribution object")

    # -- theoretical quantiles
    theoretical_quantiles = dist.ppf(obsp, *params)
    if not np.all(np.isfinite(theoretical_quantiles)):
        raise ValueError("Non-finite theoretical quantiles; check dist_params")
    
    # -- calculate theoretical and observed percentiles
    theoretical_percentiles = obsp * 100  # Convert probabilities to percentiles
    observed_percentiles = dist.cdf(sorted_data, *params) * 100  # Observed data percentiles under the model




    # -- now call the new _create_qq_plot
    qq_fig = _create_qq_plot(
        theoretical_quantiles=theoretical_quantiles,
        empirical_data=sorted_data,
        reference_line=theoretical_quantiles,
        title=title,
        name=name,
        x_label="Model Predictions (Theoretical Quantiles)",
        y_label="Observed Data (Empirical Quantiles)"
    )

    # -- optionally build the histogram of data + fitted curve
    if include_histogram:
        # you already have your own _create_histogram_plot; just call it as before
        hist_fig = _create_histogram_plot(
            #series.values, # original formulation
            sorted_data,
            dist,
            params,
            title.replace('Q–Q', 'Distribution Comparison'),
            name
        )

        pp_fig = _create_qq_plot(
            theoretical_quantiles = observed_percentiles,
            empirical_data = theoretical_percentiles,
            reference_line = theoretical_percentiles,
            title='Percentile-Percentile Plot',
            x_label='Theoretical Probabilities',
            y_label='Empirical Probabilities',
            name=name
        )

        cdf_fig = _create_cdf_plot(
            data=sorted_data,
            distribution=dist,
            fitted_params=params,
            title='Cumulative Distribution Function',
            name=name
        )
        return qq_fig, hist_fig, pp_fig, cdf_fig
    

    return qq_fig

def _setup_distribution(model: Union[str, object], data: np.ndarray, 
                       dist_params: Union[tuple, dict, None]) -> tuple:
    """
    Set up the statistical distribution and parameters for Q-Q plot.
    
    Args:
        model: Distribution model specification.
        data: Clean data array for parameter estimation.
        dist_params: User-provided distribution parameters.
        
    Returns:
        Tuple of (distribution object, fitted parameters).
        
    Raises:
        ValueError: For unsupported models or missing parameters.
    """
    if isinstance(model, str):
        if model.lower() == 'normal':
            distribution = stats.norm
            if dist_params is None:
                # Use maximum likelihood estimation
                fitted_params = (np.mean(data), np.std(data, ddof=1))
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 2:
                    raise ValueError("Normal distribution requires exactly 2 parameters (mean, std)")
                fitted_params = tuple(dist_params)
        elif model.lower() == 'lognormal':
            distribution = stats.lognorm
            if dist_params is None:
                # Fit lognormal distribution (returns shape, loc, scale)
                fitted_params = stats.lognorm.fit(data, floc=0)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 3:
                    raise ValueError("Lognormal distribution requires exactly 3 parameters (shape, loc, scale)")
                fitted_params = tuple(dist_params)
        elif model.lower() == 'weibull':
            distribution = stats.weibull_min
            if dist_params is None:
                # Fit Weibull distribution (returns shape, loc, scale)
                fitted_params = stats.weibull_min.fit(data, floc=0)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 3:
                    raise ValueError("Weibull distribution requires exactly 3 parameters (shape, loc, scale)")
                fitted_params = tuple(dist_params)
        elif model.lower() == 'gumbel':
            distribution = stats.gumbel_r
            if dist_params is None:
                # Fit Gumbel distribution (returns loc, scale)
                fitted_params = stats.gumbel_r.fit(data)
            else:
                if not isinstance(dist_params, (tuple, list)) or len(dist_params) != 2:
                    raise ValueError("Gumbel distribution requires exactly 2 parameters (loc, scale)")
                fitted_params = tuple(dist_params)
        else:
            raise ValueError(f"Unsupported string model: '{model}'. Currently supported: 'normal', 'lognormal', 'weibull', 'gumbel'")
    
    elif hasattr(model, 'ppf') and hasattr(model, 'cdf'):
        # Custom scipy.stats distribution
        distribution = model
        if dist_params is None:
            raise ValueError("Custom distributions require explicit dist_params")
        
        if isinstance(dist_params, dict):
            # Convert dict to tuple for scipy.stats compatibility
            fitted_params = tuple(dist_params.values())
        else:
            fitted_params = tuple(dist_params)
    
    else:
        raise ValueError("Model must be a string ('normal') or scipy.stats distribution object")
    
    return distribution, fitted_params

def _create_cdf_plot(
    data: np.ndarray,
    distribution: object,
    fitted_params: tuple,
    title: str,
    name: str
) -> go.Figure:
    """
    Create a cumulative distribution function comparison plot with empirical vs theoretical CDF.
    
    Args:
        data: Sorted data array.
        distribution: Scipy.stats distribution object.
        fitted_params: Fitted distribution parameters.
        title: Plot title.
        name: Data series name.
        
    Returns:
        Configured Plotly Figure object with empirical and theoretical CDFs.
    """
    fig = go.Figure()
    
    # Sort data for CDF calculation
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Calculate empirical CDF (step function)
    empirical_cdf = np.arange(1, n + 1) / n
    
    # Create theoretical CDF over extended range for smooth curve
    x_min, x_max = sorted_data.min(), sorted_data.max()
    x_range = np.linspace(x_min, x_max, 200)
    theoretical_cdf = distribution.cdf(x_range, *fitted_params)
    
    # Add empirical CDF as step plot
    fig.add_trace(go.Scatter(
        x=sorted_data,
        y=empirical_cdf,
        mode='markers+lines',
        name=f'{name} (Empirical)',
        line=dict(shape='hv', color='dodgerblue', width=2),
        marker=dict(size=4, opacity=0.7, color='dodgerblue'),
        hovertemplate=(
            f"<b>{name} Empirical CDF</b><br>"
            "Value: %{x:.3f}<br>"
            "CDF: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add theoretical CDF as smooth curve
    dist_name = getattr(distribution, 'name', 'Theoretical')
    param_str = ', '.join([f'{p:.3f}' for p in fitted_params])
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=theoretical_cdf,
        mode='lines',
        name=f'{dist_name.title()} (Theoretical)',
        line=dict(color='red', width=3),
        hovertemplate=(
            f"<b>{dist_name.title()} Theoretical CDF</b><br>"
            "Value: %{x:.3f}<br>"
            "CDF: %{y:.3f}<br>"
            f"Parameters: {param_str}<br>"
            "<extra></extra>"
        )
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Cumulative Probability',
            showgrid=True,
            gridcolor='lightgray',
            range=[0, 1]
        ),
        template='plotly_white',
        height=500,
        width=700,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig

def _create_qq_plot(
    theoretical_quantiles: np.ndarray,
    empirical_data: np.ndarray,
    reference_line: np.ndarray,
    title: str,
    name: str,
    x_label: str = 'Theoretical Quantiles',
    y_label: str = 'Empirical Quantiles',
    
) -> go.Figure:
    """
    Create the actual Q-Q plot using Plotly, matching R's qqcomp(style='graphics').

    Args:
        theoretical_quantiles: X-axis values (theoretical quantiles from the model).
        empirical_data: Y-axis values (the sorted raw data).
        reference_line:   The values to draw the y=x reference (should == theoretical_quantiles).
        title:            Plot title.
        name:             Data series name (legend).
    Returns:
        A Plotly Figure with points (x=theoretical, y=data) and a y=x line.
    """
    fig = go.Figure()

    # 1) scatter of data points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=empirical_data,
        mode='markers',
        name=name,
        marker=dict(size=6, opacity=0.7, color='dodgerblue'),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "Theoretical: %{x:.3f}<br>"
            "Empirical: %{y:.3f}<extra></extra>"
        )
    ))

    # 2) identity line y = x
    fig.add_trace(go.Scatter(
        x=reference_line,
        y=reference_line,
        mode='lines',
        name='y = x',
        line=dict(dash='dash', color='red', width=1.5),
        hoverinfo='skip'
    ))

    # layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title=x_label, showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='lightgray'),
        template='plotly_white',
        height=500,
        width=700,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
        hovermode='closest'
    )

    return fig

def _create_histogram_plot(data: np.ndarray, distribution: object, fitted_params: tuple,
                          title: str, name: str) -> go.Figure:
    """
    Create a histogram with overlaid theoretical distribution curve.
    
    Args:
        data: Original data array.
        distribution: Scipy.stats distribution object.
        fitted_params: Fitted distribution parameters.
        title: Plot title.
        name: Data series name.
        
    Returns:
        Configured Plotly Figure object with histogram and fitted curve.
    """
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=75,
        histnorm='probability density',
        name=f'{name} (Empirical)',
        marker=dict(
            color='dodgerblue',
            opacity=0.5,
            #line=dict(color='black', width=1)
        ),
        hovertemplate=(
            "Bin Center: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Create theoretical distribution curve
    x_range = np.linspace(data.min(), data.max(), 200)
    theoretical_pdf = distribution.pdf(x_range, *fitted_params)
    
    # Get distribution name for legend
    dist_name = getattr(distribution, 'name', 'Theoretical')
    param_str = ', '.join([f'{p:.3f}' for p in fitted_params])
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=theoretical_pdf,
        mode='lines',
        name=f'{dist_name.title()} ({param_str})',
        line=dict(
            color='red',
            width=3
        ),
        hovertemplate=(
            f"<b>{dist_name.title()} Distribution</b><br>"
            "Value: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Calculate statistics
    empirical_mean = np.mean(data)
    empirical_median = np.median(data)
    empirical_95th = np.percentile(data, 95)
    theoretical_mean = distribution.mean(*fitted_params)
    theoretical_median = distribution.median(*fitted_params)
    theoretical_95th = distribution.ppf(0.95, *fitted_params)

    abs_dev = np.abs(data - empirical_median)
    median_absolute_deviation = np.median(abs_dev)
    
    #st.write(abs_dev)
    

    # Calculate mode (most frequent value for empirical, ppf(0.5) or custom for theoretical)
    # For empirical mode, use the bin center with highest frequency
    hist_counts, bin_edges = np.histogram(data, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    empirical_mode = bin_centers[np.argmax(hist_counts)]
    
    # For theoretical mode, find the value that maximizes the PDF
    try:
        # Use optimization to find the mode
        result = minimize_scalar(lambda x: -distribution.pdf(x, *fitted_params), 
                               bounds=(data.min(), data.max()), method='bounded')
        theoretical_mode = result.x
    except:
        # Fallback to median if mode calculation fails
        theoretical_mode = theoretical_median
    
    # Get y-axis range for vertical lines
    y_max = max(theoretical_pdf) * 1.1
    
    # Add vertical lines for empirical statistics
    fig.add_trace(go.Scatter(
        x=[empirical_mean, empirical_mean],
        y=[0, y_max],
        mode='lines',
        name=f'Empirical Mean ({empirical_mean:.3f})',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate=(
            f"Empirical Mean<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=[empirical_median, empirical_median],
        y=[0, y_max],
        mode='lines',
        name=f'Empirical Median ({empirical_median:.3f})',
        line=dict(color='indianred', width=2, dash='dash'),
        hovertemplate=(
            f"Empirical Median<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=[empirical_mode, empirical_mode],
        y=[0, y_max],
        mode='lines',
        name=f'Empirical Mode ({empirical_mode:.3f})',
        line=dict(color='magenta', width=2, dash='dash'),
        hovertemplate=(
            f"Empirical Mode<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=[empirical_95th, empirical_95th],
        y=[0, y_max],
        mode='lines',
        name=f'Empirical 95th Percentile ({empirical_95th:.3f})',
        line=dict(color='darkred', width=2, dash='dot'),
        hovertemplate=(
            f"Empirical 95th Percentile<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=[theoretical_95th, theoretical_95th],
        y=[0, y_max],
        mode='lines',
        name=f'Theoretical 95th Percentile ({theoretical_95th:.3f})',
        line=dict(color='darkblue', width=2, dash='dot'),
        hovertemplate=(
            f"Theoretical 95th Percentile<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    mad_threshold = 4.45
    n_mad = (mad_threshold * median_absolute_deviation)  + empirical_median
    st.write(f'Median Absolute Deviation: {median_absolute_deviation:.3f}')

    # Calculate what percentage of the data is greater than n_mad
    mad_count = np.count_nonzero(data > n_mad)
    #mad_over = data[data > n_mad]  # Get the actual data points greater than n_mad
    #len_mad_over = len(mad_over)  # Count of data points greater than n_mad
    #st.write(f'Number of data points greater than {n_mad:.3f}: {len_mad_over}, or {mad_count}')
    # count the number of data points greater than n_mad

    
    mad_percent = (mad_count / len(data)) * 100
    st.write(f'Number of data points greater than {mad_threshold} MAD: {mad_count} ({mad_percent:.2f}%)')

    fig.add_trace(go.Scatter(
        x=[n_mad, n_mad],
        y=[0, y_max],
        mode='lines',
        name=f'{mad_threshold} Median Absolute Deviations from Median ({n_mad:.3f})',
        line=dict(color='indianred', width=2, dash='solid'),
        hovertemplate=(
            f"{n_mad} Median Absolute Deviation<br>"
            "Value: %{x:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Probability Density',
            showgrid=True,
            gridcolor='lightgray'
        ),
        template='plotly_white',
        height=500,
        width=700,
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
        bargap=0.1
    )
    
    return fig













