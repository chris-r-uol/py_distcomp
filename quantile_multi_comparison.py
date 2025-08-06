from typing import Union, Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize_scalar
import streamlit as st


# Color palette for multiple distributions
DISTRIBUTION_COLORS = [
    'steelblue', 'indianred', 'green', 'orange', 'purple', 
    'brown', 'pink', 'gray', 'olive', 'cyan'
]

# Supported distributions mapping
SUPPORTED_DISTRIBUTIONS = {
    'normal': stats.norm,
    'lognormal': stats.lognorm,
    'weibull': stats.weibull_min,
    'gumbel': stats.gumbel_r,
    'exponential': stats.expon,
    'gamma': stats.gamma,
    'beta': stats.beta,
    'uniform': stats.uniform,
    'logistic': stats.logistic,
    'laplace': stats.laplace,
    'chi2': stats.chi2,
    'student_t': stats.t,
    'f': stats.f,
    'pareto': stats.pareto,
    'rayleigh': stats.rayleigh,
    #'poisson': stats.poisson,
    #'binomial': stats.binom,
    #'negative_binomial': stats.nbinom,
    #'geometric': stats.geom,
}

def cullen_and_frey_plot(
    data: Union[pd.Series, np.ndarray, list],
    title: str = 'Cullen and Frey Graph',
    data_name: str = 'Data',
    n_bootstrap: int = 100,
    show_bootstrap: bool = True,
    show_theoretical: bool = True
) -> go.Figure:
    """
    Create a Cullen and Frey plot for empirical data.
    
    A Cullen and Frey plot is a statistical diagnostic chart used to assess which 
    probability distributions best fit a given dataset by plotting sample skewness 
    squared on the horizontal axis and sample kurtosis on the vertical axis.
    
    The plot includes:
    - A point representing the observed data's (skewness², kurtosis)
    - Theoretical regions showing where common probability distributions fall
    - Optional bootstrap cloud showing sampling variability
    
    Parameters
    ----------
    data : array-like
        Input data as pandas Series, numpy array, or list.
    title : str, default='Cullen and Frey Graph'
        Plot title.
    data_name : str, default='Data'
        Name for the empirical data point.
    n_bootstrap : int, default=100
        Number of bootstrap samples for variability estimation.
    show_bootstrap : bool, default=True
        Whether to show bootstrap confidence cloud.
    show_theoretical : bool, default=True
        Whether to show theoretical distribution regions.
        
    Returns
    -------
    go.Figure
        Plotly Figure object containing the Cullen and Frey plot.
        
    Raises
    ------
    ValueError
        If data is empty, contains non-numeric values, or has insufficient points.
    TypeError
        If data type is not supported.
        
    Examples
    --------
    >>> # Basic Cullen and Frey plot
    >>> cf_fig = cullen_and_frey_plot(data)
    
    >>> # With custom parameters
    >>> cf_fig = cullen_and_frey_plot(
    ...     data, 
    ...     title='Distribution Assessment',
    ...     n_bootstrap=200,
    ...     show_bootstrap=True
    ... )
    """
    # Validate and prepare data
    clean_data = _validate_and_prepare_data(data)
    
    # Calculate observed skewness and kurtosis
    obs_skew_sq, obs_kurtosis = _calculate_skewness_kurtosis(clean_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add theoretical distribution regions if requested
    if show_theoretical:
        _add_theoretical_distributions(fig)
    
    # Add bootstrap cloud if requested
    if show_bootstrap:
        _add_bootstrap_cloud(fig, clean_data, n_bootstrap)
    
    # Add observed data point
    fig.add_trace(go.Scatter(
        x=[obs_skew_sq],
        y=[obs_kurtosis],
        mode='markers',
        name=f'{data_name} (Observed)',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        hovertemplate=(
            f"<b>{data_name}</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title='Skewness²',
            showgrid=True,
            gridcolor='lightgray',
            range=[-0.5, max(10, obs_skew_sq + 2)]
        ),
        yaxis=dict(
            title='Kurtosis',
            showgrid=True,
            gridcolor='lightgray',
            range=[max(20, obs_kurtosis + 2), 0]
        ),
        template='plotly_white',
        height=600,
        width=800,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig


def _calculate_skewness_kurtosis(data: np.ndarray) -> Tuple[float, float]:
    """
    Calculate sample skewness squared and kurtosis.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
        
    Returns
    -------
    Tuple[float, float]
        Skewness squared and kurtosis values.
    """
    # Calculate moments
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    n = len(data)
    
    # Calculate skewness (using sample skewness formula)
    skewness = stats.skew(data, bias=False)
    skewness_squared = skewness ** 2
    
    # Calculate kurtosis (using sample excess kurtosis + 3)
    kurtosis = stats.kurtosis(data, bias=False) + 3
    
    return skewness_squared, kurtosis


def _add_theoretical_distributions(fig: go.Figure) -> None:
    """
    Add theoretical distribution points and regions to the plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add theoretical distributions to.
    """
    # Theoretical distribution points (skewness², kurtosis)
    theoretical_points = {
        'Normal': (0, 3),
        'Uniform': (0, 1.8),
        'Exponential': (4, 9),
        'Chi-squared (df=1)': (8, 15),
        'Chi-squared (df=4)': (2, 4.5),
        'Logistic': (0, 4.2)
    }
    
    # Add theoretical points
    for dist_name, (skew_sq, kurt) in theoretical_points.items():
        fig.add_trace(go.Scatter(
            x=[skew_sq],
            y=[kurt],
            mode='markers',
            name=dist_name,
            marker=dict(
                size=8,
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hovertemplate=(
                f"<b>{dist_name}</b><br>"
                "Skewness²: %{x:.3f}<br>"
                "Kurtosis: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))
    
    # Add lognormal curve
    _add_lognormal_curve(fig)
    
    # Add gamma curve
    _add_gamma_curve(fig)
    
    # Add beta curve
    _add_beta_curve(fig)


def _add_lognormal_curve(fig: go.Figure) -> None:
    """
    Add lognormal distribution curve to the plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add the curve to.
    """
    # Generate lognormal curve points
    # For lognormal with log(X) ~ N(0, σ²), skewness² and kurtosis depend on σ
    sigma_values = np.linspace(0.1, 2, 50)
    skew_sq_values = []
    kurtosis_values = []
    
    for sigma in sigma_values:
        # Theoretical formulas for lognormal distribution
        exp_sigma_sq = np.exp(sigma**2)
        skewness = (exp_sigma_sq + 2) * np.sqrt(exp_sigma_sq - 1)
        kurtosis = exp_sigma_sq**4 + 2*exp_sigma_sq**3 + 3*exp_sigma_sq**2 - 3
        
        skew_sq_values.append(skewness**2)
        kurtosis_values.append(kurtosis)
    
    fig.add_trace(go.Scatter(
        x=skew_sq_values,
        y=kurtosis_values,
        mode='lines',
        name='Lognormal',
        line=dict(color='blue', width=2, dash='dot'),
        hovertemplate=(
            "<b>Lognormal Distribution</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))


def _add_gamma_curve(fig: go.Figure) -> None:
    """
    Add gamma distribution curve to the plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add the curve to.
    """
    # Generate gamma curve points
    # For gamma distribution, skewness = 2/√k, kurtosis = 3 + 6/k
    k_values = np.linspace(0.5, 10, 50)
    skew_sq_values = []
    kurtosis_values = []
    
    for k in k_values:
        skewness = 2 / np.sqrt(k)
        kurtosis = 3 + 6 / k
        
        skew_sq_values.append(skewness**2)
        kurtosis_values.append(kurtosis)
    
    fig.add_trace(go.Scatter(
        x=skew_sq_values,
        y=kurtosis_values,
        mode='lines',
        name='Gamma',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate=(
            "<b>Gamma Distribution</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))


def _add_beta_curve(fig: go.Figure) -> None:
    """
    Add beta distribution curve to the plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add the curve to.
    """
    # Generate beta curve points for symmetric beta distributions (α = β)
    start = 0.0 # default = 0.5
    finish = 10.0 # default = 5
    alpha_values = np.linspace(start, finish, 120)
    skew_sq_values = []
    kurtosis_values = []
    
    for alpha in alpha_values:
        # For symmetric beta (α = β), skewness = 0, kurtosis = 3(1+2α)/(2+3α)
        skewness = 0
        kurtosis = 3 * (1 + 2*alpha) / (2 + 3*alpha)
        
        skew_sq_values.append(skewness**2)
        kurtosis_values.append(kurtosis)
    
    fig.add_trace(go.Scatter(
        x=skew_sq_values,
        y=kurtosis_values,
        mode='lines',
        name='Beta (symmetric)',
        line=dict(color='purple', width=2, dash='dashdot'),
        hovertemplate=(
            "<b>Beta Distribution (symmetric)</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))

    # ---- Asymmetric Beta: example curve with β = r * α (r ≠ 1)
    ratios = [2, 5, 10]  # change this ratio to add different asymmetric curves
    for ratio in ratios:
        a = np.linspace(start, finish, 120)
        b = ratio * a

        # Skewness and (Pearson) kurtosis for Beta(a, b)
        # skew = 2(b - a) * sqrt(a + b + 1) / ((a + b + 2) * sqrt(a b))
        # kurt = excess + 3, where
        # excess = 6[(a - b)^2 (a + b + 1) - a b (a + b + 2)] / [a b (a + b + 2)(a + b + 3)]
        numerator_skew = 2.0 * (b - a) * np.sqrt(a + b + 1.0)
        denom_skew = (a + b + 2.0) * np.sqrt(a * b)
        skew = numerator_skew / denom_skew

        excess_num = 6.0 * ((a - b)**2 * (a + b + 1.0) - a * b * (a + b + 2.0))
        excess_den = (a * b * (a + b + 2.0) * (a + b + 3.0))
        kurtosis = (excess_num / excess_den) + 3.0

        fig.add_trace(go.Scatter(
            x=(skew**2),
            y=kurtosis,
            mode='lines',
            name=f'Beta (β = {ratio}·α)',
            line=dict(color='purple',width=2),
            hovertemplate=(
                "<b>Beta (asymmetric)</b><br>"
                "β/α: " + f"{ratio:.2f}" + "<br>"
                "Skewness²: %{x:.3f}<br>"
                "Kurtosis: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))


def _add_bootstrap_cloud(
    fig: go.Figure, 
    data: np.ndarray, 
    n_bootstrap: int
) -> None:
    """
    Add bootstrap confidence cloud to the plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add bootstrap cloud to.
    data : np.ndarray
        Original data for bootstrap sampling.
    n_bootstrap : int
        Number of bootstrap samples to generate.
    """
    bootstrap_skew_sq = []
    bootstrap_kurtosis = []
    n = len(data)
    
    # Generate bootstrap samples
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        
        # Calculate skewness² and kurtosis for bootstrap sample
        skew_sq, kurt = _calculate_skewness_kurtosis(bootstrap_sample)
        
        bootstrap_skew_sq.append(skew_sq)
        bootstrap_kurtosis.append(kurt)
    
    # Add bootstrap cloud
    fig.add_trace(go.Scatter(
        x=bootstrap_skew_sq,
        y=bootstrap_kurtosis,
        mode='markers',
        name='Bootstrap samples',
        marker=dict(
            size=4,
            opacity=0.95,
            color='orange',
            symbol='circle'
        ),
        hovertemplate=(
            "<b>Bootstrap Sample</b><br>"
            "Skewness²: %{x:.3f}<br>"
            "Kurtosis: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))

def quantile_comparison_plot(
    data: Union[pd.Series, np.ndarray, list],
    models: Union[str, List[str], object, List[object]] = 'normal',
    title: str = 'Q-Q Plot',
    data_name: str = 'Data',
    dist_params: Optional[Union[tuple, dict, List[tuple], List[dict]]] = None,
    include_histogram: bool = True
) -> Union[go.Figure, Tuple[go.Figure, ...]]:

    """
    Create quantile-quantile plots comparing empirical data against theoretical distributions.
    
    This function can handle single or multiple distributions for comparison,
    following R's qqcomp(plotstyle='graphics') behavior.
    
    Parameters
    ----------
    data : array-like
        Input data as pandas Series, numpy array, or list.
    models : str, object, or list thereof, default='normal'
        Distribution model(s) to compare against. Can be:
        - String: 'normal', 'lognormal', 'weibull', 'gumbel', 'exponential', 'gamma'
        - Scipy.stats distribution object
        - List of strings or distribution objects
    title : str, default='Q-Q Plot'
        Plot title.
    data_name : str, default='Data'
        Name for the empirical data series.
    dist_params : tuple, dict, list thereof, or None, default=None
        Distribution parameters. If None, parameters are estimated from data.
        For multiple distributions, provide a list of parameter tuples/dicts.
    include_histogram : bool, default=True
        Whether to include histogram and additional plots.
        
    Returns
    -------
    Union[go.Figure, Tuple[go.Figure, ...]]
        Plotly Figure(s): Q-Q plot, and optionally histogram, P-P plot, and CDF plot.
        
    Raises
    ------
    TypeError
        If data is not a supported type.
    ValueError
        If insufficient data points or unsupported distribution model.
        
    Examples
    --------
    >>> # Single distribution
    >>> qq_fig = quantile_comparison_plot(data, 'normal')
    
    >>> # Multiple distributions
    >>> qq_fig = quantile_comparison_plot(data, ['normal', 'lognormal', 'weibull'])
    
    >>> # With custom parameters
    >>> qq_fig = quantile_comparison_plot(
    ...     data, 
    ...     ['normal', 'lognormal'], 
    ...     dist_params=[(0, 1), (1, 0, 1)]
    ... )
    """
    # Data validation and preprocessing
    empirical_data = _validate_and_prepare_data(data)
    
    # Normalize models to list format for consistent processing
    model_list = _normalize_models_input(models)
    
    # Normalize parameters to list format
    param_list = _normalize_params_input(dist_params, len(model_list))
    
    # Setup distributions with parameters
    distributions = _setup_distributions(model_list, empirical_data, param_list)
    
    # Create Q-Q plot with all distributions
    qq_fig = _create_multi_qq_plot(
        empirical_data=empirical_data,
        distributions=distributions,
        title=title,
        data_name=data_name
    )
    
    if not include_histogram:
        return qq_fig
    
    # Create additional plots for the first (primary) distribution
    primary_dist, primary_params = distributions[0]
    
    hist_fig = _create_multi_histogram_plot(
        data=empirical_data,
        distributions=distributions,
        title='PDF Distribution Comparison Plot',
        name=data_name
    )
    
    pp_fig = _create_multi_pp_plot(
        empirical_data=empirical_data,
        distributions=distributions,
        title='Percentile-Percentile Plot',
        data_name=data_name
    )
    
    cdf_fig = _create_multi_cdf_plot(
        empirical_data=empirical_data,
        distributions=distributions,
        title='Cumulative Distribution Function',
        data_name=data_name
    )
    
    return qq_fig, hist_fig, pp_fig, cdf_fig


def _validate_and_prepare_data(
    data: Union[pd.Series, np.ndarray, list]
) -> np.ndarray:
    """
    Validate input data and return sorted numpy array.
    
    Parameters
    ----------
    data : array-like
        Input data to validate and prepare.
        
    Returns
    -------
    np.ndarray
        Sorted array of clean (non-NaN) data.
        
    Raises
    ------
    TypeError
        If data type is not supported.
    ValueError
        If insufficient data points after cleaning.
    """
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Data must be a pandas Series, numpy array, or list")
    
    # Convert to pandas Series for consistent handling
    series = pd.Series(data).dropna()
    
    if len(series) < 3:
        raise ValueError("At least 3 non-NA data points are required")
    
    # Convert to numpy array and sort
    return np.sort(series.to_numpy())


def _normalize_models_input(
    models: Union[str, List[str], object, List[object]]
) -> List[Union[str, object]]:
    """
    Normalize models input to a consistent list format.
    
    Parameters
    ----------
    models : str, object, or list thereof
        Models specification to normalize.
        
    Returns
    -------
    List[Union[str, object]]
        List of model specifications.
    """
    if isinstance(models, (str, object)) and not isinstance(models, list):
        return [models]
    return list(models)


def _normalize_params_input(
    params: Optional[Union[tuple, dict, List[tuple], List[dict]]],
    n_models: int
) -> List[Optional[Union[tuple, dict]]]:
    """
    Normalize parameters input to match number of models.
    
    Parameters
    ----------
    params : tuple, dict, list thereof, or None
        Parameters specification to normalize.
    n_models : int
        Number of models to create parameters for.
        
    Returns
    -------
    List[Optional[Union[tuple, dict]]]
        List of parameter specifications, one per model.
    """
    if params is None:
        return [None] * n_models
    
    if isinstance(params, (tuple, dict)):
        return [params] + [None] * (n_models - 1)
    
    if isinstance(params, list):
        # Extend with None if list is shorter than n_models
        return params + [None] * max(0, n_models - len(params))
    
    return [None] * n_models


def _setup_distributions(
    models: List[Union[str, object]],
    data: np.ndarray,
    params_list: List[Optional[Union[tuple, dict]]]
) -> List[Tuple[object, tuple]]:
    """
    Setup distribution objects and their fitted parameters.
    
    Parameters
    ----------
    models : List[Union[str, object]]
        List of distribution specifications.
    data : np.ndarray
        Empirical data for parameter estimation.
    params_list : List[Optional[Union[tuple, dict]]]
        List of parameter specifications.
        
    Returns
    -------
    List[Tuple[object, tuple]]
        List of (distribution_object, fitted_parameters) tuples.
        
    Raises
    ------
    ValueError
        If unsupported distribution model is specified.
    """
    distributions = []
    
    for model, params in zip(models, params_list):
        distribution, fitted_params = _setup_single_distribution(model, data, params)
        distributions.append((distribution, fitted_params))
    
    return distributions


def _setup_single_distribution(
    model: Union[str, object],
    data: np.ndarray,
    dist_params: Optional[Union[tuple, dict]]
) -> Tuple[object, tuple]:
    """
    Setup a single distribution with parameters.
    
    Parameters
    ----------
    model : str or object
        Distribution specification.
    data : np.ndarray
        Data for parameter estimation if needed.
    dist_params : tuple, dict, or None
        Pre-specified parameters or None for estimation.
        
    Returns
    -------
    Tuple[object, tuple]
        Distribution object and fitted parameters.
        
    Raises
    ------
    ValueError
        If unsupported distribution model or invalid parameters.
    """
    if isinstance(model, str):
        model_lower = model.lower()
        
        if model_lower not in SUPPORTED_DISTRIBUTIONS:
            supported = ', '.join(SUPPORTED_DISTRIBUTIONS.keys())
            raise ValueError(
                f"Unsupported string model: '{model}'. "
                f"Currently supported: {supported}"
            )
        
        distribution = SUPPORTED_DISTRIBUTIONS[model_lower]
        
        if dist_params is None:
            # Estimate parameters from data
            fitted_params = _estimate_distribution_parameters(distribution, data)
        else:
            fitted_params = _validate_distribution_parameters(
                distribution, dist_params, model_lower
            )
    
    elif hasattr(model, 'ppf') and hasattr(model, 'cdf'):
        # Custom scipy.stats distribution
        distribution = model
        
        if dist_params is None:
            raise ValueError("Custom distributions require explicit dist_params")
        
        if isinstance(dist_params, dict):
            fitted_params = tuple(dist_params.values())
        else:
            fitted_params = tuple(dist_params)
    
    else:
        raise ValueError(
            "Model must be a string (e.g., 'normal') or scipy.stats distribution object"
        )
    
    return distribution, fitted_params


def _estimate_distribution_parameters(
    distribution: object,
    data: np.ndarray
) -> tuple:
    """
    Estimate distribution parameters from data.
    
    Parameters
    ----------
    distribution : object
        Scipy.stats distribution object.
    data : np.ndarray
        Data to estimate parameters from.
        
    Returns
    -------
    tuple
        Estimated parameters.
    """
    dist_name = getattr(distribution, 'name', '')
    
    if dist_name == 'norm':
        return (np.mean(data), np.std(data, ddof=1))
    elif dist_name in ['lognorm', 'weibull_min']:
        return distribution.fit(data, floc=0)
    elif dist_name == 'gumbel_r':
        return distribution.fit(data)
    elif dist_name in ['expon', 'gamma']:
        return distribution.fit(data, floc=0)
    else:
        # General case: use scipy's fit method
        return distribution.fit(data)


def _validate_distribution_parameters(
    distribution: object,
    params: Union[tuple, dict],
    model_name: str
) -> tuple:
    """
    Validate and convert distribution parameters to tuple format.
    
    Parameters
    ----------
    distribution : object
        Scipy.stats distribution object.
    params : tuple or dict
        Parameters to validate.
    model_name : str
        Name of the distribution model for error messages.
        
    Returns
    -------
    tuple
        Validated parameters in tuple format.
        
    Raises
    ------
    ValueError
        If parameters have wrong format or count for the distribution.
    """
    if isinstance(params, dict):
        return tuple(params.values())
    
    params_tuple = tuple(params)
    
    # Validate parameter count for known distributions
    expected_counts = {
        'normal': 2, 'lognormal': 3, 'weibull': 3, 
        'gumbel': 2, 'exponential': 2, 'gamma': 3
    }
    
    if model_name in expected_counts:
        expected = expected_counts[model_name]
        if len(params_tuple) != expected:
            raise ValueError(
                f"{model_name.title()} distribution requires exactly "
                f"{expected} parameters, got {len(params_tuple)}"
            )
    
    return params_tuple

def _create_multi_qq_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple]],
    title: str,
    data_name: str
) -> go.Figure:
    """
    Create Q-Q plot comparing empirical data against multiple theoretical distributions.
    
    Parameters
    ----------
    empirical_data : np.ndarray
        Sorted empirical data.
    distributions : List[Tuple[object, tuple]]
        List of (distribution_object, fitted_parameters) tuples.
    title : str
        Plot title.
    data_name : str
        Name for the empirical data series.
        
    Returns
    -------
    go.Figure
        Plotly Figure with Q-Q plot for all distributions.
    """
    fig = go.Figure()
    n = len(empirical_data)
    
    # Calculate plotting probabilities (matching R's ppoints with a=0.5)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    
    for i, (distribution, params) in enumerate(distributions):
        # Calculate theoretical quantiles
        theoretical_quantiles = distribution.ppf(probabilities, *params)
        
        if not np.all(np.isfinite(theoretical_quantiles)):
            st.warning(f"Non-finite theoretical quantiles for distribution {i+1}")
            continue
        
        # Get distribution name for legend
        dist_name = getattr(distribution, 'name', f'Distribution {i+1}')
        color = DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=empirical_data,
            mode='markers',
            name=f'{dist_name.title()}',
            marker=dict(size=6, opacity=0.7, color=color),
            hovertemplate=(
                f"<b>{dist_name.title()}</b><br>"
                "Theoretical: %{x:.3f}<br>"
                "Empirical: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add identity line for first distribution only
        
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=theoretical_quantiles,
        mode='lines',
        name='y = x',
        line=dict(dash='dash', color='black', width=1.5),
        hoverinfo='skip'
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title='Theoretical Quantiles',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Empirical Quantiles',
            showgrid=True,
            gridcolor='lightgray'
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


def _create_multi_pp_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple]],
    title: str,
    data_name: str
) -> go.Figure:
    """
    Create P-P plot comparing empirical and theoretical probabilities.
    
    Parameters
    ----------
    empirical_data : np.ndarray
        Sorted empirical data.
    distributions : List[Tuple[object, tuple]]
        List of (distribution_object, fitted_parameters) tuples.
    title : str
        Plot title.
    data_name : str
        Name for the empirical data series.
        
    Returns
    -------
    go.Figure
        Plotly Figure with P-P plot for all distributions.
    """
    fig = go.Figure()
    n = len(empirical_data)
    
    # Empirical probabilities
    empirical_probs = np.arange(1, n + 1) / n
    
    for i, (distribution, params) in enumerate(distributions):
        # Calculate theoretical probabilities for empirical data
        theoretical_probs = distribution.cdf(empirical_data, *params)
        
        # Get distribution name and color
        dist_name = getattr(distribution, 'name', f'Distribution {i+1}')
        color = DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=theoretical_probs,
            y=empirical_probs,
            mode='markers',
            name=f'{dist_name.title()}',
            marker=dict(size=6, opacity=0.7, color=color),
            hovertemplate=(
                f"<b>{dist_name.title()}</b><br>"
                "Theoretical Prob: %{x:.3f}<br>"
                "Empirical Prob: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))
    
    # Add identity line
    identity_line = np.linspace(0, 1, 100)
    fig.add_trace(go.Scatter(
        x=identity_line,
        y=identity_line,
        mode='lines',
        name='y = x',
        line=dict(dash='dash', color='black', width=1.5),
        hoverinfo='skip'
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title='Theoretical Probabilities',
            showgrid=True,
            gridcolor='lightgray',
            range=[0, 1]
        ),
        yaxis=dict(
            title='Empirical Probabilities',
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


def _create_multi_cdf_plot(
    empirical_data: np.ndarray,
    distributions: List[Tuple[object, tuple]],
    title: str,
    data_name: str
) -> go.Figure:
    """
    Create CDF comparison plot with empirical vs multiple theoretical CDFs.
    
    Parameters
    ----------
    empirical_data : np.ndarray
        Sorted empirical data.
    distributions : List[Tuple[object, tuple]]
        List of (distribution_object, fitted_parameters) tuples.
    title : str
        Plot title.
    data_name : str
        Name for the empirical data series.
        
    Returns
    -------
    go.Figure
        Plotly Figure with CDF comparison for all distributions.
    """
    fig = go.Figure()
    n = len(empirical_data)
    
    # Calculate empirical CDF
    empirical_cdf = np.arange(1, n + 1) / n
    
    # Add empirical CDF
    fig.add_trace(go.Scatter(
        x=empirical_data,
        y=empirical_cdf,
        mode='markers+lines',
        name=f'{data_name} (Empirical)',
        line=dict(shape='hv', color='black', width=2),
        marker=dict(size=4, opacity=0.7, color='black'),
        hovertemplate=(
            f"<b>{data_name} Empirical CDF</b><br>"
            "Value: %{x:.3f}<br>"
            "CDF: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Create extended range for smooth theoretical curves
    x_min, x_max = empirical_data.min(), empirical_data.max()
    x_range = np.linspace(x_min, x_max, 200)
    
    # Add theoretical CDFs
    for i, (distribution, params) in enumerate(distributions):
        theoretical_cdf = distribution.cdf(x_range, *params)
        
        # Get distribution name and color
        dist_name = getattr(distribution, 'name', f'Distribution {i+1}')
        color = DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]
        param_str = ', '.join([f'{p:.3f}' for p in params])
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=theoretical_cdf,
            mode='lines',
            name=f'{dist_name.title()} (Theoretical)',
            line=dict(color=color, width=3),
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
        title=dict(text=title, x=0.5, font=dict(size=16)),
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


# Legacy function name for backward compatibility
def _create_qq_plot(
    theoretical_quantiles: np.ndarray,
    empirical_data: np.ndarray,
    reference_line: np.ndarray,
    title: str,
    name: str,
    x_label: str = 'Theoretical Quantiles',
    y_label: str = 'Empirical Quantiles'
) -> go.Figure:
    """
    Create a single Q-Q plot (legacy function for backward compatibility).
    
    Parameters
    ----------
    theoretical_quantiles : np.ndarray
        X-axis values (theoretical quantiles from the model).
    empirical_data : np.ndarray
        Y-axis values (the sorted raw data).
    reference_line : np.ndarray
        Values for the y=x reference line.
    title : str
        Plot title.
    name : str
        Data series name for legend.
    x_label : str, default='Theoretical Quantiles'
        X-axis label.
    y_label : str, default='Empirical Quantiles'
        Y-axis label.
        
    Returns
    -------
    go.Figure
        Plotly Figure with Q-Q plot.
    """
    fig = go.Figure()

    # Scatter plot of data points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=empirical_data,
        mode='markers',
        name=name,
        marker=dict(size=6, opacity=0.7, color='dodgerblue'),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "Theoretical: %{x:.3f}<br>"
            "Empirical: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))

    # Identity line y = x
    fig.add_trace(go.Scatter(
        x=reference_line,
        y=reference_line,
        mode='lines',
        name='y = x',
        line=dict(dash='dash', color='red', width=1.5),
        hoverinfo='skip'
    ))

    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title=x_label, showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='lightgray'),
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


def _create_histogram_plot(
    data: np.ndarray, 
    distribution: object, 
    fitted_params: tuple,
    title: str, 
    name: str
) -> go.Figure:
    """
    Create a histogram with overlaid theoretical distribution curve.
    
    Parameters
    ----------
    data : np.ndarray
        Original data array.
    distribution : object
        Scipy.stats distribution object.
    fitted_params : tuple
        Fitted distribution parameters.
    title : str
        Plot title.
    name : str
        Data series name.
        
    Returns
    -------
    go.Figure
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
        line=dict(color='red', width=3),
        hovertemplate=(
            f"<b>{dist_name.title()} Distribution</b><br>"
            "Value: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title='Value', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Probability Density', showgrid=True, gridcolor='lightgray'),
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


def _create_multi_histogram_plot(
    data: np.ndarray,
    distributions: List[Tuple[object, tuple]],
    title: str,
    name: str
) -> go.Figure:
    """
    Create a histogram with multiple overlaid theoretical distribution curves.
    
    Parameters
    ----------
    data : np.ndarray
        Original data array.
    distributions : List[Tuple[object, tuple]]
        List of (distribution_object, fitted_parameters) tuples.
    title : str
        Plot title.
    name : str
        Data series name.
        
    Returns
    -------
    go.Figure
        Configured Plotly Figure object with histogram and multiple fitted curves.
    """
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=75,
        histnorm='probability density',
        name=f'{name} (Empirical)',
        marker=dict(
            color='gray',
            opacity=0.7,
        ),
        hovertemplate=(
            "Bin Center: %{x:.3f}<br>"
            "Density: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    
    # Create extended range for smooth theoretical curves
    x_range = np.linspace(data.min(), data.max(), 200)
    
    # Add theoretical distribution curves
    for i, (distribution, fitted_params) in enumerate(distributions):
        theoretical_pdf = distribution.pdf(x_range, *fitted_params)
        
        # Get distribution name and color
        dist_name = getattr(distribution, 'name', f'Distribution {i+1}')
        color = DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]
        param_str = ', '.join([f'{p:.3f}' for p in fitted_params])
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=theoretical_pdf,
            mode='lines',
            name=f'{dist_name.title()}',
            line=dict(color=color, width=3),
            hovertemplate=(
                f"<b>{dist_name.title()} Distribution</b><br>"
                "Value: %{x:.3f}<br>"
                "Density: %{y:.3f}<br>"
                f"Parameters: {param_str}<br>"
                "<extra></extra>"
            )
        ))
    
    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title='Value', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Probability Density', showgrid=True, gridcolor='lightgray'),
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













