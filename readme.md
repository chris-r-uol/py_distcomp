
# PyDistComp: Python Distribution Comparison Tool

A professional Python library for comprehensive statistical distribution comparison and visualization. PyDistComp provides advanced Q-Q plots, P-P plots, CDF comparisons, histogram overlays, Cullen and Frey plots, and empirical distribution analysis to help analysts and researchers evaluate how well theoretical distributions fit their empirical data.

## 🌟 Features

- **Multi-Distribution Comparison**: Compare your data against multiple theoretical distributions simultaneously
- **Interactive Visualizations**: Professional-quality plots using Plotly with hover information and zoom capabilities
- **Comprehensive Distribution Support**: 15+ built-in distributions including Normal, Log-normal, Weibull, Gamma, Beta, and more
- **Custom Distribution Support**: Use any scipy.stats distribution object
- **Multiple Plot Types**: Q-Q plots, P-P plots, CDF comparisons, histogram overlays, Cullen and Frey plots
- **Empirical Data Analysis**: Dedicated empirical CDF and density plots with kernel density estimation
- **Statistical Diagnostic Tools**: Cullen and Frey plots for distribution family identification with bootstrap confidence regions
- **Statistical Analysis**: Automatic parameter estimation with support for custom parameters
- **Streamlit Demo App**: Interactive web application for exploring functionality with real-time parameter adjustment
- **Professional Documentation**: Comprehensive docstrings and type hints

## 📊 Supported Distributions

| Distribution | String Key | Parameters |
|-------------|------------|------------|
| Normal | `'normal'` | mean, std |
| Log-normal | `'lognormal'` | shape, loc, scale |
| Weibull | `'weibull'` | shape, loc, scale |
| Gumbel | `'gumbel'` | loc, scale |
| Exponential | `'exponential'` | loc, scale |
| Gamma | `'gamma'` | shape, loc, scale |
| Beta | `'beta'` | a, b, loc, scale |
| Uniform | `'uniform'` | loc, scale |
| Logistic | `'logistic'` | loc, scale |
| Laplace | `'laplace'` | loc, scale |
| Chi-squared | `'chi2'` | df, loc, scale |
| Student's t | `'student_t'` | df, loc, scale |
| F-distribution | `'f'` | dfn, dfd, loc, scale |
| Pareto | `'pareto'` | b, loc, scale |
| Rayleigh | `'rayleigh'` | loc, scale |

## 🚀 Installation

### Install from GitHub

Since this package is not yet published to PyPI, install directly from GitHub:

```bash
pip install git+https://github.com/chris-r-uol/py_distcomp.git
```

### Development Installation

For development or to run the demo app:

```bash
git clone https://github.com/chris-r-uol/py_distcomp.git
cd py_distcomp
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- pandas
- plotly
- scipy
- streamlit (for demo app)

## 📖 Quick Start

### Basic Usage

```python
import numpy as np
from py_distcomp import quantile_comparison_plot, cullen_and_frey_plot, empirical_cdf_plot, empirical_density_plot

# Generate sample data
data = np.random.normal(0, 1, 1000)

# Single distribution comparison
qq_fig = quantile_comparison_plot(data, models='normal')
qq_fig.show()

# Multiple distribution comparison
qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data, 
    models=['normal', 'lognormal', 'weibull'],
    title='Distribution Comparison',
    data_name='Sample Data'
)

# Display all plots
qq_fig.show()      # Q-Q plot
hist_fig.show()    # Histogram with fitted curve
pp_fig.show()      # P-P plot
cdf_fig.show()     # CDF comparison

# Cullen and Frey plot for distribution family identification
cf_fig = cullen_and_frey_plot(data, title='Distribution Assessment')
cf_fig.show()

# Empirical data analysis
emp_cdf_fig = empirical_cdf_plot(data, name='Sample Data')
emp_density_fig = empirical_density_plot(data, name='Sample Data')
emp_cdf_fig.show()
emp_density_fig.show()
```

### Advanced Usage

```python
from scipy import stats
import pandas as pd
from py_distcomp import quantile_comparison_plot, cullen_and_frey_plot

# Load your data
data = pd.read_csv('your_data.csv')['column_name']

# Compare against multiple distributions with custom parameters
models = ['normal', 'weibull', stats.gamma]
params = [
    (0, 1),           # Normal: mean=0, std=1
    (2, 0, 1),        # Weibull: shape=2, loc=0, scale=1
    (2, 0, 1)         # Gamma: shape=2, loc=0, scale=1
]

qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data=data,
    models=models,
    dist_params=params,
    title='Custom Distribution Analysis',
    data_name='My Data'
)

# Use Cullen and Frey plot to identify distribution families
cf_fig = cullen_and_frey_plot(
    data=data,
    title='Distribution Family Assessment',
    data_name='My Data',
    n_bootstrap=200,
    show_bootstrap=True,
    show_theoretical=True
)
```

## 🎛️ Demo Application

Run the interactive Streamlit demo to explore functionality:

```bash
streamlit run app.py
```

The demo app provides:
- Interactive data generation with various distributions
- Real-time parameter adjustment
- Multiple distribution comparison
- Cullen and Frey plot for distribution assessment
- Empirical data visualization with CDF and density plots
- Export capabilities for generated plots

## 📚 API Reference

### Main Functions

#### `quantile_comparison_plot`

```python
def quantile_comparison_plot(
    data: Union[pd.Series, np.ndarray, list],
    models: Union[str, List[str], object, List[object]] = 'normal',
    title: str = 'Q-Q Plot',
    data_name: str = 'Data',
    dist_params: Optional[Union[tuple, dict, List[tuple], List[dict]]] = None,
    include_histogram: bool = True
) -> Union[go.Figure, Tuple[go.Figure, ...]]
```

**Parameters:**
- `data`: Input data as pandas Series, numpy array, or list
- `models`: Distribution model(s) to compare against
- `title`: Plot title
- `data_name`: Name for the empirical data series
- `dist_params`: Distribution parameters (None for auto-estimation)
- `include_histogram`: Whether to include additional plots

**Returns:**
- Single figure (if `include_histogram=False`)
- Tuple of figures: (Q-Q plot, histogram, P-P plot, CDF plot)

#### `cullen_and_frey_plot`

```python
def cullen_and_frey_plot(
    data: Union[pd.Series, np.ndarray, list],
    title: str = 'Cullen and Frey Graph',
    data_name: str = 'Data',
    n_bootstrap: int = 100,
    show_bootstrap: bool = True,
    show_theoretical: bool = True
) -> go.Figure
```

A Cullen and Frey plot for distribution family identification based on sample skewness and kurtosis.

**Parameters:**
- `data`: Input data
- `title`: Plot title
- `data_name`: Name for the empirical data point
- `n_bootstrap`: Number of bootstrap samples for confidence region
- `show_bootstrap`: Whether to show bootstrap confidence cloud
- `show_theoretical`: Whether to show theoretical distribution regions

#### `empirical_cdf_plot`

```python
def empirical_cdf_plot(
    data: Union[np.ndarray, pd.Series, List[float]], 
    name: str = "Data",
    color: str = 'seagreen',
    width: int = 700,
    height: int = 500,
    show_percentiles: bool = True,
    percentile_lines: Optional[List[float]] = None,
    show_annotations: bool = True
) -> go.Figure
```

Create an empirical cumulative distribution function plot with optional percentile markers.

#### `empirical_density_plot`

```python
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
) -> go.Figure
```

Create an empirical density plot combining histogram with kernel density estimation.

## 📈 Plot Types

### Q-Q Plot (Quantile-Quantile)
Compares quantiles of your data against theoretical distribution quantiles. Points falling on the diagonal line indicate good fit.

### P-P Plot (Probability-Probability)  
Compares cumulative probabilities. More sensitive to differences in the center of the distribution.

### CDF Comparison
Shows empirical vs theoretical cumulative distribution functions. Good for visualizing overall distribution shape.

### Histogram with Fitted Curves
Overlays theoretical probability density functions on your data histogram, with statistical markers.

### Cullen and Frey Plot
Statistical diagnostic chart plotting sample skewness² vs kurtosis to identify which distribution families are most appropriate for your data. Includes theoretical regions for common distributions and optional bootstrap confidence regions.

### Empirical CDF Plot
Visualizes the empirical cumulative distribution function as a step function, with optional percentile markers and annotations.

### Empirical Density Plot
Combines histogram representation with smooth kernel density estimation to show the empirical probability density function.

## 🎯 Use Cases

- **Quality Control**: Assess if manufacturing data follows expected distributions
- **Risk Analysis**: Validate assumptions about return distributions in finance
- **Reliability Engineering**: Test if failure times follow Weibull or exponential distributions
- **Environmental Science**: Analyze if measurements follow normal or log-normal distributions
- **Research**: Validate distributional assumptions before statistical modeling
- **Exploratory Data Analysis**: Use Cullen and Frey plots to identify candidate distribution families
- **Data Preprocessing**: Visualize empirical distributions before transformation or modeling

## 🔧 Examples

### Example 1: Financial Returns Analysis

```python
import yfinance as yf
from py_distcomp import quantile_comparison_plot, cullen_and_frey_plot

# Download stock data
stock = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
returns = stock['Close'].pct_change().dropna()

# First, identify appropriate distribution families
cf_fig = cullen_and_frey_plot(
    data=returns,
    title='Stock Return Distribution Assessment',
    data_name='AAPL Returns'
)

# Compare against common financial distributions
qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data=returns,
    models=['normal', 'student_t', 'laplace'],
    title='Stock Return Distribution Analysis',
    data_name='AAPL Returns'
)
```

### Example 2: Manufacturing Quality Control

```python
# Simulate manufacturing measurements
measurements = np.random.normal(100, 2, 500)  # Target: 100mm ± 2mm

# Check if process is in control
qq_fig = quantile_comparison_plot(
    data=measurements,
    models='normal',
    dist_params=(100, 2),  # Expected parameters
    title='Manufacturing Process Control',
    data_name='Part Dimensions'
)
```

### Example 3: Reliability Analysis

```python
# Simulate failure times
failure_times = np.random.weibull(2, 1000) * 100

# Test against reliability distributions
qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data=failure_times,
    models=['weibull', 'exponential', 'gamma'],
    title='Component Reliability Analysis',
    data_name='Time to Failure'
)
```

### Example 4: Exploratory Data Analysis Workflow

```python
import numpy as np
from py_distcomp import quantile_comparison_plot, cullen_and_frey_plot, empirical_cdf_plot, empirical_density_plot

# Generate mixed data for demonstration
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 800),
    np.random.exponential(1, 200)
])

# Step 1: Visualize empirical distribution
emp_density_fig = empirical_density_plot(data, name='Mixed Data')
emp_cdf_fig = empirical_cdf_plot(data, name='Mixed Data')

# Step 2: Use Cullen and Frey plot for distribution family identification
cf_fig = cullen_and_frey_plot(
    data=data,
    title='Distribution Family Assessment',
    data_name='Mixed Data',
    n_bootstrap=200
)

# Step 3: Compare against candidate distributions
qq_fig, hist_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data=data,
    models=['normal', 'lognormal', 'weibull', 'gamma'],
    title='Distribution Comparison Analysis',
    data_name='Mixed Data'
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Chris Rushton** - University of Leeds  
GitHub: [@chris-r-uol](https://github.com/chris-r-uol)

## 🙏 Acknowledgments

- Built with [Plotly](https://plotly.com/python/) for interactive visualizations
- Statistical distributions provided by [SciPy](https://scipy.org/)
- Inspired by R's `fitdistrplus` package: [FitDistrPlus](https://cran.r-project.org/web/packages/fitdistrplus/index.html)
- Demo app powered by [Streamlit](https://streamlit.io/)

## 📊 Roadmap

- [x] **Multi-distribution comparison with Q-Q, P-P, and CDF plots**
- [x] **Cullen and Frey plots for distribution family identification**
- [x] **Empirical CDF and density plots with KDE**
- [x] **Interactive Streamlit demo application**
- [x] **Bootstrap confidence regions for Cullen and Frey plots**
- [ ] Add more distribution types (mixture models, custom distributions)
- [ ] Implement goodness-of-fit statistics (KS test, Anderson-Darling, etc.)
- [ ] Add confidence bands for Q-Q plots
- [ ] Support for censored data analysis
- [ ] Integration with statistical testing frameworks
- [ ] Publication to PyPI
- [ ] R package integration

---

⭐ **Star this repository if you find it useful!** ⭐