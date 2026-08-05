
# PyDistComp: Python Distribution Comparison Tool

A Python port of the distribution-fitting and comparison workflow provided by R's
[**fitdistrplus**](https://lbbe-software.github.io/fitdistrplus/) (Delignette-Muller & Dutang).
PyDistComp provides Q-Q plots, P-P plots, CDF comparisons, histogram overlays, Cullen and Frey
graphs, maximum-likelihood fitting and goodness-of-fit statistics, so that analysis begun in R can
be continued in Python without changing the numbers.

**fitdistrplus is the reference implementation.** Plotting positions, moment estimators, parameter
estimates, axis limits, test statistics and critical values follow the R package. Where a choice had
to be made, R's behaviour was taken. The test suite checks the outputs against the values printed in
the fitdistrplus vignette, using the package's own `groundbeef` dataset.

### Mapping to fitdistrplus

| PyDistComp | fitdistrplus |
|---|---|
| `descdist` | `descdist(..., graph = FALSE)` |
| `cullen_and_frey_plot` | `descdist(..., graph = TRUE)` |
| `fit_distributions` | `fitdist(..., method = "mle")` |
| `gofstat` | `gofstat` |
| `quantile_comparison_plot` | `qqcomp` + `denscomp` + `ppcomp` + `cdfcomp` |
| `empirical_cdf_plot`, `empirical_density_plot` | `plotdist` (data only) |

## 🌟 Features

- **Multi-Distribution Comparison**: Compare your data against multiple theoretical distributions simultaneously
- **Maximum-Likelihood Fitting**: Parameter estimation matching `fitdist(..., method = "mle")`, reported in R's parameterisation
- **Goodness-of-Fit Statistics**: Kolmogorov-Smirnov, Cramér-von Mises, Anderson-Darling and chi-squared statistics, with R's test decisions, plus AIC and BIC
- **Interactive Visualizations**: Plotly plots with hover information and zoom
- **Comprehensive Distribution Support**: 16 built-in distributions
- **Custom Distribution Support**: Use any scipy.stats distribution object
- **Multiple Plot Types**: Q-Q plots, P-P plots, CDF comparisons, histogram overlays, Cullen and Frey graphs
- **Empirical Data Analysis**: Dedicated empirical CDF and density plots with kernel density estimation
- **Streamlit Demo App**: Interactive web application for exploring functionality
- **No streamlit dependency**: The library itself needs only numpy, pandas, scipy and plotly

## 📊 Supported Distributions

Parameters are listed as **R** reports them, which is what `fit_distributions` returns in
`FitResult.estimate`. Several R densities have no location or scale argument at all, so PyDistComp
pins scipy's `loc`/`scale` accordingly rather than silently estimating a larger model — this is what
makes the estimates, log-likelihoods and AIC/BIC values agree with R.

| Distribution | String Key | R name | Estimated parameters | Pinned in scipy |
|---|---|---|---|---|
| Normal | `'normal'` | `norm` | mean, sd | — |
| Log-normal | `'lognormal'` | `lnorm` | meanlog, sdlog | `loc = 0` |
| Weibull | `'weibull'` | `weibull` | shape, scale | `loc = 0` |
| Gamma | `'gamma'` | `gamma` | shape, rate | `loc = 0` |
| Exponential | `'exponential'` | `exp` | rate | `loc = 0` |
| Uniform | `'uniform'` | `unif` | min, max | — |
| Logistic | `'logistic'` | `logis` | location, scale | — |
| Beta | `'beta'` | `beta` | shape1, shape2 | `loc = 0`, `scale = 1` |
| Cauchy | `'cauchy'` | `cauchy` | location, scale | — |
| Gumbel | `'gumbel'` | `gumbel` | loc, scale | — |
| Laplace | `'laplace'` | `laplace` | location, scale | — |
| Pareto | `'pareto'` | `pareto` | shape, scale | `loc = 0` |
| Rayleigh | `'rayleigh'` | `rayleigh` | scale | `loc = 0` |
| Chi-squared | `'chi2'` | `chisq` | df | `loc = 0`, `scale = 1` |
| Student's t | `'student_t'` | `t` | df | `loc = 0`, `scale = 1` |
| F-distribution | `'f'` | `f` | df1, df2 | `loc = 0`, `scale = 1` |

R's own names are accepted as aliases, so `'lnorm'`, `'exp'`, `'norm'` and `'unif'` work too. As in
R, `beta` is defined on [0, 1] and raises an error for data outside that range.

Passing `dist_params` explicitly bypasses fitting; those must be the **full scipy** tuple
(shapes, `loc`, `scale`), e.g. `(2.0, 0.0, 83.0)` for a Weibull.

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

`streamlit` is needed only for the demo app: `pip install py_distcomp[app]`.

## 📖 Quick Start

The workflow mirrors the one in the fitdistrplus vignette: describe the data, pick candidate
families from the Cullen and Frey graph, fit them, then compare fits both graphically and
numerically.

```python
import numpy as np
from py_distcomp import (
    descdist, cullen_and_frey_plot, fit_distributions, gofstat,
    quantile_comparison_plot, empirical_cdf_plot, empirical_density_plot,
)

data = np.random.default_rng(0).gamma(shape=3, scale=2, size=500)

# 1. Describe the data -- R's descdist()
descdist(data)
# {'min': 0.42, 'max': 22.8, 'median': 5.28, 'mean': 6.03,
#  'sd': 3.51, 'skewness': 1.10, 'kurtosis': 4.65, 'method': 'unbiased'}

# 2. Identify candidate families -- R's descdist(graph = TRUE)
cullen_and_frey_plot(data, seed=42).show()

# 3. Fit by maximum likelihood -- R's fitdist(..., method = "mle")
fits = fit_distributions(data, ['gamma', 'lognormal', 'weibull'])
for fit in fits:
    print(fit.r_name, fit.estimate, round(fit.aic, 1))

# 4. Compare graphically -- R's qqcomp / denscomp / ppcomp / cdfcomp
qq_fig, dens_fig, pp_fig, cdf_fig = quantile_comparison_plot(
    data, models=['gamma', 'lognormal', 'weibull'],
)
qq_fig.show()

# 5. Compare numerically -- R's gofstat()
gofstat(fits)
#            ks       ks_test   cvm  cvm_test    ad   ad_test  ...  aic   bic
# gamma      0.021  not rejected  ...
```

### Empirical data on its own

```python
empirical_cdf_plot(data, name='Sample Data').show()      # step ECDF at ppoints(n)
empirical_density_plot(data, name='Sample Data').show()  # histogram + KDE
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
    title: str = 'Q-Q plot',
    data_name: str = 'Data',
    dist_params: Optional[Union[tuple, dict, List[tuple], List[dict]]] = None,
    include_histogram: bool = True,
    a_ppoints: float = 0.5,
    ynoise: bool = False,
    bins: Union[int, str, Sequence[float]] = 'sturges',
    fitnbpts: int = 101,
    seed: Optional[int] = None,
) -> Union[go.Figure, Tuple[go.Figure, ...]]
```

**Parameters:**
- `data`: Input data as pandas Series, numpy array, or list
- `models`: Distribution model(s) to compare against
- `title`: Title of the Q-Q plot
- `data_name`: Name for the empirical data series
- `dist_params`: Full scipy parameter tuples (None for maximum-likelihood estimation)
- `include_histogram`: Whether to return the density, P-P and CDF plots too
- `a_ppoints`: Offset for the plotting positions `(i - a) / (n + 1 - 2a)`; R's `*comp` functions use 0.5
- `ynoise`: Jitter later series by `U(-0.02, 0.02)`, as R does to separate overlapping fits. R defaults this to `TRUE`; it is `False` here because the series are separable interactively. Hover always reports the un-jittered value.
- `bins`: Histogram binning for the density plot; Sturges' rule is also R's `hist()` default
- `fitnbpts`: Points used to draw fitted curves, as R's `fitnbpts`
- `seed`: Seed for the `ynoise` jitter

**Returns:**
- Single figure (if `include_histogram=False`)
- Tuple of figures: (Q-Q plot, density plot, P-P plot, CDF plot)

#### `descdist`

```python
def descdist(
    data: Union[pd.Series, np.ndarray, list],
    method: str = 'unbiased',
) -> Dict[str, Any]
```

The non-graphical half of R's `descdist`. Returns `min`, `max`, `median`, `mean`, `sd`, `skewness`,
`kurtosis` and `method`. Skewness and kurtosis use the Fisher (1930) corrections by default, as R
does; `kurtosis` is **not** excess kurtosis, so a normal distribution gives 3. At least four values
are required, as in R.

#### `cullen_and_frey_plot`

```python
def cullen_and_frey_plot(
    data: Union[pd.Series, np.ndarray, list],
    title: str = 'Cullen and Frey graph',
    data_name: str = 'Data',
    discrete: bool = False,
    method: str = 'unbiased',
    n_bootstrap: int = 100,
    show_bootstrap: bool = True,
    show_theoretical: bool = True,
    seed: Optional[int] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure
```

Square of skewness against kurtosis, with the kurtosis axis inverted, against the regions occupied
by common distribution families — the graph R's `descdist` draws.

**Parameters:**
- `data`: Input data; at least four values, as in R
- `title`, `data_name`: Plot title and legend label for the observed point
- `discrete`: Draw the discrete overlays (negative binomial region, Poisson line) instead of the continuous ones (beta region, gamma and lognormal lines), as R's `discrete` argument
- `method`: `'unbiased'` (R's default) or `'sample'` moment estimators
- `n_bootstrap`: Bootstrap resamples, R's `boot`; at least 10
- `show_bootstrap`: Whether to draw the bootstrap cloud. As in R, the bootstrap also widens the axis limits to cover the resampled points
- `show_theoretical`: Whether to draw the theoretical points, curves and regions
- `seed`: Seed for the bootstrap. An isolated generator is used, so the global numpy random state is untouched

Axis limits follow R: `xmax = max(4, ceiling(skewness²))` and `kurtmax = max(10, ceiling(kurtosis))`,
taken over the bootstrap sample when there is one.

#### `fit_distributions`

```python
def fit_distributions(
    data: Union[np.ndarray, pd.Series, list],
    models: Union[str, object, Sequence] = 'normal',
) -> List[FitResult]
```

Maximum-likelihood fitting, the equivalent of one `fitdist(data, distname)` call per distribution.
Each `FitResult` carries `estimate` (R's parameterisation), `params` (scipy's), `loglik`, `aic`,
`bic`, `n` and `n_free_params`.

#### `gofstat`

```python
def gofstat(
    fits: Union[FitResult, Sequence[FitResult]],
    chisqbreaks: Optional[Sequence[float]] = None,
    meancount: Optional[int] = None,
) -> pd.DataFrame
```

Goodness-of-fit statistics, a port of R's `gofstat`. Returns a DataFrame with one row per fit:
`ks`, `cvm`, `ad` and their test decisions, `chisq` with `chisq_df` and `chisq_pvalue`, and
`loglik`, `aic`, `bic`.

Test columns carry the same strings R prints: `'rejected'`, `'not rejected'` or `'not computed'`.
As in fitdistrplus, only the exponential, gamma, Weibull and logistic distributions have tabulated
critical values, and the Kolmogorov-Smirnov decision needs n ≥ 30; everything else reports
`'not computed'`. Chi-squared cell boundaries default to R's rule, targeting
`round(n / (4n)^(2/5))` observations per cell.

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
    models=['normal', 'laplace'],
    title='Stock Return Distribution Analysis',
    data_name='AAPL Returns'
)
```

> **Note on `'student_t'`, `'chi2'` and `'f'`.** R's `dt`, `dchisq` and `df` take only degrees of
> freedom, so PyDistComp pins scipy's `loc = 0` and `scale = 1` to match. That makes them a poor fit
> for data that is not already standardised, such as returns. To scale a t-distribution, pass the
> scipy object with explicit parameters instead — this is outside what fitdistrplus does:
>
> ```python
> from scipy import stats
> quantile_comparison_plot(returns, stats.t, dist_params=(4, 0.0, 0.02))
> ```

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
- [x] **Goodness-of-fit statistics (KS, Cramér-von Mises, Anderson-Darling, chi-squared, AIC, BIC)**
- [x] **Maximum-likelihood fitting in R's parameterisation**
- [ ] Standard errors of the estimates (R's `fitdist` reports these from the Hessian)
- [ ] Discrete distributions (Poisson, negative binomial, geometric) for fitting as well as plotting
- [ ] Other estimation methods: moment matching (`mmedist`), quantile matching (`qmedist`), maximum goodness-of-fit (`mgedist`)
- [ ] Bootstrap of the fitted parameters (`bootdist`)
- [ ] Add more distribution types (mixture models, custom distributions)
- [ ] Add confidence bands for Q-Q plots
- [ ] Support for censored data analysis
- [ ] Integration with statistical testing frameworks
- [ ] Publication to PyPI
- [ ] R package integration

---

⭐ **Star this repository if you find it useful!** ⭐