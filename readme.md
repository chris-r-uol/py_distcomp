
# PyDistComp: Python Distribution Comparison Tool

[![tests](https://github.com/chris-r-uol/py_distcomp/actions/workflows/tests.yml/badge.svg)](https://github.com/chris-r-uol/py_distcomp/actions/workflows/tests.yml)

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
| `fit_distributions` | `fitdist(..., method = "mle"/"mme"/"qme"/"mge")` |
| `moment_match`, `quantile_match`, `maximum_goodness_of_fit` | `mmedist`, `qmedist`, `mgedist` |
| `FitResult.std_error` / `.vcov` / `.correlation` | `fitdist`'s `sd`, `vcov`, `cor` |
| `bootdist` | `bootdist` |
| `gofstat` | `gofstat` |
| `quantile_comparison_plot` | `qqcomp` + `denscomp` + `ppcomp` + `cdfcomp` |
| `empirical_cdf_plot`, `empirical_density_plot` | `plotdist` (data only) |

### Beyond fitdistrplus: the off-model fraction

`off_model_fraction` and its plots implement the method of

> Rushton, C.E., Tate, J.E. and Shepherd, S.P. (2021) "A novel method for comparing passenger car
> fleets and identifying high-chance gross emitting vehicles using kerbside remote sensing data",
> *Science of the Total Environment*, **750**, 142088.
> [doi:10.1016/j.scitotenv.2020.142088](https://doi.org/10.1016/j.scitotenv.2020.142088)

which extends the fitdistrplus workflow rather than reproducing it. See
[Off-model fraction analysis](#-off-model-fraction-analysis) below.

`fit_spliced` is that method's principled successor: rather than searching percentiles for a cut, it
estimates the threshold by likelihood alongside both sides — see
[Spliced distributions](#-spliced-distributions-one-heavier-tail-not-two-populations).

`fit_mixture` estimates the same superposition jointly by expectation-maximisation, which is the
principled version of the paper's percentile cut — see
[Mixture fitting](#-mixture-fitting-superposition-of-distributions).

## 🌟 Features

- **Multi-Distribution Comparison**: Compare your data against multiple theoretical distributions simultaneously
- **Four Estimation Methods**: maximum likelihood, moment matching, quantile matching and maximum goodness-of-fit, matching `fitdist(..., method = ...)` and reported in R's parameterisation
- **Q-Q Confidence Bands**: bootstrap bands showing whether a departure from the line is larger than sampling noise
- **Goodness-of-Fit Statistics**: Kolmogorov-Smirnov, Cramér-von Mises, Anderson-Darling and chi-squared statistics, with R's test decisions, plus AIC and BIC
- **Uncertainty**: standard errors and correlations from the observed information, plus bootstrap confidence intervals on parameters and quantiles
- **Interactive Visualizations**: Plotly plots with hover information and zoom
- **Comprehensive Distribution Support**: 19 built-in distributions, continuous and discrete
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

### Discrete

| Distribution | String Key | R name | Estimated parameters | Pinned in scipy |
|---|---|---|---|---|
| Poisson | `'poisson'` | `pois` | lambda | `loc = 0` |
| Negative binomial | `'negative_binomial'` | `nbinom` | size, mu | `loc = 0` |
| Geometric | `'geometric'` | `geom` | prob | `loc = -1` |

These live on the non-negative integers and raise an error for negative or non-integer data, as R
does. Three scipy differences are handled for you:

- scipy's discrete distributions have **no `fit` method**, so the estimators are implemented here —
  closed forms for the Poisson and geometric, and a one-dimensional search for the negative
  binomial, whose `mu` is the sample mean whatever `size` turns out to be.
- scipy exposes **`logpmf` rather than `logpdf`**, so everything that scores a fit goes through a
  helper that picks the right one.
- **scipy's `geom` starts at 1** where R's `dgeom` starts at 0, hence the `loc = -1`. Without it
  every probability would be shifted by one.

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

To include the demo app:

```bash
pip install "py_distcomp[app] @ git+https://github.com/chris-r-uol/py_distcomp.git"
```

### Development Installation

```bash
git clone https://github.com/chris-r-uol/py_distcomp.git
cd py_distcomp
pip install -e ".[app,dev]"
```

That gives an editable install with streamlit for the demo and pytest for the test suite:

```bash
pytest
```

### Requirements

- Python 3.9+
- numpy
- pandas
- plotly
- scipy

`streamlit` is needed only for the demo app, and is pulled in by the `[app]` extra. The library
itself imports none of it.

## 📖 Quick Start

The workflow mirrors the one in the fitdistrplus vignette: describe the data, pick candidate
families from the Cullen and Frey graph, fit them, then compare fits both graphically and
numerically.

```python
import numpy as np
from py_distcomp import (
    descdist, cullen_and_frey_plot, fit_distributions, gofstat, bootdist,
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

# 6. Put uncertainty on the winner -- R's fitdist std errors and bootdist
fits[0].summary()             # estimate and std_error, as fitdist prints
bootdist(fits[0], seed=1).summary()   # percentile confidence intervals
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

## 🧭 Worked examples

Three complete walkthroughs. Every number and figure below was produced by the code shown, on the
data shown — `python docs/make_figures.py` regenerates all of it.

---

### 1. Which distribution fits? — the fitdistrplus workflow

Using `groundbeef`, the serving-size dataset shipped with fitdistrplus, so these are the numbers the
R vignette prints.

```python
import numpy as np
import py_distcomp as pdc

serving = np.load("tests/groundbeef.npy")     # 254 servings, in grams
```

**Describe the data.** `descdist` gives the same seven summaries R does. Kurtosis is not excess
kurtosis, so a normal distribution would give 3.

```python
pdc.descdist(serving)
# {'min': 10.0, 'max': 200.0, 'median': 79.0, 'mean': 73.646, 'sd': 35.885,
#  'skewness': 0.7353, 'kurtosis': 3.5514, 'method': 'unbiased'}
```

**Pick candidate families.** Right-skewed with kurtosis near 3.5 puts the data in the gamma /
lognormal / Weibull region of the Cullen and Frey graph. The orange cloud is the bootstrap, showing
how far that position could move on a resample.

```python
pdc.cullen_and_frey_plot(serving, data_name="Serving size", seed=42)
```

![Cullen and Frey graph](docs/images/cullen-frey.png)

**Fit them.** Estimates come back in R's parameterisation — a gamma `rate`, not scipy's scale —
beside the standard errors `fitdist` prints.

```python
fits = pdc.fit_distributions(serving, ["weibull", "gamma", "lognormal"])
fits[0].summary()
#        estimate  std_error
# shape    2.1856     0.1046      <- R prints 2.186 (0.1046)
# scale   83.3467     2.5271      <- R prints 83.348 (2.5269)
```

**Compare them graphically.** One call returns all four fitdistrplus comparison plots.

```python
qq, density, pp, cdf = pdc.quantile_comparison_plot(
    serving, ["weibull", "gamma", "lognormal"], data_name="Serving size",
)
```

![Q-Q comparison](docs/images/qq-comparison.png)

![Density comparison](docs/images/density-comparison.png)

![CDF comparison](docs/images/cdf-comparison.png)

The empirical CDF is drawn at `ppoints(n, a = 0.5)`, as `cdfcomp` does, not at `(1:n)/n`.

**And numerically.**

```python
pdc.gofstat(fits)
#                   ks   ks_test     cvm      ad        aic        bic
# weibull       0.1396  rejected  0.6840  3.5731  2514.4494  2521.5241
# gamma         0.1280  rejected  0.6937  3.5671  2511.2502  2518.3249
# lognormal     0.1493  rejected  0.8277  4.5437  2526.6386  2533.7133
```

The gamma wins on AIC — but **all three are rejected**, and a confidence band shows why. 78 of 254
points fall outside the 95% simultaneous band, so the departure is far beyond sampling noise. The
serving sizes are heavily rounded, and no smooth two-parameter density is going to absorb that.

```python
pdc.quantile_comparison_plot(
    serving, "weibull", include_histogram=False, confidence_band=0.95, seed=1,
)
```

![Q-Q plot with a confidence band](docs/images/qq-band.png)

**Put an interval on it.** The bootstrap agrees closely with the Wald intervals here, which is
itself reassurance that both are valid.

```python
boot = pdc.bootdist(fits[0], niter=1001, seed=1)
boot.summary()
#        estimate   median     2.5%    97.5%
# shape    2.1856   2.1936   2.0030   2.4194
# scale   83.3467  83.1504  78.5767  88.3180

boot.quantile_ci([0.5, 0.95])         # intervals on the distribution's own quantiles
#       median    2.5%   97.5%
# 0.50   70.38   65.96   75.17
# 0.95  137.27  128.95  145.75
```

![Bootstrapped parameter cloud](docs/images/bootdist.png)

The diagonal smear says `shape` and `scale` trade off against one another — they cannot be read
independently.

---

### 2. A bulk with a heavy tail — off-model fraction and mixtures

The structure Rushton et al. (2021) describe: most of a fleet well behaved, a small subset emitting
far more. Simulated here so the truth is known — 95% Gumbel(11.2, 8.3), 5% Gumbel(70, 25).

```python
from scipy import stats

rng = np.random.default_rng(0)
emissions = np.concatenate([
    stats.gumbel_r.rvs(11.2, 8.3, size=950, random_state=rng),
    stats.gumbel_r.rvs(70.0, 25.0, size=50, random_state=rng),
])
```

**The published method: cut, refit, and find where the fit is best.**

```python
result = pdc.off_model_fraction(emissions, "gumbel")
result
# OffModelResult(gumbel: P_off=96, fraction=4%, R²=0.9959, n_off_model=40)

result.fit.estimate        # {'loc': 11.81, 'scale': 8.69}   the bulk
result.tail_fit.estimate   # {'loc': 84.93, 'scale': 16.22}  the off-model tail
result.threshold           # 63.65
```

![R-squared sweep](docs/images/r2-sweep.png)

The sweep peaks at the 96th percentile — a 4% off-model fraction against a true 5%. The lowest of
the injected tail overlaps the bulk and is genuinely not separable, so recovery lands just under.

![Off-model superposition](docs/images/off-model-density.png)

**The likelihood-based version.** A mixture estimates weights, both components and every
observation's assignment jointly, so nothing is discarded and each observation gets a *probability*
rather than a label.

```python
mixture = pdc.fit_mixture(emissions, ("gumbel", "gumbel"))
mixture.confint()
#          estimate  std_error   lower   upper
# weight1     0.959      0.008   0.944   0.974
# loc1       11.799      0.308  11.194  12.403      <- truth 11.2
# scale1      8.689      0.245   8.209   9.170      <- truth 8.3
# weight2     0.041      0.008   0.026   0.056      <- truth 0.05
# loc2       82.415      5.726  71.192  93.638      <- truth 70
# scale2     18.898      3.506  12.026  25.770      <- truth 25

mixture.n_starts_at_best, mixture.n_starts_converged   # (9, 9) — every start agreed
mixture.degenerate                                      # False
```

**Read that table honestly.** The weight and the bulk are recovered tightly. The upper component is
not: its interval on `loc2` is 22 units wide and the true 70 sits at its very edge. That is the data,
not the algorithm — a small tail buried under a long-tailed bulk is weakly identified, and the value
of these intervals is that they *say so* rather than leaving you to assume otherwise.

![Mixture density](docs/images/mixture-density.png)

The mixture is also a far better description than any single distribution, and `gofstat` will say so
in one table:

```python
pdc.gofstat(pdc.fit_distributions(emissions, ["gumbel", "normal"]) + [mixture])
#                      ks      cvm       ad        aic
# gumbel           0.0613   1.3736  10.4844  8037.3185
# normal           0.1771  12.2556  71.0721  8744.5995
# gumbel + gumbel  0.0156   0.0380   0.2954  7856.1005      <- ΔAIC 181
```

**Per-observation probabilities** are the replacement for a hard cut:

```python
p = mixture.component_probability()
(p > 0.5).sum(), (p > 0.9).sum()      # (41, 31)
```

![Component probability](docs/images/component-probability.png)

41 observations are more likely than not to belong to the high component; 31 are near-certain. The
curve is the answer to "how likely is *this* vehicle to be a gross emitter", which a percentile cut
cannot give.

---

### 3. Count data

Discrete fits go through the same functions.

```python
counts = np.random.default_rng(1).negative_binomial(4, 0.4, 2000).astype(float)

fits = pdc.fit_distributions(counts, ["poisson", "negative_binomial", "geometric"])
pdc.gofstat(fits)
#                       chisq  chisq_pvalue        aic
# poisson            1727.000         0.000  11600.581
# negative_binomial     6.525         0.769  10600.379      <- the only one that survives
# geometric           675.319         0.000  11424.994

fits[1].estimate      # {'size': 4.18, 'mu': 5.903}   truth: size 4, mu 6
```

![Count density comparison](docs/images/counts-density.png)

Only the chi-squared statistic appears for discrete fits — that is R's behaviour, because the
Kolmogorov-Smirnov, Cramér-von Mises and Anderson-Darling statistics compare an empirical step
function against a *continuous* one, and their critical values do not apply when the fitted
distribution has jumps.

---

### Comparing population subsets

The comparison a table of bare estimates cannot support: whether two fleets actually differ.

```python
fits = {label: pdc.bootdist(pdc.fit_distributions(subset, "gumbel")[0], niter=1001)
        for label, subset in fleet_subsets.items()}
pdc.confint_plot(fits, parameter="loc")
```

![Confidence intervals by subset](docs/images/confint.png)

Non-overlapping intervals; the step change between Euro classes is real rather than sampling noise.

## 🎛️ Demo Application

If you installed the `[app]` extra, the demo has its own command:

```bash
py-distcomp-demo
```

From a checkout, this also works:

```bash
streamlit run app.py
```

The demo app provides:
- Interactive data generation with various distributions
- Real-time parameter adjustment
- Multiple distribution comparison
- Cullen and Frey plot for distribution assessment
- Empirical data visualization with CDF and density plots
- Off-model fraction analysis with the R² sweep and superposition plots
- Mixture fitting with per-observation component probabilities
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

## 🔢 Count data

Discrete fits go through the same functions; pass a discrete name and everything downstream adapts.

```python
from py_distcomp import fit_distributions, gofstat, quantile_comparison_plot

counts = np.random.default_rng(0).negative_binomial(4, 0.4, 2000)

fits = fit_distributions(counts, ['poisson', 'negative_binomial', 'geometric'])
gofstat(fits)
#                     ks  ks_test    chisq  chisq_pvalue      aic
# poisson            NaN  not computed  312.4      0.000    9421.6
# negative_binomial  NaN  not computed    9.1      0.334    9016.3
# geometric          NaN  not computed  180.7      0.000    9284.1

quantile_comparison_plot(counts, ['poisson', 'negative_binomial'])
```

**`gofstat` reports the chi-squared statistic alone for a discrete fit.** That is R's behaviour, and
the reason is real: the Kolmogorov-Smirnov, Cramér-von Mises and Anderson-Darling statistics all
compare an empirical step function against a *continuous* one, and their tabulated critical values
do not apply when the fitted distribution has jumps. AIC and BIC are still comparable across
families, so use those and the chi-squared p-value.

The plots adapt too: the density panel draws probability masses at the integers against bars of
relative frequency, and the CDF panel plots the empirical points at `(1:n)/n` rather than at
`ppoints`, as `cdfcomp` does when `discrete = TRUE` — `ppoints` would place the steps off the
integers where the mass actually sits.

`empirical_density_plot(counts, discrete=True)` does the same for data on its own, and omits the
kernel density estimate, which assumes a continuous distribution.

### Over-dispersed counts as a mixture

Two Poisson regimes are often a better description of over-dispersed counts than one negative
binomial, and say something more interpretable — that the population contains two kinds of thing:

```python
from py_distcomp import fit_mixture

mix = fit_mixture(counts, ('poisson', 'poisson'))
mix.estimate            # {'weight1': 0.70, 'lambda1': 1.94, 'weight2': 0.30, 'lambda2': 11.5}
mix.component_probability()   # per-observation chance of the high-rate regime
```

`descdist` and `cullen_and_frey_plot(discrete=True)` remain the way to choose a family in the first
place — the graph's negative binomial region and Poisson line are drawn for exactly this case.

## 🎚️ Estimation methods

`fit_distributions(..., method=)` dispatches exactly as R's `fitdist` does.

| `method=` | fitdistrplus | minimises |
|---|---|---|
| `'mle'` (default) | `mledist` | the negative log-likelihood |
| `'mme'` | `mmedist` | the distance between theoretical and sample moments |
| `'qme'` | `qmedist` | the distance between theoretical and sample quantiles |
| `'mge'` | `mgedist` | a goodness-of-fit statistic directly |

```python
fit_distributions(x, 'weibull', method='mme')                    # moment matching
fit_distributions(x, 'weibull', method='qme', probs=[0.9, 0.99]) # match the tail
fit_distributions(x, 'weibull', method='mge', gof='ADR')         # right-tail weighted
```

**When each earns its place.** `'mme'` has closed forms for ten distributions — normal, lognormal,
Poisson, exponential, gamma, negative binomial, geometric, beta, uniform and logistic — so it cannot
fail to converge, which makes it a dependable fallback when likelihood optimisation misbehaves.
`'qme'` fits where you tell it to: match at 0.9 and 0.99 and the tail is what gets fitted, at the
cost of the middle. `'mge'` optimises the statistic a reader will judge the fit by, and its
`gof='ADR'` and `'AD2R'` variants weight the right tail specifically — directly relevant when the
tail is the thing of interest.

`gof` accepts `'CvM'` (default), `'KS'`, `'AD'`, and the tail-weighted `'ADR'`, `'ADL'`, `'AD2R'`,
`'AD2L'`, `'AD2'` — `R`/`L` for the right and left tail, `2` for the squared variants that weight
the extremes harder still.

Each estimator does what it claims: on the `groundbeef` Weibull, `method='mge', gof='KS'` gives a
lower KS statistic than maximum likelihood does, and maximum likelihood gives a higher
log-likelihood than any of them. `bootdist` refits with whichever method produced the original, so
bootstrapping an `'mge'` fit describes the sampling behaviour of *that* estimator.

## 📐 Q-Q confidence bands

A Q-Q plot shows where the data departs from the fitted distribution, but not whether the departure
is bigger than sampling noise would produce anyway. A bootstrap band answers that.

```python
from py_distcomp import qq_confidence_band, quantile_comparison_plot

# as a plot
quantile_comparison_plot(x, 'weibull', confidence_band=0.95)

# or as a table, to count the excursions
band = qq_confidence_band(fit, level=0.95, kind='simultaneous')
outside = (band.observed < band.lower) | (band.observed > band.upper)
outside.sum()
```

**`kind='simultaneous'`** (the plot's default) keeps the *whole* curve inside with the stated
probability, so a single excursion is evidence of a real departure. **`kind='pointwise'`** gives
each point its own interval, so with *n* points a handful will fall outside by chance even under a
perfect fit — useful for seeing *where* a fit strains, not *whether* it does.

**`refit=True`** (the default) re-estimates the parameters on every simulated sample. This matters
more than it looks: the real points are plotted against quantiles fitted to the real data, so the
fit has already absorbed part of the discrepancy, and the null distribution has to absorb the same
part to be comparable. On data genuinely from the fitted family this gives 4.9% of points outside a
nominal 5% pointwise band. Turning it off treats the estimated parameters as if they were known to
be correct, which makes the band far too wide — 0.2% outside on the same check — so real departures
can hide inside it.

The band pairs naturally with the off-model method: where `off_model_fraction` finds the cut by
maximising R², the band says whether the points beyond it are outside what the fitted distribution
would produce.

## 📏 Uncertainty on the estimates

A fitted parameter without an interval can be reported but not compared. Two routes are available.

### Standard errors, as `fitdist` reports them

```python
fit = fit_distributions(serving, 'weibull')[0]

fit.summary()
#        estimate  std_error
# shape    2.1856     0.1046
# scale   83.3467     2.5271

fit.confint(0.95)     # Wald: estimate ± z · std_error
fit.vcov              # variance-covariance matrix
fit.correlation       # R's cov2cor(vcov)
```

These come from the observed information — the Hessian of the log-likelihood at the maximum, taken
in **R's** parameterisation, so the standard error of a gamma `rate` is the rate's own, not that of
scipy's `scale`. `vcov` is `None` where the Hessian is singular, as R returns `NULL`; the uniform is
the usual case, its likelihood being flat inside the support.

### Bootstrap intervals

```python
from py_distcomp import bootdist

boot = bootdist(fit, niter=1001, seed=1)      # 'param' by default
boot.summary()
#        estimate     median      2.5%     97.5%
# shape     2.186      2.189     2.003     2.419
# scale    83.348     83.150    78.577    88.318

boot.quantile_ci([0.5, 0.95])   # intervals on the distribution's own quantiles
boot.estimates                  # every resample's parameters
boot.n_converged, boot.n_failed
```

`bootmethod='param'` draws each resample from the fitted distribution, so the interval reflects
sampling variability *under the assumed model*. `bootmethod='nonparam'` resamples the observed data
with replacement and assumes nothing — the safer choice when the fit is imperfect.

**Which to use.** The standard errors assume the log-likelihood is quadratic near its maximum. That
is an asymptotic argument: for a small sample, or a parameter whose sampling distribution is skewed
— a shape, or a scale near zero — the symmetric Wald interval can be poor, and can even run below
zero for a parameter that cannot be negative. The bootstrap makes no such assumption but costs
`niter` refits (about a second for 1001, on a simple distribution). They agree closely when both are
valid, which is itself a useful check.

### Comparing population subsets

This is the comparison a table of bare estimates cannot support — whether two subsets really differ:

```python
from py_distcomp import confint_plot

fleets = {
    label: bootdist(fit_distributions(subset, 'gumbel')[0], niter=1001, seed=1)
    for label, subset in fleet_subsets.items()
}
confint_plot(fleets, parameter='loc').show()
```

`bootdist_plot(boot)` shows the resampled parameter cloud (R's `plot.bootdist`). A diagonal smear
means the parameters trade off against one another and cannot be read independently.

### Bootstrapping a mixture

`bootdist` also accepts a `MixtureResult`, which is the only route to uncertainty there — the weights
and component parameters come out of expectation-maximisation rather than a single optimisation, so
there is no Hessian to invert. This turns the identifiability caveat below from a warning into a
number:

```python
bootdist(mix, niter=400, seed=1).summary()
#          estimate  median    2.5%   97.5%
# weight1     0.959   0.959   0.943   0.973
# loc1       11.799  11.794  11.215  12.431
# weight2     0.041   0.041   0.027   0.057    <- the fraction is well determined
# loc2       82.415  82.281  68.752  90.696    <- its location is not
```

The refit is warm-started from the original fit, which is both far faster and keeps the component
ordering stable — without it components could swap between resamples and the percentiles would be
meaningless.

## 🎯 Off-model fraction analysis

Sometimes most of a population follows one distribution but a small, high-valued subset does not.
Fitting everything at once then gives parameters that describe neither group. The method of Rushton
et al. (2021) finds that subset: cut the data at successively lower percentiles, refit, and measure
how well the Q-Q relationship follows the 1:1 line. The percentile that maximises the fit is the
**off-model percentile** `P_off`; the remaining `100 - P_off` per cent is the **off-model fraction**.
In the paper those are candidate gross-emitting vehicles.

```python
from py_distcomp import (
    off_model_fraction, qq_r_squared,
    r_squared_sweep_plot, percentile_cut_qq_plot, off_model_density_plot,
)

result = off_model_fraction(emission_ratios, model='gumbel')

result.percentile        # 95.0  -- P_off
result.fraction          # 5.0   -- per cent off-model
result.r_squared         # 0.991
result.threshold         # data value at the cut
result.fit.estimate      # {'loc': 11.2, 'scale': 8.3}  parameters of the bulk
result.tail_fit.estimate # parameters of the off-model subset
result.off_model_values  # the observations above the cut
result.summary()         # one row, in the style of the paper's Table 1

r_squared_sweep_plot(result).show()         # Figure 5
percentile_cut_qq_plot(data, 'gumbel').show()  # Figure 6
off_model_density_plot(result).show()       # bulk + tail superposition
```

Comparing population subsets is the point of the method, so the sweep plot takes a mapping:

```python
results = {
    f'{euro} {fuel}': off_model_fraction(subset, 'gumbel')
    for (euro, fuel), subset in fleet_subsets.items()
}
r_squared_sweep_plot(results, reference_line=0.98).show()
```

The Gumbel is the default because its location is the modal value and its scale describes the
spread, so both parameters are directly comparable between subsets — the paper's reason for
choosing it. Any registered distribution works.

### `qq_r_squared`

```python
def qq_r_squared(
    data, model='gumbel', dist_params=None,
    method='identity', a_ppoints=0.5,
) -> float
```

The goodness-of-fit measure behind the sweep. `'identity'` measures deviation from the 1:1 line,
`1 - Σ(y - x)² / Σ(y - ȳ)²`, so location or scale error is penalised and the value can go negative.
`'pearson'` is the squared correlation of the two quantile sets, which only measures how *straight*
the Q-Q relationship is.

> **On the choice of default.** The paper reports "the R² value of the relationship between the
> empirical and theoretical quantiles" without giving a formula, but describes fit throughout as
> agreement with the 1:1 line, so `'identity'` is the default. On contaminated samples the two agree
> on `P_off` to within a percentile; `'identity'` simply falls further when the fit is poor. Pass
> `method='pearson'` if you want the other reading.

### `off_model_fraction`

```python
def off_model_fraction(
    data, model='gumbel', percentiles=None, method='identity',
    min_points=10, fit_tail=True, a_ppoints=0.5,
) -> OffModelResult
```

`percentiles` defaults to every integer percentile from 1 to 100, where 100 means no cut — a clean
sample returns `P_off = 100` and a zero fraction, as the paper found for pre-Euro 6 diesels.
`fit_tail=True` runs the paper's second iteration, fitting the same family to the off-model
observations so the population can be described as a superposition of two distributions.

`OffModelResult.curve` holds the whole sweep — percentile, retained `n`, threshold, R² and the
fitted parameters at each cut — which is the data behind Figure 5.

## 🔬 Measurement error: fitting the distribution behind the readings

Some quantities cannot be observed directly. A remote sensing device measures a concentration ratio
that is physically non-negative, but propagates enough error that individual readings come back
**negative**. Discarding those biases the distribution high — the error works both ways — so they are
kept, and what is actually observed is

<p align="center"><i>Y = X + ε</i></p>

with *X* the true value and *ε* symmetric instrument error. Fitting a family directly to *Y*
estimates the **smeared** distribution: its spread is the true spread inflated by the error, and its
shape has to accommodate values the true quantity can never take.

`fit_convolved` fits *X* instead.

```python
from py_distcomp import fit_convolved, convolved_density_plot

fit = fit_convolved(readings, true_family="gamma", error="normal")

fit.true_estimate     # {'shape': 2.07, 'rate': 0.169}  — the quantity of interest
fit.scale             # 5.90  the error scale, estimated
fit.true_sd           # 8.51  spread of the truth
fit.observed_sd       # 10.35 spread of the readings
fit.inflation         # +21.7%  how much the noise widened them
```

![Measurement-error model](docs/images/convolved-density.png)

### Why this matters more than it looks

**The inflation lands on the spread, which is usually the parameter being compared.** Fit a Gumbel
directly to readings of a gamma-distributed quantity smeared by a normal error, and its scale comes
out **45% too high**. Comparing that number between groups compares emissions *and* instrument
noise.

**It unlocks families the observations forbid.** A gamma or lognormal cannot be fitted to data
containing negatives — scipy raises — which is why an unbounded family is often chosen by default.
But a lognormal can perfectly well be *the truth behind* such data, and the convolution recovers it:
meanlog 2.287 against a true 2.303, sdlog 0.709 against 0.700, from a sample where a direct fit is
impossible.

**It beats the alternatives on their own terms.** The fitted model goes into `gofstat` like anything
else:

```
                     ks    cvm      ad        aic
gumbel            0.023  0.330   2.187  21996.180
normal            0.097  9.958  59.351  22788.103
logistic          0.058  2.695  28.399  22439.794
lognormal+normal  0.013  0.060   0.425  21970.117    <- AIC prefers it
```

### Supplying the error scale, or estimating it

`scale=None` (the default) estimates it alongside everything else, which works because a symmetric
error and a skewed signal are separable — on simulated data it recovers 5.90 against a true 6.0.
Pass `scale=` to fix it from calibration instead.

**Getting a supplied scale wrong distorts the spread far more than the mean:**

| σ assumed | fitted sd | vs truth 8.49 |
|---|---|---|
| 3.0 | 10.49 | +24% |
| 6.0 ✓ | 8.44 | — |
| 9.0 | 6.00 | −29% |

When the calibration is not solid, estimating is the safer choice.

`error` accepts `'normal'` (default), `'laplace'`, `'logistic'` and `'cauchy'`, all taken as
symmetric and centred on zero. The error is assumed **constant** — a per-observation scale is not yet
supported.

## ✂️ Spliced distributions: one heavier tail, not two populations

`fit_spliced` answers a question a mixture cannot. A mixture says the tail is a **second population**
sitting on top of the first — both components have mass everywhere, and every observation belongs to
each with some probability. A splice says there is **one population whose tail is heavier** than its
body implies: below a threshold θ one family governs, above it another, each renormalised so the
whole remains a proper density.

<p align="center"><i>f(x) = w · f₁(x)/F₁(θ) for x ≤ θ &nbsp;&nbsp;|&nbsp;&nbsp; (1−w) · f₂(x)/(1−F₂(θ)) for x &gt; θ</i></p>

This is the principled successor to the percentile cut of Rushton et al. (2021). That method
*searches* percentiles for the cut that maximises a Q-Q R². This one **estimates θ by likelihood**
alongside both sides, and returns it with a profile-likelihood confidence interval.

```python
from py_distcomp import fit_spliced, spliced_density_plot, threshold_profile_plot

fit = fit_spliced(emissions, lower="gumbel", upper="pareto")

fit.threshold          # the estimated join, θ
fit.threshold_ci()     # profile-likelihood interval for it
fit.weight             # mass below θ — determined by continuity, not fitted
fit.lower_estimate     # {'loc': ..., 'scale': ...}
fit.upper_estimate     # {'shape': ..., 'scale': ...}
fit.n_above            # observations the upper family governs
```

![Spliced density](docs/images/spliced-density.png)

### Continuity determines the weight

By default the two pieces are required to **meet** at θ, which is what makes the result one density
rather than two stitched fragments. That constraint fixes `w` rather than leaving it free:

<p align="center"><i>w = b / (a + b),&nbsp;&nbsp; a = f₁(θ)/F₁(θ),&nbsp;&nbsp; b = f₂(θ)/(1−F₂(θ))</i></p>

So a continuous splice of two two-parameter families estimates **five** quantities: θ and two per
side. `continuous=False` frees the weight and permits a jump, which `SplicedDistribution.jump`
reports — a large one means the two families do not belong together.

### The threshold is profiled, not optimised

The likelihood changes shape whenever θ crosses an observation, so it is not differentiable there
and cannot be optimised smoothly. It is profiled over a grid instead, which also produces the
interval for free:

![Threshold profile](docs/images/threshold-profile.png)

**Look at this plot before trusting the threshold.** A sharp peak means the data locates the join; a
flat ridge — as above, because that example's data is really a mixture — means it does not, whatever
the point estimate says.

### Which model is right? Let the data decide

The two constructions are **distinguishable**. Fit both to data generated each way and the right one
wins:

| data generated as | single | mixture | splice | AIC prefers |
|---|---|---|---|---|
| **mixture** (two populations) | 8037.3 | **7856.1** | 7876.8 | mixture ✓ |
| **splice** (one heavy tail) | 15373.0 | 15079.1 | **15067.9** | splice ✓ |

```python
gofstat([
    fit_distributions(x, "gumbel")[0],
    fit_mixture(x, ("gumbel", "gumbel")),
    fit_spliced(x, "gumbel", "pareto"),
])
```

So "is the tail a second population, or the same population with a heavier tail?" is no longer a
choice inherited from whichever tool was to hand — it is a model comparison the data can settle.

### A note on the tail parameters

Only the **shape** of a Pareto tail is identifiable once it is truncated above θ: a Pareto truncated
above θ is a Pareto starting at θ with the same shape, so the scale carries no separate information.
The shape is the quantity that governs how heavy the tail is, and it is recovered well — 2.2 → 2.10
on simulated data.

## 🔀 Mixture fitting: superposition of distributions

The off-model method splits a population with a hard cut, then fits each side separately. That
works, but the cut is chosen by a percentile search rather than by the likelihood, an observation
belongs wholly to one side or the other, and the two fits never inform each other.

`fit_mixture` estimates the same superposition properly:

<p align="center"><i>f(x) = w<sub>1</sub> f<sub>1</sub>(x; θ<sub>1</sub>) + w<sub>2</sub> f<sub>2</sub>(x; θ<sub>2</sub>)</i></p>

by expectation-maximisation, so the weights, both components' parameters and the assignment of
observations are estimated jointly. Nothing is discarded, and every observation gets a **probability**
of belonging to each component rather than a hard label — which is what "high-chance gross emitting
vehicle" actually asks for.

```python
from py_distcomp import fit_mixture, mixture_density_plot, component_probability_plot

mix = fit_mixture(emission_ratios, ('gumbel', 'gumbel'))

mix.weights                  # array([0.959, 0.041])
mix.estimate                 # {'weight1': 0.959, 'loc1': 11.8, 'scale1': 8.7,
                             #  'weight2': 0.041, 'loc2': 82.4, 'scale2': 18.9}
mix.component_probability()  # per-observation chance of being in the upper component
mix.classify(threshold=0.9)  # boolean flags at a chosen confidence
mix.expected_counts()        # array([959.4, 40.6])
mix.aic, mix.bic, mix.loglik

mixture_density_plot(mix).show()          # histogram, weighted components, their sum
component_probability_plot(mix).show()    # P(upper component) against value
```

Components need not share a family — `('gumbel', 'normal')` is fine — and more than two are
supported. The fitted mixture behaves like a scipy distribution, so it drops into everything else:

```python
from py_distcomp import fit_distributions, gofstat, quantile_comparison_plot

# Compare a mixture against single distributions on the same footing
gofstat(fit_distributions(data, ['gumbel', 'normal']) + [mix])
#                        ks       cvm        ad    loglik      aic      bic
# gumbel            0.061349  1.373640 10.484441 -4016.66  8037.32  8047.13
# normal            0.177074 12.255646 71.072144 -4370.30  8744.60  8754.42
# gumbel + gumbel   0.015558  0.037965  0.295380 -3923.05  7856.10  7880.64

# And it plots like any other model
quantile_comparison_plot(data, ['gumbel', mix.dist])
```

### `fit_mixture`

```python
def fit_mixture(
    data, models=('gumbel', 'gumbel'), init='auto',
    max_iter=500, tol=1e-8, min_points=5, min_weight=1e-4,
) -> MixtureResult
```

`init` controls the starting partition, which matters because EM finds a *local* optimum:
`'off_model'` seeds from the paper's percentile cut, `'quantile'` from equal-mass and tail-heavy
splits, and `'auto'` (the default) tries both and keeps whichever reaches the higher likelihood.

AIC and BIC count both components' free parameters plus `K - 1` weights, so a mixture can be
compared against a single distribution honestly — the extra parameters are paid for.

### Which should you use?

| | Off-model cut | Mixture |
|---|---|---|
| Split chosen by | percentile search maximising Q-Q R² | maximum likelihood |
| Observation assignment | hard, above/below a threshold | probability per observation |
| Components inform each other | no | yes |
| Reproduces the paper | yes | no — it's the improved version |
| Comparable on AIC/BIC | not directly | yes |

They agree closely in practice: on a sample with 5% injected contamination the cut gives a 4%
off-model fraction and the mixture an upper weight of 0.041, with bulk parameters within a few per
cent of each other. Use the cut to reproduce or extend the published analysis, and the mixture when
you want a per-vehicle probability or a likelihood-based model comparison.

### Uncertainty on mixture parameters

Standard errors come from the observed information, as for a single fit — but the weights need
care. They sum to one, so differentiating with respect to all of them gives a singular matrix. The
Hessian is taken over `K − 1` log-odds against the last component and mapped back to the weights by
the delta method.

```python
mix.std_error     # every weight and component parameter
mix.confint()     # Wald intervals
mix.vcov          # variance-covariance
mix.correlation
```

On a well-separated two-component normal fit these agree with `bootdist` to within a few per cent
on every parameter, and the weight's standard error lands near `sqrt(w(1-w)/n)` — the value it would
take if classification were certain. The weight block of `vcov` is singular by construction: with
two components their correlation is exactly −1, which is right rather than a defect.

A **degenerate fit reports no uncertainty at all** — `vcov` is `None` and the standard errors are
`nan`. A collapsed component has none to quote.

Mixtures test the Wald assumptions harder than single fits do, so where these and `bootdist`
disagree, believe the bootstrap.

### Choosing among local optima

EM finds a *local* optimum, so the useful output is not only the best fit but **how many independent
starts agreed on it**:

```python
mix.n_starts            # deterministic partitions plus random restarts
mix.n_starts_converged
mix.n_starts_at_best    # how many reached the winning likelihood
mix.start_logliks       # every start's result, best first
```

Restarts are seeded by k-means++ so they genuinely differ from one another and from the quantile
splits. They default to on for two reasons, both measured:

- **Beyond two components the deterministic strategy offers a single starting point.** Without
  restarts a three-component fit has exactly one, and no way to know whether it is any good.
- On three overlapping components, restarts found a **strictly better optimum in 12 of 25 trials**.

Agreement tracks reliability directly. Well-separated data: 9 of 9 starts reached the same
likelihood, spread 0.000. Overlapping data: 3 of 8, spread 1.775 — a signal to distrust the fit that
nothing else in the output would have given you.

They are not free: a three-component fit costs roughly 20 s with restarts against under a second
without. `n_start=0` turns them off. The seed is fixed by default so a fit is reproducible.

### Degenerate components

A mixture likelihood is frequently **unbounded**: shrink a component onto a few points and it goes
to infinity, so the highest-likelihood fit can be one that describes nothing. Four near-identical
values among 300 standard normal draws is enough to trigger it.

Every fit is checked for this, by two independent routes:

- **too little support** — a component's effective sample size, `sum_i P(component | x_i)`, against
  the parameters it has to estimate; three per parameter, never fewer than five
- **collapsed width** — a component orders of magnitude narrower than the data's own spread, which
  is what repeated or rounded values produce

```python
mix.degenerate      # True if any component fails
mix.diagnostics     # per-component: weight, n_effective, n_required, width, reason
```

A degenerate fit **warns**, prints `DEGENERATE` in its repr, and reports the flag in `summary()`.
Where the starting partitions offer a choice, a well-supported fit is preferred over a
higher-likelihood collapsed one. `on_degenerate='raise'` turns the warning into an error;
`'ignore'` silences it but leaves the flag set.

The AIC of a degenerate fit is deliberately **not** adjusted — quietly altering a number would hide
the problem rather than surface it. The flag is there so you know not to compare on it.

### A caveat on identifiability

When the two components overlap heavily — a small tail buried under a long-tailed bulk, which is
exactly the vehicle-emissions case — the upper component's parameters are weakly identified even
though its *weight* is recovered well. On the 5%-contamination example the injected tail is
Gumbel(70, 25) and the fit returns Gumbel(82, 19): the weight and the classification are reliable,
the individual tail parameters less so. This is a property of the data, not the algorithm; the EM
is verified against `sklearn.mixture.GaussianMixture` to 4 decimal places where the models coincide.
Run `bootdist(mix)` to see how wide that uncertainty actually is, and check `mix.converged`.

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

### R² Sweep Plot
R² against percentile cut, with the selected off-model percentile marked. Several population subsets can be overlaid for comparison.

### Percentile-Cut Q-Q Grid
A grid of Q-Q plots, one per cut, showing how the fitted location and scale respond as observations are removed from the top.

### Off-Model Density Plot
The population as a superposition of the distribution fitted to the retained data and the distribution fitted to the off-model tail, each weighted by its share of the population.

### Measurement-Error Density Plot
The readings, the smeared density that describes them, and the narrower true density recovered from behind it — with the physically impossible region marked.

### Spliced Density Plot
The fitted density drawn as the two pieces it is made of, with the join marked. A log density axis is usually the only way to see a tail beside a body many times taller.

### Threshold Profile Plot
The profile log-likelihood the threshold was read from, with its confidence interval shaded — a flat ridge is a warning the join is not located.

### Mixture Density Plot
Histogram with each weighted component of a jointly fitted mixture and their sum.

### Bootstrap Parameter Cloud
The resampled parameter estimates, one panel per pair, with the original estimate marked. R's `plot.bootdist`.

### Q-Q Plot with Confidence Band
A Q-Q plot shaded with a bootstrap band, so departures larger than sampling noise are visible at a glance.

### Confidence Interval Plot
Estimates with their intervals, several fits side by side, for comparing population subsets.

### Component Probability Plot
Posterior probability of belonging to a component against value — the per-observation replacement for a hard cut.

## 🎯 Use Cases

- **Quality Control**: Assess if manufacturing data follows expected distributions
- **Risk Analysis**: Validate assumptions about return distributions in finance
- **Reliability Engineering**: Test if failure times follow Weibull or exponential distributions
- **Environmental Science**: Analyze if measurements follow normal or log-normal distributions
- **Research**: Validate distributional assumptions before statistical modeling
- **Exploratory Data Analysis**: Use Cullen and Frey plots to identify candidate distribution families
- **Data Preprocessing**: Visualize empirical distributions before transformation or modeling

## 🔧 More examples

See [Worked examples](#-worked-examples) above for three complete walkthroughs with output and
figures. The snippets below cover a few other shapes of problem.

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

## 📦 Releasing

Publishing runs on PyPI's trusted publishing: GitHub mints a short-lived OIDC token for
`.github/workflows/publish.yml` running in the `pypi` environment, and PyPI accepts it in place of
an API token. There is no secret stored anywhere.

To cut a release:

1. Bump the version in **both** `pyproject.toml` and `py_distcomp/__init__.py`.
2. Merge that to `main`.
3. Publish a GitHub Release tagged `v<version>` — e.g. `v0.7.0`.

The workflow then builds the sdist and wheel, checks the metadata renders on PyPI
(`twine check --strict`), verifies the tag matches the packaged version, installs the wheel into a
clean environment and imports it, and only then uploads. A tag that disagrees with the version fails
the run before anything irreversible happens.

`workflow_dispatch` runs the same build and checks but defaults to a dry run, so the button cannot
publish by accident.

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
- Ported from R's `fitdistrplus` package: [FitDistrPlus](https://cran.r-project.org/web/packages/fitdistrplus/index.html)
- Off-model fraction method from [Rushton, Tate & Shepherd (2021)](https://doi.org/10.1016/j.scitotenv.2020.142088)
- Demo app powered by [Streamlit](https://streamlit.io/)

## 📊 Roadmap

- [x] **Multi-distribution comparison with Q-Q, P-P, and CDF plots**
- [x] **Cullen and Frey plots for distribution family identification**
- [x] **Empirical CDF and density plots with KDE**
- [x] **Interactive Streamlit demo application**
- [x] **Bootstrap confidence regions for Cullen and Frey plots**
- [x] **Goodness-of-fit statistics (KS, Cramér-von Mises, Anderson-Darling, chi-squared, AIC, BIC)**
- [x] **Maximum-likelihood fitting in R's parameterisation**
- [x] **Off-model fraction analysis (Rushton et al., 2021)**
- [x] **Mixture fitting by expectation-maximisation, with per-observation component probabilities**
- [x] **Standard errors, variance-covariance and correlation of the estimates**
- [x] **Bootstrap confidence intervals (`bootdist`), including on mixture weights**
- [x] **Discrete distributions: Poisson, negative binomial and geometric**
- [x] **Moment, quantile and maximum goodness-of-fit estimation (`mmedist`, `qmedist`, `mgedist`)**
- [x] **Bootstrap confidence bands for Q-Q plots**
- [x] **Spliced (composite) distributions, with the threshold estimated by likelihood**
- [x] **Measurement-error (convolution) models, recovering the distribution behind noisy readings**
- [ ] Support for censored data analysis
- [ ] Integration with statistical testing frameworks
- [ ] Publication to PyPI
- [ ] R package integration

---

⭐ **Star this repository if you find it useful!** ⭐