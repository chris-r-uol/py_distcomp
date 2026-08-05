"""
Plots for the off-model fraction method of Rushton et al. (2021).

``r_squared_sweep_plot``      Figure 5: R² against percentile cut.
``percentile_cut_qq_plot``    Figure 6: Q-Q plots as the cut is varied.
``off_model_density_plot``    the fleet as a superposition of two distributions.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .distributions import fit_distribution, ppoints, resolve_distribution
from .off_model import OffModelResult, off_model_fraction
from .quantile_multi_comparison import DISTRIBUTION_COLORS, _validate_and_prepare_data

__all__ = [
    "r_squared_sweep_plot",
    "percentile_cut_qq_plot",
    "off_model_density_plot",
]


def r_squared_sweep_plot(
    results: Union[OffModelResult, Dict[str, OffModelResult]],
    title: str = "Maximising the fit values of the data cut",
    reference_line: Optional[float] = 0.98,
    show_selected: bool = True,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """R² against percentile cut -- Figure 5 of Rushton et al. (2021).

    Parameters
    ----------
    results : OffModelResult or dict of str -> OffModelResult
        A single sweep, or several keyed by label so that population subsets
        (Euro class, fuel type, ...) can be compared on one axes, as the paper
        does.
    title : str
        Plot title.
    reference_line : float or None, default=0.98
        Horizontal reference, drawn as a red dashed line in the paper.
    show_selected : bool, default=True
        Mark the selected off-model percentile on each curve.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    if isinstance(results, OffModelResult):
        results = {results.model: results}
    if not results:
        raise ValueError("At least one result is required")

    fig = go.Figure()

    for i, (label, result) in enumerate(results.items()):
        colour = DISTRIBUTION_COLORS[i % len(DISTRIBUTION_COLORS)]
        curve = result.curve
        fig.add_trace(go.Scatter(
            x=curve["percentile"],
            y=curve["r_squared"],
            mode="lines+markers",
            name=str(label),
            line=dict(color=colour, width=1.5),
            marker=dict(size=4, color=colour),
            customdata=curve["n"],
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Percentile: %{x}<br>"
                "R²: %{y:.4f}<br>"
                "n retained: %{customdata}<extra></extra>"
            ),
        ))

        if show_selected:
            fig.add_trace(go.Scatter(
                x=[result.percentile],
                y=[result.r_squared],
                mode="markers",
                name=f"{label} P_off = {result.percentile:g}",
                marker=dict(size=13, color=colour, symbol="star",
                            line=dict(width=1, color="black")),
                hovertemplate=(
                    f"<b>{label} selected</b><br>"
                    "P_off: %{x}<br>"
                    "R²: %{y:.4f}<br>"
                    f"Off-model fraction: {result.fraction:g}%<extra></extra>"
                ),
            ))

    if reference_line is not None:
        fig.add_hline(
            y=reference_line,
            line=dict(color="red", width=1.5, dash="dash"),
            annotation=dict(text=f"R² = {reference_line:g}", font=dict(size=11)),
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="Percentile", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="R²", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        width=width,
        height=height,
        # Outside the panel, as in the paper: the interesting low-percentile
        # part of the curve sits exactly where an inset legend would go.
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        margin=dict(r=180),
        hovermode="closest",
    )
    return fig


def percentile_cut_qq_plot(
    data: Union[np.ndarray, pd.Series, list],
    model: Union[str, object] = "gumbel",
    percentiles: Sequence[float] = (99, 98, 97, 96, 95, 94, 93, 92, 91),
    ncols: int = 3,
    title: str = "Q-Q plots for varying percentile cuts",
    a_ppoints: float = 0.5,
    subplot_size: int = 280,
) -> go.Figure:
    """A grid of Q-Q plots, one per percentile cut -- Figure 6 of the paper.

    Each panel cuts the data at its percentile, refits ``model`` to what
    remains, and plots the resulting Q-Q against the 1:1 line, so the effect of
    the cut on the fitted location and scale can be seen directly.

    Parameters
    ----------
    data : array-like
        Input data.
    model : str or scipy distribution, default='gumbel'
        Distribution to fit at each cut.
    percentiles : sequence of float
        Cuts to show, in panel order.
    ncols : int, default=3
        Panels per row.
    title : str
        Overall title.
    a_ppoints : float, default=0.5
        Offset for the plotting positions.
    subplot_size : int, default=280
        Approximate size of each panel in pixels.

    Returns
    -------
    go.Figure
    """
    sorted_data = _validate_and_prepare_data(data)
    dist, _ = resolve_distribution(model)

    percentiles = list(percentiles)
    nrows = int(np.ceil(len(percentiles) / ncols))

    panels = []
    for pct in percentiles:
        threshold = float(np.percentile(sorted_data, pct))
        retained = sorted_data[sorted_data <= threshold]
        if len(retained) < 3:
            panels.append((pct, None, None, np.nan))
            continue
        _, params = fit_distribution(model, retained)
        theoretical = dist.ppf(ppoints(len(retained), a=a_ppoints), *params)
        ss_res = np.sum((retained - theoretical) ** 2)
        ss_tot = np.sum((retained - retained.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        panels.append((pct, theoretical, retained, r2))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            f"{_ordinal(pct)} percentile (R² = {r2:.3f})" if np.isfinite(r2)
            else f"{_ordinal(pct)} percentile"
            for pct, _, _, r2 in panels
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for i, (pct, theoretical, retained, _) in enumerate(panels):
        row, col = i // ncols + 1, i % ncols + 1
        if theoretical is None:
            continue

        fig.add_trace(go.Scatter(
            x=theoretical, y=retained,
            mode="markers",
            name=f"{pct:g}%",
            showlegend=False,
            marker=dict(size=4, opacity=0.6, color="dimgray"),
            hovertemplate=("Theoretical: %{x:.3f}<br>"
                           "Empirical: %{y:.3f}<extra></extra>"),
        ), row=row, col=col)

        lo = float(min(theoretical.min(), retained.min()))
        hi = float(max(theoretical.max(), retained.max()))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            showlegend=False,
            line=dict(color="black", width=1),
            hoverinfo="skip",
        ), row=row, col=col)

        fig.update_xaxes(title_text="Theoretical quantiles", row=row, col=col,
                         showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(title_text="Empirical quantiles", row=row, col=col,
                         showgrid=True, gridcolor="lightgray")

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template="plotly_white",
        width=subplot_size * ncols + 120,
        height=subplot_size * nrows + 120,
        hovermode="closest",
    )
    fig.update_annotations(font_size=12)
    return fig


def off_model_density_plot(
    result: OffModelResult,
    data_name: str = "Data",
    title: Optional[str] = None,
    bins: Union[int, str] = "sturges",
    fitnbpts: int = 201,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """The population as a superposition of the main fit and the off-model tail.

    Shows the histogram of all observations, the distribution fitted to the
    retained data, and -- when one was fitted -- the distribution fitted to the
    off-model observations, each weighted by its share of the population. The
    cut is marked.

    Parameters
    ----------
    result : OffModelResult
        Output of :func:`~py_distcomp.off_model_fraction`.
    data_name : str, default='Data'
        Legend label for the observations.
    title : str, optional
        Plot title.  Defaults to a description of the off-model fraction.
    bins : int or str, default='sturges'
        Histogram binning, passed to ``numpy.histogram_bin_edges``.
    fitnbpts : int, default=201
        Points used to draw the fitted curves.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    data = result.data
    if title is None:
        title = (
            f"{result.model.title()} fit with a "
            f"{result.fraction:g}% off-model fraction"
        )

    fig = go.Figure()

    edges = np.histogram_bin_edges(data, bins=bins)
    counts, edges = np.histogram(data, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres, y=counts, width=np.diff(edges),
        name=f"{data_name} (empirical)",
        marker=dict(color="gray", opacity=0.6, line=dict(width=0.5, color="white")),
        hovertemplate="Bin centre: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    sfin = np.linspace(float(data.min()), float(data.max()), fitnbpts)

    # Each component is scaled by its share of the population, so the two curves
    # sum to the mixture density rather than each integrating to one.
    main_weight = result.n_retained / result.n_total
    fig.add_trace(go.Scatter(
        x=sfin,
        y=result.fit.dist.pdf(sfin, *result.fit.params) * main_weight,
        mode="lines",
        name=f"On-model ({main_weight * 100:.0f}%)",
        line=dict(color=DISTRIBUTION_COLORS[0], width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    if result.tail_fit is not None:
        tail_weight = result.n_off_model / result.n_total
        fig.add_trace(go.Scatter(
            x=sfin,
            y=result.tail_fit.dist.pdf(sfin, *result.tail_fit.params) * tail_weight,
            mode="lines",
            name=f"Off-model ({tail_weight * 100:.0f}%)",
            line=dict(color=DISTRIBUTION_COLORS[1], width=2.5, dash="dash"),
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sfin,
            y=(result.fit.dist.pdf(sfin, *result.fit.params) * main_weight
               + result.tail_fit.dist.pdf(sfin, *result.tail_fit.params) * tail_weight),
            mode="lines",
            name="Superposition",
            line=dict(color="black", width=1.5, dash="dot"),
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        ))

    fig.add_vline(
        x=result.threshold,
        line=dict(color="red", width=1.5, dash="dash"),
        annotation=dict(
            text=f"P{result.percentile:g} = {result.threshold:.3g}",
            font=dict(size=11),
        ),
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="data", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        width=width,
        height=height,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
        bargap=0,
    )
    return fig


def _ordinal(value: float) -> str:
    """1 -> '1st', 92 -> '92nd', as the paper labels its panels."""
    n = int(value)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
