"""
Plots for measurement-error models.

``convolved_density_plot`` shows what the fit separates: the observations, the
smeared density that describes them, and the narrower true density recovered
from behind it.
"""

from typing import Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from .convolution import ConvolvedResult
from .quantile_multi_comparison import DISTRIBUTION_COLORS

__all__ = ["convolved_density_plot"]


def convolved_density_plot(
    result: ConvolvedResult,
    data_name: str = "Observed",
    title: Optional[str] = None,
    bins: Union[int, str, Sequence[float]] = "sturges",
    show_true: bool = True,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """The observations, the fitted convolution, and the density behind it.

    Three things are worth seeing together: the observations as they came off
    the instrument, the convolved density that describes them, and the true
    density the model recovers.  The gap between the last two is the
    measurement error -- and it is usually wider than people expect.

    Parameters
    ----------
    result : ConvolvedResult
        Output of :func:`~py_distcomp.fit_convolved`.
    data_name : str, default='Observed'
        Legend label for the observations.
    title : str, optional
        Plot title.  Defaults to naming the model and the inflation it found.
    bins : int, str or sequence, default='sturges'
        Histogram binning, passed to ``numpy.histogram_bin_edges``.
    show_true : bool, default=True
        Draw the recovered density of the unobserved quantity.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    data = result.data
    if title is None:
        title = (f"{result.true_model} behind {result.error_model} error "
                 f"(observations {result.inflation:+.0%} wider than the truth)")

    fig = go.Figure()

    edges = np.histogram_bin_edges(data, bins=bins)
    counts, edges = np.histogram(data, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres, y=counts, width=np.diff(edges),
        name=f"{data_name} (n = {len(data)})",
        marker=dict(color="gray", opacity=0.55, line=dict(width=0.5, color="white")),
        hovertemplate="Bin centre: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    grid = np.linspace(float(np.min(data)), float(np.max(data)), 500)
    fig.add_trace(go.Scatter(
        x=grid, y=result.dist.pdf(grid),
        mode="lines", name="fitted, with error",
        line=dict(color=DISTRIBUTION_COLORS[0], width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Density: %{y:.5f}<extra></extra>",
    ))

    if show_true:
        with np.errstate(invalid="ignore", divide="ignore"):
            true_density = result.dist.true_dist.pdf(grid, *result.dist.true_params)
        fig.add_trace(go.Scatter(
            x=grid, y=np.nan_to_num(true_density),
            mode="lines", name="recovered, error removed",
            line=dict(color=DISTRIBUTION_COLORS[2], width=2.5, dash="dash"),
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.5f}<extra></extra>",
        ))

    # Values the true quantity cannot take, but the instrument still reports.
    if float(np.min(data)) < 0:
        fig.add_vrect(
            x0=float(np.min(data)) * 1.05, x1=0.0,
            fillcolor="rgba(196,56,79,0.10)", line_width=0,
            annotation=dict(text="physically impossible, kept anyway",
                            font=dict(size=10)),
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(title="value", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        width=width, height=height,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
        bargap=0,
    )
    return fig
