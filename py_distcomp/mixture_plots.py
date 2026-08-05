"""
Plots for fitted mixtures.

``mixture_density_plot``       the histogram with each weighted component and their sum.
``component_probability_plot`` the posterior probability of belonging to a component.
"""

from typing import Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from .mixture import MixtureResult
from .quantile_multi_comparison import DISTRIBUTION_COLORS

__all__ = ["mixture_density_plot", "component_probability_plot"]


def mixture_density_plot(
    result: MixtureResult,
    data_name: str = "Data",
    title: Optional[str] = None,
    bins: Union[int, str, Sequence[float]] = "sturges",
    fitnbpts: int = 301,
    show_components: bool = True,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Histogram with the fitted mixture and its weighted components.

    Each component is scaled by its mixing weight, so the components sum to the
    mixture density drawn over them rather than each integrating to one.

    Parameters
    ----------
    result : MixtureResult
        Output of :func:`~py_distcomp.fit_mixture`.
    data_name : str, default='Data'
        Legend label for the observations.
    title : str, optional
        Plot title.  Defaults to a description of the fitted mixture.
    bins : int, str or sequence, default='sturges'
        Histogram binning, passed to ``numpy.histogram_bin_edges``.
    fitnbpts : int, default=301
        Points used to draw the curves.
    show_components : bool, default=True
        Draw the individual weighted components as well as their sum.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    data = result.data
    if title is None:
        weights = ", ".join(f"{w * 100:.1f}%" for w in result.weights)
        title = f"{result.name} mixture ({weights})"

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

    if show_components:
        for k, ((dist, params), weight, name) in enumerate(
            zip(result.components, result.weights, result.model_names)
        ):
            fig.add_trace(go.Scatter(
                x=sfin,
                y=dist.pdf(sfin, *params) * weight,
                mode="lines",
                name=f"Component {k + 1}: {name} ({weight * 100:.1f}%)",
                line=dict(color=DISTRIBUTION_COLORS[k % len(DISTRIBUTION_COLORS)],
                          width=2, dash="dash"),
                hovertemplate=(f"<b>Component {k + 1}</b><br>"
                               "Value: %{x:.3f}<br>"
                               "Weighted density: %{y:.4f}<extra></extra>"),
            ))

    fig.add_trace(go.Scatter(
        x=sfin,
        y=result.dist.pdf(sfin),
        mode="lines",
        name="Mixture",
        line=dict(color="black", width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))

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


def component_probability_plot(
    result: MixtureResult,
    component: int = -1,
    threshold: Optional[float] = 0.5,
    data_name: str = "Data",
    title: Optional[str] = None,
    show_observations: bool = True,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Posterior probability of belonging to a component, against value.

    This is the per-observation version of a hard cut: rather than everything
    above a percentile being labelled off-model, each observation carries a
    probability of having come from the component.  With the default
    ``component=-1`` that is the high-valued component, so the curve reads as
    the chance an observation is off-model.

    Parameters
    ----------
    result : MixtureResult
        Output of :func:`~py_distcomp.fit_mixture`.
    component : int, default=-1
        Which component to show the probability for.  ``-1`` is the last.
    threshold : float or None, default=0.5
        Draw a decision threshold and report how many observations exceed it.
    data_name : str, default='Data'
        Legend label for the observations.
    title : str, optional
        Plot title.
    show_observations : bool, default=True
        Mark the observed values along the curve.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    data = result.data
    index = component % result.n_components
    label = f"component {index + 1} ({result.model_names[index]})"
    if title is None:
        title = f"Probability of belonging to {label}"

    # A smooth curve over the range, plus the observations themselves.
    grid = np.linspace(float(data.min()), float(data.max()), 400)
    grid_prob = result.dist.responsibilities(grid)[:, index]
    obs_prob = result.component_probability(component=index)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grid, y=grid_prob,
        mode="lines",
        name=f"P({label})",
        line=dict(color=DISTRIBUTION_COLORS[index % len(DISTRIBUTION_COLORS)], width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Probability: %{y:.3f}<extra></extra>",
    ))

    if show_observations:
        fig.add_trace(go.Scatter(
            x=data, y=obs_prob,
            mode="markers",
            name=f"{data_name} (n = {len(data)})",
            marker=dict(size=5, color="black", opacity=0.35),
            hovertemplate="Value: %{x:.3f}<br>Probability: %{y:.3f}<extra></extra>",
        ))

    if threshold is not None:
        n_above = int(np.sum(obs_prob >= threshold))
        fig.add_hline(
            y=threshold,
            line=dict(color="red", width=1.5, dash="dash"),
            annotation=dict(
                text=(f"P ≥ {threshold:g}: {n_above} observations "
                      f"({n_above / len(data) * 100:.1f}%)"),
                font=dict(size=11),
            ),
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="data", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Posterior probability", showgrid=True,
                   gridcolor="lightgray", range=[-0.02, 1.02]),
        template="plotly_white",
        width=width,
        height=height,
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
    )
    return fig
