"""
Plots for spliced distributions.

``spliced_density_plot``    the two pieces meeting at the threshold.
``threshold_profile_plot``  the profile log-likelihood the threshold is read from.
"""

from typing import Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from .quantile_multi_comparison import DISTRIBUTION_COLORS
from .spliced import SplicedResult

__all__ = ["spliced_density_plot", "threshold_profile_plot"]


def spliced_density_plot(
    result: SplicedResult,
    data_name: str = "Data",
    title: Optional[str] = None,
    bins: Union[int, str, Sequence[float]] = "sturges",
    fitnbpts: int = 401,
    log_y: bool = False,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """The fitted density, drawn as the two pieces it is made of.

    Each side is coloured separately and the join is marked, so the reader can
    see where one family hands over to the other -- and, if the fit was made
    with ``continuous=False``, whether the density jumps there.

    Parameters
    ----------
    result : SplicedResult
        Output of :func:`~py_distcomp.fit_spliced`.
    data_name : str, default='Data'
        Legend label for the observations.
    title : str, optional
        Plot title.
    bins : int, str or sequence, default='sturges'
        Histogram binning, passed to ``numpy.histogram_bin_edges``.
    fitnbpts : int, default=401
        Points used to draw the fitted curve.
    log_y : bool, default=False
        Log the density axis, which is usually the only way to see a tail at
        all next to a body many times taller.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    data = result.data
    threshold = result.threshold
    if title is None:
        title = (f"{result.name} spliced at {threshold:.4g} "
                 f"({result.n_above} of {result.n} above)")

    fig = go.Figure()

    edges = np.histogram_bin_edges(data, bins=bins)
    counts, edges = np.histogram(data, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres, y=counts, width=np.diff(edges),
        name=f"{data_name} (empirical)",
        marker=dict(color="gray", opacity=0.55, line=dict(width=0.5, color="white")),
        hovertemplate="Bin centre: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    # Each side drawn over its own range, so the handover is visible rather
    # than smoothed across by a single curve.
    low, high = float(np.min(data)), float(np.max(data))
    n_below = max(2, int(fitnbpts * max(0.15, min(0.85, result.weight))))
    below = np.linspace(low, threshold, n_below)
    above = np.linspace(threshold, high, fitnbpts - n_below)

    fig.add_trace(go.Scatter(
        x=below, y=result.dist.pdf(below),
        mode="lines", name=f"{result.lower_model} (below)",
        line=dict(color=DISTRIBUTION_COLORS[0], width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Density: %{y:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=above, y=result.dist.pdf(above),
        mode="lines", name=f"{result.upper_model} (above)",
        line=dict(color=DISTRIBUTION_COLORS[1], width=2.5),
        hovertemplate="Value: %{x:.3f}<br>Density: %{y:.5f}<extra></extra>",
    ))

    fig.add_vline(
        x=threshold,
        line=dict(color="black", width=1.5, dash="dash"),
        annotation=dict(text=f"θ = {threshold:.4g}", font=dict(size=11)),
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="data", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="lightgray",
                   type="log" if log_y else "linear"),
        template="plotly_white",
        width=width, height=height,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray", borderwidth=1),
        hovermode="closest",
        bargap=0,
    )
    return fig


def threshold_profile_plot(
    result: SplicedResult,
    level: float = 0.95,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 460,
) -> go.Figure:
    """The profile log-likelihood the threshold was chosen from.

    The threshold is not estimated by a smooth optimisation -- the likelihood
    changes shape whenever it crosses an observation -- so it is profiled over a
    grid instead.  That grid is worth looking at: a sharp peak means the data
    locates the join, and a flat ridge means it does not, whatever the point
    estimate says.

    Parameters
    ----------
    result : SplicedResult
        Output of :func:`~py_distcomp.fit_spliced`.
    level : float, default=0.95
        Confidence level for the shaded interval, from the usual
        ``chi2(1)`` cutoff.
    title : str, optional
        Plot title.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    profile = result.profile.sort_values("threshold")
    cutoff = result.loglik - stats.chi2.ppf(level, 1) / 2.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile["threshold"], y=profile["loglik"],
        mode="lines+markers",
        name="profile log-likelihood",
        line=dict(color=DISTRIBUTION_COLORS[2], width=2),
        marker=dict(size=4),
        customdata=profile["n_above"],
        hovertemplate=("θ: %{x:.4g}<br>log-likelihood: %{y:.3f}<br>"
                       "observations above: %{customdata}<extra></extra>"),
    ))

    inside = profile[profile["loglik"] >= cutoff]["threshold"]
    if not inside.empty:
        fig.add_vrect(
            x0=float(inside.min()), x1=float(inside.max()),
            fillcolor="rgba(150,150,150,0.18)", line_width=0,
            annotation=dict(text=f"{level:.0%} interval", font=dict(size=11)),
            annotation_position="top right",
        )
    fig.add_hline(
        y=cutoff, line=dict(color="red", width=1.2, dash="dot"),
        annotation=dict(text=f"max − χ²({level:.0%})/2", font=dict(size=10)),
        annotation_position="bottom right",
    )
    # Kept clear of the interval's own label, which sits at the other end.
    fig.add_vline(
        x=result.threshold, line=dict(color="black", width=1.5, dash="dash"),
        annotation=dict(text=f"θ̂ = {result.threshold:.4g}", font=dict(size=11)),
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text=title or "Profile log-likelihood for the threshold",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="threshold θ", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="log-likelihood", showgrid=True, gridcolor="lightgray"),
        template="plotly_white",
        width=width, height=height,
        showlegend=False,
        hovermode="closest",
    )
    return fig
