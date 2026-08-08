"""
Plots of bootstrapped parameter estimates.

``bootdist_plot``     the bootstrapped parameter cloud, R's ``plot.bootdist``.
``confint_plot``      estimates with their intervals, several fits side by side.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bootdist import BootdistResult
from .gofstat import FitResult
from .mixture import MixtureResult
from .quantile_multi_comparison import DISTRIBUTION_COLORS

__all__ = ["bootdist_plot", "confint_plot"]


def bootdist_plot(
    result: BootdistResult,
    title: Optional[str] = None,
    parameters: Optional[Sequence[str]] = None,
    subplot_size: int = 300,
) -> go.Figure:
    """The bootstrapped parameter cloud -- R's ``plot.bootdist``.

    One panel per parameter pair, scattering the resampled estimates, with the
    original estimate marked. A single-parameter fit gets a histogram instead.
    The shape of the cloud is the point: a diagonal smear means the parameters
    trade off against each other and cannot be interpreted independently.

    Parameters
    ----------
    result : BootdistResult
        Output of :func:`~py_distcomp.bootdist`.
    title : str, optional
        Plot title.
    parameters : sequence of str, optional
        Restrict to these parameters. Useful for a mixture, where the full
        grid of six is unwieldy.
    subplot_size : int, default=300
        Approximate panel size in pixels.

    Returns
    -------
    go.Figure
    """
    estimates = result.estimates
    names = list(parameters) if parameters else list(estimates.columns)
    missing = [n for n in names if n not in estimates.columns]
    if missing:
        raise ValueError(f"Unknown parameter(s): {', '.join(missing)}")
    if title is None:
        title = (
            f"Bootstrapped estimates for {result.fit.name} "
            f"({result.n_converged} {result.method} resamples)"
        )

    original = result.fit.estimate

    if len(names) == 1:
        name = names[0]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=estimates[name],
            nbinsx=40,
            name="bootstrap",
            marker=dict(color=DISTRIBUTION_COLORS[0], opacity=0.75),
            hovertemplate=f"{name}: %{{x:.4g}}<br>count: %{{y}}<extra></extra>",
        ))
        _mark_original(fig, original[name], result.ci.loc[name])
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title=name,
            yaxis_title="count",
            template="plotly_white",
            width=700,
            height=460,
            bargap=0.02,
        )
        return fig

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"{b} vs {a}" for a, b in pairs],
        horizontal_spacing=0.10, vertical_spacing=0.14,
    )

    for i, (a, b) in enumerate(pairs):
        row, col = i // ncols + 1, i % ncols + 1
        fig.add_trace(go.Scatter(
            x=estimates[a], y=estimates[b],
            mode="markers",
            showlegend=False,
            marker=dict(size=4, opacity=0.35, color=DISTRIBUTION_COLORS[0]),
            hovertemplate=f"{a}: %{{x:.4g}}<br>{b}: %{{y:.4g}}<extra></extra>",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[original[a]], y=[original[b]],
            mode="markers",
            showlegend=False,
            marker=dict(size=12, color="red", symbol="x-thin",
                        line=dict(width=3, color="red")),
            hovertemplate=(f"<b>estimate</b><br>{a}: %{{x:.4g}}<br>"
                           f"{b}: %{{y:.4g}}<extra></extra>"),
        ), row=row, col=col)
        fig.update_xaxes(title_text=a, row=row, col=col,
                         showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(title_text=b, row=row, col=col,
                         showgrid=True, gridcolor="lightgray")

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template="plotly_white",
        width=subplot_size * ncols + 140,
        height=subplot_size * nrows + 140,
        hovermode="closest",
    )
    fig.update_annotations(font_size=12)
    return fig


def _mark_original(fig: go.Figure, estimate: float, limits: pd.Series) -> None:
    fig.add_vline(x=estimate, line=dict(color="red", width=2),
                  annotation=dict(text="estimate", font=dict(size=11)),
                  annotation_position="top right")
    for label, value in limits.items():
        if label == "median":
            continue
        fig.add_vline(x=value, line=dict(color="black", width=1, dash="dash"),
                      annotation=dict(text=label, font=dict(size=10)),
                      annotation_position="bottom right")


def confint_plot(
    fits: Union[Dict[str, object], Sequence[object]],
    parameter: Optional[str] = None,
    level: float = 0.95,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 450,
) -> go.Figure:
    """Estimates with confidence intervals, several fits side by side.

    This is the comparison the paper needs and could not make: whether two
    population subsets really differ in a fitted parameter, or whether their
    intervals overlap.

    Parameters
    ----------
    fits : mapping of label to fit, or sequence of fits
        :class:`~py_distcomp.FitResult` or
        :class:`~py_distcomp.BootdistResult` objects. A ``BootdistResult``
        contributes its percentile interval; a bare fit contributes the Wald
        interval from the observed information.
    parameter : str, optional
        Which parameter to plot. Defaults to the first one the fits share.
    level : float, default=0.95
        Coverage, used for the Wald intervals. A ``BootdistResult`` uses the
        level it was computed at.
    title : str, optional
        Plot title.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
    """
    if isinstance(fits, dict):
        items = list(fits.items())
    else:
        items = [(getattr(f, "fit", f).name, f) for f in fits]
    if not items:
        raise ValueError("At least one fit is required")

    rows = []
    for label, item in items:
        if isinstance(item, BootdistResult):
            if parameter is None:
                parameter = list(item.estimates.columns)[0]
            if parameter not in item.ci.index:
                raise ValueError(f"'{parameter}' was not estimated for {label}")
            limits = item.ci.loc[parameter]
            lower, upper = limits.iloc[1], limits.iloc[2]
            estimate = item.fit.estimate[parameter]
            source = f"{item.conf_level:.0%} percentile"
        else:
            if parameter is None:
                parameter = list(item.estimate)[0]
            if parameter not in item.estimate:
                raise ValueError(f"'{parameter}' was not estimated for {label}")
            interval = item.confint(level)
            estimate = item.estimate[parameter]
            lower = interval.loc[parameter, "lower"]
            upper = interval.loc[parameter, "upper"]
            source = f"{level:.0%} Wald"
        rows.append((str(label), estimate, lower, upper, source))

    if title is None:
        title = f"{parameter} with {rows[0][4]} intervals"

    labels = [r[0] for r in rows]
    estimates = [r[1] for r in rows]
    lowers = [r[2] for r in rows]
    uppers = [r[3] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=estimates,
        y=labels,
        mode="markers",
        name=parameter,
        marker=dict(size=11, color=DISTRIBUTION_COLORS[0], symbol="circle"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[u - e for e, u in zip(estimates, uppers)],
            arrayminus=[e - l for e, l in zip(estimates, lowers)],
            thickness=1.5,
            width=8,
        ),
        hovertemplate=("%{y}<br>" + parameter + ": %{x:.4g}<extra></extra>"),
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title=parameter, showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="", showgrid=True, gridcolor="lightgray",
                   autorange="reversed"),
        template="plotly_white",
        width=width,
        height=height,
        showlegend=False,
        hovermode="closest",
    )
    return fig
