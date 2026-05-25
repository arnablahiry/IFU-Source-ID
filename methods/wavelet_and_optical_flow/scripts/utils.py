"""Shared plotting utilities for IFU source identification notebooks."""
from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches


def add_beam(
    ax,
    bmin_pix: float,
    bmaj_pix: float,
    bpa_deg: float,
    xy_offset: tuple[float, float] = (10, 10),
    color: str = "white",
    crosshair: bool = True,
) -> None:
    """Draw a synthesized beam ellipse on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bmin_pix : float
        Minor-axis FWHM in pixels.
    bmaj_pix : float
        Major-axis FWHM in pixels (≥ bmin_pix).
    bpa_deg : float
        Beam position angle in degrees, CCW from the x-axis (East).
    xy_offset : (float, float)
        Offset from the bottom-left corner of the current axes limits, in
        data units (pixels).  Default (10, 10).
    color : str
        Colour of the ellipse outline and crosshairs.
    crosshair : bool
        Draw major/minor axis crosshairs inside the ellipse.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + xy_offset[0]
    y0 = ylim[0] + xy_offset[1]

    beam = mpatches.Ellipse(
        (x0, y0),
        width=bmaj_pix,
        height=bmin_pix,
        angle=bpa_deg,
        edgecolor=color,
        facecolor="none",
        linewidth=1.2,
        zorder=8,
    )
    ax.add_patch(beam)

    if crosshair:
        theta = np.deg2rad(-bpa_deg)
        dx_minor = 0.5 * bmin_pix * np.sin(theta)
        dy_minor = 0.5 * bmin_pix * np.cos(theta)
        dx_major = 0.5 * bmaj_pix * np.cos(theta)
        dy_major = -0.5 * bmaj_pix * np.sin(theta)
        ax.plot(
            [x0 - dx_minor, x0 + dx_minor],
            [y0 - dy_minor, y0 + dy_minor],
            color=color, linewidth=1.0, alpha=0.7, zorder=9,
        )
        ax.plot(
            [x0 - dx_major, x0 + dx_major],
            [y0 - dy_major, y0 + dy_major],
            color=color, linewidth=1.0, alpha=0.7, zorder=9,
        )
