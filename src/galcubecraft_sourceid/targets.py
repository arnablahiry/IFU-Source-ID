"""Ground-truth target construction for 3D source identification."""
from __future__ import annotations

import numpy as np


CLASS_NAMES = ("central", "satellite")


def galaxy_channels(systemic_vels: np.ndarray, average_vels: np.ndarray) -> np.ndarray:
    """Map each galaxy's systemic LOS velocity to the nearest spectral channel.

    Parameters
    ----------
    systemic_vels : (n_gals,) array of galaxy systemic velocities (km/s).
    average_vels  : (n_ch,)   array of channel centre velocities (km/s).

    Returns
    -------
    (n_gals,) int array of channel indices.
    """
    diff = np.abs(systemic_vels[:, None] - average_vels[None, :])
    return np.argmin(diff, axis=1)


def build_heatmap(
    cube_shape,
    centers_cyx,
    classes,
    sigma=1.5,
    n_classes=2,
):
    """Build a per-class 3D Gaussian heatmap target.

    Parameters
    ----------
    cube_shape : (n_ch, n_y, n_x) — the cube's shape.
    centers_cyx : (n_gals, 3) array of float coordinates in (channel, y, x).
    classes : (n_gals,) int array in [0, n_classes).
    sigma : Gaussian blob radius (in voxels, isotropic in (ch, y, x)).
    n_classes : number of channels in the heatmap (default 2: central, sat).

    Returns
    -------
    np.ndarray (n_classes, n_ch, n_y, n_x), float32 in [0, 1].
    """
    n_ch, n_y, n_x = cube_shape
    out = np.zeros((n_classes, n_ch, n_y, n_x), dtype=np.float32)
    if len(centers_cyx) == 0:
        return out

    # Grid axes reused across galaxies
    cc = np.arange(n_ch)[:, None, None]
    yy = np.arange(n_y)[None, :, None]
    xx = np.arange(n_x)[None, None, :]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    for (c, y, x), cls in zip(np.asarray(centers_cyx, dtype=float), classes):
        cls = int(cls)
        if cls < 0 or cls >= n_classes:
            continue
        d2 = (cc - c) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
        blob = np.exp(-d2 * inv2s2).astype(np.float32)
        # Take max over overlapping blobs so peaks remain at 1.
        np.maximum(out[cls], blob, out=out[cls])
    return out


