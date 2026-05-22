"""Per-channel starlet (à trous IUWT) source detection for 3-D spectral cubes.

Strategy
--------
The scarlet2 starlet transform decomposes each 2-D spectral slice into detail
bands at increasing angular scales plus a coarse residual.  Compact sources
produce strong coefficients in the fine-scale bands; diffuse emission sits in
the coarser bands and is naturally suppressed.

For each channel slice, ``wavelet_footprints_scarlet2``:

1. Computes the starlet transform via ``scarlet2.wavelets.starlet_transform``.
2. Thresholds each detail scale independently using a per-scale MAD noise
   estimate, so the threshold adapts to the actual signal level at that scale
   rather than collapsing on noise-free or low-signal data.
3. Applies an absolute floor of 10 % of the detection-plane peak to suppress
   float32 rounding artefacts in noise-free cubes.
4. Extracts connected blobs on the chosen detection scale and returns peak
   coordinates, binary footprint masks, and bounding boxes.

Input formats
-------------
``load_cube`` accepts .h5/.hdf5, .fits/.fit, .npy, .npz.

Usage (standalone)::

    python wavelet_detections.py \\
        --cube  data/clean_cube.npy \\
        --out   /tmp/detections \\
        --channels 70,74 \\
        --k-sigma 5 --scales 6 --use-scale 5 --min-area 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scarlet2 as sc
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Multi-format cube loader
# ---------------------------------------------------------------------------

def load_cube(path: str | Path) -> np.ndarray:
    """Load a spectral cube from HDF5, FITS, .npy, or .npz.

    Always returns float32 (n_ch, H, W).  NaNs are replaced with 0.
    """
    path = Path(path)
    suf  = path.suffix.lower()

    if suf in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            cube = f["cube"][:].astype(np.float32)

    elif suf in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data
        if data is None:
            raise ValueError(f"No data in primary HDU of {path}")
        data = np.squeeze(data).astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError(
                f"Cannot interpret FITS array with shape {data.shape} as (n_ch,H,W)"
            )
        cube = data

    elif suf == ".npy":
        cube = np.load(path).astype(np.float32)
        if cube.ndim == 2:
            cube = cube[np.newaxis]

    elif suf == ".npz":
        arch = np.load(path)
        key  = "cube" if "cube" in arch else list(arch.keys())[0]
        cube = arch[key].astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported extension {suf!r}.  Use .h5/.hdf5, .fits/.fit, .npy, or .npz"
        )

    if cube.ndim != 3:
        raise ValueError(f"Loaded array has shape {cube.shape}; expected (n_ch, H, W)")

    np.nan_to_num(cube, copy=False, nan=0.0)
    return cube


def active_channels(cube: np.ndarray, threshold_frac: float = 0.05) -> list[int]:
    """Return indices of channels whose positive flux exceeds *threshold_frac* × max.

    Uses only positive flux so noise-dominated channels (where positive and
    negative values roughly cancel) do not inflate the total.
    """
    flux   = np.nansum(np.clip(cube, 0.0, None), axis=(1, 2))
    thresh = threshold_frac * float(flux.max())
    return [int(i) for i in np.where(flux >= thresh)[0]]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ChannelDetection(NamedTuple):
    """All detection results for a single spectral channel."""

    channel: int
    # Raw channel slice (H, W) float32.
    image: np.ndarray
    # One (H, W) bool mask per detected source blob.
    footprint_masks: list
    # (row, col) integer tuples, one per blob.
    peaks: list
    # (y0, x0, y1, x1) integer tuples, one per blob.
    boxes: list
    # Thresholded starlet coefficient cube (n_scales+1, H, W) float32.
    detect_coeffs: np.ndarray


# ---------------------------------------------------------------------------
# Per-channel detection
# ---------------------------------------------------------------------------

def wavelet_footprints_scarlet2(
    image: np.ndarray,
    scales: int = 4,
    k_sigma: float = 2.3,
    use_scale: int = 2,
    min_area: int = 10,
    thresh: float | None = None,
) -> ChannelDetection:
    """Detect compact-source footprints in a single 2-D image via starlet thresholding.

    Parameters
    ----------
    image :
        2-D spectral slice (H, W).
    scales :
        Total number of starlet scales (including the coarse residual plane).
    k_sigma :
        Detection threshold in units of per-scale MAD noise.
    use_scale :
        1-based index of the detail band used for blob detection.
        Scale 1 is the finest (sub-pixel structure); higher scales capture
        progressively larger compact sources.
    min_area :
        Minimum blob area in pixels; smaller blobs are discarded as artefacts.
    thresh :
        Absolute lower bound on the detection-plane value.  ``None`` (default)
        sets it to 10 % of the detection-plane maximum, which prevents
        float32 rounding noise from triggering detections in signal-free channels.

    Returns
    -------
    ChannelDetection
        channel is set to -1 here; callers should replace it via ``._replace``.

    Notes
    -----
    Each scale is thresholded with its *own* MAD estimate rather than a single
    image-level sigma.  At coarser scales the B3-spline propagation factor is
    ~0.086× the image noise, so a single sigma would either be 11× too strict
    (missing real sources) or, when the image is nearly empty and sigma → 0,
    collapse to zero and trigger detections everywhere from rounding errors.
    Per-scale MAD avoids both failure modes.
    """
    img    = np.asarray(image, dtype=np.float32)
    coeffs = np.asarray(
        sc.wavelets.starlet_transform(img, scales=scales), dtype=np.float32
    )

    # Per-scale thresholding — noise level differs by ~10× across scales.
    detect = np.zeros_like(coeffs)
    for i in range(coeffs.shape[0] - 1):
        # MAD of coefficients at this scale, +epsilon to guard against all-zero planes.
        sigma_i = 1.4826 * np.median(np.abs(coeffs[i] - np.median(coeffs[i]))) + 1e-12
        detect[i] = np.where(np.abs(coeffs[i]) > k_sigma * sigma_i, coeffs[i], 0.0)
    detect[-1] = coeffs[-1]   # coarse residual kept as-is
    detect[detect < 0] = 0    # positive emission only

    scale_idx = int(np.clip(use_scale - 1, 0, detect.shape[0] - 1))
    plane = detect[scale_idx]

    # 10 % of peak prevents float32 rounding artefacts (~1e-7) from
    # triggering detections when a signal-free channel makes sigma → 0.
    effective_thresh = 0.1 * float(plane.max()) if thresh is None else thresh
    binary           = plane > effective_thresh
    labeled, _       = label(binary)
    regions = [
        r for r in regionprops(labeled, intensity_image=plane) if r.area >= min_area
    ]

    peaks, footprint_masks, boxes = [], [], []
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        patch = plane[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        peaks.append((int(y0 + py), int(x0 + px)))
        footprint_masks.append(labeled == r.label)
        boxes.append((y0, x0, y1, x1))

    # Fallback: if the wavelet finds nothing, pin to the raw-image peak so
    # downstream callers always have at least one position to work with.
    if not peaks:
        fb = peak_local_max(img, min_distance=3, num_peaks=1, exclude_border=False)
        if len(fb):
            yy, xx = int(fb[0, 0]), int(fb[0, 1])
            peaks = [(yy, xx)]
            m     = np.zeros(img.shape, dtype=bool)
            m[max(0, yy-2):min(img.shape[0], yy+3),
              max(0, xx-2):min(img.shape[1], xx+3)] = True
            footprint_masks = [m]
            boxes = [(max(0,yy-2), max(0,xx-2),
                      min(img.shape[0],yy+3), min(img.shape[1],xx+3))]

    return ChannelDetection(
        channel=-1, image=img,
        footprint_masks=footprint_masks, peaks=peaks, boxes=boxes,
        detect_coeffs=detect,
    )


def detect_cube_per_channel(
    cube: np.ndarray,
    channel_list: list[int] | None = None,
    scales: int = 4,
    k_sigma: float = 2.3,
    use_scale: int = 2,
    min_area: int = 10,
    thresh: float | None = None,
) -> list[ChannelDetection]:
    """Run ``wavelet_footprints_scarlet2`` on every channel in *channel_list*.

    Returns an ordered list of :class:`ChannelDetection` objects with the
    ``channel`` field correctly set to the cube channel index.
    """
    if channel_list is None:
        channel_list = list(range(cube.shape[0]))

    results = []
    for ch in channel_list:
        det = wavelet_footprints_scarlet2(
            cube[ch],
            scales=scales, k_sigma=k_sigma,
            use_scale=use_scale, min_area=min_area, thresh=thresh,
        )
        results.append(det._replace(channel=ch))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--cube",             required=True,
                    help="Cube file: .h5/.hdf5, .fits/.fit, .npy, .npz")
    ap.add_argument("--out",              required=True,
                    help="Output directory")
    ap.add_argument("--channels",         default=None,
                    help="Comma-separated channel indices; default: auto active")
    ap.add_argument("--active-threshold", type=float, default=0.05)
    ap.add_argument("--scales",           type=int,   default=6)
    ap.add_argument("--k-sigma",          type=float, default=5.0)
    ap.add_argument("--use-scale",        type=int,   default=5)
    ap.add_argument("--min-area",         type=int,   default=20)
    ap.add_argument("--thresh",           type=float, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cube = load_cube(args.cube)
    print(f"Cube: {cube.shape}  range [{cube.min():.3e}, {cube.max():.3e}]")

    if args.channels:
        channel_list = [int(c) for c in args.channels.split(",")]
    else:
        channel_list = active_channels(cube, threshold_frac=args.active_threshold)
        print(f"Auto-selected {len(channel_list)} active channels "
              f"(ch {channel_list[0]}–{channel_list[-1]})")

    detections = detect_cube_per_channel(
        cube, channel_list=channel_list,
        scales=args.scales, k_sigma=args.k_sigma,
        use_scale=args.use_scale, min_area=args.min_area, thresh=args.thresh,
    )

    for det in detections:
        print(f"  ch {det.channel:4d}  {len(det.peaks)} blobs")

    summary = {
        "cube": str(args.cube),
        "channels": channel_list,
        "n_detections_per_channel": [len(d.peaks) for d in detections],
        "params": vars(args),
    }
    (out / "detections_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}/detections_summary.json")


if __name__ == "__main__":
    main()
