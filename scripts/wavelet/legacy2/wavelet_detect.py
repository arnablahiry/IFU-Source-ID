"""Per-slice wavelet source detection above diffuse emission for 3-D spectral cubes.

Strategy — compact sources vs. diffuse emission
------------------------------------------------
The starlet (à trous) transform decomposes an image into detail bands at
increasing angular scales plus a coarse residual.  Diffuse emission lives in
the coarse residual and in the large-scale detail bands; compact sources appear
exclusively in the fine-scale detail bands.

By thresholding *only* the fine scales (controlled by `detail_scales`, default
0..2) we detect compact sources *above* the diffuse emission without the diffuse
halo inflating the significance map or creating spurious large-footprint detections.
Optionally (`subtract_diffuse=True`) the coarse reconstruction is subtracted from
the slice before detection, which further suppresses the diffuse pedestal.

Input formats
-------------
`load_cube()` accepts:
  *.h5 / *.hdf5   — HDF5 file with a 'cube' dataset  (n_ch, H, W)
  *.fits           — FITS file; 4-D Stokes cubes are squeezed automatically
  *.npy            — raw numpy array saved with np.save
  *.npz            — numpy archive; first array or 'cube' key is used

Usage (standalone):
    python scripts/wavelet/wavelet_detect.py \\
        --cube data/all_cubes/cube_10001.h5 \\
        --out /tmp/wavelet_detect \\
        --k-sigma 3.0 --scales 5 --detail-scales 0,1,2 --min-area 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import center_of_mass, label as nd_label
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Multi-format cube loader
# ---------------------------------------------------------------------------

def load_cube(path: str | Path) -> np.ndarray:
    """Load a spectral cube from HDF5, FITS, .npy, or .npz.

    Always returns float32 (n_ch, H, W).
    """
    path = Path(path)
    suf = path.suffix.lower()

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
        # After squeezing, expect 3-D (n_ch, H, W).
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Cannot interpret FITS array with shape {data.shape} as (n_ch,H,W)")
        cube = data

    elif suf == ".npy":
        cube = np.load(path).astype(np.float32)
        if cube.ndim == 2:
            cube = cube[np.newaxis]

    elif suf == ".npz":
        arch = np.load(path)
        if "cube" in arch:
            cube = arch["cube"].astype(np.float32)
        else:
            key = list(arch.keys())[0]
            cube = arch[key].astype(np.float32)

    else:
        raise ValueError(f"Unsupported file extension: {suf!r}. "
                         "Use .h5/.hdf5, .fits/.fit, .npy, or .npz")

    if cube.ndim != 3:
        raise ValueError(f"Loaded array has shape {cube.shape}; expected (n_ch, H, W)")

    # Replace NaNs with 0 so downstream arithmetic stays finite.
    np.nan_to_num(cube, copy=False, nan=0.0)
    return cube


def active_channels(cube: np.ndarray, threshold_frac: float = 0.05) -> list[int]:
    """Return indices of channels whose positive flux exceeds `threshold_frac` of the max.

    Uses only positive flux (clipped at zero) so that noise-dominated channels
    whose positive and negative values roughly cancel don't inflate the total.
    """
    flux = np.nansum(np.clip(cube, 0.0, None), axis=(1, 2))
    thresh = threshold_frac * float(flux.max())
    return [int(i) for i in np.where(flux >= thresh)[0]]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(out_dir: Path, name: str = "wavelet_detect") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Starlet (IUWT, à trous B3-spline)
# ---------------------------------------------------------------------------

_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def _atrous_conv2d(img: np.ndarray, step: int) -> np.ndarray:
    from scipy.ndimage import convolve1d
    h = np.zeros(step * 4 + 1, dtype=np.float64)
    h[::step] = _B3
    out = convolve1d(img.astype(np.float64), h, axis=0, mode="reflect")
    return convolve1d(out, h, axis=1, mode="reflect")


def starlet_transform(image: np.ndarray, n_scales: int = 5) -> np.ndarray:
    """Return (n_scales+1, H, W) coefficients.  Last plane = coarse residual."""
    img = image.astype(np.float64)
    coeffs = np.empty((n_scales + 1, *img.shape))
    c = img
    for j in range(n_scales):
        c_next = _atrous_conv2d(c, step=2 ** j)
        coeffs[j] = c - c_next
        c = c_next
    coeffs[-1] = c
    return coeffs


def starlet_coarse(image: np.ndarray, n_scales: int = 5) -> np.ndarray:
    """Return the coarse residual only (the 'diffuse' component)."""
    c = image.astype(np.float64)
    for j in range(n_scales):
        c = _atrous_conv2d(c, step=2 ** j)
    return c.astype(np.float32)


# Tabulated B3-spline noise propagation factors σ_j / σ_0 for j = 0..6.
_PROPAGATION = np.array([0.889, 0.201, 0.086, 0.041, 0.020, 0.010, 0.005])


def mad_noise(arr: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))) + 1e-12)


# ---------------------------------------------------------------------------
# Per-slice significance (no detection — just the significance map)
# ---------------------------------------------------------------------------

def significance_slice(
    image: np.ndarray,
    n_scales: int = 5,
    k_sigma: float = 3.0,
    detail_scales: tuple[int, ...] = (0, 1, 2),
    subtract_diffuse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (diffuse, significance) maps for one spectral slice.

    No blob detection is done here — detection happens globally across the
    collapsed cube in `detect_cube`.
    """
    H, W = image.shape
    img = image.astype(np.float64)

    diffuse = starlet_coarse(img, n_scales=n_scales).astype(np.float32)
    if subtract_diffuse:
        img = img - diffuse.astype(np.float64)

    coeffs = starlet_transform(img, n_scales=n_scales)
    noise0 = mad_noise(coeffs[0])

    significance = np.zeros((H, W), dtype=np.float32)
    for j in detail_scales:
        if j >= coeffs.shape[0] - 1:
            continue
        e_j = _PROPAGATION[min(j, len(_PROPAGATION) - 1)]
        thresh = k_sigma * noise0 * e_j
        plane = coeffs[j].astype(np.float32)
        significance += np.where(plane > thresh, plane, 0.0)

    return diffuse, significance


# ---------------------------------------------------------------------------
# Global source detection (collapsed across channels) + per-channel measurement
# ---------------------------------------------------------------------------

class SourceRegion(NamedTuple):
    y: float           # sub-pixel centroid (row)
    x: float           # sub-pixel centroid (col)
    flux: float        # sum of original flux within footprint
    snr: float         # peak significance / global significance max
    area_px: int       # footprint size in pixels
    mask: np.ndarray   # (H, W) bool — full-image footprint mask


# keep detect_slice as a thin wrapper for callers that use it directly
def detect_slice(
    image: np.ndarray,
    n_scales: int = 5,
    k_sigma: float = 3.0,
    min_area: int = 4,
    detail_scales: tuple[int, ...] = (0, 1, 2),
    subtract_diffuse: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[SourceRegion]]:
    """Legacy single-slice detection (used in wavelet_scales plot).

    Returns (diffuse, significance, regions).
    Prefer `detect_cube` for full cubes — it detects globally.
    """
    from scipy.ndimage import center_of_mass, label as nd_label
    from skimage.measure import regionprops

    H, W = image.shape
    diffuse, significance = significance_slice(
        image, n_scales=n_scales, k_sigma=k_sigma,
        detail_scales=detail_scales, subtract_diffuse=subtract_diffuse,
    )

    binary = significance > 0
    labeled, _ = nd_label(binary)
    regions: list[SourceRegion] = []
    for reg in regionprops(labeled, intensity_image=significance):
        if reg.area < min_area:
            continue

        y0, x0, y1, x1 = reg.bbox
        patch = significance[y0:y1, x0:x1]
        cy, cx = center_of_mass(patch)
        cy = float(np.clip(cy + y0, 0, H - 1))
        cx = float(np.clip(cx + x0, 0, W - 1))

        mask = labeled == reg.label
        regions.append(SourceRegion(
            y=cy, x=cx,
            flux=float(significance[mask].sum()),
            snr=float(significance[mask].max() / (noise0 + 1e-12)),
            area_px=int(reg.area),
            mask=mask,
        ))

    regions.sort(key=lambda r: -r.flux)
    return diffuse, significance.astype(np.float32), regions


# ---------------------------------------------------------------------------
# Global source detection + per-channel measurement
# ---------------------------------------------------------------------------

class CubeDetections:
    """Container for per-channel detection results.

    Sources are detected once on the max-collapsed significance map, giving
    exactly N global blobs.  Per-channel, each source's flux and centroid are
    measured within its global footprint mask.
    """

    def __init__(
        self,
        global_sources: list[SourceRegion],   # N global source blobs
        channel_regions: list[list[SourceRegion]],  # [ch_idx][src_idx]
        significance: np.ndarray,             # (n_proc_ch, H, W) float32
        diffuse_maps: np.ndarray,             # (n_proc_ch, H, W) float32
        global_significance: np.ndarray,      # (H, W) float32 — max across channels
        channel_list: list[int],
        cube_shape: tuple[int, int, int],
    ) -> None:
        self.global_sources = global_sources
        self.channel_regions = channel_regions
        self.significance = significance
        self.diffuse_maps = diffuse_maps
        self.global_significance = global_significance
        self.channel_list = channel_list
        self.cube_shape = cube_shape

    def n_sources(self) -> int:
        return len(self.global_sources)

    def n_channels(self) -> int:
        return len(self.channel_list)

    def total_detections(self) -> int:
        return sum(len(r) for r in self.channel_regions)

    def n_per_channel(self) -> list[int]:
        return [len(r) for r in self.channel_regions]

    def union_mask(self, ch_idx: int) -> np.ndarray:
        """Boolean union of all source masks for a processed channel index."""
        _, H, W = self.cube_shape
        m = np.zeros((H, W), dtype=bool)
        for reg in self.channel_regions[ch_idx]:
            m |= reg.mask
        return m


def detect_cube(
    cube: np.ndarray,
    channel_list: list[int] | None = None,
    n_scales: int = 5,
    k_sigma: float = 3.0,
    min_area: int = 9,
    max_area: int | None = None,
    detail_scales: tuple[int, ...] = (0, 1, 2),
    subtract_diffuse: bool = True,
    peak_min_distance: int = 20,
    peak_threshold_rel: float = 0.05,
    log: logging.Logger | None = None,
) -> CubeDetections:
    """Detect sources globally via peak finding, then measure per channel.

    Pipeline
    --------
    1. Compute per-channel significance maps (fine starlet scales above diffuse).
    2. Max-project across channels → global significance map.
    3. Find N peaks on the global map with `peak_local_max` (one peak per
       distinct source regardless of how they connect at the base).
    4. Assign every significant pixel to its nearest peak (Voronoi).
       Each Voronoi region is one source's spatial territory.
    5. Per channel: the source footprint = significant pixels in that channel
       that fall inside the source's territory.  Centroid and flux are
       measured within this per-channel footprint.

    This approach finds exactly N sources where N is the number of distinct
    emission peaks, not the number of connected-component fragments.
    """
    from skimage.feature import peak_local_max
    from scipy.ndimage import binary_dilation as _dilate

    n_ch, H, W = cube.shape
    if channel_list is None:
        channel_list = list(range(n_ch))
    n_proc = len(channel_list)

    # Step 1: per-channel significance + diffuse maps
    if log:
        log.info("  Computing per-channel significance maps ...")
    sig_maps  = np.zeros((n_proc, H, W), dtype=np.float32)
    diff_maps = np.zeros((n_proc, H, W), dtype=np.float32)

    for idx, ch in enumerate(channel_list):
        diff, sig = significance_slice(
            cube[ch],
            n_scales=n_scales,
            k_sigma=k_sigma,
            detail_scales=detail_scales,
            subtract_diffuse=subtract_diffuse,
        )
        sig_maps[idx]  = sig
        diff_maps[idx] = diff
        if log and (idx % max(1, n_proc // 10) == 0 or idx == n_proc - 1):
            log.info("  ch %3d/%3d  sig_max=%.4f", ch, channel_list[-1], float(sig.max()))

    # Step 2: global significance = max across channels
    global_sig = sig_maps.max(axis=0)   # (H, W)

    # Step 3: find peaks on the global map → one peak per distinct source
    if log:
        log.info("  Finding peaks (min_distance=%d, threshold_rel=%.2f) ...",
                 peak_min_distance, peak_threshold_rel)

    peak_coords = peak_local_max(
        global_sig,
        min_distance=peak_min_distance,
        threshold_rel=peak_threshold_rel,
    )  # (N, 2) in (row, col)

    if len(peak_coords) == 0:
        if log:
            log.warning("  No peaks found — try lowering k_sigma or peak_threshold_rel")
        return CubeDetections(
            global_sources=[], channel_regions=[[] for _ in channel_list],
            significance=sig_maps, diffuse_maps=diff_maps,
            global_significance=global_sig, channel_list=channel_list,
            cube_shape=(n_ch, H, W),
        )

    # Step 4: Voronoi assignment — each significant pixel goes to its nearest peak
    # Build a distance map for each peak and assign by argmin.
    yy, xx = np.mgrid[:H, :W]
    dist_stack = np.stack([
        np.hypot(yy - py, xx - px) for py, px in peak_coords
    ], axis=0)  # (N, H, W)
    assignment = dist_stack.argmin(axis=0)  # (H, W) — which peak owns each pixel

    # Build territory masks: significant region closest to each peak
    # Dilate slightly so channels with shifted emission don't fall outside.
    significant_anywhere = global_sig > 0
    territories = []
    for k in range(len(peak_coords)):
        base = significant_anywhere & (assignment == k)
        territories.append(_dilate(base, iterations=5))

    # Build global SourceRegion for each peak
    global_sources: list[SourceRegion] = []
    for k, (py, px) in enumerate(peak_coords):
        m = territories[k] & significant_anywhere
        if m.sum() < min_area:
            continue
        if max_area is not None and m.sum() > max_area:
            continue
        global_sources.append(SourceRegion(
            y=float(py), x=float(px),
            flux=float(global_sig[m].sum()),
            snr=float(global_sig[int(py), int(px)] / (global_sig.max() + 1e-12)),
            area_px=int(m.sum()),
            mask=m,
        ))

    # Keep territories aligned with accepted global_sources
    accepted_indices = [
        k for k, (py, px) in enumerate(peak_coords)
        if (territories[k] & significant_anywhere).sum() >= min_area
        and (max_area is None or (territories[k] & significant_anywhere).sum() <= max_area)
    ]
    territories = [territories[k] for k in accepted_indices]

    global_sources.sort(key=lambda r: -r.flux)
    # Re-sort territories to match
    order = sorted(range(len(global_sources)), key=lambda i: -global_sources[i].flux)
    global_sources = [global_sources[i] for i in order]
    territories    = [territories[i] for i in order]

    if log:
        log.info("  → %d global sources", len(global_sources))
        for i, src in enumerate(global_sources):
            log.info("    src %d  y=%.0f x=%.0f  area=%d  snr=%.2f",
                     i, src.y, src.x, src.area_px, src.snr)

    # Step 5: per-channel footprint within each Voronoi territory.
    #
    # Significant pixels in this channel that belong to this source's territory
    # form the per-channel footprint.  This can shrink, grow, or shift as the
    # source changes spectrally.  When a source is below threshold in a channel,
    # the territory mask is kept as a fall-back so flow tracking still has a
    # region to work with.
    channel_regions: list[list[SourceRegion]] = []
    for idx, ch in enumerate(channel_list):
        ch_regs: list[SourceRegion] = []
        sig = sig_maps[idx]
        for src, territory in zip(global_sources, territories):
            ch_mask = territory & (sig > 0)
            if ch_mask.any():
                # Keep only the connected component that contains the global peak.
                # If the peak pixel is below threshold in this channel, treat it
                # as a non-detection rather than guessing with a random component.
                labeled_ch, _ = nd_label(ch_mask)
                py, px = int(round(src.y)), int(round(src.x))
                peak_lbl = labeled_ch[py, px]
                if peak_lbl > 0:
                    ch_mask = labeled_ch == peak_lbl
                else:
                    ch_mask = np.zeros((H, W), dtype=bool)

            if ch_mask.any():
                sig_in_ch = np.zeros((H, W), dtype=np.float32)
                sig_in_ch[ch_mask] = sig[ch_mask]
                cy, cx = center_of_mass(sig_in_ch)
                cy = float(np.clip(cy, 0, H - 1))
                cx = float(np.clip(cx, 0, W - 1))
                raw_flux = float(cube[ch][ch_mask].sum())
                peak_snr  = float(sig[ch_mask].max() / (global_sig.max() + 1e-12))
                area = int(ch_mask.sum())
            else:
                cy, cx = src.y, src.x
                raw_flux = 0.0
                peak_snr  = 0.0
                area = 0
                ch_mask = territory  # fall back for flow masking

            ch_regs.append(SourceRegion(
                y=cy, x=cx,
                flux=raw_flux,
                snr=peak_snr,
                area_px=area,
                mask=ch_mask,
            ))
        channel_regions.append(ch_regs)

    return CubeDetections(
        global_sources=global_sources,
        channel_regions=channel_regions,
        significance=sig_maps,
        diffuse_maps=diff_maps,
        global_significance=global_sig,
        channel_list=channel_list,
        cube_shape=(n_ch, H, W),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cube", required=True,
                    help="Cube file: .h5/.hdf5, .fits/.fit, .npy, or .npz")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--scales", type=int, default=5,
                    help="Total starlet scales (default: 5)")
    ap.add_argument("--detail-scales", type=str, default="0,1,2",
                    help="Comma-separated fine scales used for detection (default: 0,1,2)")
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--max-area", type=int, default=None,
                    help="Maximum blob area in pixels (default: no limit)")
    ap.add_argument("--peak-min-distance", type=int, default=20,
                    help="Min pixel separation between source peaks (default: 20)")
    ap.add_argument("--peak-threshold-rel", type=float, default=0.05,
                    help="Peak detection threshold relative to global max (default: 0.05)")
    ap.add_argument("--no-subtract-diffuse", action="store_true",
                    help="Skip diffuse subtraction (old behaviour)")
    ap.add_argument("--channels", type=str, default=None,
                    help="Comma-separated channel indices; default: auto-detect active")
    ap.add_argument("--active-threshold", type=float, default=0.05,
                    help="Fraction of peak flux to define 'active' channels (default: 0.05)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    log.info("Loading cube: %s", args.cube)
    cube = load_cube(args.cube)
    n_ch, H, W = cube.shape
    log.info("Cube: ch=%d H=%d W=%d  range=[%.3e, %.3e]",
             n_ch, H, W, float(cube.min()), float(cube.max()))

    detail_scales = tuple(int(s) for s in args.detail_scales.split(","))

    if args.channels is not None:
        channel_list = [int(c) for c in args.channels.split(",")]
    else:
        channel_list = active_channels(cube, threshold_frac=args.active_threshold)
        log.info("Auto-detected %d active channels (%.0f%% threshold)",
                 len(channel_list), args.active_threshold * 100)

    dets = detect_cube(
        cube, channel_list=channel_list,
        n_scales=args.scales, k_sigma=args.k_sigma,
        min_area=args.min_area,
        detail_scales=detail_scales,
        subtract_diffuse=not args.no_subtract_diffuse,
        log=log,
    )

    np.savez_compressed(
        out_dir / "significance.npz",
        significance=dets.significance,
        diffuse=dets.diffuse_maps,
        channels=np.array(dets.channel_list, dtype=np.int32),
    )

    all_ch, all_y, all_x, all_flux, all_snr, all_area = [], [], [], [], [], []
    for idx, ch in enumerate(dets.channel_list):
        for reg in dets.channel_regions[idx]:
            all_ch.append(ch); all_y.append(reg.y); all_x.append(reg.x)
            all_flux.append(reg.flux); all_snr.append(reg.snr)
            all_area.append(reg.area_px)

    np.savez(out_dir / "detections.npz",
             channel=np.array(all_ch, dtype=np.int32),
             y=np.array(all_y, dtype=np.float32),
             x=np.array(all_x, dtype=np.float32),
             flux=np.array(all_flux, dtype=np.float32),
             snr=np.array(all_snr, dtype=np.float32),
             area_px=np.array(all_area, dtype=np.int32))

    log.info("Total detections: %d  (mean %.1f / ch)",
             dets.total_detections(), float(np.mean(dets.n_per_channel())))

    summary = {
        "cube": str(args.cube),
        "n_channels": len(dets.channel_list),
        "total_detections": dets.total_detections(),
        "mean_per_channel": float(np.mean(dets.n_per_channel())),
        "params": {
            "scales": args.scales,
            "detail_scales": list(detail_scales),
            "k_sigma": args.k_sigma,
            "min_area": args.min_area,
            "subtract_diffuse": not args.no_subtract_diffuse,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
