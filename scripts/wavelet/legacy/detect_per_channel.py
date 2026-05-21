"""Per-channel source detection using starlet (à trous wavelet) transform.

For each spectral channel of an IFU cube:
  1. Compute the starlet transform (scarlet2).
  2. Estimate per-scale noise from the finest scale (MAD estimator).
  3. Build a multi-scale significance mask: keep coefficients > k_sigma * noise.
  4. Reconstruct a denoised "significant signal" image by summing masked scales.
  5. Find connected footprints in the significance map and extract one detection
     per footprint (position, flux, SNR, morphology patch).

Output per cube:
  <out>/detections.npz   -- arrays: channel, y, x, flux, snr, scale_peak
  <out>/channel_maps.npz -- denoised significance images, one per channel
  <out>/summary.json     -- aggregate stats (n_detections per channel, etc.)

Example:
    python scripts/wavelet/detect_per_channel.py \\
        --cube data/all_cubes/cube_10001.h5 \\
        --out /tmp/detect_test \\
        --k-sigma 3.0 --min-area 4 --scales 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import label as nd_label, center_of_mass
from skimage.measure import regionprops


def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("detect_per_channel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "detect.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def mad_noise(arr: np.ndarray) -> float:
    """Robust noise estimate: 1.4826 * MAD of the array."""
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))) + 1e-12)


def detect_channel(
    image: np.ndarray,
    scales: int = 4,
    k_sigma: float = 3.0,
    min_area: int = 4,
    patch_radius: int = 7,
) -> tuple[np.ndarray, list[dict]]:
    """Run starlet detection on a single 2D channel image.

    Parameters
    ----------
    image      : (H, W) float32 observed channel slice.
    scales     : number of wavelet scales (starlet levels).
    k_sigma    : significance threshold in units of noise per scale.
    min_area   : minimum footprint area in pixels to keep a detection.
    patch_radius: half-width of the morphology patch saved per source.

    Returns
    -------
    significance : (H, W) float32 — sum of masked wavelet planes (denoised signal).
    detections   : list of dicts, one per source found, with keys:
                     y, x          — peak position (float, sub-pixel via CoM)
                     flux          — sum of significance image in footprint
                     snr           — peak / noise_scale0
                     area_px       — footprint area in pixels
                     patch         — (2*patch_radius+1)^2 normalised morphology
    """
    import scarlet2 as sc

    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape

    # Starlet transform: returns (scales+1, H, W) — last plane is the coarse residual.
    coeffs = np.array(sc.wavelets.starlet_transform(img, scales=scales), dtype=np.float32)

    # Noise estimated from the finest scale (scale 0 ≈ white noise on detector).
    noise_scale0 = mad_noise(coeffs[0])

    # Per-scale thresholding.
    # Noise propagation factor for the à trous starlet: σ_j ≈ σ_0 * e_j
    # where e_j = [0.889, 0.201, 0.086, 0.041, 0.020, ...] (tabulated for B3 spline).
    propagation = np.array([0.889, 0.201, 0.086, 0.041, 0.020, 0.010, 0.005])
    significance = np.zeros((H, W), dtype=np.float32)
    for s in range(coeffs.shape[0] - 1):   # skip the coarse residual (last plane)
        e_s = propagation[min(s, len(propagation) - 1)]
        noise_s = noise_scale0 * e_s
        thresh = k_sigma * noise_s
        plane = coeffs[s]
        mask = plane > thresh
        significance += np.where(mask, plane, 0.0)

    # Connected-component footprints on the significance image.
    binary = significance > 0
    labeled, n_labels = nd_label(binary)

    detections = []
    for reg in regionprops(labeled, intensity_image=significance):
        if reg.area < min_area:
            continue

        # Sub-pixel centroid via intensity-weighted centre of mass within bbox.
        y0, x0, y1, x1 = reg.bbox
        patch_sig = significance[y0:y1, x0:x1]
        cy, cx = center_of_mass(patch_sig)
        cy += y0
        cx += x0
        cy = float(np.clip(cy, 0, H - 1))
        cx = float(np.clip(cx, 0, W - 1))

        flux = float(reg.image_intensity.sum())
        snr = float(reg.image_intensity.max() / (noise_scale0 + 1e-12))

        # Morphology patch from the original (not significance) image.
        r = patch_radius
        yi, xi = int(round(cy)), int(round(cx))
        y0p = max(0, yi - r); y1p = min(H, yi + r + 1)
        x0p = max(0, xi - r); x1p = min(W, xi + r + 1)
        patch = img[y0p:y1p, x0p:x1p].copy()
        patch = np.clip(patch, 0.0, None)
        patch_sum = patch.sum()
        if patch_sum > 0:
            patch = patch / patch_sum

        detections.append({
            "y": cy,
            "x": cx,
            "flux": flux,
            "snr": snr,
            "area_px": int(reg.area),
            "patch": patch,
        })

    # Sort by flux descending.
    detections.sort(key=lambda d: -d["flux"])
    return significance, detections


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--cube", required=True, help="HDF5 cube file (dataset: 'cube')")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--scales", type=int, default=4,
                    help="Number of starlet scales (default: 4)")
    ap.add_argument("--k-sigma", type=float, default=3.0,
                    help="Detection threshold in units of per-scale noise (default: 3.0)")
    ap.add_argument("--min-area", type=int, default=4,
                    help="Minimum footprint area in pixels (default: 4)")
    ap.add_argument("--patch-radius", type=int, default=7,
                    help="Half-width of saved morphology patch (default: 7)")
    ap.add_argument("--channels", type=str, default=None,
                    help="Comma-separated channel indices to process, e.g. '10,20,30'. "
                         "Default: all channels.")
    ap.add_argument("--show-gt", action="store_true",
                    help="If the HDF5 has galaxy ground truth, compare detections to GT.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    log.info("Loading cube: %s", args.cube)
    with h5py.File(args.cube, "r") as f:
        cube = f["cube"][:].astype(np.float32)
        has_gt = "galaxies" in f
        if has_gt and args.show_gt:
            gt_positions = f["galaxies/positions_xyz_px"][:]   # (n_gals, 3) x,y,channel
            gt_cubes = f["galaxies/cubes"][:].astype(np.float32)
            n_gals = int(f.attrs["n_gals"])
        else:
            has_gt = False

    n_channels, H, W = cube.shape
    log.info("Cube shape: channels=%d  H=%d  W=%d", n_channels, H, W)
    log.info("Detection params: scales=%d  k_sigma=%.1f  min_area=%d",
             args.scales, args.k_sigma, args.min_area)

    if args.channels is not None:
        channel_list = [int(c) for c in args.channels.split(",")]
    else:
        channel_list = list(range(n_channels))

    # Per-channel detection loop.
    all_det_ch, all_det_y, all_det_x = [], [], []
    all_det_flux, all_det_snr, all_det_area = [], [], []
    sig_maps = np.zeros((len(channel_list), H, W), dtype=np.float32)
    n_per_channel = []

    for idx, ch in enumerate(channel_list):
        img = cube[ch]
        sig, dets = detect_channel(
            img,
            scales=args.scales,
            k_sigma=args.k_sigma,
            min_area=args.min_area,
            patch_radius=args.patch_radius,
        )
        sig_maps[idx] = sig
        n_per_channel.append(len(dets))

        for d in dets:
            all_det_ch.append(ch)
            all_det_y.append(d["y"])
            all_det_x.append(d["x"])
            all_det_flux.append(d["flux"])
            all_det_snr.append(d["snr"])
            all_det_area.append(d["area_px"])

        if (idx + 1) % max(1, len(channel_list) // 8) == 0 or idx == len(channel_list) - 1:
            log.info("  channel %3d: %d detections  (sig_max=%.5f)",
                     ch, len(dets), float(sig.max()))

    # Save detections array.
    det_arrays = dict(
        channel=np.array(all_det_ch, dtype=np.int32),
        y=np.array(all_det_y, dtype=np.float32),
        x=np.array(all_det_x, dtype=np.float32),
        flux=np.array(all_det_flux, dtype=np.float32),
        snr=np.array(all_det_snr, dtype=np.float32),
        area_px=np.array(all_det_area, dtype=np.int32),
    )
    np.savez(out_dir / "detections.npz", **det_arrays)
    np.savez_compressed(out_dir / "channel_maps.npz",
                        significance=sig_maps,
                        channels=np.array(channel_list, dtype=np.int32))

    total_dets = len(all_det_ch)
    log.info("Total detections across %d channels: %d", len(channel_list), total_dets)
    log.info("Mean detections/channel: %.1f  max: %d",
             np.mean(n_per_channel), max(n_per_channel) if n_per_channel else 0)

    # Optional GT comparison.
    if has_gt:
        log.info("--- GT comparison (match radius = 3 px) ---")
        det_yx = np.stack([all_det_y, all_det_x, all_det_ch], axis=1) if total_dets else np.zeros((0, 3))
        for g in range(n_gals):
            gx, gy, gz = gt_positions[g]
            # Find all channels where this galaxy has significant flux.
            gal_flux_per_ch = gt_cubes[g].sum(axis=(1, 2))
            active_chs = np.where(gal_flux_per_ch > 0.01 * gal_flux_per_ch.max())[0]
            hits = 0
            for ach in active_chs:
                if ach not in channel_list:
                    continue
                if total_dets == 0:
                    continue
                mask = det_arrays["channel"] == ach
                if not mask.any():
                    continue
                dy = det_arrays["y"][mask] - gy
                dx = det_arrays["x"][mask] - gx
                if np.hypot(dy, dx).min() <= 3.0:
                    hits += 1
            recall = hits / max(len(active_chs), 1)
            log.info("  galaxy %d  center=(ch=%d y=%d x=%d)  active_channels=%d  "
                     "hit_channels=%d  recall=%.2f",
                     g, gz, gy, gx, len(active_chs), hits, recall)

    summary = {
        "cube": str(args.cube),
        "n_channels_processed": len(channel_list),
        "total_detections": total_dets,
        "mean_detections_per_channel": float(np.mean(n_per_channel)) if n_per_channel else 0.0,
        "max_detections_per_channel": int(max(n_per_channel)) if n_per_channel else 0,
        "params": {
            "scales": args.scales,
            "k_sigma": args.k_sigma,
            "min_area": args.min_area,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved detections.npz, channel_maps.npz, summary.json → %s", out_dir)


if __name__ == "__main__":
    main()
