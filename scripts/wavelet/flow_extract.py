"""Wavelet-based source extraction with optical flow tracking.

Two-stage pipeline per cube:

  Stage 1 — ROI detection on the mean map
    Collapse the cube spectrally (mean) → run starlet wavelet on the 2D mean
    image using broader scales (roi_scales) → each connected component above
    k_sigma_roi becomes a Region Of Interest (ROI).  ROI bounding boxes are
    padded by roi_pad pixels and clamped to the cube edges.

  Stage 2 — Per-ROI per-channel detection + flow tracking
    For each ROI sub-cube (spatially restricted):
      a. Per channel: starlet transform → threshold fine scales (use_scales) →
         connected component footprints (in local coordinates).
      b. TV-L1 optical flow between adjacent channels (in local crop).
      c. Hungarian tracking of footprints across channels; kinematic splits
         (same source, two velocity components) recorded as child tracks.
      d. Source flux cubes assembled: cube[ch, y0:y1, x0:x1] * mask.
    Results are mapped back to full-cube coordinates.

Output:
  source_cubes.h5   — (N_sources, n_ch, H, W) float32
  rois.json         — list of detected ROI bounding boxes
  tracks.csv        — track lineage table
  summary.json      — aggregate stats
  flow_extract.log  — run log

Usage:
    python scripts/wavelet/flow_extract.py \\
        --cube data/all_cubes/cube_10001.h5 \\
        --out /tmp/flow_extract_test \\
        --show-gt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import center_of_mass, label as nd_label
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("flow_extract")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "flow_extract.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Starlet helpers
# ---------------------------------------------------------------------------

# Noise propagation factors for the B3-spline à trous starlet at each scale.
# σ_scale_j ≈ σ_0 * PROPAGATION[j]  where σ_0 is the finest-scale MAD noise.
_PROPAGATION = np.array([0.889, 0.201, 0.086, 0.041, 0.020, 0.010], dtype=np.float32)


def _mad_noise(arr: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))) + 1e-12)


def _starlet_significance(
    image: np.ndarray,
    scales: int,
    use_scales: tuple[int, ...],
    k_sigma: float,
) -> tuple[np.ndarray, float]:
    """Return (sig_map, noise0) where sig_map sums wavelet planes that pass threshold."""
    import scarlet2 as sc

    img = np.asarray(image, dtype=np.float32)
    coeffs = np.array(sc.wavelets.starlet_transform(img, scales=scales), dtype=np.float32)
    noise0 = _mad_noise(coeffs[0])

    H, W = img.shape
    sig_map = np.zeros((H, W), dtype=np.float32)
    for s in use_scales:
        if s >= coeffs.shape[0] - 1:
            continue
        noise_s = noise0 * _PROPAGATION[min(s, len(_PROPAGATION) - 1)]
        plane = coeffs[s]
        sig_map += np.where(plane > k_sigma * noise_s, plane, 0.0)
    return sig_map, noise0


# ---------------------------------------------------------------------------
# Stage 1 — ROI detection on max map using scarlet2 Box footprints
# ---------------------------------------------------------------------------

def detect_rois(
    cube: np.ndarray,
    scales: int = 5,
    roi_scales: tuple[int, ...] = (1, 2),
    k_sigma_roi: float = 3.0,
    roi_min_area: int = 4,
    roi_pad: int = 10,
) -> list[dict]:
    """Detect ROIs from the pixel-wise maximum projection of the cube.

    Uses the max map (more sensitive to spectrally narrow sources than the mean)
    and detects sources as connected components in the fine-scale starlet
    significance image.  Each component's bounding box is grown by `roi_pad`
    pixels and stored as a `scarlet2.Box`.

    Parameters
    ----------
    cube         : (n_ch, H, W) cube.
    scales       : total starlet scales for the max-map transform.
    roi_scales   : fine scales used for significance (default: 1,2 for compact sources).
    k_sigma_roi  : per-scale detection threshold.
    roi_min_area : minimum connected component area to keep.
    roi_pad      : pixels of padding around each component's natural bbox.

    Returns
    -------
    List of dicts with keys: y0, x0, y1, x1, cy, cx, box, max_flux.
    Coordinates are in full-cube pixel space (row=y, col=x).
    """
    from scarlet2.bbox import Box

    _, H, W = cube.shape
    # Max map: pixel-wise maximum over channels — preserves peak signal of
    # spectrally narrow sources that average to noise in the mean map.
    max_map = cube.max(axis=0)

    sig_map, _ = _starlet_significance(max_map, scales, roi_scales, k_sigma_roi)

    # Connected components on the significance image.
    binary = sig_map > 0
    labeled, _ = nd_label(binary)

    rois = []
    for reg in regionprops(labeled, intensity_image=sig_map):
        if reg.area < roi_min_area:
            continue

        # Natural bbox from connected component extent.
        r0, c0, r1, c1 = reg.bbox          # (row_min, col_min, row_max, col_max)

        # Pad and clamp to cube edges.
        y0 = max(0, r0 - roi_pad)
        x0 = max(0, c0 - roi_pad)
        y1 = min(H, r1 + roi_pad)
        x1 = min(W, c1 + roi_pad)

        # scarlet2.Box(shape=(H, W), origin=(y0, x0)) — shape in (rows, cols).
        box = Box(shape=(y1 - y0, x1 - x0), origin=(y0, x0))

        # Intensity-weighted centroid in full-cube coordinates.
        patch = sig_map[r0:r1, c0:c1]
        cy_local, cx_local = center_of_mass(patch)
        cy = float(cy_local + r0)
        cx = float(cx_local + c0)

        rois.append({
            "y0": int(y0), "x0": int(x0),
            "y1": int(y1), "x1": int(x1),
            "cy": cy, "cx": cx,
            "box": box,
            "max_flux": float(max_map[r0:r1, c0:c1].max()),
        })

    rois.sort(key=lambda r: -r["max_flux"])
    return rois


# ---------------------------------------------------------------------------
# Stage 2 — Per-channel detection within a spatial crop
# ---------------------------------------------------------------------------

def detect_channel(
    image: np.ndarray,
    scales: int = 4,
    use_scales: tuple[int, ...] = (1, 2),
    k_sigma: float = 3.0,
    min_area: int = 4,
) -> tuple[np.ndarray, list[dict]]:
    """Starlet-threshold a single 2D image (local crop or full channel).

    Returns
    -------
    sig_map    : (H, W) float32 significance image.
    footprints : list of dicts (cy, cx, flux, mask) in local coordinates.
    """
    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape

    sig_map, _ = _starlet_significance(img, scales, use_scales, k_sigma)
    binary = sig_map > 0
    labeled, _ = nd_label(binary)

    footprints = []
    for reg in regionprops(labeled, intensity_image=sig_map):
        if reg.area < min_area:
            continue
        y0, x0, y1, x1 = reg.bbox
        patch = sig_map[y0:y1, x0:x1]
        cy_local, cx_local = center_of_mass(patch)
        cy = float(np.clip(cy_local + y0, 0, H - 1))
        cx = float(np.clip(cx_local + x0, 0, W - 1))

        fp_mask = np.zeros((H, W), dtype=bool)
        fp_mask[labeled == reg.label] = True

        footprints.append({
            "cy": cy,
            "cx": cx,
            "flux": float(reg.image_intensity.sum()),
            "mask": fp_mask,
        })

    footprints.sort(key=lambda d: -d["flux"])
    return sig_map, footprints


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def _normalize01(img: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def compute_flow(prev: np.ndarray, curr: np.ndarray, fast: bool = False) -> np.ndarray:
    """Return (2, H, W) flow field [vy, vx] from prev→curr."""
    p, c = _normalize01(prev), _normalize01(curr)
    if fast:
        return optical_flow_ilk(p, c).astype(np.float32)
    return optical_flow_tvl1(p, c, num_warp=5, num_iter=10).astype(np.float32)


def flow_warp_point(cy: float, cx: float, flow: np.ndarray) -> tuple[float, float]:
    H, W = flow.shape[1:]
    yi = int(np.clip(round(cy), 0, H - 1))
    xi = int(np.clip(round(cx), 0, W - 1))
    return cy + float(flow[0, yi, xi]), cx + float(flow[1, yi, xi])


# ---------------------------------------------------------------------------
# Tracking data structures
# ---------------------------------------------------------------------------

@dataclass
class Obs:
    channel: int
    cy: float      # in full-cube coordinates
    cx: float
    flux: float
    mask: np.ndarray   # (H_full, W_full) bool in full-cube coordinates


@dataclass
class Track:
    tid: int
    parent: int | None
    roi_idx: int
    obs: list[Obs] = field(default_factory=list)

    def last(self) -> Obs:
        return self.obs[-1]

    @property
    def total_flux(self) -> float:
        return sum(o.flux for o in self.obs)


# ---------------------------------------------------------------------------
# Tracker (operates in local ROI coordinates)
# ---------------------------------------------------------------------------

def track_roi(
    all_footprints: list[list[dict]],
    max_motion_px: float = 10.0,
    split_radius_px: float = 8.0,
    split_flux_frac: float = 0.1,
    tid_offset: int = 0,
    roi: dict | None = None,
    H_full: int = 0,
    W_full: int = 0,
) -> dict[int, Track]:
    """Associate per-channel footprints (local coords) into tracks.

    Footprint masks are lifted back to full-cube coordinates using roi offsets.
    """
    oy = roi["y0"] if roi else 0
    ox = roi["x0"] if roi else 0
    roi_idx = roi.get("_idx", 0) if roi else 0

    def _lift_mask(local_mask: np.ndarray) -> np.ndarray:
        full = np.zeros((H_full, W_full), dtype=bool)
        h, w = local_mask.shape
        full[oy:oy + h, ox:ox + w] = local_mask
        return full

    tracks: dict[int, Track] = {}
    nxt = tid_offset
    n_channels = len(all_footprints)

    for ch, fps in enumerate(all_footprints):
        if fps:
            for fp in fps:
                tracks[nxt] = Track(
                    tid=nxt, parent=None, roi_idx=roi_idx,
                    obs=[Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                             fp["flux"], _lift_mask(fp["mask"]))])
                nxt += 1
            start = ch
            break
    else:
        return tracks

    # Precompute flows lazily: flow_fields[ch] = flow between ch and ch+1.
    # We receive all_footprints already — flows must be passed separately.
    # (flows are computed by the caller and passed via closure — see process_roi)
    return tracks, nxt, start, oy, ox, roi_idx, _lift_mask


def process_roi(
    roi: dict,
    roi_idx: int,
    sub_cube: np.ndarray,
    H_full: int,
    W_full: int,
    scales: int,
    use_scales: tuple[int, ...],
    k_sigma: float,
    min_area: int,
    max_motion_px: float,
    split_radius_px: float,
    split_flux_frac: float,
    fast_flow: bool,
    tid_offset: int,
) -> tuple[dict[int, Track], int]:
    """Run per-channel detection + flow tracking inside one ROI sub-cube.

    Returns updated tracks dict and next available tid.
    """
    oy, ox = roi["y0"], roi["x0"]
    n_ch, Hl, Wl = sub_cube.shape

    def _lift_mask(local_mask: np.ndarray) -> np.ndarray:
        full = np.zeros((H_full, W_full), dtype=bool)
        full[oy:oy + Hl, ox:ox + Wl] = local_mask
        return full

    # Per-channel detection in local coords.
    all_fps: list[list[dict]] = []
    for ch in range(n_ch):
        _, fps = detect_channel(
            sub_cube[ch],
            scales=scales,
            use_scales=use_scales,
            k_sigma=k_sigma,
            min_area=min_area,
        )
        all_fps.append(fps)

    # Optical flow between adjacent channels (local crop).
    flows: list[np.ndarray] = []
    for ch in range(n_ch - 1):
        flows.append(compute_flow(sub_cube[ch], sub_cube[ch + 1], fast=fast_flow))

    # Track.
    tracks: dict[int, Track] = {}
    nxt = tid_offset

    # Seed from first non-empty channel.
    start = None
    for ch, fps in enumerate(all_fps):
        if fps:
            for fp in fps:
                tracks[nxt] = Track(
                    tid=nxt, parent=None, roi_idx=roi_idx,
                    obs=[Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                             fp["flux"], _lift_mask(fp["mask"]))])
                nxt += 1
            start = ch
            break

    if start is None:
        return tracks, nxt

    for ch in range(start + 1, n_ch):
        fps = all_fps[ch]
        if not fps:
            continue

        flow = flows[ch - 1]
        active = [t for t in tracks.values() if t.last().channel == ch - 1]

        if not active:
            for fp in fps:
                tracks[nxt] = Track(
                    tid=nxt, parent=None, roi_idx=roi_idx,
                    obs=[Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                             fp["flux"], _lift_mask(fp["mask"]))])
                nxt += 1
            continue

        # Cost matrix in local coordinates.
        cost = np.full((len(active), len(fps)), 1e6, dtype=np.float32)
        pred_yx = []
        for i, tr in enumerate(active):
            # Convert back to local coords for flow lookup.
            ly = tr.last().cy - oy
            lx = tr.last().cx - ox
            py_l, px_l = flow_warp_point(ly, lx, flow)
            pred_yx.append((py_l, px_l))
            for j, fp in enumerate(fps):
                d = np.hypot(fp["cy"] - py_l, fp["cx"] - px_l)
                if d <= max_motion_px:
                    cost[i, j] = d

        rows, cols = linear_sum_assignment(cost)
        matched_tracks, matched_fps = set(), set()

        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e5:
                continue
            fp = fps[c]
            active[r].obs.append(Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                                     fp["flux"], _lift_mask(fp["mask"])))
            matched_tracks.add(r)
            matched_fps.add(c)

        for c, fp in enumerate(fps):
            if c in matched_fps:
                continue
            best_parent, best_dist = None, 1e9
            for r in matched_tracks:
                py_l, px_l = pred_yx[r]
                d = np.hypot(fp["cy"] - py_l, fp["cx"] - px_l)
                if d < best_dist:
                    best_dist, best_parent = d, r

            if (best_parent is not None
                    and best_dist <= split_radius_px
                    and fp["flux"] >= split_flux_frac * active[best_parent].last().flux):
                parent_tid = active[best_parent].tid
                tracks[nxt] = Track(
                    tid=nxt, parent=parent_tid, roi_idx=roi_idx,
                    obs=[Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                             fp["flux"], _lift_mask(fp["mask"]))])
                nxt += 1
            else:
                tracks[nxt] = Track(
                    tid=nxt, parent=None, roi_idx=roi_idx,
                    obs=[Obs(ch, fp["cy"] + oy, fp["cx"] + ox,
                             fp["flux"], _lift_mask(fp["mask"]))])
                nxt += 1

    return tracks, nxt


# ---------------------------------------------------------------------------
# Tree aggregation + flux extraction
# ---------------------------------------------------------------------------

def _root_map(tracks: dict[int, Track]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {tid: [] for tid in tracks}
    for tid, tr in tracks.items():
        if tr.parent is not None and tr.parent in children:
            children[tr.parent].append(tid)

    roots = {tid: [tid] for tid, tr in tracks.items() if tr.parent is None}
    changed = True
    while changed:
        changed = False
        for root, members in roots.items():
            grown = sorted(set(members + [c for m in members for c in children.get(m, [])]))
            if len(grown) != len(members):
                roots[root] = grown
                changed = True
    return roots


def extract_source_cubes(
    tracks: dict[int, Track],
    cube: np.ndarray,
    n_sources: int | None = None,
    min_channels: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """Build per-source flux cubes (full spatial extent) by masking the cube."""
    n_ch, H, W = cube.shape
    roots = _root_map(tracks)

    root_stats = {}
    for root, members in roots.items():
        total_flux = sum(tracks[m].total_flux for m in members)
        all_channels = {o.channel for m in members for o in tracks[m].obs}
        root_stats[root] = (total_flux, len(all_channels))

    valid_roots = [r for r, (_, nc) in root_stats.items() if nc >= min_channels]
    valid_roots.sort(key=lambda r: -root_stats[r][0])
    if n_sources is not None:
        valid_roots = valid_roots[:n_sources]

    source_cubes = np.zeros((len(valid_roots), n_ch, H, W), dtype=np.float32)
    for out_idx, root in enumerate(valid_roots):
        for m in roots[root]:
            for obs in tracks[m].obs:
                ch = obs.channel
                source_cubes[out_idx, ch] += cube[ch] * obs.mask.astype(np.float32)

    return source_cubes, valid_roots


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_rois(
    cube: np.ndarray,
    rois: list[dict],
    gt_pos=None,
    roi_scales: tuple = (1, 2),
    k_sigma_roi: float = 3.0,
    scales: int = 4,
    out_path: Path = Path("roi_detection.png"),
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    mean_map = cube.mean(axis=0)
    max_map  = cube.max(axis=0)
    sig_mean, _ = _starlet_significance(mean_map, scales, roi_scales, k_sigma_roi)
    sig_max,  _ = _starlet_significance(max_map,  scales, roi_scales, k_sigma_roi)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"ROI Detection Pipeline  ({out_path.parent.name})", fontsize=12)

    def _show(ax, img, title):
        ax.imshow(img, cmap="viridis", aspect="equal",
                  extent=[0, img.shape[1], 0, img.shape[0]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (col)")
        ax.set_ylabel("y (row)")
        if gt_pos is not None:
            for g, (gx, gy, gz) in enumerate(gt_pos.astype(int)):
                # gx = column, gy = row; with extent=[0,W,0,H] these map directly
                ax.plot(gx, gy, "r+", ms=12, mew=2)
                ax.text(gx + 1, gy + 1, str(g), color="red", fontsize=8)

    _show(axes[0, 0], mean_map, "Mean map")
    _show(axes[0, 1], max_map,  "Max map")
    _show(axes[0, 2], sig_mean, f"Sig(mean map)  scales={roi_scales}  k={k_sigma_roi}σ")
    _show(axes[1, 0], sig_max,  f"Sig(max map)   scales={roi_scales}  k={k_sigma_roi}σ")

    # ROI bboxes (from scarlet2 Box) on sig(max map).
    ax = axes[1, 1]
    _show(ax, sig_max, "ROI bboxes on Sig(max map)")
    colors = plt.cm.Set2.colors
    for i, roi in enumerate(rois):
        c = colors[i % len(colors)]
        # scarlet2 Box origin=(y0, x0), shape=(H, W)
        box = roi.get("box")
        if box is not None:
            y0b, x0b = box.origin
            h_b, w_b = box.shape
        else:
            y0b, x0b = roi["y0"], roi["x0"]
            h_b = roi["y1"] - roi["y0"]
            w_b = roi["x1"] - roi["x0"]
        rect = mpatches.Rectangle(
            (x0b, y0b), w_b, h_b,
            lw=2, edgecolor=c, facecolor=c, alpha=0.15)
        ax.add_patch(rect)
        rect2 = mpatches.Rectangle(
            (x0b, y0b), w_b, h_b,
            lw=2, edgecolor=c, facecolor="none")
        ax.add_patch(rect2)
        # Centroid: cy=row, cx=col → plot as (cx, cy)
        ax.plot(roi["cx"], roi["cy"], marker="x", color=c, ms=9, mew=2)
        ax.text(x0b + 1, y0b + 1, f"ROI{i}", color=c, fontsize=8, fontweight="bold")

    # Max map per-channel: show the channel where max_map peaks for each galaxy.
    ax = axes[1, 2]
    ax.set_title("Per-galaxy peak channel slice (max map pixel)")
    ax.axis("off")
    if gt_pos is not None:
        n_gals = len(gt_pos)
        ncols  = min(n_gals, 5)
        nrows  = (n_gals + ncols - 1) // ncols
        inner  = ax.inset_axes([0, 0, 1, 1])
        inner.axis("off")
        for g in range(n_gals):
            gx, gy, gz = gt_pos[g].astype(int)
            ch_slice = cube[gz]
            r, c = divmod(g, ncols)
            subax = inner.inset_axes([c / ncols, 1 - (r + 1) / nrows,
                                      1 / ncols, 1 / nrows])
            r_size = 15
            y0c = max(0, gy - r_size); y1c = min(cube.shape[1], gy + r_size + 1)
            x0c = max(0, gx - r_size); x1c = min(cube.shape[2], gx + r_size + 1)
            subax.imshow(ch_slice[y0c:y1c, x0c:x1c], origin="lower",
                         cmap="inferno", aspect="equal")
            subax.set_title(f"g{g} ch{gz}", fontsize=7)
            subax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_sources(
    cube: np.ndarray,
    source_cubes: np.ndarray,
    gt_pos=None,
    gt_cubes=None,
    out_path: Path = Path("source_overview.png"),
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_src = source_cubes.shape[0]
    n_gals = len(gt_pos) if gt_pos is not None else 0
    ncols = max(n_src, n_gals, 1)
    nrows = 3  # row0=pred, row1=GT, row2=diff

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 9))
    fig.suptitle("Source extraction overview (spectral projection)", fontsize=11)

    def _proj(sc_arr): return sc_arr.sum(axis=0)  # sum over channels

    for s in range(ncols):
        ax0 = axes[0, s] if ncols > 1 else axes[0]
        ax1 = axes[1, s] if ncols > 1 else axes[1]
        ax2 = axes[2, s] if ncols > 1 else axes[2]

        if s < n_src:
            proj = _proj(source_cubes[s])
            ax0.imshow(np.log1p(proj / (proj.max() + 1e-12) * 100),
                       origin="lower", cmap="inferno", aspect="equal")
            ax0.set_title(f"Pred src {s}\n flux={source_cubes[s].sum():.3f}", fontsize=8)
        else:
            ax0.axis("off")
        ax0.axis("off")

        if gt_pos is not None and s < n_gals:
            gt_proj = _proj(gt_cubes[s])
            ax1.imshow(np.log1p(gt_proj / (gt_proj.max() + 1e-12) * 100),
                       origin="lower", cmap="inferno", aspect="equal")
            gx, gy, gz = gt_pos[s].astype(int)
            ax1.set_title(f"GT gal {s}\n flux={gt_cubes[s].sum():.3f}", fontsize=8)
        else:
            ax1.axis("off")
        ax1.axis("off")

        # Difference: best matching pred vs GT.
        if gt_pos is not None and s < n_gals and n_src > 0:
            best_s, best_iou = 0, 0.0
            for ps in range(n_src):
                overlap = np.minimum(source_cubes[ps], gt_cubes[s]).sum()
                union   = np.maximum(source_cubes[ps], gt_cubes[s]).sum()
                iou = float(overlap / (union + 1e-8))
                if iou > best_iou:
                    best_iou, best_s = iou, ps
            diff = _proj(source_cubes[best_s]) - _proj(gt_cubes[s])
            vmax = max(abs(diff).max(), 1e-8)
            ax2.imshow(diff, origin="lower", cmap="bwr",
                       vmin=-vmax, vmax=vmax, aspect="equal")
            ax2.set_title(f"Pred{best_s}-GT{s}  IoU={best_iou:.3f}", fontsize=8)
        else:
            ax2.axis("off")
        ax2.axis("off")

    axes[0, 0].set_ylabel("Predicted") if ncols > 1 else None
    axes[1, 0].set_ylabel("Ground truth") if ncols > 1 else None
    axes[2, 0].set_ylabel("Difference") if ncols > 1 else None

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--cube", required=True, help="HDF5 cube (dataset: 'cube')")
    ap.add_argument("--out", required=True, help="Output directory")

    # ROI detection (Stage 1)
    g1 = ap.add_argument_group("Stage 1 — ROI detection (mean map)")
    g1.add_argument("--roi-scales", type=str, default="1,2",
                    help="Fine starlet scales for max-map ROI detection (default: '1,2')")
    g1.add_argument("--k-sigma-roi", type=float, default=3.0,
                    help="Significance threshold for ROI detection (default: 3.0)")
    g1.add_argument("--roi-min-area", type=int, default=4,
                    help="Min connected-component area in the max-map sig image (default: 4)")
    g1.add_argument("--roi-pad", type=int, default=10,
                    help="Padding around each component's natural bbox in pixels (default: 10)")

    # Per-channel detection (Stage 2)
    g2 = ap.add_argument_group("Stage 2 — per-channel detection + flow tracking")
    g2.add_argument("--scales", type=int, default=4,
                    help="Total starlet scales per channel (default: 4)")
    g2.add_argument("--use-scales", type=str, default="1,2",
                    help="Fine scales for per-channel detection (default: '1,2')")
    g2.add_argument("--k-sigma", type=float, default=3.0,
                    help="Per-channel detection threshold (default: 3.0)")
    g2.add_argument("--min-area", type=int, default=4,
                    help="Min footprint area in pixels (default: 4)")
    g2.add_argument("--max-motion", type=float, default=10.0,
                    help="Max centroid motion per channel in px (default: 10.0)")
    g2.add_argument("--split-radius", type=float, default=8.0,
                    help="Max distance for kinematic split child (default: 8.0)")
    g2.add_argument("--split-flux-frac", type=float, default=0.1,
                    help="Min flux ratio for a split child vs parent (default: 0.1)")
    g2.add_argument("--min-channels", type=int, default=3,
                    help="Discard sources seen in fewer channels (default: 3)")
    g2.add_argument("--n-sources", type=int, default=None,
                    help="Keep top-N sources by flux. Default: all.")
    g2.add_argument("--fast-flow", action="store_true",
                    help="Use ILK flow (faster) instead of TV-L1")

    ap.add_argument("--show-gt", action="store_true",
                    help="Print GT comparison if HDF5 has galaxy ground truth.")
    ap.add_argument("--plot", action="store_true",
                    help="Save diagnostic plots (roi_detection.png, source_overview.png).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    roi_scales = tuple(int(s) for s in args.roi_scales.split(","))
    use_scales = tuple(int(s) for s in args.use_scales.split(","))

    log.info("Loading cube: %s", args.cube)
    with h5py.File(args.cube, "r") as f:
        cube = f["cube"][:].astype(np.float32)
        has_gt = "galaxies" in f and args.show_gt
        if has_gt:
            gt_pos   = f["galaxies/positions_xyz_px"][:]
            gt_cubes = f["galaxies/cubes"][:].astype(np.float32)
            n_gals   = int(f.attrs["n_gals"])

    n_ch, H, W = cube.shape
    log.info("Cube: channels=%d  H=%d  W=%d", n_ch, H, W)

    # ---- Stage 1: ROI detection ----
    log.info("Stage 1: ROI detection on max map "
             "(scales=%s  k_sigma=%.1f  roi_pad=%d  min_area=%d) ...",
             roi_scales, args.k_sigma_roi, args.roi_pad, args.roi_min_area)
    rois = detect_rois(
        cube,
        scales=max(roi_scales) + 2,
        roi_scales=roi_scales,
        k_sigma_roi=args.k_sigma_roi,
        roi_min_area=args.roi_min_area,
        roi_pad=args.roi_pad,
    )
    log.info("  Found %d ROIs", len(rois))
    for i, roi in enumerate(rois):
        log.info("  ROI %d: y=[%d,%d) x=[%d,%d)  size=%dx%d  max_flux=%.4f",
                 i, roi["y0"], roi["y1"], roi["x0"], roi["x1"],
                 roi["y1"] - roi["y0"], roi["x1"] - roi["x0"], roi["max_flux"])

    if not rois:
        log.warning("No ROIs detected — try lowering --k-sigma-roi or --roi-min-dist")
        return

    # Save ROI info (strip non-serialisable Box objects).
    rois_json = [{k: v for k, v in r.items() if k != "box"} for r in rois]
    (out_dir / "rois.json").write_text(json.dumps(rois_json, indent=2))

    if args.plot:
        _plot_rois(cube, rois,
                   gt_pos=gt_pos if has_gt else None,
                   roi_scales=roi_scales,
                   k_sigma_roi=args.k_sigma_roi,
                   scales=max(roi_scales) + 2,
                   out_path=out_dir / "roi_detection.png")
        log.info("Saved roi_detection.png")

    # ---- Stage 2: per-ROI tracking ----
    log.info("Stage 2: per-channel detection + flow tracking inside each ROI ...")
    all_tracks: dict[int, Track] = {}
    tid_offset = 0

    for roi_idx, roi in enumerate(rois):
        y0, x0, y1, x1 = roi["y0"], roi["x0"], roi["y1"], roi["x1"]
        sub_cube = cube[:, y0:y1, x0:x1]
        log.info("  ROI %d: processing sub-cube %s ...", roi_idx, sub_cube.shape)

        roi_tracks, tid_offset = process_roi(
            roi=roi,
            roi_idx=roi_idx,
            sub_cube=sub_cube,
            H_full=H,
            W_full=W,
            scales=args.scales,
            use_scales=use_scales,
            k_sigma=args.k_sigma,
            min_area=args.min_area,
            max_motion_px=args.max_motion,
            split_radius_px=args.split_radius,
            split_flux_frac=args.split_flux_frac,
            fast_flow=args.fast_flow,
            tid_offset=tid_offset,
        )

        n_rt = sum(1 for t in roi_tracks.values() if t.parent is None)
        n_sp = sum(1 for t in roi_tracks.values() if t.parent is not None)
        log.info("    %d tracks (%d roots, %d splits)", len(roi_tracks), n_rt, n_sp)
        all_tracks.update(roi_tracks)

    n_roots  = sum(1 for t in all_tracks.values() if t.parent is None)
    n_splits = sum(1 for t in all_tracks.values() if t.parent is not None)
    log.info("All tracks: %d total  %d roots  %d splits", len(all_tracks), n_roots, n_splits)

    # ---- Extract source cubes ----
    log.info("Extracting source flux cubes ...")
    source_cubes, root_ids = extract_source_cubes(
        all_tracks, cube,
        n_sources=args.n_sources,
        min_channels=args.min_channels,
    )
    log.info("Output sources: %d", len(root_ids))

    # ---- Save outputs ----
    with h5py.File(out_dir / "source_cubes.h5", "w") as f:
        f.create_dataset("source_cubes", data=source_cubes, compression="gzip")
        f.create_dataset("root_track_ids", data=np.array(root_ids, dtype=np.int32))

    with open(out_dir / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "parent", "roi_idx", "n_channels", "total_flux",
                    "ch_start", "ch_end", "cy_mean", "cx_mean"])
        for tid, tr in sorted(all_tracks.items()):
            chs = [o.channel for o in tr.obs]
            w.writerow([tid,
                        tr.parent if tr.parent is not None else -1,
                        tr.roi_idx,
                        len(tr.obs), f"{tr.total_flux:.5g}",
                        min(chs), max(chs),
                        f"{np.mean([o.cy for o in tr.obs]):.2f}",
                        f"{np.mean([o.cx for o in tr.obs]):.2f}"])

    if args.plot:
        _plot_sources(cube, source_cubes,
                      gt_pos=gt_pos if has_gt else None,
                      gt_cubes=gt_cubes if has_gt else None,
                      out_path=out_dir / "source_overview.png")
        log.info("Saved source_overview.png")

    # ---- GT comparison ----
    if has_gt:
        log.info("--- GT comparison ---")
        for g in range(n_gals):
            gx, gy, gz = gt_pos[g]
            gt_f = float(gt_cubes[g].sum())
            best_iou, best_src = 0.0, -1
            for s in range(source_cubes.shape[0]):
                overlap = np.minimum(source_cubes[s], gt_cubes[g]).sum()
                union   = np.maximum(source_cubes[s], gt_cubes[g]).sum()
                iou = float(overlap / (union + 1e-8))
                if iou > best_iou:
                    best_iou, best_src = iou, s
            pred_f = float(source_cubes[best_src].sum()) if best_src >= 0 else 0.0
            log.info("  galaxy %d  center=(ch=%d y=%d x=%d)  GT_flux=%.4f  "
                     "best_src=%d  pred_flux=%.4f  IoU=%.3f",
                     g, int(gz), int(gy), int(gx), gt_f, best_src, pred_f, best_iou)

    summary = {
        "cube": str(args.cube),
        "n_channels": n_ch, "H": H, "W": W,
        "n_rois": len(rois),
        "n_tracks": len(all_tracks),
        "n_root_tracks": n_roots,
        "n_split_tracks": n_splits,
        "n_output_sources": len(root_ids),
        "params": {
            "roi_scales": list(roi_scales),
            "k_sigma_roi": args.k_sigma_roi,
            "roi_pad": args.roi_pad,
            "roi_min_area": args.roi_min_area,
            "use_scales": list(use_scales),
            "k_sigma": args.k_sigma,
            "min_area": args.min_area,
            "max_motion_px": args.max_motion,
            "split_radius_px": args.split_radius,
            "min_channels": args.min_channels,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved source_cubes.h5, rois.json, tracks.csv, summary.json → %s", out_dir)


if __name__ == "__main__":
    main()
