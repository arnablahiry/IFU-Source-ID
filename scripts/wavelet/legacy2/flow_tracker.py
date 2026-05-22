"""Masked optical-flow source tracker for 3-D spectral cubes.

Relies on `wavelet_detect.detect_cube` to produce per-channel source
masks.  Only pixels that fall inside a detected source in *both* the
reference and the target channel are used when computing the optical flow
field, which makes the estimate sharper for compact sources and prevents
the bright halo of a dominant galaxy from dominating the displacement.

Pipeline
--------
1. Run `wavelet_detect.detect_cube` on the cube (or accept pre-computed
   `CubeDetections`).
2. For each pair of consecutive channels (ch_n, ch_{n+1}):
   a. Form the union source mask in each channel.
   b. Restrict the two slices to the intersection of both masks.
   c. Compute TV-L1 optical flow on the masked crops (OpenCV or skimage).
3. For each channel, propagate source centroids forward/backward through
   the flow field → per-source trajectory (channel, row, col).
4. Assemble per-source sub-cubes: for each channel, copy the flux inside
   the source's footprint mask; pixels outside are set to zero.

Output
------
  source_cubes.h5  — (N_sources, n_ch, H, W) float32, one cube per track
  tracks.csv       — source_id, channel, y, x, flux, snr, area_px
  flow.npz         — (n_ch-1, 2, H, W) flow fields (v, u)
  summary.json     — aggregate stats

Usage
-----
    python scripts/wavelet/flow_tracker.py \\
        --cube data/all_cubes/cube_10001.h5 \\
        --out /tmp/flow_tracker_test \\
        --k-sigma 3.0 --scales 4 --min-area 4 --flow-method tvl1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from wavelet_detect import (
    CubeDetections, SourceRegion, detect_cube,
    load_cube, active_channels,
)

try:
    from skimage.registration import optical_flow_tvl1
    _HAS_TVL1 = True
except Exception:
    _HAS_TVL1 = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("flow_tracker")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "flow_tracker.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# Masked optical flow
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float64)
    lo, hi = np.percentile(a, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _masked_flow_tvl1(
    ref: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """TV-L1 flow restricted to `mask` pixels.

    Outside the mask, both images are zeroed so the TV-L1 solver sees
    a blank field and returns near-zero displacements there.  This
    prevents structure outside detected sources from leaking into the
    flow estimate inside sources.

    Returns (2, H, W) float32 with flow[0]=v (row), flow[1]=u (col).
    """
    if not _HAS_TVL1:
        raise ImportError("skimage.registration.optical_flow_tvl1 not available")
    H, W = ref.shape
    r = (ref * mask).astype(np.float64)
    t = (tgt * mask).astype(np.float64)
    v, u = optical_flow_tvl1(r, t, **kwargs)
    out = np.zeros((2, H, W), dtype=np.float32)
    out[0] = v.astype(np.float32)
    out[1] = u.astype(np.float32)
    # Zero out displacements in the non-source region so downstream code
    # never propagates centroids along artefact flow vectors.
    out[:, ~mask] = 0.0
    return out


def _masked_flow_farneback(
    ref: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    **defaults,
) -> np.ndarray:
    if not _HAS_CV2:
        raise ImportError("opencv-python (cv2) not installed")
    params = dict(pyr_scale=0.5, levels=3, winsize=15,
                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    params.update(defaults)
    H, W = ref.shape
    a = _to_uint8(ref * mask)
    b = _to_uint8(tgt * mask)
    f = cv2.calcOpticalFlowFarneback(a, b, None, **params)
    out = np.zeros((2, H, W), dtype=np.float32)
    out[0] = f[..., 1]
    out[1] = f[..., 0]
    out[:, ~mask] = 0.0
    return out


def compute_masked_flow(
    cube: np.ndarray,
    dets: CubeDetections,
    method: str = "tvl1",
    **kwargs,
) -> np.ndarray:
    """Compute (n_proc_ch - 1, 2, H, W) optical flow using only source pixels.

    For each consecutive pair of processed channels, the intersection of
    their source masks is used.  Both slices are zeroed outside that
    intersection before flow estimation.

    Returns array indexed by *processed* channel pairs, not raw channel
    indices.  The caller can map back via `dets.channel_list`.
    """
    n_ch, H, W = dets.cube_shape
    n_proc = dets.n_channels()
    flow = np.zeros((n_proc - 1, 2, H, W), dtype=np.float32)

    _flow_fn = _masked_flow_tvl1 if method == "tvl1" else _masked_flow_farneback

    for idx in range(n_proc - 1):
        ch_ref = dets.channel_list[idx]
        ch_tgt = dets.channel_list[idx + 1]

        mask_ref = dets.union_mask(idx)
        mask_tgt = dets.union_mask(idx + 1)
        # Intersection: only pixels that are sources in both channels.
        joint_mask = mask_ref & mask_tgt

        if not joint_mask.any():
            # No overlapping source pixels — leave flow as zero.
            continue

        flow[idx] = _flow_fn(
            cube[ch_ref].astype(np.float32),
            cube[ch_tgt].astype(np.float32),
            joint_mask,
            **kwargs,
        )

    return flow


# ---------------------------------------------------------------------------
# Centroid propagation
# ---------------------------------------------------------------------------

def _bilinear_sample(field: np.ndarray, y: float, x: float) -> float:
    H, W = field.shape
    y = np.clip(y, 0.0, H - 1.001)
    x = np.clip(x, 0.0, W - 1.001)
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    fy = y - y0; fx = x - x0
    return float(
        (1 - fy) * (1 - fx) * field[y0,     x0]
      + (1 - fy) * fx       * field[y0,     min(x0 + 1, W - 1)]
      + fy       * (1 - fx) * field[min(y0 + 1, H - 1), x0]
      + fy       * fx       * field[min(y0 + 1, H - 1), min(x0 + 1, W - 1)]
    )


def propagate_centroid(
    y0: float,
    x0: float,
    start_idx: int,
    flow: np.ndarray,
    n_proc: int,
) -> np.ndarray:
    """Track one centroid forward and backward through `flow`.

    flow : (n_proc-1, 2, H, W)  — flow[i] maps proc_ch[i] → proc_ch[i+1]
    start_idx : index in the processed channel list where (y0, x0) lives.

    Returns (n_proc, 2) trajectory in (row, col).
    """
    traj = np.zeros((n_proc, 2), dtype=np.float64)
    traj[start_idx] = (y0, x0)

    # Forward pass
    for i in range(start_idx, n_proc - 1):
        cy, cx = traj[i]
        v = _bilinear_sample(flow[i, 0], cy, cx)
        u = _bilinear_sample(flow[i, 1], cy, cx)
        traj[i + 1] = (cy + v, cx + u)

    # Backward pass (approximate inverse via negative flow at current pos)
    for i in range(start_idx, 0, -1):
        cy, cx = traj[i]
        v = _bilinear_sample(flow[i - 1, 0], cy, cx)
        u = _bilinear_sample(flow[i - 1, 1], cy, cx)
        traj[i - 1] = (cy - v, cx - u)

    return traj


# ---------------------------------------------------------------------------
# Source matching: link detections across channels
# ---------------------------------------------------------------------------

def _match_regions(
    regions_a: list[SourceRegion],
    regions_b: list[SourceRegion],
    max_dist: float = 10.0,
) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour matching between two sets of SourceRegions.

    Returns list of (idx_a, idx_b) pairs whose centroid distance <= max_dist.
    Uses the Hungarian algorithm for optimal bipartite assignment.
    """
    from scipy.optimize import linear_sum_assignment

    if not regions_a or not regions_b:
        return []

    ya = np.array([r.y for r in regions_a])
    xa = np.array([r.x for r in regions_a])
    yb = np.array([r.y for r in regions_b])
    xb = np.array([r.x for r in regions_b])

    dy = ya[:, None] - yb[None, :]
    dx = xa[:, None] - xb[None, :]
    cost = np.hypot(dy, dx)

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] <= max_dist
    ]


# ---------------------------------------------------------------------------
# Main tracking routine
# ---------------------------------------------------------------------------

class Track:
    """A source track: centroid trajectory + per-channel flux masks."""

    def __init__(self, track_id: int, n_proc: int, H: int, W: int) -> None:
        self.track_id = track_id
        self.trajectory: np.ndarray = np.full((n_proc, 2), np.nan)   # (n_proc, 2)
        self.snr: np.ndarray = np.zeros(n_proc)
        self.flux: np.ndarray = np.zeros(n_proc)
        self.area: np.ndarray = np.zeros(n_proc, dtype=np.int32)
        # Per-processed-channel source mask (H, W) bool.
        self._masks: list[np.ndarray | None] = [None] * n_proc

    def set_channel(self, proc_idx: int, reg: SourceRegion) -> None:
        self.trajectory[proc_idx] = (reg.y, reg.x)
        self.snr[proc_idx] = reg.snr
        self.flux[proc_idx] = reg.flux
        self.area[proc_idx] = reg.area_px
        self._masks[proc_idx] = reg.mask

    def mask(self, proc_idx: int) -> np.ndarray | None:
        return self._masks[proc_idx]


def build_tracks(
    dets: CubeDetections,
    flow: np.ndarray,
    max_match_dist: float = 10.0,
) -> list[Track]:
    """Build one track per global source.

    Since `detect_cube` already assigns every channel's detections to the N
    global sources (index-matched), each source maps directly to one track.
    The flow field is used to refine the centroid trajectory via bilinear
    sampling for channels where the source is below threshold (flux == 0),
    filling gaps without inventing new tracks.

    Returns list of Track objects sorted by mean flux (descending).
    """
    n_proc = dets.n_channels()
    n_ch, H, W = dets.cube_shape
    N = dets.n_sources()

    tracks = []
    for src_idx in range(N):
        t = Track(src_idx, n_proc, H, W)
        for ch_idx in range(n_proc):
            regs = dets.channel_regions[ch_idx]
            if src_idx < len(regs):
                t.set_channel(ch_idx, regs[src_idx])
        tracks.append(t)

    # Gap-fill: for channels where flux==0, propagate the last known centroid
    # forward via optical flow so the trajectory is continuous.
    for t in tracks:
        # Forward pass
        for ch_idx in range(1, n_proc):
            if t.flux[ch_idx] == 0 and not np.isnan(t.trajectory[ch_idx - 1, 0]):
                cy, cx = t.trajectory[ch_idx - 1]
                flow_idx = ch_idx - 1
                if flow_idx < flow.shape[0]:
                    v = _bilinear_sample(flow[flow_idx, 0], cy, cx)
                    u = _bilinear_sample(flow[flow_idx, 1], cy, cx)
                    t.trajectory[ch_idx] = (cy + v, cx + u)
        # Backward pass (for leading channels below threshold)
        for ch_idx in range(n_proc - 2, -1, -1):
            if t.flux[ch_idx] == 0 and not np.isnan(t.trajectory[ch_idx + 1, 0]):
                cy, cx = t.trajectory[ch_idx + 1]
                flow_idx = ch_idx
                if flow_idx < flow.shape[0]:
                    v = _bilinear_sample(flow[flow_idx, 0], cy, cx)
                    u = _bilinear_sample(flow[flow_idx, 1], cy, cx)
                    t.trajectory[ch_idx] = (cy - v, cx - u)

    tracks.sort(key=lambda t: -float(np.nanmean(t.flux)))
    return tracks


def _build_tracks_legacy(
    dets: CubeDetections,
    flow: np.ndarray,
    max_match_dist: float = 10.0,
) -> list[Track]:
    """Legacy Hungarian-matching tracker (kept for reference).
    Use build_tracks() instead — it relies on the global-detection alignment.
    """
    n_proc = dets.n_channels()
    n_ch, H, W = dets.cube_shape

    tracks: dict[int, Track] = {}
    next_id = 0
    active: dict[int, int] = {}

    if not dets.channel_regions[0]:
        seed_idx = next((i for i, r in enumerate(dets.channel_regions) if r), None)
        if seed_idx is None:
            return []
        first_idx = seed_idx
    else:
        first_idx = 0

    for ridx, reg in enumerate(dets.channel_regions[first_idx]):
        t = Track(next_id, n_proc, H, W)
        t.set_channel(first_idx, reg)
        tracks[next_id] = t
        active[next_id] = ridx
        next_id += 1

    for proc_idx in range(first_idx + 1, n_proc):
        regs = dets.channel_regions[proc_idx]
        prev_regs = dets.channel_regions[proc_idx - 1]

        # Build predicted positions for each active track by flowing
        # the previous centroid one step forward.
        predicted: dict[int, tuple[float, float]] = {}
        for tid in list(active.keys()):
            prev_idx = proc_idx - 1
            flow_idx = prev_idx           # flow[i] maps proc_ch[i] -> proc_ch[i+1]
            if flow_idx >= flow.shape[0]:
                predicted[tid] = tracks[tid].trajectory[prev_idx].tolist()
                continue
            cy, cx = tracks[tid].trajectory[prev_idx]
            if np.isnan(cy):
                del active[tid]
                continue
            v = _bilinear_sample(flow[flow_idx, 0], cy, cx)
            u = _bilinear_sample(flow[flow_idx, 1], cy, cx)
            predicted[tid] = (cy + v, cx + u)

        if not regs:
            # No detections: advance trajectory via flow only (no mask update).
            for tid, (py, px) in predicted.items():
                tracks[tid].trajectory[proc_idx] = (py, px)
            continue

        # Match active tracks to current detections.
        pred_tids = list(predicted.keys())
        pred_yx = np.array([predicted[tid] for tid in pred_tids])
        det_yx = np.array([(r.y, r.x) for r in regs])

        from scipy.optimize import linear_sum_assignment
        dy = pred_yx[:, 0:1] - det_yx[None, :, 0]
        dx = pred_yx[:, 1:2] - det_yx[None, :, 1]
        cost = np.hypot(dy.squeeze(1) if dy.ndim == 3 else dy,
                        dx.squeeze(1) if dx.ndim == 3 else dx)
        # Reshape properly
        cost = np.hypot(pred_yx[:, 0:1] - det_yx[:, 0],
                        pred_yx[:, 1:2] - det_yx[:, 1])  # (n_pred, n_det)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det = set()
        for r, c in zip(row_ind, col_ind):
            tid = pred_tids[r]
            if cost[r, c] <= max_match_dist:
                tracks[tid].set_channel(proc_idx, regs[c])
                active[tid] = c
                matched_det.add(c)
            else:
                # Track lost — extrapolate position but mark no mask.
                tracks[tid].trajectory[proc_idx] = predicted[tid]

        # Unmatched predicted tracks: extrapolate via flow.
        matched_pred = set(row_ind[cost[row_ind, col_ind] <= max_match_dist])
        for r, tid in enumerate(pred_tids):
            if r not in matched_pred:
                tracks[tid].trajectory[proc_idx] = predicted[tid]
                if tid in active:
                    del active[tid]

        # Unmatched detections in current channel → new tracks.
        for c, reg in enumerate(regs):
            if c not in matched_det:
                t = Track(next_id, n_proc, H, W)
                t.set_channel(proc_idx, reg)
                tracks[next_id] = t
                active[next_id] = c
                next_id += 1

    track_list = sorted(tracks.values(), key=lambda t: -np.nanmean(t.flux))
    return track_list


# ---------------------------------------------------------------------------
# Assemble per-source sub-cubes
# ---------------------------------------------------------------------------

def assemble_source_cubes(
    cube: np.ndarray,
    tracks: list[Track],
    dets: CubeDetections,
) -> np.ndarray:
    """Build (N_tracks, n_ch, H, W) float32 source cubes.

    For each track and each processed channel, the original cube flux is
    copied inside the track's detection mask; undetected channels are zeros.
    The full cube shape is used so all sources share the same coordinate frame.
    """
    n_ch, H, W = dets.cube_shape
    n_tracks = len(tracks)
    out = np.zeros((n_tracks, n_ch, H, W), dtype=np.float32)

    for tid, track in enumerate(tracks):
        for proc_idx, ch in enumerate(dets.channel_list):
            m = track.mask(proc_idx)
            if m is not None:
                out[tid, ch][m] = cube[ch].astype(np.float32)[m]

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cube", required=True,
                    help="Cube file: .h5/.hdf5, .fits/.fit, .npy, or .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=int, default=5)
    ap.add_argument("--detail-scales", type=str, default="0,1,2",
                    help="Fine starlet scales used for detection (default: 0,1,2)")
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--no-subtract-diffuse", action="store_true")
    ap.add_argument("--channels", type=str, default=None,
                    help="Comma-separated channel indices; default: auto-detect active")
    ap.add_argument("--active-threshold", type=float, default=0.05)
    ap.add_argument("--flow-method", choices=["tvl1", "farneback"], default="tvl1")
    ap.add_argument("--max-match-dist", type=float, default=10.0)
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
        log.info("Auto-detected %d active channels", len(channel_list))

    # Stage 1: wavelet detection
    log.info("Stage 1: wavelet per-channel detection ...")
    dets = detect_cube(
        cube, channel_list=channel_list,
        n_scales=args.scales, k_sigma=args.k_sigma,
        min_area=args.min_area,
        detail_scales=detail_scales,
        subtract_diffuse=not args.no_subtract_diffuse,
        log=log,
    )
    log.info("  %d total detections across %d channels",
             dets.total_detections(), dets.n_channels())

    # Stage 2: masked optical flow
    log.info("Stage 2: masked optical flow (%s) ...", args.flow_method)
    flow = compute_masked_flow(cube, dets, method=args.flow_method)
    np.savez_compressed(
        out_dir / "flow.npz",
        flow=flow,
        channels=np.array(dets.channel_list, dtype=np.int32),
    )
    log.info("  flow computed; median |v|=%.3f px",
             float(np.median(np.abs(flow[:, 0]))))

    # Stage 3: track linking
    log.info("Stage 3: linking tracks (max_match_dist=%.1f px) ...",
             args.max_match_dist)
    tracks = build_tracks(dets, flow, max_match_dist=args.max_match_dist)
    log.info("  %d tracks found", len(tracks))

    # Stage 4: assemble source cubes
    log.info("Stage 4: assembling source cubes ...")
    source_cubes = assemble_source_cubes(cube, tracks, dets)

    with h5py.File(out_dir / "source_cubes.h5", "w") as f:
        f.create_dataset("source_cubes", data=source_cubes,
                         compression="gzip", compression_opts=4)
        f.attrs["n_sources"] = len(tracks)
        f.attrs["n_channels"] = n_ch

    # Write tracks CSV
    with open(out_dir / "tracks.csv", "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["source_id", "proc_ch_idx", "channel",
                         "y", "x", "flux", "snr", "area_px"])
        for tid, track in enumerate(tracks):
            for pidx, ch in enumerate(dets.channel_list):
                y, x = track.trajectory[pidx]
                writer.writerow([
                    tid, pidx, ch,
                    f"{y:.3f}", f"{x:.3f}",
                    f"{track.flux[pidx]:.4f}",
                    f"{track.snr[pidx]:.2f}",
                    int(track.area[pidx]),
                ])

    summary = {
        "cube": str(args.cube),
        "n_channels_processed": dets.n_channels(),
        "total_detections": dets.total_detections(),
        "n_tracks": len(tracks),
        "params": {
            "scales": args.scales,
            "k_sigma": args.k_sigma,
            "min_area": args.min_area,
            "flow_method": args.flow_method,
            "max_match_dist": args.max_match_dist,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved source_cubes.h5, tracks.csv, flow.npz, summary.json → %s", out_dir)


if __name__ == "__main__":
    main()
