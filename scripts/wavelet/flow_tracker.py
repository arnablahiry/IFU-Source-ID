"""Flow-guided source tracker for per-channel wavelet detections.

Takes the list of per-channel detections produced by
``wavelet_detections.detect_cube_per_channel`` and runs a four-stage pipeline:

Stage 1 — Masked optical flow
    TV-L1 flow is computed between every consecutive channel pair, but only
    inside the intersection of the two channels' union footprint masks.
    Zeroing the images outside detected sources prevents artefact-level flow
    vectors from leaking into the tracking step.

Stage 2 — Track linking with split/merge detection
    Each active track's last centroid is propagated forward through the flow
    field via bilinear interpolation to predict its position in the next
    channel.  Hungarian assignment (optimal bipartite matching) then links
    predictions to actual detections within MAX_LINK_DIST pixels.

    Unmatched detections are classified by proximity to a predicted position:
    - Within MAX_SPLIT_DIST px → flagged as a **split** of the nearest parent.
    - Beyond MAX_SPLIT_DIST px → new independent source track.

    Unmatched predictions are classified by proximity to an already-claimed
    detection:
    - Within MAX_LINK_DIST px of a matched detection → flagged as a **merge**
      into that detection's track; centroid extrapolated via flow.
    - Otherwise → centroid extrapolated via flow (gap in detection coverage).

Stage 3 — Kinematic classification
    A track is **kinematically active** if its cumulative centroid displacement
    across channels exceeds MIN_DISPLACEMENT pixels, or if it was involved in
    a split or merge event.

Stage 4 — Source grouping
    Tracks connected by split_from / merge_into relationships are grouped into
    **sources** via union-find.  A source represents one physical object whose
    emission footprint may split into several blobs across channels (due to
    kinematics / Doppler shear) and later rejoin.

Output
------
``run_flow_tracker`` returns
  ``detections`` — list[ChannelDetection], one per processed channel
  ``flow_seq``   — list of (ch_ref, ch_tgt, flow (2,H,W), joint_mask) tuples
  ``tracks``     — list of track dicts, each containing:
      ``id``           — unique integer identifier
      ``source_id``    — which source this track belongs to
      ``trajectory``   — list of (channel, row, col) centroid tuples
      ``masks``        — {channel: (H,W) bool footprint mask}
      ``split_at``     — channels where this track split into a child
      ``split_from``   — parent track id if this is a split product, else None
      ``merge_into``   — list of (channel, track_id) merge events
      ``displacement`` — total centroid travel in pixels
      ``has_split``    — bool: involved in any split/merge event?
      ``kinematic``    — bool: kinematically active?
  ``sources``    — list of source dicts, each containing:
      ``id``           — unique integer identifier
      ``track_ids``    — list of track ids that belong to this source
      ``channels``     — sorted list of channels the source spans
      ``n_channels``   — number of channels spanned
      ``split_events`` — channels where the source footprint split
      ``merge_events`` — channels where sub-tracks merged back together

Usage (standalone)::

    python flow_tracker.py \\
        --cube  data/clean_cube.npy \\
        --out   /tmp/tracks \\
        --channels 70,74 \\
        --min-match-overlap 5 --min-split-overlap 3 --min-displacement 3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from skimage.registration import optical_flow_tvl1

from wavelet_detections import (
    ChannelDetection,
    active_channels,
    detect_cube_per_channel,
    load_cube,
)


# ---------------------------------------------------------------------------
# Stage 1 — Masked optical flow
# ---------------------------------------------------------------------------

def masked_flow_tvl1(
    img_ref: np.ndarray,
    img_tgt: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """TV-L1 optical flow restricted to *mask* pixels.

    Both images are zeroed outside *mask* before the solver runs, so emission
    structure outside detected source footprints never influences the flow
    estimate inside them.

    Parameters
    ----------
    img_ref, img_tgt :
        2-D float32 channel images, shape (H, W).
    mask :
        Boolean (H, W) — True where flow should be estimated.

    Returns
    -------
    np.ndarray
        Shape (2, H, W) float32.  ``flow[0]`` = v (row displacement),
        ``flow[1]`` = u (col displacement).  Zero everywhere outside *mask*.
    """
    r = (img_ref * mask).astype(np.float64)
    t = (img_tgt * mask).astype(np.float64)
    v, u = optical_flow_tvl1(r, t)
    flow = np.zeros((2, *img_ref.shape), dtype=np.float32)
    flow[0] = v
    flow[1] = u
    flow[:, ~mask] = 0.0
    return flow


def compute_flow_sequence(
    detections: list[ChannelDetection],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Compute masked TV-L1 flow for every consecutive detection pair.

    The joint mask is the *union* of the source footprints from both channels.
    Using the union (rather than the intersection) is critical for split
    detection: when a source splits into a new spatial location between
    channels, the two blobs may not overlap at all.  With an intersection
    mask the flow would be zero everywhere and the predicted centroid would
    not move — causing the split-off blob to be mis-classified as a new
    independent source.  With the union mask the TV-L1 solver sees the
    source signal on both sides and produces flow vectors that point from
    the pre-split footprint toward the post-split footprint, allowing
    :func:`link_tracks` to attribute the new blob to the correct parent.

    Parameters
    ----------
    detections :
        Ordered list of :class:`~wavelet_detections.ChannelDetection` objects.

    Returns
    -------
    list of (ch_ref, ch_tgt, flow, joint_mask) tuples.
    """
    H, W = detections[0].image.shape
    results = []

    for i in range(len(detections) - 1):
        d_ref, d_tgt = detections[i], detections[i + 1]

        union_ref = np.zeros((H, W), dtype=bool)
        for m in d_ref.footprint_masks:
            union_ref |= m

        union_tgt = np.zeros((H, W), dtype=bool)
        for m in d_tgt.footprint_masks:
            union_tgt |= m

        # Union: flow is estimated wherever either channel has source signal.
        joint_mask = union_ref | union_tgt

        if joint_mask.any():
            flow = masked_flow_tvl1(d_ref.image, d_tgt.image, joint_mask)
        else:
            flow = np.zeros((2, H, W), dtype=np.float32)

        results.append((d_ref.channel, d_tgt.channel, flow, joint_mask))

    return results


# ---------------------------------------------------------------------------
# Catmull-Rom flow sampling helper
# ---------------------------------------------------------------------------

def _sample_flow(field: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic interpolation of a 2-D scalar field at (ys, xs).

    Uses scipy.ndimage.map_coordinates with order=3 (cubic spline, equivalent
    to Catmull-Rom for smooth fields).  ys and xs are 1-D float arrays.
    """
    coords = np.stack([
        np.clip(ys, 0, field.shape[0] - 1),
        np.clip(xs, 0, field.shape[1] - 1),
    ])
    return map_coordinates(field, coords, order=3, mode='nearest')


# ---------------------------------------------------------------------------
# Ghost mask advection helper
# ---------------------------------------------------------------------------

def _advect_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Advect a boolean source footprint forward through a flow field.

    Every True pixel at (y, x) in *mask* is displaced by the Catmull-Rom
    sampled flow vector at that pixel.  Displaced pixel positions are
    accumulated into a float32 weight map (raw hit counts, not normalised,
    not dilated).

    Overlap between this weight map and a blob footprint mask is computed as
        (weight_map * blob_mask).sum()
    A non-zero overlap means the flow carries source pixels into the blob.

    Parameters
    ----------
    mask : (H, W) bool
    flow : (2, H, W) float32 — flow[0]=row disp, flow[1]=col disp

    Returns
    -------
    weight_map : (H, W) float32 — raw advection hit counts
    """
    H, W = mask.shape
    ys, xs = np.where(mask)
    weight_map = np.zeros((H, W), dtype=np.float32)
    if ys.size == 0:
        return weight_map

    v = _sample_flow(flow[0], ys.astype(float), xs.astype(float))
    u = _sample_flow(flow[1], ys.astype(float), xs.astype(float))

    pred_ys = np.clip(np.round(ys + v).astype(int), 0, H - 1)
    pred_xs = np.clip(np.round(xs + u).astype(int), 0, W - 1)

    # np.add.at handles duplicate destination pixels correctly (unlike +=)
    np.add.at(weight_map, (pred_ys, pred_xs), 1.0)
    return weight_map


# ---------------------------------------------------------------------------
# Stage 2 — Track linking with split/merge detection
# ---------------------------------------------------------------------------

def link_tracks(
    detections: list[ChannelDetection],
    flow_seq: list[tuple[int, int, np.ndarray, np.ndarray]],
    min_match_overlap: int = 5,
    min_split_overlap: int = 3,
    ghost_threshold: float = 0.5,
) -> list[dict]:
    """Link per-channel blob detections into multi-channel tracks.

    Uses stateful ghost masks and pixel-overlap matching — no Euclidean distance.

    Algorithm
    ---------
    Each track maintains a *ghost mask*: its most recently known wavelet
    footprint, advected channel-by-channel through the flow via Catmull-Rom
    cubic interpolation.  Matching and split attribution both use pixel-overlap
    between the advected ghost mask and new blob masks; Euclidean distance is
    never used.

    For each consecutive channel pair (ref → tgt):

    A. Advect every active track's ghost mask through the flow → adv_maps.
    B. Hungarian matching on negative-overlap cost matrix.  Pairs with overlap
       ≥ min_match_overlap are matched; ghost reset to the matched detection mask.
    C. Unmatched predictions: check for merge via overlap, then extrapolate
       centroid via flow; ghost drifts forward (binarized at ghost_threshold).
    D. Unmatched detections: attributed as splits of the track whose advected
       ghost has the highest overlap ≥ min_split_overlap.  No distance fallback.
       Blobs with zero overlap to any source start a new independent track.

    Parameters
    ----------
    detections :
        Per-channel detection results in channel order.
    flow_seq :
        Output of :func:`compute_flow_sequence`.
    min_match_overlap :
        Minimum pixel overlap (advected ghost ∩ blob mask) to accept a
        continuation match.
    min_split_overlap :
        Minimum pixel overlap to attribute an unmatched detection as a split
        of an existing source.
    ghost_threshold :
        Threshold on the advected weight map used to binarize the ghost mask
        for gap channels.  0.5 means a pixel needs ≥1 advected hit to survive.

    Returns
    -------
    list[dict]
        One dict per track with keys: ``id``, ``trajectory``, ``masks``,
        ``split_at``, ``split_from``, ``merge_into``, ``active``.
    """
    def _new_track(tid, ch, y, x, mask):
        return dict(
            id=tid, trajectory=[(ch, y, x)], masks={ch: mask},
            split_at=[], split_from=None, merge_into=[], active=True,
        )

    tracks: list[dict] = []

    # Seed one track per blob in the first channel.
    d0 = detections[0]
    for mask, (y, x) in zip(d0.footprint_masks, d0.peaks):
        tracks.append(_new_track(len(tracks), d0.channel, float(y), float(x), mask))

    # Stateful ghost masks: each source's current advected footprint.
    ghost_masks: dict[int, np.ndarray] = {
        t['id']: t['masks'][d0.channel].copy() for t in tracks
    }

    for fi, (ch_ref, ch_tgt, flow, _) in enumerate(flow_seq):
        d_tgt  = detections[fi + 1]
        active = [t for t in tracks if t['active']]

        # A. Advect every active source's ghost mask through the flow field.
        adv_maps: dict[int, np.ndarray] = {
            t['id']: _advect_mask(ghost_masks[t['id']], flow)
            for t in active
        }

        # No detections: drift all ghosts forward and extrapolate centroids.
        if not d_tgt.peaks:
            for t in active:
                adv = adv_maps[t['id']]
                new_ghost = (adv >= ghost_threshold)
                ghost_masks[t['id']] = new_ghost if new_ghost.any() else (adv > 0)
                cy, cx = t['trajectory'][-1][1:]
                t['trajectory'].append((ch_tgt, cy, cx))
            continue

        # B. Overlap cost matrix → Hungarian matching (continuation).
        n_active = len(active)
        n_blobs  = len(d_tgt.footprint_masks)
        cost = np.zeros((n_active, n_blobs), dtype=float)
        for r, t in enumerate(active):
            for c, blob_mask in enumerate(d_tgt.footprint_masks):
                cost[r, c] = -float((adv_maps[t['id']] * blob_mask).sum())

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_pred: set[int] = set()
        matched_det:  set[int] = set()
        det_to_track: dict[int, dict] = {}

        for r, c in zip(row_ind, col_ind):
            if -cost[r, c] >= min_match_overlap:
                t = active[r]
                t['trajectory'].append(
                    (ch_tgt, float(d_tgt.peaks[c][0]), float(d_tgt.peaks[c][1]))
                )
                t['masks'][ch_tgt]   = d_tgt.footprint_masks[c]
                ghost_masks[t['id']] = d_tgt.footprint_masks[c]
                matched_pred.add(r)
                matched_det.add(c)
                det_to_track[c] = t

        # C. Unmatched predictions: merge check + ghost drift + centroid extrapolation.
        for r, t in enumerate(active):
            if r in matched_pred:
                continue
            cy, cx = t['trajectory'][-1][1:]
            ys_a = np.array([cy], dtype=float)
            xs_a = np.array([cx], dtype=float)
            py = float(cy + _sample_flow(flow[0], ys_a, xs_a)[0])
            px = float(cx + _sample_flow(flow[1], ys_a, xs_a)[0])

            # Merge: advected mask overlaps a detection already owned by another track.
            for c_det, owner in det_to_track.items():
                ov = float((adv_maps[t['id']] * d_tgt.footprint_masks[c_det]).sum())
                if ov >= min_match_overlap:
                    t['merge_into'].append((ch_tgt, owner['id']))
                    break

            t['trajectory'].append((ch_tgt, py, px))

            # Drift ghost through flow (gap channel — no wavelet detection).
            adv = adv_maps[t['id']]
            new_ghost = (adv >= ghost_threshold)
            ghost_masks[t['id']] = new_ghost if new_ghost.any() else (adv > 0)

        # D. Unmatched detections: split attribution via flow overlap (NO distance fallback).
        for c, (dy, dx) in enumerate(d_tgt.peaks):
            if c in matched_det:
                continue
            blob_mask    = d_tgt.footprint_masks[c]
            best_overlap = 0.0
            best_parent  = None

            for t in active:
                ov = float((adv_maps[t['id']] * blob_mask).sum())
                if ov >= min_split_overlap and ov > best_overlap:
                    best_overlap = ov
                    best_parent  = t

            parent_id = None
            if best_parent is not None:
                parent_id = best_parent['id']
                best_parent['split_at'].append(ch_tgt)

            new_t = _new_track(
                len(tracks), ch_tgt, float(dy), float(dx), blob_mask,
            )
            new_t['split_from'] = parent_id
            ghost_masks[new_t['id']] = blob_mask.copy()
            tracks.append(new_t)

    return tracks


# ---------------------------------------------------------------------------
# Stage 3 — Kinematic classification
# ---------------------------------------------------------------------------

def classify_kinematic(
    tracks: list[dict],
    min_displacement: float = 3.0,
) -> list[dict]:
    """Add kinematic classification fields to each track dict (in-place).

    A track is **kinematically active** if:
    - Its cumulative centroid displacement across channels ≥ *min_displacement*, or
    - It was involved in a split event (either as parent or as split-off child).

    Adds keys ``displacement`` (float, px), ``has_split`` (bool),
    and ``kinematic`` (bool) to each track dict.
    """
    for t in tracks:
        traj = t['trajectory']
        # Sum of step-wise displacements — captures curved trajectories better
        # than straight-line start-to-end distance.
        disp = sum(
            np.hypot(traj[i+1][1] - traj[i][1], traj[i+1][2] - traj[i][2])
            for i in range(len(traj) - 1)
        )
        split = bool(t['split_at']) or t['split_from'] is not None or bool(t['merge_into'])
        t['displacement'] = float(disp)
        t['has_split']    = split
        t['kinematic']    = disp >= min_displacement or split

    return tracks


# ---------------------------------------------------------------------------
# Stage 4 — Source grouping
# ---------------------------------------------------------------------------

def group_into_sources(tracks: list[dict]) -> list[dict]:
    """Group related tracks into sources via union-find over split/merge edges.

    Two tracks belong to the same source if they are connected by any chain of
    ``split_from`` (child→parent) or ``merge_into`` (merging track → target)
    relationships.  The result is one source per connected component.

    Annotates each track dict in-place with a ``source_id`` key.

    Parameters
    ----------
    tracks :
        Output of :func:`classify_kinematic` (or :func:`link_tracks`).

    Returns
    -------
    list[dict]
        One dict per source, sorted by ascending ``id``, with keys:
        ``id``, ``track_ids``, ``channels``, ``n_channels``,
        ``split_events``, ``merge_events``.
    """
    from collections import defaultdict

    # Path-compressed union-find.
    parent = {t['id']: t['id'] for t in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for t in tracks:
        if t['split_from'] is not None:
            union(t['id'], t['split_from'])
        for _ch, target_id in t['merge_into']:
            union(t['id'], target_id)

    track_by_id = {t['id']: t for t in tracks}
    groups: dict[int, list[dict]] = defaultdict(list)
    for t in tracks:
        groups[find(t['id'])].append(t)

    sources = []
    for sid, group_tracks in enumerate(groups.values()):
        track_ids    = sorted(t['id'] for t in group_tracks)
        channels     = sorted({ch for t in group_tracks for ch, *_ in t['trajectory']})
        split_events = sorted({ch for t in group_tracks for ch in t['split_at']})
        merge_events = sorted({ch for t in group_tracks for ch, _ in t['merge_into']})
        src = dict(
            id=sid,
            track_ids=track_ids,
            channels=channels,
            n_channels=len(channels),
            split_events=split_events,
            merge_events=merge_events,
        )
        sources.append(src)
        for tid in track_ids:
            track_by_id[tid]['source_id'] = sid

    return sources


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_flow_tracker(
    cube: np.ndarray,
    channel_list: list[int],
    scales: int = 6,
    k_sigma: float = 5.0,
    use_scale: int = 5,
    min_area: int = 20,
    thresh: float | None = None,
    min_match_overlap: int = 5,
    min_split_overlap: int = 3,
    ghost_threshold: float = 0.5,
    min_displacement: float = 3.0,
) -> tuple[list[ChannelDetection], list[tuple], list[dict], list[dict]]:
    """Detect → flow → track → classify → group sources.

    Convenience wrapper that chains all four stages.

    Returns
    -------
    detections : list[ChannelDetection]
    flow_seq   : list of (ch_ref, ch_tgt, flow, joint_mask)
    tracks     : list of classified track dicts (each annotated with source_id)
    sources    : list of source dicts grouping tracks by physical object
    """
    detections = detect_cube_per_channel(
        cube, channel_list=channel_list,
        scales=scales, k_sigma=k_sigma,
        use_scale=use_scale, min_area=min_area, thresh=thresh,
    )
    flow_seq = compute_flow_sequence(detections)
    tracks   = link_tracks(detections, flow_seq,
                           min_match_overlap=min_match_overlap,
                           min_split_overlap=min_split_overlap,
                           ghost_threshold=ghost_threshold)
    classify_kinematic(tracks, min_displacement=min_displacement)
    sources = group_into_sources(tracks)
    return detections, flow_seq, tracks, sources


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
    ap.add_argument("--min-match-overlap", type=int,   default=5,
                    help="Min pixel overlap (advected ghost ∩ blob) to match a continuation")
    ap.add_argument("--min-split-overlap", type=int,  default=3,
                    help="Min pixel overlap to attribute an unmatched blob as a split")
    ap.add_argument("--ghost-threshold",  type=float, default=0.5,
                    help="Threshold on advected weight map to binarize ghost mask in gap channels")
    ap.add_argument("--min-displacement", type=float, default=3.0,
                    help="Min centroid travel (px) to call a track kinematic")
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

    detections, flow_seq, tracks, sources = run_flow_tracker(
        cube, channel_list=channel_list,
        scales=args.scales, k_sigma=args.k_sigma,
        use_scale=args.use_scale, min_area=args.min_area, thresh=args.thresh,
        min_match_overlap=args.min_match_overlap,
        min_split_overlap=args.min_split_overlap,
        ghost_threshold=args.ghost_threshold,
        min_displacement=args.min_displacement,
    )

    n_kin = sum(1 for t in tracks if t['kinematic'])
    print(f"\n{len(tracks)} tracks  ({n_kin} kinematic)  →  {len(sources)} sources")
    for src in sources:
        print(f"  source {src['id']:2d}"
              f"  tracks={src['track_ids']}"
              f"  channels {src['channels'][0]}–{src['channels'][-1]}"
              f"  ({src['n_channels']} ch)"
              f"  splits={src['split_events']}"
              f"  merges={src['merge_events']}")

    # Write tracks CSV — one row per (track, channel) pair.
    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "track_id", "channel", "y", "x",
                    "displacement", "kinematic", "has_split", "split_from"])
        for t in tracks:
            for ch, y, x in t['trajectory']:
                w.writerow([
                    t.get('source_id', ""), t['id'], ch, f"{y:.2f}", f"{x:.2f}",
                    f"{t['displacement']:.3f}",
                    int(t['kinematic']), int(t['has_split']),
                    "" if t['split_from'] is None else t['split_from'],
                ])

    # Write sources CSV — one row per source.
    with open(out / "sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "track_ids", "ch_start", "ch_end",
                    "n_channels", "split_events", "merge_events"])
        for src in sources:
            w.writerow([
                src['id'],
                ";".join(str(i) for i in src['track_ids']),
                src['channels'][0], src['channels'][-1],
                src['n_channels'],
                ";".join(str(c) for c in src['split_events']),
                ";".join(str(c) for c in src['merge_events']),
            ])

    summary = {
        "cube": str(args.cube),
        "channels": channel_list,
        "n_tracks": len(tracks),
        "n_kinematic": n_kin,
        "n_sources": len(sources),
        "params": {k: v for k, v in vars(args).items()
                   if k not in ("cube", "out", "channels")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved tracks.csv, sources.csv, summary.json → {out}")


if __name__ == "__main__":
    main()
