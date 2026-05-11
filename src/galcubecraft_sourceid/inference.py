"""Decode predicted 3D heatmaps into discrete galaxy detections."""
from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max


def decode_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.2,
    min_distance: int = 2,
    max_detections_per_class: int = 8,
):
    """Extract peaks from a multi-class 3D heatmap.

    Parameters
    ----------
    heatmap : (n_classes, n_ch, n_y, n_x) model output in [0, 1].
    threshold : minimum peak value.
    min_distance : spacing for NMS in voxels.
    max_detections_per_class : cap on peaks kept per class.

    Returns
    -------
    list of dicts, each with keys
      'class' (int), 'channel' (int), 'y' (int), 'x' (int), 'score' (float).
    """
    out = []
    for cls in range(heatmap.shape[0]):
        peaks = peak_local_max(
            heatmap[cls],
            min_distance=min_distance,
            threshold_abs=threshold,
            num_peaks=max_detections_per_class,
        )
        for c, y, x in peaks:
            out.append({
                "class": int(cls),
                "channel": int(c),
                "y": int(y),
                "x": int(x),
                "score": float(heatmap[cls, c, y, x]),
            })
    out.sort(key=lambda d: -d["score"])
    return out


def match_detections(
    detections, gt_centers_cyx, gt_classes, tol: float = 2.5,
):
    """Greedy nearest-neighbour match of detections to ground truth.

    Returns the list of matched (detection, gt_index, distance) triples and
    the indices of unmatched ground-truth galaxies.
    """
    gt_centers_cyx = np.asarray(gt_centers_cyx, dtype=float)
    gt_classes = np.asarray(gt_classes, dtype=int)
    unmatched = set(range(len(gt_centers_cyx)))
    matches = []
    for det in detections:
        p = np.array([det["channel"], det["y"], det["x"]], dtype=float)
        best_j, best_d = -1, np.inf
        for j in unmatched:
            if gt_classes[j] != det["class"]:
                continue
            d = float(np.linalg.norm(gt_centers_cyx[j] - p))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= tol:
            matches.append((det, best_j, best_d))
            unmatched.remove(best_j)
    return matches, sorted(unmatched)
