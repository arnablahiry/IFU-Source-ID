"""Detection metrics for source-identification evaluation."""
from __future__ import annotations

import numpy as np

from .inference import match_detections


def detection_stats(detections, gt_centers_cyx, gt_classes, tol: float = 2.5):
    """Compute precision / recall / mean distance error for one cube.

    A detection counts as a true positive if it matches a ground-truth
    galaxy of the same class within `tol` voxels.
    """
    matches, unmatched = match_detections(detections, gt_centers_cyx, gt_classes, tol=tol)
    tp = len(matches)
    fp = len(detections) - tp
    fn = len(unmatched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    mean_err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "mean_distance_error": mean_err,
    }
