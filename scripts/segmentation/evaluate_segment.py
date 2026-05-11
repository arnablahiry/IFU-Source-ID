"""Evaluate a trained `EmbeddingUNet3D` on the test split.

Two modes for recovering per-voxel source IDs from the predicted embeddings:

  --seeds oracle   Use catalog centers as cluster seeds. Each foreground voxel
                   is assigned to the seed nearest in embedding space.
                   Sanity check — proves the embedding geometry is right.

  --seeds meanshift  Run mean-shift on foreground embeddings (no centers
                     needed). This is the deployment mode: at test time the
                     model only sees the cube, never the catalog.

Foreground voxels are those with `input >= flux_threshold * input.max()` per
cube. Background voxels are excluded from clustering and reported as ID = -1.

Writes per-cube metrics (instance count, segmentation IoU vs ground truth via
Hungarian, per-source flux MSE) and dumps a few sample arrays for plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path


def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("evaluate_segment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "eval.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _aggregate(rows, keys):
    import numpy as np
    out = {}
    for k in keys:
        vals = [r[k] for r in rows
                if r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return out


def _foreground_mask(cube_3d, frac):
    """Voxels above `frac * cube.max()` — clustering target."""
    import numpy as np
    if cube_3d.max() <= 0:
        return np.zeros_like(cube_3d, dtype=bool)
    return cube_3d >= (frac * float(cube_3d.max()))


def _cluster_meanshift(emb_fg, bandwidth):
    """Mean-shift on foreground embeddings."""
    from sklearn.cluster import MeanShift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(emb_fg)
    return ms.labels_, ms.cluster_centers_


def _hungarian_iou(pred_labels, gt_labels, n_pred, n_gt):
    """Best-match IoU between predicted and ground-truth segmentation masks.

    Builds an N_pred x N_gt IoU matrix, runs Hungarian on (1 - IoU), returns
    the matched mean IoU.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if n_pred == 0 or n_gt == 0:
        return float("nan"), []
    iou = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i in range(n_pred):
        pi = (pred_labels == i)
        for j in range(n_gt):
            gj = (gt_labels == j)
            inter = float((pi & gj).sum())
            union = float((pi | gj).sum())
            iou[i, j] = inter / max(union, 1e-8)
    rows, cols = linear_sum_assignment(-iou)
    matched = [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols)]
    if not matched:
        return float("nan"), []
    return float(np.mean([m[2] for m in matched])), matched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--seeds", default="meanshift", choices=["oracle", "meanshift"],
                    help="oracle uses ground-truth centers; meanshift is fully unseeded")
    ap.add_argument("--flux-threshold", type=float, default=0.05,
                    help="Foreground voxels are those with input >= frac * input.max()")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="Mean-shift bandwidth (default: delta_d/2 from train config)")
    ap.add_argument("--oracle-dist-threshold", type=float, default=None,
                    help="Oracle mode: voxels farther than this from all seeds are background "
                         "(default: 2*delta_d from train config)")
    ap.add_argument("--min-cluster-flux-frac", type=float, default=0.05,
                    help="Drop clusters with total flux < frac * (max cluster flux). Voxels in "
                         "dropped clusters are reassigned to the diffuse channel.")
    ap.add_argument("--min-cluster-voxels", type=int, default=50,
                    help="Drop clusters with fewer than this many voxels (likely noise).")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    log = _setup_logging(out_dir)

    train_cfg = json.loads((run_dir / "config.json").read_text())
    log.info("Train config: %s", json.dumps(train_cfg, indent=2))

    import numpy as np
    import torch
    from galcubecraft_sourceid import (
        CubeDataset, EmbeddingUNet3D, voxel_instance_labels, add_coord_channels,
    )

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s   seeds=%s", device, args.seeds)

    ds = CubeDataset(args.data)
    splits = json.loads((run_dir / "splits.json").read_text())
    eval_indices = list(splits[args.split])
    log.info("%s split: %d cubes", args.split, len(eval_indices))

    delta_v = float(train_cfg.get("delta_v", 0.5))
    delta_d = float(train_cfg.get("delta_d", 1.5))
    threshold_frac = float(train_cfg.get("threshold_frac", 0.1))
    bandwidth = args.bandwidth if args.bandwidth is not None else delta_d / 2.0
    oracle_dt = args.oracle_dist_threshold if args.oracle_dist_threshold is not None else 2.0 * delta_d
    log.info("delta_v=%.2f  delta_d=%.2f  bandwidth=%.2f  oracle_dt=%.2f",
             delta_v, delta_d, bandwidth, oracle_dt)

    use_coords = bool(train_cfg.get("coord_conv", False))
    in_channels = 4 if use_coords else 1
    model = EmbeddingUNet3D(
        in_channels=in_channels,
        embedding_dim=int(train_cfg["embedding_dim"]),
        base=int(train_cfg["base"]),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / args.checkpoint, map_location=device))
    model.eval()
    log.info("Loaded %s  in_channels=%d  coord_conv=%s", run_dir / args.checkpoint, in_channels, use_coords)

    rows = []
    samples_written = 0
    t0 = time.time()
    with torch.no_grad():
        for k, idx in enumerate(eval_indices):
            item = ds[idx]
            cube = item["cube"].unsqueeze(0).to(device)               # (1, 1, C, Y, X)
            gc = item["galaxy_cubes"]                                  # (M, C, Y, X)
            valid = item["galaxy_valid"]                               # (M,)
            centers = item["centers_cyx"].numpy()                      # (M, 3)
            n_g = int(valid.sum())

            model_input = add_coord_channels(cube) if use_coords else cube
            emb = model(model_input)[0]                                # (D, C, Y, X)
            D, C, Y, X = emb.shape
            emb_np = emb.cpu().numpy()
            cube_np = cube.cpu().numpy()[0, 0]                         # (C, Y, X)

            # Ground-truth voxel labels (for IoU eval).
            gt_lab = voxel_instance_labels(gc, valid, threshold_frac).numpy()   # (C, Y, X)

            fg = _foreground_mask(cube_np, args.flux_threshold)
            fg_flat = fg.reshape(-1)
            emb_flat = emb_np.reshape(D, -1)
            emb_fg = emb_flat[:, fg_flat].T                            # (n_fg, D)

            if emb_fg.shape[0] == 0:
                pred_lab = -np.ones((C, Y, X), dtype=np.int64)
                n_pred = 0
            elif args.seeds == "oracle" and n_g > 0:
                # Sample seed embeddings at the rounded catalog positions.
                seeds = []
                for g in range(n_g):
                    cz = int(np.clip(round(centers[g, 0]), 0, C - 1))
                    cy = int(np.clip(round(centers[g, 1]), 0, Y - 1))
                    cx = int(np.clip(round(centers[g, 2]), 0, X - 1))
                    seeds.append(emb_np[:, cz, cy, cx])
                seeds = np.stack(seeds, axis=0)                        # (n_g, D)
                # nearest-seed assignment for each foreground voxel
                d2 = ((emb_fg[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)  # (n_fg, n_g)
                nearest = d2.argmin(axis=1)
                min_d = np.sqrt(d2.min(axis=1))
                lab_fg = np.where(min_d <= oracle_dt, nearest, -1)
                pred_lab = -np.ones((C, Y, X), dtype=np.int64)
                pred_lab.reshape(-1)[fg_flat] = lab_fg
                n_pred = n_g
            else:
                lab_fg, _ = _cluster_meanshift(emb_fg, bandwidth)
                pred_lab = -np.ones((C, Y, X), dtype=np.int64)
                pred_lab.reshape(-1)[fg_flat] = lab_fg
                n_pred = int(lab_fg.max() + 1) if lab_fg.size > 0 else 0

            # Filter spurious clusters: anything below the flux/voxel cut goes
            # to the diffuse channel. Built-in for the meanshift mode; oracle
            # mode is left untouched (its cluster count is fixed by n_g).
            if args.seeds == "meanshift" and n_pred > 0:
                cluster_total_flux = np.zeros(n_pred, dtype=np.float64)
                cluster_voxels = np.zeros(n_pred, dtype=np.int64)
                for kk in range(n_pred):
                    mk = (pred_lab == kk)
                    cluster_total_flux[kk] = float((cube_np * mk).sum())
                    cluster_voxels[kk] = int(mk.sum())
                if cluster_total_flux.max() > 0:
                    flux_cut = args.min_cluster_flux_frac * float(cluster_total_flux.max())
                    keep = (cluster_total_flux >= flux_cut) & (cluster_voxels >= args.min_cluster_voxels)
                else:
                    keep = np.zeros(n_pred, dtype=bool)
                # Remap kept cluster ids to a contiguous 0..K-1 range; dropped
                # clusters become -1 (diffuse).
                remap = -np.ones(n_pred, dtype=np.int64)
                new_idx = 0
                for kk in range(n_pred):
                    if keep[kk]:
                        remap[kk] = new_idx
                        new_idx += 1
                relabelled = pred_lab.copy()
                fg_pred = relabelled >= 0
                relabelled[fg_pred] = remap[relabelled[fg_pred]]
                pred_lab = relabelled
                n_pred = int(new_idx)

            # Diffuse mask = foreground voxels not assigned to a kept source.
            diffuse_mask = fg & (pred_lab < 0)
            iou_mean, matches = _hungarian_iou(pred_lab, gt_lab, n_pred, n_g)

            # Per-source flux: input * (pred_lab == k), compared to clean target.
            per_src_mse = []
            flux_rel_terms = []
            tgt_total = 0.0
            for r, c, _iou in matches:
                pred_mask = (pred_lab == r)
                pred_flux = cube_np * pred_mask
                tgt_flux = gc[c].numpy()
                per_src_mse.append(float(((pred_flux - tgt_flux) ** 2).mean()))
                flux_rel_terms.append(abs(float(pred_flux.sum()) - float(tgt_flux.sum())))
                tgt_total += abs(float(tgt_flux.sum()))
            avg_mse = float(np.mean(per_src_mse)) if per_src_mse else float("nan")
            flux_rel = float(sum(flux_rel_terms) / max(tgt_total, 1e-8)) if flux_rel_terms else float("nan")

            diffuse_flux_total = float((cube_np * diffuse_mask).sum())
            input_flux_total = float(cube_np.sum())
            diffuse_frac = diffuse_flux_total / max(input_flux_total, 1e-12)

            row = {
                "path": item["path"],
                "n_gt": n_g,
                "n_pred": n_pred,
                "matched_iou_mean": iou_mean,
                "per_source_mse": avg_mse,
                "flux_relative_error": flux_rel,
                "diffuse_flux_fraction": diffuse_frac,
            }
            rows.append(row)

            if samples_written < args.n_samples:
                np.savez(
                    samples_dir / f"{item['path']}.npz",
                    input=cube_np,
                    embedding=emb_np,
                    pred_labels=pred_lab,
                    gt_labels=gt_lab,
                    diffuse_mask=diffuse_mask,
                    centers_cyx=centers,
                    valid=valid.numpy(),
                    target=gc.numpy(),
                    matches=np.array(matches, dtype=np.float32) if matches else np.zeros((0, 3), dtype=np.float32),
                    seed_mode=np.array(args.seeds),
                )
                samples_written += 1

            if (k + 1) % max(1, len(eval_indices) // 10) == 0:
                log.info("  %d / %d  (%.1fs)", k + 1, len(eval_indices), time.time() - t0)

    csv_path = out_dir / "per_cube_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = ["matched_iou_mean", "per_source_mse", "flux_relative_error",
            "diffuse_flux_fraction", "n_pred", "n_gt"]
    summary = _aggregate(rows, keys)
    summary["n_cubes"] = len(rows)
    summary["seeds"] = args.seeds
    summary["split"] = args.split
    summary["checkpoint"] = str(run_dir / args.checkpoint)
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s and summary.json", csv_path.name)
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
