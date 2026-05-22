"""Evaluate a trained `SeparationUNet3D` checkpoint on the validation split.

Replays the same train/val split used during training (same seed and
val_fraction from `<run>/config.json`), runs inference on the val cubes,
and writes:

    <out>/eval.log               -- run-time log
    <out>/per_cube_metrics.csv   -- one row per validation cube
    <out>/summary.json           -- aggregated statistics
    <out>/samples/<stem>.npz     -- input/pred/target arrays for a few cubes

Default `--out` is `<run>/eval`.
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
    logger = logging.getLogger("evaluate")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Directory with cube_*.h5 (typically the same one used for training)")
    ap.add_argument("--run", required=True, help="Train run directory containing best.pt + config.json")
    ap.add_argument("--out", default=None, help="Eval output dir (default: <run>/eval)")
    ap.add_argument("--checkpoint", default="best.pt", help="Which checkpoint inside --run to load")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="Which split to evaluate (uses splits.json from the run dir).")
    ap.add_argument("--n-samples", type=int, default=8, help="Number of qualitative example arrays to dump")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    log = _setup_logging(out_dir)

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        log.error("Missing %s", cfg_path)
        sys.exit(1)
    train_cfg = json.loads(cfg_path.read_text())
    log.info("Train config: %s", json.dumps(train_cfg, indent=2))

    import numpy as np
    import torch
    from torch.utils.data import random_split
    from galcubecraft_sourceid import (
        CubeDataset, SeparationUNet3D, MaskedSeparationUNet3D,
        PositionGuidedMaskedSeparationUNet3D, TwoStageUNet3D,
        JointDetSegUNet3D, InstanceSegUNet3D, BinarySegUNet3D,
    )

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    log.info("Loading dataset from %s", args.data)
    ds = CubeDataset(args.data)
    log.info("Dataset size = %d  max_n_gals = %d", len(ds), ds.max_n_gals)

    splits_path = run_dir / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        if args.split not in splits:
            log.error("splits.json has no '%s' split (keys: %s)", args.split, list(splits.keys()))
            sys.exit(1)
        eval_indices = list(splits[args.split])
        log.info("Using %s split from splits.json (%d cubes)", args.split, len(eval_indices))
    else:
        # Backward compat: replay legacy 2-way split (no test set was saved).
        log.warning("No splits.json found in %s — replaying legacy 2-way split.", run_dir)
        n_val = max(1, int(train_cfg["val_fraction"] * len(ds)))
        n_train = len(ds) - n_val
        _, val_subset = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(int(train_cfg["seed"])),
        )
        eval_indices = list(val_subset.indices)
        log.info("Validation split has %d cubes (replay seed=%d)", len(eval_indices), int(train_cfg["seed"]))

    model_kind = train_cfg.get("model", "baseline")
    has_diffuse_slot = model_kind in ("hungarian_diffuse", "mask", "two_stage", "pos_guided")
    permutation_invariant = model_kind in ("hungarian", "hungarian_diffuse", "mask", "two_stage", "pos_guided")
    out_slots = ds.max_n_gals + (1 if has_diffuse_slot else 0)
    base = int(train_cfg.get("base", 16))
    log.info("Model kind: %s  out_slots=%d  permutation_invariant=%s",
             model_kind, out_slots, permutation_invariant)
    if model_kind == "mask":
        model = MaskedSeparationUNet3D(max_n_gals=ds.max_n_gals, base=base).to(device)
    elif model_kind == "pos_guided":
        model = PositionGuidedMaskedSeparationUNet3D(
            max_n_gals=ds.max_n_gals, base=base,
            center_sigma=float(train_cfg.get("center_sigma", 3.0)),
            bias_scale=float(train_cfg.get("bias_scale", 5.0)),
        ).to(device)
    elif model_kind == "two_stage":
        model = TwoStageUNet3D(max_n_gals=ds.max_n_gals, base=base).to(device)
    elif model_kind == "joint":
        model = JointDetSegUNet3D(
            max_n_gals=ds.max_n_gals, base=base,
            center_sigma=float(train_cfg.get("center_sigma", 1.5)),
        ).to(device)
    elif model_kind == "instance":
        model = InstanceSegUNet3D(max_n_gals=ds.max_n_gals, base=base).to(device)
    elif model_kind == "binseg":
        model = BinarySegUNet3D(max_n_gals=ds.max_n_gals, base=base).to(device)
    else:
        model = SeparationUNet3D(max_n_gals=out_slots, base=base).to(device)
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        log.error("Missing checkpoint %s", ckpt_path)
        sys.exit(1)
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        log.error("Failed to load %s — likely a max_n_gals/base mismatch.\n%s", ckpt_path, e)
        sys.exit(1)
    model.eval()
    log.info("Loaded checkpoint %s", ckpt_path)

    rows = []
    samples_written = 0
    t0 = time.time()
    with torch.no_grad():
        for k, idx in enumerate(eval_indices):
            item = ds[idx]
            cube = item["cube"].unsqueeze(0).to(device)               # (1, 1, C, Y, X)
            target = item["galaxy_cubes"].unsqueeze(0).to(device)      # (1, M, C, Y, X)
            valid = item["galaxy_valid"].unsqueeze(0).to(device)       # (1, M)
            if model_kind == "pos_guided":
                centers_t = item["centers_cyx"].unsqueeze(0).to(device)  # (1, M, 3)
                pred_full = model(cube, valid, centers_t)
            elif model_kind in ("mask", "two_stage"):
                pred_full = model(cube, valid)
            elif model_kind == "joint":
                centers_t = item["centers_cyx"].unsqueeze(0).to(device)
                out = model(cube, centers_t)
                pred_full = out[0]                                        # masks only
            elif model_kind == "instance":
                logits = model(cube)                                      # (1, M+1, C, Y, X)
                labels = logits.argmax(dim=1, keepdim=True)               # (1, 1, C, Y, X)
                # Convert argmax label map to per-galaxy flux cubes.
                pred_full = torch.zeros(1, ds.max_n_gals, *cube.shape[2:],
                                        device=device, dtype=cube.dtype)
                for g in range(ds.max_n_gals):
                    mask = (labels == g).float()                          # (1, 1, C, Y, X)
                    pred_full[:, g] = (cube * mask).squeeze(1)
            elif model_kind == "binseg":
                masks = model(cube)                                        # (1, M, C, Y, X) sigmoid
                pred_full = masks * cube                                   # flux predictions
            else:
                pred_full = model(cube)                                   # (1, M[+1], C, Y, X)
            pred_diffuse = pred_full[:, ds.max_n_gals:ds.max_n_gals+1] if has_diffuse_slot else None
            pred_gal = pred_full[:, :ds.max_n_gals]                     # (1, M, C, Y, X)

            # Hungarian-align pred galaxy slots to target slots so saved
            # arrays and per-slot metrics are comparable across runs.
            if permutation_invariant:
                from galcubecraft_sourceid.models import _pairwise_voxel_mse
                from scipy.optimize import linear_sum_assignment
                costs = _pairwise_voxel_mse(pred_gal, target).cpu().numpy()
                aligned = torch.zeros_like(target)
                n_v = int(valid.sum())
                if n_v > 0:
                    rows_, cols_ = linear_sum_assignment(costs[0, :, :n_v])
                    for r_, c_ in zip(rows_.tolist(), cols_.tolist()):
                        aligned[0, c_] = pred_gal[0, r_]
                pred = aligned
            else:
                pred = pred_gal

            # Per-slot MSE (masked, voxel-wise).
            w = valid[:, :, None, None, None]
            num_vox = pred.shape[2] * pred.shape[3] * pred.shape[4]
            denom = valid.sum().clamp(min=1.0) * num_vox
            per_slot_mse = float((((pred - target) ** 2) * w).sum() / denom)

            # Flux conservation per cube.
            tgt_sum = (target * w).sum(dim=(2, 3, 4))                   # (1, M)
            prd_sum = (pred * w).sum(dim=(2, 3, 4))
            flux_rel = float(((prd_sum - tgt_sum).abs() * valid).sum()
                             / ((tgt_sum.abs() * valid).sum() + 1e-8))

            # Residual energy: how much of the input is NOT explained by the
            # sum of predicted slots (this is the implicit diffuse field).
            sum_pred = pred_full.sum(dim=1)                              # include diffuse slot if present
            input_2d = cube.squeeze(1)
            res_energy = float(((input_2d - sum_pred) ** 2).mean())
            input_energy = float((input_2d ** 2).mean())
            res_frac = res_energy / max(input_energy, 1e-12)

            # Peak position error (in voxels) per valid slot.
            n_g = int(valid.sum())
            centers = item["centers_cyx"].numpy()                       # (M, 3)
            peak_dists = []
            for g in range(n_g):
                pg = pred[0, g].cpu().numpy()
                if pg.max() < 1e-8:
                    peak_dists.append(float("nan"))
                    continue
                peak = np.unravel_index(int(pg.argmax()), pg.shape)
                peak_dists.append(float(np.linalg.norm(np.asarray(peak) - centers[g])))
            mean_peak_err = float(np.nanmean(peak_dists)) if peak_dists else float("nan")

            row = {
                "path": item["path"],
                "n_gals": int(valid.sum()),
                "per_slot_mse": per_slot_mse,
                "flux_relative_error": flux_rel,
                "residual_energy": res_energy,
                "residual_fraction_of_input": res_frac,
                "mean_peak_distance_px": mean_peak_err,
            }
            rows.append(row)

            if samples_written < args.n_samples:
                save_kwargs = dict(
                    input=cube.cpu().numpy()[0, 0],
                    pred=pred.cpu().numpy()[0],
                    target=target.cpu().numpy()[0],
                    valid=valid.cpu().numpy()[0],
                    centers_cyx=centers,
                    classes=item["classes"].numpy(),
                )
                if pred_diffuse is not None:
                    save_kwargs["pred_diffuse"] = pred_diffuse.cpu().numpy()[0, 0]
                np.savez(samples_dir / f"{item['path']}.npz", **save_kwargs)
                samples_written += 1

            if (k + 1) % max(1, len(eval_indices) // 10) == 0:
                log.info("  %d / %d cubes", k + 1, len(eval_indices))

    csv_path = out_dir / "per_cube_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = ["per_slot_mse", "flux_relative_error", "residual_energy",
            "residual_fraction_of_input", "mean_peak_distance_px"]
    summary = _aggregate(rows, keys)
    summary["n_cubes"] = len(rows)
    summary["split"] = args.split
    summary["checkpoint"] = str(ckpt_path)
    summary["data"] = str(args.data)
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("Wrote %s (%d rows) and summary.json (%d sample arrays)",
             csv_path.name, len(rows), samples_written)
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
