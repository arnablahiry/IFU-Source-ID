"""Oracle evaluation of a trained `ExtractorUNet3D` on the test split.

For each test cube, queries the extractor at each *ground-truth* galaxy
position (i.e. assumes a perfect Stage-1 detector). This measures the
extractor's separation quality in isolation; end-to-end pipeline performance
also depends on detection accuracy.

Writes:
    <out>/eval.log
    <out>/per_cube_metrics.csv
    <out>/summary.json
    <out>/samples/<stem>.npz       -- input/per-galaxy pred/target arrays
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
    logger = logging.getLogger("evaluate_extract")
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="Train run directory containing best.pt + config.json")
    ap.add_argument("--out", default=None, help="Eval output dir (default: <run>/eval)")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
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
    from galcubecraft_sourceid import CubeDataset, ExtractorUNet3D, position_query_volume

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    ds = CubeDataset(args.data)
    log.info("Dataset size = %d  max_n_gals = %d", len(ds), ds.max_n_gals)

    splits = json.loads((run_dir / "splits.json").read_text())
    eval_indices = list(splits[args.split])
    log.info("Using %s split (%d cubes)", args.split, len(eval_indices))

    model = ExtractorUNet3D(base=int(train_cfg["base"])).to(device)
    state = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded %s", run_dir / args.checkpoint)

    query_sigma = float(train_cfg.get("query_sigma", 2.0))

    rows = []
    samples_written = 0
    t0 = time.time()
    with torch.no_grad():
        for k, idx in enumerate(eval_indices):
            item = ds[idx]
            cube = item["cube"].unsqueeze(0).to(device)              # (1, 1, C, Y, X)
            target = item["galaxy_cubes"].to(device)                  # (M, C, Y, X)
            valid = item["galaxy_valid"].to(device)                   # (M,)
            centers = item["centers_cyx"].to(device)                  # (M, 3)
            n_g = int(valid.sum())

            preds = []
            for g in range(n_g):
                q = position_query_volume(centers[g], cube.shape[2:], query_sigma)  # (1, C, Y, X)
                x = torch.cat([cube, q.unsqueeze(0)], dim=1)         # (1, 2, C, Y, X)
                pred_g = model(x)[0, 0]                              # (C, Y, X)
                preds.append(pred_g)
            pred_stack = torch.stack(preds, dim=0) if preds else torch.zeros((0,) + cube.shape[2:], device=device)

            tgt_g = target[:n_g]
            mse = float(((pred_stack - tgt_g) ** 2).mean()) if n_g > 0 else float("nan")

            tgt_sums = tgt_g.sum(dim=(1, 2, 3))                       # (n_g,)
            prd_sums = pred_stack.sum(dim=(1, 2, 3))
            flux_rel = float(((prd_sums - tgt_sums).abs()).sum() / (tgt_sums.abs().sum() + 1e-8)) if n_g > 0 else float("nan")

            sum_pred = pred_stack.sum(dim=0)                          # (C, Y, X)
            input_2d = cube.squeeze(1).squeeze(0)                     # (C, Y, X)
            res_energy = float(((input_2d - sum_pred) ** 2).mean())
            input_energy = float((input_2d ** 2).mean())
            res_frac = res_energy / max(input_energy, 1e-12)

            peak_dists = []
            centers_np = centers.cpu().numpy()
            for g in range(n_g):
                pg = pred_stack[g].cpu().numpy()
                if pg.max() < 1e-8:
                    peak_dists.append(float("nan"))
                    continue
                peak = np.unravel_index(int(pg.argmax()), pg.shape)
                peak_dists.append(float(np.linalg.norm(np.asarray(peak) - centers_np[g])))
            mean_peak_err = float(np.nanmean(peak_dists)) if peak_dists else float("nan")

            row = {
                "path": item["path"],
                "n_gals": n_g,
                "per_slot_mse": mse,
                "flux_relative_error": flux_rel,
                "residual_energy": res_energy,
                "residual_fraction_of_input": res_frac,
                "mean_peak_distance_px": mean_peak_err,
            }
            rows.append(row)

            if samples_written < args.n_samples and n_g > 0:
                pad_pred = np.zeros((ds.max_n_gals,) + tuple(cube.shape[2:]), dtype=np.float32)
                pad_pred[:n_g] = pred_stack.cpu().numpy()
                np.savez(
                    samples_dir / f"{item['path']}.npz",
                    input=cube.cpu().numpy()[0, 0],
                    pred=pad_pred,
                    target=target.cpu().numpy(),
                    valid=valid.cpu().numpy(),
                    centers_cyx=centers_np,
                    classes=item["classes"].numpy(),
                )
                samples_written += 1

            if (k + 1) % max(1, len(eval_indices) // 10) == 0:
                log.info("  %d / %d cubes  (%.1fs elapsed)", k + 1, len(eval_indices), time.time() - t0)

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
    summary["checkpoint"] = str(run_dir / args.checkpoint)
    summary["data"] = str(args.data)
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s and summary.json", csv_path.name)
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
