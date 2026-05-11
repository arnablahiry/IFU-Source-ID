"""Train SeparationUNet3D on a folder of GalCubeCraft HDF5 cubes.

Logs train/val loss, flux-conservation and per-slot residuals every epoch to
both stderr and `runs/<name>/train.log`. Writes `best.pt` + `last.pt` and
dumps the full config to `config.json`.

Example:
    python scripts/train_separation.py \\
        --data data/train --out runs/sep_v1 \\
        --epochs 30 --batch-size 2 --base 16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_separation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _flux_metrics(pred, target, valid):
    """Per-cube / per-slot diagnostics.

    Returns a dict of scalars averaged over the batch:
      - per_slot_mse:          masked MSE per voxel per valid slot
      - flux_relative_error:   |sum(pred)-sum(target)| / sum(target), averaged over valid slots
      - residual_vs_input_mse: mean((input - sum(pred))**2) — small means the
                                model also fit the diffuse component
    """
    import torch
    from scipy.optimize import linear_sum_assignment
    from galcubecraft_sourceid.models import _pairwise_voxel_mse

    # Drop the diffuse slot (last channel) if pred has more slots than target.
    M = target.shape[1]
    pred_gal = pred[:, :M]

    # Hungarian-aligned metrics: training is permutation-invariant, so naive
    # per-slot MSE on raw output order is meaningless. Match preds to targets
    # by min-cost assignment first, then compute per-slot MSE / flux error.
    costs = _pairwise_voxel_mse(pred_gal, target).detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    aligned = torch.zeros_like(target)
    for b in range(pred_gal.shape[0]):
        n_v = int(valid_np[b].sum())
        if n_v == 0:
            continue
        rows, cols = linear_sum_assignment(costs[b, :, :n_v])
        for r, c in zip(rows.tolist(), cols.tolist()):
            aligned[b, c] = pred_gal[b, r].detach()

    w = valid[:, :, None, None, None]
    num_vox = pred_gal.shape[2] * pred_gal.shape[3] * pred_gal.shape[4]
    denom = valid.sum().clamp(min=1.0) * num_vox
    per_slot_mse = (((aligned - target) ** 2) * w).sum() / denom

    tgt_sum = (target * w).sum(dim=(2, 3, 4))
    prd_sum = (aligned * w).sum(dim=(2, 3, 4))
    flux_rel = ((prd_sum - tgt_sum).abs() * valid).sum() / ((tgt_sum.abs() * valid).sum() + 1e-8)

    return {
        "per_slot_mse": float(per_slot_mse.detach().cpu()),
        "flux_relative_error": float(flux_rel.detach().cpu()),
    }


def _compute_loss(loss_fn, needs_cube, pred, target, valid, cube):
    if needs_cube:
        return loss_fn(pred, target, valid, cube)
    return loss_fn(pred, target, valid)


def train_one_epoch(model, loader, opt, loss_fn, needs_cube, device, log, epoch, log_every):
    import torch
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        cube = batch["cube"].to(device, non_blocking=True)
        target = batch["galaxy_cubes"].to(device, non_blocking=True)
        valid = batch["galaxy_valid"].to(device, non_blocking=True)
        pred = model(cube)
        loss = _compute_loss(loss_fn, needs_cube, pred, target, valid, cube)
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.detach().cpu()) * cube.size(0)
        n += cube.size(0)
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            log.info("  epoch %d  step %4d/%d  loss %.4e  (%.1f samples/s)",
                     epoch, i + 1, len(loader), float(loss.detach().cpu()),
                     n / max(elapsed, 1e-6))
    return total / max(n, 1)


def validate(model, loader, loss_fn, needs_cube, device):
    import torch
    model.eval()
    metrics = {"val_loss": 0.0, "per_slot_mse": 0.0, "flux_relative_error": 0.0, "n": 0}
    with torch.no_grad():
        for batch in loader:
            cube = batch["cube"].to(device, non_blocking=True)
            target = batch["galaxy_cubes"].to(device, non_blocking=True)
            valid = batch["galaxy_valid"].to(device, non_blocking=True)
            pred = model(cube)
            loss = _compute_loss(loss_fn, needs_cube, pred, target, valid, cube)
            m = _flux_metrics(pred, target, valid)
            bs = cube.size(0)
            metrics["val_loss"] += float(loss.detach().cpu()) * bs
            metrics["per_slot_mse"] += m["per_slot_mse"] * bs
            metrics["flux_relative_error"] += m["flux_relative_error"] * bs
            metrics["n"] += bs
    n = max(metrics.pop("n"), 1)
    return {k: v / n for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Directory with cube_*.h5")
    ap.add_argument("--out", default="runs/sep", help="Output directory (checkpoints + logs)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels")
    ap.add_argument(
        "--model",
        choices=["baseline", "hungarian", "hungarian_diffuse", "mask"],
        default="baseline",
        help=(
            "baseline: fixed-slot masked MSE (sep_v1).  "
            "hungarian: permutation-invariant matching + per-slot rebalancing.  "
            "hungarian_diffuse: hungarian + extra diffuse output slot + reconstruction term.  "
            "mask: softmax-over-slots mask decoder (slots compete per voxel) + "
            "Hungarian + diffuse slot. Σ pred ≡ input by construction."
        ),
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1,
                    help="Fraction of cubes held out for final evaluation (never seen during training).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=20, help="Log every N training steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.info("Run directory: %s", run_dir)
    log.info("Config: %s", json.dumps(vars(args), indent=2))

    import torch
    from torch.utils.data import DataLoader, random_split
    from galcubecraft_sourceid import (
        CubeDataset,
        SeparationUNet3D,
        MaskedSeparationUNet3D,
        masked_separation_loss,
        hungarian_separation_loss,
        hungarian_separation_loss_with_diffuse,
    )

    torch.manual_seed(args.seed)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    log.info("Scanning dataset at %s", args.data)
    ds = CubeDataset(args.data)
    log.info("Loaded %d cubes.  max_n_gals inferred = %d", len(ds), ds.max_n_gals)

    n_test = max(1, int(args.test_fraction * len(ds)))
    n_val = max(1, int(args.val_fraction * len(ds)))
    n_train = len(ds) - n_val - n_test
    if n_train <= 0:
        log.error("val_fraction + test_fraction (%.2f + %.2f) leaves no training data.",
                  args.val_fraction, args.test_fraction)
        sys.exit(1)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    # Save the explicit index lists so downstream tools (eval, analysis) can
    # use the same partitions deterministically without re-seeding the RNG.
    splits = {
        "train": [int(i) for i in train_ds.indices],
        "val":   [int(i) for i in val_ds.indices],
        "test":  [int(i) for i in test_ds.indices],
    }
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    log.info("Wrote split index lists to %s", run_dir / "splits.json")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    if args.model == "mask":
        out_slots = ds.max_n_gals + 1   # +1 diffuse slot is built into the mask decoder
        # Mask decoder enforces Σ pred = input, so no recon term is needed.
        loss_fn = lambda p, t, v, c: hungarian_separation_loss_with_diffuse(
            p, t, v, c, recon_weight=0.0,
        )
        needs_cube = True
        model = MaskedSeparationUNet3D(max_n_gals=ds.max_n_gals, base=args.base).to(device)
        log.info("Model: MaskedSeparationUNet3D(galaxy_slots=%d + 1 diffuse, base=%d)",
                 ds.max_n_gals, args.base)
    elif args.model == "hungarian_diffuse":
        out_slots = ds.max_n_gals + 1
        loss_fn = hungarian_separation_loss_with_diffuse
        needs_cube = True
        model = SeparationUNet3D(max_n_gals=out_slots, base=args.base).to(device)
        log.info("Model: SeparationUNet3D(out_slots=%d, base=%d)", out_slots, args.base)
    elif args.model == "hungarian":
        out_slots = ds.max_n_gals
        loss_fn = hungarian_separation_loss
        needs_cube = False
        model = SeparationUNet3D(max_n_gals=out_slots, base=args.base).to(device)
        log.info("Model: SeparationUNet3D(out_slots=%d, base=%d)", out_slots, args.base)
    else:
        out_slots = ds.max_n_gals
        loss_fn = masked_separation_loss
        needs_cube = False
        model = SeparationUNet3D(max_n_gals=out_slots, base=args.base).to(device)
        log.info("Model: SeparationUNet3D(out_slots=%d, base=%d)", out_slots, args.base)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loss: %s  | %d trainable params", args.model, n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = math.inf
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info("=== epoch %d/%d  lr=%.2e ===", epoch, args.epochs, opt.param_groups[0]["lr"])
        tl = train_one_epoch(model, train_loader, opt, loss_fn, needs_cube, device,
                             log, epoch, args.log_every)
        vm = validate(model, val_loader, loss_fn, needs_cube, device)
        sched.step()
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tl,
            "val_loss": vm["val_loss"],
            "val_per_slot_mse": vm["per_slot_mse"],
            "val_flux_relative_error": vm["flux_relative_error"],
            "elapsed_sec": dt,
            "lr": opt.param_groups[0]["lr"],
        }
        metrics_history.append(row)
        (run_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")

        improved = vm["val_loss"] < best_val
        if improved:
            best_val = vm["val_loss"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save(model.state_dict(), run_dir / "last.pt")

        log.info(
            "epoch %d  train %.4e  val %.4e  per_slot_mse %.4e  flux_rel_err %.3f  %s  (%.1fs)",
            epoch, tl, vm["val_loss"], vm["per_slot_mse"], vm["flux_relative_error"],
            "[best]" if improved else "      ", dt,
        )

    log.info("Training complete. Best val loss: %.4e", best_val)
    log.info("Checkpoints: %s", [p.name for p in run_dir.glob("*.pt")])


if __name__ == "__main__":
    main()
