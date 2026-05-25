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


def train_one_epoch(model, loader, opt, loss_fn, needs_cube, needs_centers,
                    is_seg_mask, is_instance, is_binseg, device, log, epoch, log_every):
    import torch
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        cube = batch["cube"].to(device, non_blocking=True)
        target = batch["galaxy_cubes"].to(device, non_blocking=True)
        valid = batch["galaxy_valid"].to(device, non_blocking=True)
        centers = batch["centers_cyx"].to(device, non_blocking=True)
        if is_binseg:
            masks = model(cube)
            loss = loss_fn(masks, target, valid)
        elif is_instance:
            logits = model(cube)
            loss = loss_fn(logits, target, valid)
        elif is_seg_mask:
            out = model(cube, centers, valid)
            if isinstance(out, tuple):  # joint model returns (masks, heatmap)
                gt_hmap = batch["heatmap"].to(device, non_blocking=True)
                loss = loss_fn((out[0], out[1], gt_hmap), target, valid)
            else:
                loss = loss_fn(out, target, valid)
        elif needs_centers:
            pred = model(cube, valid, centers)
            loss = _compute_loss(loss_fn, needs_cube, pred, target, valid, cube)
        else:
            pred = model(cube, valid)
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


def validate(model, loader, loss_fn, needs_cube, needs_centers,
             is_seg_mask, is_instance, is_binseg, device):
    import torch
    model.eval()
    metrics = {"val_loss": 0.0, "per_slot_mse": 0.0, "flux_relative_error": 0.0, "n": 0}
    with torch.no_grad():
        for batch in loader:
            cube = batch["cube"].to(device, non_blocking=True)
            target = batch["galaxy_cubes"].to(device, non_blocking=True)
            valid = batch["galaxy_valid"].to(device, non_blocking=True)
            centers = batch["centers_cyx"].to(device, non_blocking=True)
            if is_binseg:
                masks = model(cube)
                loss = loss_fn(masks, target, valid)
                pred = masks * cube   # (B, M, C, Y, X) flux predictions
                m = _flux_metrics(pred, target, valid)
            elif is_instance:
                logits = model(cube)
                loss = loss_fn(logits, target, valid)
                # Convert argmax label map to flux predictions for metrics.
                label_map = logits.argmax(dim=1, keepdim=True)  # (B, 1, C, Y, X)
                M = target.shape[1]
                pred = torch.stack([
                    (label_map[:, 0] == k).float() * cube[:, 0]
                    for k in range(M)
                ], dim=1)  # (B, M, C, Y, X)
                m = _flux_metrics(pred, target, valid)
            elif is_seg_mask:
                out = model(cube, centers, valid)
                if isinstance(out, tuple):
                    gt_hmap = batch["heatmap"].to(device, non_blocking=True)
                    loss = loss_fn((out[0], out[1], gt_hmap), target, valid)
                    masks = out[0]
                else:
                    masks = out
                    loss = loss_fn(masks, target, valid)
                pred = masks * cube
                m = _flux_metrics(pred, target, valid)
            elif needs_centers:
                pred = model(cube, valid, centers)
                loss = _compute_loss(loss_fn, needs_cube, pred, target, valid, cube)
                m = _flux_metrics(pred, target, valid)
            else:
                pred = model(cube, valid)
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
    ap.add_argument("--n-cubes", type=int, default=None,
                    help="Use only the first N cubes (numeric order). Default: all.")
    ap.add_argument("--out", default="runs/sep", help="Output directory (checkpoints + logs)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels")
    ap.add_argument(
        "--model",
        choices=["baseline", "hungarian", "hungarian_diffuse", "mask", "two_stage", "pos_guided", "seg_mask", "joint", "instance", "binseg"],
        default="baseline",
        help=(
            "baseline: fixed-slot masked MSE (sep_v1).  "
            "hungarian: permutation-invariant matching + per-slot rebalancing.  "
            "hungarian_diffuse: hungarian + extra diffuse output slot + reconstruction term.  "
            "mask: softmax-over-slots mask decoder (slots compete per voxel) + "
            "Hungarian + diffuse slot. Σ pred ≡ input by construction.  "
            "pos_guided: mask model + per-slot Gaussian logit bias from GT centers."
        ),
    )
    ap.add_argument("--cross-weight", type=float, default=0.0,
                    help="Cross-contamination penalty weight (mask model).")
    ap.add_argument("--fg-weight", type=float, default=1.0,
                    help="Foreground BCE loss weight (two_stage model).")
    ap.add_argument("--fg-threshold", type=float, default=0.05,
                    help="Flux fraction threshold for GT foreground mask (two_stage model).")
    ap.add_argument("--center-sigma", type=float, default=1.5,
                    help="Gaussian sigma (voxels) for positional slot bias (pos_guided model).")
    ap.add_argument("--bias-scale", type=float, default=30.0,
                    help="Logit scale for positional Gaussian bias (pos_guided model).")
    ap.add_argument("--center-noise", type=float, default=2.0,
                    help="Std dev of Gaussian noise added to GT centers during training (joint model). "
                         "Bridges gap between GT and predicted centers at inference.")
    ap.add_argument("--det-weight", type=float, default=1.0,
                    help="Weight for detection heatmap loss (joint model).")
    ap.add_argument("--entropy-weight", type=float, default=0.0,
                    help="Per-voxel softmax entropy penalty weight (mask/pos_guided models). "
                         "Encourages hard slot assignment — reduces hallucination.")
    ap.add_argument("--pos-weight", type=float, default=10.0,
                    help="Positive class weight for binary BCE loss (binseg model).")
    ap.add_argument("--patience", type=int, default=100,
                    help="Early stopping patience (epochs without val improvement). 0 = disabled.")
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
        PositionGuidedMaskedSeparationUNet3D,
        TwoStageUNet3D,
        masked_separation_loss,
        hungarian_separation_loss,
        hungarian_separation_loss_with_diffuse,
        two_stage_separation_loss,
        mask_entropy_loss,
        SegMaskUNet3D,
        seg_mask_loss,
        JointDetSegUNet3D,
        joint_det_seg_loss,
        InstanceSegUNet3D,
        instance_seg_loss,
        BinarySegUNet3D,
        binary_seg_loss,
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
    ds = CubeDataset(args.data, max_cubes=args.n_cubes)
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

    needs_centers = False
    is_seg_mask = False
    is_instance = False
    is_binseg = False

    def _make_mask_loss_fn(m, cw, ew):
        """Wrap separation loss + optional entropy penalty for mask-type models."""
        def _loss(p, t, v, c):
            L = hungarian_separation_loss_with_diffuse(p, t, v, c, recon_weight=0.0, cross_weight=cw)
            if ew > 0:
                L = L + ew * mask_entropy_loss(m)
            return L
        return _loss

    if args.model == "binseg":
        is_binseg = True
        needs_cube = False
        _ft = args.fg_threshold
        _pw = args.pos_weight
        model = BinarySegUNet3D(max_n_gals=ds.max_n_gals, base=args.base).to(device)
        loss_fn = lambda masks, t, v: binary_seg_loss(masks, t, v, fg_threshold=_ft, pos_weight=_pw)
        log.info("Model: BinarySegUNet3D(M=%d slots, base=%d, fg_threshold=%.2f, pos_weight=%.1f)",
                 ds.max_n_gals, args.base, _ft, _pw)
    elif args.model == "instance":
        is_instance = True
        needs_cube = False
        _ft = args.fg_threshold
        model = InstanceSegUNet3D(max_n_gals=ds.max_n_gals, base=args.base).to(device)
        loss_fn = lambda logits, t, v: instance_seg_loss(logits, t, v, fg_threshold=_ft)
        log.info("Model: InstanceSegUNet3D(classes=%d galaxies + 1 bg, base=%d, fg_threshold=%.2f)",
                 ds.max_n_gals, args.base, args.fg_threshold)
    elif args.model == "seg_mask":
        is_seg_mask = True
        needs_centers = True
        needs_cube = False
        _ft = args.fg_threshold
        model = SegMaskUNet3D(
            max_n_gals=ds.max_n_gals, base=args.base,
            center_sigma=args.center_sigma,
        ).to(device)
        loss_fn = lambda masks, t, v: seg_mask_loss(masks, t, v, fg_threshold=_ft)
        log.info("Model: SegMaskUNet3D(slots=%d, base=%d, sigma=%.1f, fg_threshold=%.2f)",
                 ds.max_n_gals, args.base, args.center_sigma, args.fg_threshold)
    elif args.model == "joint":
        is_seg_mask = True   # reuse the seg_mask training path
        needs_centers = True
        needs_cube = False
        _ft, _dw = args.fg_threshold, args.det_weight
        model = JointDetSegUNet3D(
            max_n_gals=ds.max_n_gals, base=args.base,
            center_sigma=args.center_sigma, center_noise=args.center_noise,
        ).to(device)
        loss_fn = lambda masks_hmap, t, v: joint_det_seg_loss(
            masks_hmap[0], masks_hmap[1], t, v, masks_hmap[2],
            fg_threshold=_ft, det_weight=_dw,
        )
        log.info("Model: JointDetSegUNet3D(slots=%d, base=%d, sigma=%.1f, "
                 "center_noise=%.1f, det_weight=%.1f)",
                 ds.max_n_gals, args.base, args.center_sigma,
                 args.center_noise, args.det_weight)
    elif args.model == "mask":
        needs_cube = True
        model = MaskedSeparationUNet3D(max_n_gals=ds.max_n_gals, base=args.base).to(device)
        loss_fn = _make_mask_loss_fn(model, args.cross_weight, args.entropy_weight)
        log.info("Model: MaskedSeparationUNet3D(galaxy_slots=%d + 1 diffuse, base=%d, "
                 "entropy_weight=%.2f)", ds.max_n_gals, args.base, args.entropy_weight)
    elif args.model == "pos_guided":
        needs_cube = True
        needs_centers = True
        model = PositionGuidedMaskedSeparationUNet3D(
            max_n_gals=ds.max_n_gals, base=args.base,
            center_sigma=args.center_sigma, bias_scale=args.bias_scale,
        ).to(device)
        loss_fn = _make_mask_loss_fn(model, args.cross_weight, args.entropy_weight)
        log.info("Model: PositionGuidedMaskedSeparationUNet3D(slots=%d + 1 diffuse, "
                 "base=%d, sigma=%.1f, bias_scale=%.1f, entropy_weight=%.2f)",
                 ds.max_n_gals, args.base, args.center_sigma, args.bias_scale, args.entropy_weight)
    elif args.model == "two_stage":
        _fw, _ft, _cw = args.fg_weight, args.fg_threshold, args.cross_weight
        needs_cube = True
        model = TwoStageUNet3D(max_n_gals=ds.max_n_gals, base=args.base).to(device)
        loss_fn = lambda p, t, v, c: two_stage_separation_loss(
            p, t, v, c, model, fg_threshold=_ft, fg_weight=_fw, cross_weight=_cw,
        )
        log.info("Model: TwoStageUNet3D(galaxy_slots=%d + 1 diffuse, base=%d)",
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
    epochs_no_improve = 0
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info("=== epoch %d/%d  lr=%.2e ===", epoch, args.epochs, opt.param_groups[0]["lr"])
        tl = train_one_epoch(model, train_loader, opt, loss_fn, needs_cube, needs_centers,
                             is_seg_mask, is_instance, is_binseg, device, log, epoch, args.log_every)
        vm = validate(model, val_loader, loss_fn, needs_cube, needs_centers,
                      is_seg_mask, is_instance, is_binseg, device)
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_no_improve += 1
        torch.save(model.state_dict(), run_dir / "last.pt")

        log.info(
            "epoch %d  train %.4e  val %.4e  per_slot_mse %.4e  flux_rel_err %.3f  %s  (%.1fs)",
            epoch, tl, vm["val_loss"], vm["per_slot_mse"], vm["flux_relative_error"],
            "[best]" if improved else f"[no_improve={epochs_no_improve}]", dt,
        )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            log.info("Early stopping: no improvement for %d epochs.", args.patience)
            break

    log.info("Training complete. Best val loss: %.4e  (stopped at epoch %d)", best_val, epoch)
    log.info("Checkpoints: %s", [p.name for p in run_dir.glob("*.pt")])


if __name__ == "__main__":
    main()
