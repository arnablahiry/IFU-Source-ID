"""Train the Stage-2 single-source extractor for the detect-then-extract pipeline.

Stage 1 of the pipeline is the existing `UNet3D` heatmap detector (trained via
`scripts/train.py`) plus `decode_peaks` for NMS. Stage 2 — this script — trains
an `ExtractorUNet3D` that, given the raw cube and a position prior, isolates
the single source at that position. No slot indexing, no Hungarian, no slot
collapse: each source is generated independently from a 3D Gaussian "look here"
hint, so the model never has to decide "which slot is which."

Training expansion:
    Each cube of N galaxies generates N training samples. Sample i feeds the
    raw cube concatenated with a Gaussian centered on galaxy i's catalog
    position (jittered by `--center-jitter` voxels at train time so the
    extractor is robust to imperfect Stage-1 detections at inference).
    Target is galaxy i's clean per-galaxy cube.

Loss is plain MSE, optionally rebalanced by 1/sqrt(target_RMS) (the same
Fix-3 rebalance used in the Hungarian losses) so faint satellites get
comparable gradient to bright centrals.

Example:
    python scripts/train_extract.py \\
        --data experiments/sep_v1/data \\
        --out experiments/extract_v1/train \\
        --epochs 30 --batch-size 4 --base 16 --query-sigma 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from galcubecraft_sourceid import (
    CubeDataset,
    ExtractorUNet3D,
    position_query_volume,
)


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_extract")
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


class PerGalaxyDataset(Dataset):
    """Wraps `CubeDataset` to emit one example per galaxy.

    Each item is built lazily by reloading the underlying cube. We pre-scan
    n_gals from each HDF5 once at init so __len__ is exact without holding
    the data in memory.
    """

    def __init__(self, cube_ds: CubeDataset, query_sigma: float, center_jitter: float):
        self.cube_ds = cube_ds
        self.query_sigma = float(query_sigma)
        self.center_jitter = float(center_jitter)
        self.index: list[tuple[int, int]] = []
        for ci in range(len(cube_ds)):
            with h5py.File(cube_ds.items[ci], "r") as f:
                n = int(f.attrs["n_gals"])
            for gi in range(n):
                self.index.append((ci, gi))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        ci, gi = self.index[k]
        item = self.cube_ds[ci]
        cube = item["cube"]                              # (1, C, Y, X)
        target = item["galaxy_cubes"][gi].unsqueeze(0)   # (1, C, Y, X)
        center = item["centers_cyx"][gi].clone()         # (3,)
        if self.center_jitter > 0.0:
            center = center + torch.randn(3) * self.center_jitter
        q = position_query_volume(center, cube.shape[1:], self.query_sigma)  # (1, C, Y, X)
        x = torch.cat([cube, q], dim=0)                  # (2, C, Y, X)
        return {
            "input": x,
            "target": target,
            "center": center,
            "cube_idx": ci,
            "galaxy_idx": gi,
            "path": item["path"],
        }


def extract_loss(pred: torch.Tensor, target: torch.Tensor, rebalance: bool = True) -> torch.Tensor:
    """Per-sample MSE with optional 1/sqrt(target_RMS) rebalance (Fix 3)."""
    pair = ((pred - target) ** 2).mean(dim=(1, 2, 3, 4))             # (B,)
    if rebalance:
        scale = (target ** 2).mean(dim=(1, 2, 3, 4)).sqrt().detach().clamp(min=1e-2)
        pair = pair / scale
    return pair.mean()


def _flux_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-source diagnostics, averaged over the batch."""
    mse = ((pred - target) ** 2).mean()
    tgt_sum = target.sum(dim=(1, 2, 3, 4))
    prd_sum = pred.sum(dim=(1, 2, 3, 4))
    flux_rel = ((prd_sum - tgt_sum).abs()).sum() / (tgt_sum.abs().sum() + 1e-8)
    return {"mse": float(mse.detach().cpu()), "flux_relative_error": float(flux_rel.detach().cpu())}


def train_one_epoch(model, loader, opt, device, log, epoch, log_every, rebalance):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(x)
        loss = extract_loss(pred, target, rebalance=rebalance)
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            log.info("  epoch %d  step %4d/%d  loss %.4e  (%.1f samples/s)",
                     epoch, i + 1, len(loader), float(loss.detach().cpu()),
                     n / max(elapsed, 1e-6))
    return total / max(n, 1)


def validate(model, loader, device, rebalance):
    model.eval()
    metrics = {"val_loss": 0.0, "mse": 0.0, "flux_relative_error": 0.0, "n": 0}
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(x)
            loss = extract_loss(pred, target, rebalance=rebalance)
            m = _flux_metrics(pred, target)
            bs = x.size(0)
            metrics["val_loss"] += float(loss.detach().cpu()) * bs
            metrics["mse"] += m["mse"] * bs
            metrics["flux_relative_error"] += m["flux_relative_error"] * bs
            metrics["n"] += bs
    n = max(metrics.pop("n"), 1)
    return {k: v / n for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Directory with cube_*.h5")
    ap.add_argument("--out", default="runs/extract", help="Output directory (checkpoints + logs)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels")
    ap.add_argument("--query-sigma", type=float, default=2.0,
                    help="Gaussian sigma (voxels) of the position-prior channel")
    ap.add_argument("--center-jitter", type=float, default=0.5,
                    help="Stddev (voxels) of training-time noise added to the query center, "
                         "so the extractor is robust to imperfect Stage-1 detections")
    ap.add_argument("--no-rebalance", action="store_true",
                    help="Disable 1/sqrt(target_RMS) loss rebalance (Fix 3)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.info("Run directory: %s", run_dir)
    log.info("Config: %s", json.dumps(vars(args), indent=2))

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
    cube_ds = CubeDataset(args.data)
    log.info("Loaded %d cubes.  max_n_gals=%d", len(cube_ds), cube_ds.max_n_gals)

    # Split at the cube level (not per galaxy), so all galaxies from one cube
    # land in the same split — prevents leakage between train/val/test.
    n_cubes = len(cube_ds)
    n_test = max(1, int(args.test_fraction * n_cubes))
    n_val = max(1, int(args.val_fraction * n_cubes))
    n_train = n_cubes - n_val - n_test
    if n_train <= 0:
        log.error("val_fraction + test_fraction leaves no training data.")
        sys.exit(1)
    train_cubes, val_cubes, test_cubes = random_split(
        range(n_cubes), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    splits = {
        "train": [int(i) for i in train_cubes.indices],
        "val":   [int(i) for i in val_cubes.indices],
        "test":  [int(i) for i in test_cubes.indices],
    }
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    log.info("Split (cubes): train=%d  val=%d  test=%d", n_train, n_val, n_test)

    train_subset_ds = CubeDataset(args.data); train_subset_ds.items = [cube_ds.items[i] for i in splits["train"]]
    val_subset_ds   = CubeDataset(args.data); val_subset_ds.items   = [cube_ds.items[i] for i in splits["val"]]

    train_ds = PerGalaxyDataset(train_subset_ds, args.query_sigma, args.center_jitter)
    val_ds   = PerGalaxyDataset(val_subset_ds,   args.query_sigma, center_jitter=0.0)
    log.info("Per-galaxy expansion: train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = ExtractorUNet3D(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: ExtractorUNet3D(base=%d)  | %d trainable params", args.base, n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rebalance = not args.no_rebalance
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info("=== epoch %d/%d  lr=%.2e ===", epoch, args.epochs, opt.param_groups[0]["lr"])
        tl = train_one_epoch(model, train_loader, opt, device, log, epoch, args.log_every, rebalance)
        vm = validate(model, val_loader, device, rebalance)
        sched.step()
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": tl,
            "val_loss": vm["val_loss"],
            "val_mse": vm["mse"],
            "val_flux_relative_error": vm["flux_relative_error"],
            "elapsed_sec": dt,
            "lr": opt.param_groups[0]["lr"],
        }
        (run_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")
        improved = vm["val_loss"] < best_val
        if improved:
            best_val = vm["val_loss"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save(model.state_dict(), run_dir / "last.pt")
        log.info(
            "epoch %d  train %.4e  val %.4e  mse %.4e  flux_rel_err %.3f  %s  (%.1fs)",
            epoch, tl, vm["val_loss"], vm["mse"], vm["flux_relative_error"],
            "[best]" if improved else "      ", dt,
        )

    log.info("Training complete. Best val loss: %.4e", best_val)


if __name__ == "__main__":
    main()
