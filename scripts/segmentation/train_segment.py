"""Train an `EmbeddingUNet3D` (with optional CoordConv) for per-voxel
instance segmentation, with per-epoch eval+analyse pipeline.

Each epoch saves `best.pt`, `last.pt`, appends a row to `metrics.jsonl`, and
optionally fires off `evaluate_segment.py` + `analyse_segment.py` as a
subprocess writing to `<out>/eval/epoch_N/` and `<out>/analysis/epoch_N/`.

The eval subprocess runs on CPU so it doesn't contend with the main MPS/CUDA
training loop. Per-epoch eval is launched in the background — if it's slower
than the next epoch, it just keeps running while training continues.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from galcubecraft_sourceid import (
    CubeDataset,
    EmbeddingUNet3D,
    add_coord_channels,
    discriminative_loss,
    voxel_instance_labels,
)


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_segment")
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


def _build_labels(galaxy_cubes: torch.Tensor, valid: torch.Tensor, threshold_frac: float) -> torch.Tensor:
    B = galaxy_cubes.shape[0]
    labs = []
    for b in range(B):
        labs.append(voxel_instance_labels(galaxy_cubes[b], valid[b], threshold_frac))
    return torch.stack(labs, dim=0)


def augment_batch(cube: torch.Tensor, galaxy_cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random Y/X flips + cyclic channel rolls.

    NB: applying augmentation when CoordConv is enabled scrambles the
    (content, coord) joint distribution and neutralises the position cue —
    use `--no-augment` for CoordConv runs unless coords are also flipped.
    """
    B = cube.shape[0]
    n_ch = cube.shape[2]
    for b in range(B):
        if torch.rand(1).item() < 0.5:
            cube[b] = cube[b].flip(-1)
            galaxy_cubes[b] = galaxy_cubes[b].flip(-1)
        if torch.rand(1).item() < 0.5:
            cube[b] = cube[b].flip(-2)
            galaxy_cubes[b] = galaxy_cubes[b].flip(-2)
        shift = int(torch.randint(0, n_ch, (1,)).item())
        if shift > 0:
            cube[b] = cube[b].roll(shift, dims=-3)
            galaxy_cubes[b] = galaxy_cubes[b].roll(shift, dims=-3)
    return cube, galaxy_cubes


def _segmentation_metrics(emb: torch.Tensor, labels: torch.Tensor) -> dict:
    B, D = emb.shape[:2]
    spreads = []
    seps = []
    for b in range(B):
        e = emb[b].reshape(D, -1)
        lab = labels[b].reshape(-1)
        uniq = torch.unique(lab)
        uniq = uniq[uniq >= 0]
        if uniq.numel() == 0:
            continue
        means = []
        for k in uniq.tolist():
            mask = (lab == k)
            v = e[:, mask]
            mu = v.mean(dim=1)
            means.append(mu)
            spreads.append(float((v - mu[:, None]).norm(dim=0).mean().detach().cpu()))
        if len(means) > 1:
            mt = torch.stack(means, dim=0)
            pair = (mt[:, None] - mt[None, :]).norm(dim=2)
            mask = ~torch.eye(len(means), dtype=torch.bool, device=pair.device)
            seps.append(float(pair[mask].min().detach().cpu()))
    return {
        "intra_cluster_spread": float(sum(spreads) / max(len(spreads), 1)),
        "inter_cluster_min_sep": float(sum(seps) / max(len(seps), 1)) if seps else float("nan"),
    }


def train_one_epoch(model, loader, opt, device, log, epoch, log_every, threshold_frac,
                    loss_kwargs, augment, use_coords):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        cube = batch["cube"].to(device, non_blocking=True)
        gc = batch["galaxy_cubes"].to(device, non_blocking=True)
        valid = batch["galaxy_valid"].to(device, non_blocking=True)

        if augment:
            cube, gc = augment_batch(cube, gc)

        cube_input = add_coord_channels(cube) if use_coords else cube
        labels = _build_labels(gc, valid, threshold_frac).to(device, non_blocking=True)
        emb = model(cube_input)
        loss = discriminative_loss(emb, labels, **loss_kwargs)
        opt.zero_grad(); loss.backward(); opt.step()

        total += float(loss.detach().cpu()) * cube.size(0)
        n += cube.size(0)

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            log.info("  epoch %d  step %4d/%d  loss %.4e  (%.1f samples/s)",
                     epoch, i + 1, len(loader), float(loss.detach().cpu()),
                     n / max(elapsed, 1e-6))
    return total / max(n, 1)


def validate(model, loader, device, threshold_frac, loss_kwargs, use_coords):
    model.eval()
    metrics = {"val_loss": 0.0, "intra_cluster_spread": 0.0, "inter_cluster_min_sep": 0.0, "n": 0}
    n_sep = 0
    with torch.no_grad():
        for batch in loader:
            cube = batch["cube"].to(device, non_blocking=True)
            gc = batch["galaxy_cubes"].to(device, non_blocking=True)
            valid = batch["galaxy_valid"].to(device, non_blocking=True)

            cube_input = add_coord_channels(cube) if use_coords else cube
            labels = _build_labels(gc, valid, threshold_frac).to(device, non_blocking=True)
            emb = model(cube_input)
            loss = discriminative_loss(emb, labels, **loss_kwargs)
            m = _segmentation_metrics(emb, labels)

            bs = cube.size(0)
            metrics["val_loss"] += float(loss.detach().cpu()) * bs
            metrics["intra_cluster_spread"] += m["intra_cluster_spread"] * bs
            if not math.isnan(m["inter_cluster_min_sep"]):
                metrics["inter_cluster_min_sep"] += m["inter_cluster_min_sep"] * bs
                n_sep += bs
            metrics["n"] += bs
    n = max(metrics.pop("n"), 1)
    return {"val_loss": metrics["val_loss"] / n,
            "intra_cluster_spread": metrics["intra_cluster_spread"] / n,
            "inter_cluster_min_sep": metrics["inter_cluster_min_sep"] / max(n_sep, 1)}


def _spawn_per_epoch_eval(args, run_dir: Path, epoch: int, log, eval_proc):
    """Fire-and-forget eval+analyse for the just-saved last.pt.

    If the previous epoch's eval is still running, wait for it before kicking
    off a new one (single eval at a time, no pile-up). Returns the new process.
    """
    if eval_proc is not None and eval_proc.poll() is None:
        log.info("  waiting for previous eval to finish before starting epoch %d eval...", epoch)
        eval_proc.wait()
    eval_out = run_dir / "eval" / f"epoch_{epoch}"
    analyse_out = run_dir / "analysis" / f"epoch_{epoch}"
    eval_out.mkdir(parents=True, exist_ok=True)
    analyse_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-c",
        "import subprocess, sys; "
        f"r = subprocess.run([sys.executable, 'scripts/evaluate_segment.py', "
        f"'--data', '{args.data}', '--run', '{run_dir}', "
        f"'--out', '{eval_out}', '--checkpoint', 'last.pt', "
        f"'--split', 'test', '--seeds', 'meanshift', "
        f"'--bandwidth', '{args.eval_bandwidth}', "
        f"'--min-cluster-flux-frac', '{args.eval_min_cluster_flux_frac}', "
        f"'--min-cluster-voxels', '{args.eval_min_cluster_voxels}', "
        f"'--n-samples', '{args.eval_n_samples}', '--device', 'cpu']); "
        f"sys.exit(r.returncode) if r.returncode else "
        f"subprocess.run([sys.executable, 'scripts/analyse_segment.py', "
        f"'--run', '{run_dir}', '--eval-dir', '{eval_out}', '--out', '{analyse_out}'])"
    ]
    log_path = run_dir / "eval" / f"epoch_{epoch}.log"
    f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    log.info("  spawned per-epoch eval+analyse → %s (pid %d)", eval_out.relative_to(run_dir), proc.pid)
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/segment")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--threshold-frac", type=float, default=0.1)
    ap.add_argument("--coord-conv", action="store_true",
                    help="Add normalised (z,y,x) channels to the input — sources separable by position")
    ap.add_argument("--no-coord-conv", dest="coord_conv", action="store_false")
    ap.set_defaults(coord_conv=True)
    ap.add_argument("--augment", action="store_true",
                    help="Random Y/X flips + cyclic channel rolls. Off by default — incompatible with --coord-conv.")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.set_defaults(augment=False)
    ap.add_argument("--delta-v", type=float, default=0.5)
    ap.add_argument("--delta-d", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    # Per-epoch eval+analyse pipeline knobs.
    ap.add_argument("--eval-every", type=int, default=1,
                    help="Run eval+analyse every N epochs. 0 disables per-epoch eval.")
    ap.add_argument("--eval-bandwidth", type=float, default=1.5)
    ap.add_argument("--eval-min-cluster-flux-frac", type=float, default=0.05)
    ap.add_argument("--eval-min-cluster-voxels", type=int, default=50)
    ap.add_argument("--eval-n-samples", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.info("Run dir: %s", run_dir)
    log.info("Config: %s", json.dumps(vars(args), indent=2))

    if args.augment and args.coord_conv:
        log.warning("--augment + --coord-conv: augmentation will scramble the (content, coord) "
                    "pairing unless coords are flipped together. Consider --no-augment.")

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

    ds = CubeDataset(args.data)
    log.info("Loaded %d cubes  max_n_gals=%d", len(ds), ds.max_n_gals)
    n_test = max(1, int(args.test_fraction * len(ds)))
    n_val = max(1, int(args.val_fraction * len(ds)))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    splits = {"train": [int(i) for i in train_ds.indices],
              "val":   [int(i) for i in val_ds.indices],
              "test":  [int(i) for i in test_ds.indices]}
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    log.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    in_channels = 4 if args.coord_conv else 1
    model = EmbeddingUNet3D(in_channels=in_channels, embedding_dim=args.embedding_dim,
                            base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: EmbeddingUNet3D(in_ch=%d, D=%d, base=%d)  | %d params",
             in_channels, args.embedding_dim, args.base, n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_kwargs = dict(delta_v=args.delta_v, delta_d=args.delta_d,
                       alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    best_val = math.inf
    eval_proc = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info("=== epoch %d/%d  lr=%.2e ===", epoch, args.epochs, opt.param_groups[0]["lr"])
        tl = train_one_epoch(model, train_loader, opt, device, log, epoch, args.log_every,
                             args.threshold_frac, loss_kwargs, args.augment, args.coord_conv)
        vm = validate(model, val_loader, device, args.threshold_frac, loss_kwargs, args.coord_conv)
        sched.step()
        dt = time.time() - t0

        row = {"epoch": epoch, "train_loss": tl, **vm,
               "elapsed_sec": dt, "lr": opt.param_groups[0]["lr"]}
        (run_dir / "metrics.jsonl").open("a").write(json.dumps(row) + "\n")

        improved = vm["val_loss"] < best_val
        if improved:
            best_val = vm["val_loss"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save(model.state_dict(), run_dir / "last.pt")
        log.info("epoch %d  train %.4e  val %.4e  intra %.3f  inter %.3f  %s (%.1fs)",
                 epoch, tl, vm["val_loss"], vm["intra_cluster_spread"],
                 vm["inter_cluster_min_sep"], "[best]" if improved else "      ", dt)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_proc = _spawn_per_epoch_eval(args, run_dir, epoch, log, eval_proc)

    if eval_proc is not None and eval_proc.poll() is None:
        log.info("Waiting for final eval to finish...")
        eval_proc.wait()
    log.info("Training complete. Best val loss: %.4e", best_val)


if __name__ == "__main__":
    main()
