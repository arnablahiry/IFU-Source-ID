"""Training loop for 3D heatmap regression."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import CubeDataset
from .models import UNet3D, focal_mse_loss


def train(
    data_root,
    out_dir: str = "runs/default",
    epochs: int = 30,
    batch_size: int = 2,
    lr: float = 1e-3,
    heatmap_sigma: float = 1.5,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    device: str | None = None,
    base_channels: int = 16,
):
    """End-to-end training on a directory of `GalCubeCraft` cubes + catalogs."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)

    ds = CubeDataset(data_root, heatmap_sigma=heatmap_sigma)
    n_val = max(1, int(val_fraction * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = UNet3D(in_channels=1, out_channels=2, base=base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            cube = batch["cube"].to(device, non_blocking=True)       # (B, 1, C, Y, X)
            target = batch["heatmap"].to(device, non_blocking=True)   # (B, K, C, Y, X)
            pred = model(cube)
            loss = focal_mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * cube.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cube = batch["cube"].to(device, non_blocking=True)
                target = batch["heatmap"].to(device, non_blocking=True)
                pred = model(cube)
                val_loss += focal_mse_loss(pred, target).item() * cube.size(0)
        val_loss /= len(val_ds)
        sched.step()

        print(f"epoch {epoch:3d} | train {train_loss:.4e} | val {val_loss:.4e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir_p / "best.pt")
    torch.save(model.state_dict(), out_dir_p / "last.pt")
    return out_dir_p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", help="Directory with cube_*.npy and cube_*_meta.pkl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/default")
    args = ap.parse_args()
    train(args.data_root, out_dir=args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr)
