"""Create 80:10:10 train/val/test splits from all_cubes and save datasets as pkl files."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import random_split

from galcubecraft_sourceid import CubeDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/all_cubes")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading CubeDataset from {args.data} ...")
    ds = CubeDataset(args.data)
    print(f"  {len(ds)} cubes  max_n_gals={ds.max_n_gals}")

    n_test = max(1, int(args.test_fraction * len(ds)))
    n_val = max(1, int(args.val_fraction * len(ds)))
    n_train = len(ds) - n_val - n_test
    print(f"  Split: train={n_train}  val={n_val}  test={n_test}")

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    for name, subset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        path = out / f"{name}_dataset.pkl"
        with open(path, "wb") as f:
            pickle.dump(subset, f)
        print(f"  Saved {path}  ({len(subset)} samples)")

    print("Done.")


if __name__ == "__main__":
    main()
