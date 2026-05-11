"""Generate a folder of GalCubeCraft HDF5 cubes for training.

Each cube contains 2–5 galaxies (central + 1–4 satellites) by default,
with the diffuse emission (halo + bridges + tails) enabled. The per-galaxy
clean cubes are stored under `/galaxies/cubes` in each HDF5, which is the
source-separation target used by `train_separation.py`.

Example:
    python scripts/generate_cubes.py --out data/train --n 2000 \\
        --min-gals 2 --max-gals 5 --grid-size 72 --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("generate_cubes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, help="Output directory (HDF5 files are written here)")
    ap.add_argument("--n", type=int, default=2000, help="Number of cubes to generate")
    ap.add_argument("--min-gals", type=int, default=2)
    ap.add_argument("--max-gals", type=int, default=5)
    ap.add_argument("--grid-size", type=int, default=96, help="Spatial pixels per side (n_y == n_x)")
    ap.add_argument("--channels", type=int, default=64,
                    help="Number of spectral channels in the output cube (n_ch)")
    ap.add_argument("--resolution", default="resolved",
                    choices=["all", "resolved", "unresolved"],
                    help="Size regime relative to the beam (see GalCubeCraft)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir / "generate_cubes.log")

    # Defer heavy imports until after argparse so --help is fast.
    from GalCubeCraft.core import GalCubeCraft

    cfg = {
        "n_cubes": args.n,
        "n_gals_range": [args.min_gals, args.max_gals],
        "grid_size": args.grid_size,
        "channels": args.channels,
        "resolution": args.resolution,
        "seed": args.seed,
        "out_dir": str(out_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    log.info("Configuration: %s", json.dumps(cfg, indent=2))
    log.info("Instantiating GalCubeCraft — sampling per-cube parameters")
    t0 = time.time()
    g = GalCubeCraft(
        n_gals=(args.min_gals, args.max_gals),
        n_cubes=args.n,
        resolution=args.resolution,
        grid_size=args.grid_size,
        n_spectral_slices=args.channels,
        seed=args.seed,
        save=True,
        fname=str(out_dir),
        verbose=False,
    )
    # Distribution summary of n_gals.
    counts = {int(k): int((g.n_gals == k).sum()) for k in sorted(set(int(x) for x in g.n_gals))}
    log.info("n_gals distribution: %s  (total=%d)", counts, sum(counts.values()))
    log.info("Sampling done in %.1fs. Generating cubes…", time.time() - t0)

    t_gen = time.time()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # GalCubeCraft.generate_cubes() iterates internally; wrap its progress
    # by monkey-patching a small per-cube hook.
    orig_gen = g.generate_cubes
    if tqdm is not None:
        # Wrap: we still call generate_cubes but add a manual progress bar by
        # peeking at the number of entries in `g.results` from a separate
        # thread isn't worth it — simpler: split across chunks of 1 cube.
        # To keep behaviour simple and not touch core, just run once with tqdm
        # simulated via logging every 5%.
        pass

    n_done = 0
    step = max(1, args.n // 20)
    # We can't easily hook into the GalCubeCraft loop, so just call once.
    orig_gen()
    n_done = args.n
    produced = sorted(out_dir.glob("cube_*.h5"))
    log.info("Produced %d HDF5 files in %.1fs (avg %.2fs / cube)",
             len(produced), time.time() - t_gen,
             (time.time() - t_gen) / max(1, len(produced)))

    # Quick sanity check on a random file.
    if produced:
        import h5py
        sample = produced[0]
        with h5py.File(sample, "r") as f:
            log.info("Sample file: %s", sample.name)
            log.info("  /cube shape = %s, dtype = %s", f["cube"].shape, f["cube"].dtype)
            log.info("  /galaxies/cubes shape = %s", f["galaxies/cubes"].shape)
            log.info("  n_gals = %d, n_satellites = %d",
                     int(f.attrs["n_gals"]), int(f.attrs["n_satellites"]))
            log.info("  spatial_res = %.3f kpc/px, spectral_res = %.2f km/s, FOV = %.1f kpc",
                     float(f.attrs["spatial_resolution_kpc_per_px"]),
                     float(f.attrs["spectral_resolution_km_s"]),
                     float(f.attrs["fov_kpc"]))
    log.info("Done. Logs: %s", out_dir / "generate_cubes.log")


if __name__ == "__main__":
    main()
