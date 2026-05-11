"""Run multi-source kinematic separation on a single cube.

Each detected source is returned as its own (n_ch, H, W) cube together
with its centroid trajectory across spectral channels.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from galcubecraft_sourceid.mca_optical_flow import (
    MultiSourceConfig,
    estimate_flow_field,
    separate_sources_kinematic,
)


def load_cube(path: Path):
    with h5py.File(path, "r") as f:
        cube = f["cube"][...].astype(np.float32)
        gals = f["galaxies/cubes"][...].astype(np.float32)
        types = [t.decode() if isinstance(t, bytes) else str(t)
                 for t in f["galaxies/types"][...]]
        positions = f["galaxies/positions_xyz_px"][...]
    return cube, gals, types, positions


def assign_sources_to_truth(sources, gt_gals):
    """Hungarian assignment by mom-0 cosine similarity. Returns the
    permutation that maps recovered source k -> ground-truth gal[perm[k]]
    and the K-by-K cosine matrix.
    """
    from scipy.optimize import linear_sum_assignment
    K = sources.shape[0]
    G = gt_gals.shape[0]
    sM = sources.sum(1).reshape(K, -1)              # (K, H*W)
    gM = gt_gals.sum(1).reshape(G, -1)              # (G, H*W)
    sN = np.linalg.norm(sM, axis=1) + 1e-12
    gN = np.linalg.norm(gM, axis=1) + 1e-12
    cos = (sM @ gM.T) / (sN[:, None] * gN[None, :])
    # Hungarian on -cos (we want to maximize); pad to square if needed.
    n = max(K, G)
    cost = np.zeros((n, n))
    cost[:K, :G] = -cos
    row, col = linear_sum_assignment(cost)
    return row, col, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="experiments/sep_v1/data/cube_1006.h5")
    ap.add_argument("--out", default="experiments/mca_v1")
    ap.add_argument("--K", type=int, default=None,
                    help="Force the number of sources; default = auto-detect.")
    ap.add_argument("--mask-sigma", type=float, default=6.0)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--peak-threshold-rel", type=float, default=0.02)
    ap.add_argument("--min-peak-distance", type=int, default=5)
    ap.add_argument("--centroid-smooth-sigma", type=float, default=1.0)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--flow", choices=["tvl1", "farneback"], default="tvl1")
    args = ap.parse_args()

    cube_path = Path(args.cube)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    cube, gt_gals, gt_types, gt_pos = load_cube(cube_path)
    n_ch, H, W = cube.shape
    print(f"loaded {cube_path.name}: cube {cube.shape}, "
          f"{len(gt_types)} galaxies ({gt_types})")

    t0 = time.time()
    flow = estimate_flow_field(cube, method=args.flow)
    print(f"flow ({args.flow}) in {time.time() - t0:.1f}s "
          f"max|flow|={float(np.abs(flow).max()):.2f} px")

    cfg = MultiSourceConfig(
        K=args.K,
        mask_sigma=args.mask_sigma,
        n_iter=args.n_iter,
        starlet_denoise=not args.no_denoise,
        peak_threshold_rel=args.peak_threshold_rel,
        min_peak_distance=args.min_peak_distance,
        centroid_smooth_sigma=args.centroid_smooth_sigma,
        verbose=True,
    )
    t0 = time.time()
    res = separate_sources_kinematic(cube.astype(np.float64),
                                     flow=flow, config=cfg)
    print(f"separation in {time.time() - t0:.1f}s, K={res.sources.shape[0]}")

    # Per-source flux summary.
    K = res.sources.shape[0]
    for k in range(K):
        s = res.sources[k]
        print(f"  src{k}: flux={s.sum():.3e} "
              f"peak={s.max():.3e} "
              f"track_y=({res.trajectories[:,k,0].min():.1f},"
              f"{res.trajectories[:,k,0].max():.1f}) "
              f"track_x=({res.trajectories[:,k,1].min():.1f},"
              f"{res.trajectories[:,k,1].max():.1f})")

    # Assignment vs ground truth.
    row, col, cos = assign_sources_to_truth(res.sources, gt_gals)
    print("recovered -> truth assignment (Hungarian on moment-0 cosine):")
    G = gt_gals.shape[0]
    for k in range(K):
        gi = col[k]
        c = cos[k, gi] if gi < G else float("nan")
        gt_label = gt_types[gi] if gi < G else "(none)"
        print(f"  src{k} -> gt{gi} ({gt_label}): cos={c:.3f}")

    out_path = out_dir / (cube_path.stem + "_mca_multi.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("cube",         data=cube,                           compression="gzip")
        f.create_dataset("sources",      data=res.sources.astype(np.float32), compression="gzip")
        f.create_dataset("trajectories", data=res.trajectories.astype(np.float32))
        f.create_dataset("flow",         data=res.flow.astype(np.float32),    compression="gzip")
        f.create_dataset("peaks",        data=res.peaks.astype(np.int32))
        f.create_dataset("residual",     data=res.residual.astype(np.float32),compression="gzip")
        f.create_dataset("assignment",   data=col[:K].astype(np.int32))
        f.create_dataset("cosine_matrix",data=cos.astype(np.float32))
        f.attrs["mask_sigma"] = args.mask_sigma
        f.attrs["n_iter"] = args.n_iter
        f.attrs["flow_method"] = args.flow
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
