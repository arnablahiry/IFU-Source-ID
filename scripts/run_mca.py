"""Run MCA + optical-flow source separation on a single GalCubeCraft cube."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from galcubecraft_sourceid.mca_optical_flow import (
    MCAConfig,
    estimate_flow_field,
    mca_decompose_cube,
)


def load_cube(path: Path):
    with h5py.File(path, "r") as f:
        cube = f["cube"][...].astype(np.float32)
        gals = f["galaxies/cubes"][...].astype(np.float32)
        types = [t.decode() if isinstance(t, bytes) else str(t)
                 for t in f["galaxies/types"][...]]
        positions = f["galaxies/positions_xyz_px"][...]
    return cube, gals, types, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="experiments/sep_v1/data/cube_1006.h5")
    ap.add_argument("--out", default="experiments/mca_v1")
    ap.add_argument("--n-iter", type=int, default=80)
    ap.add_argument("--n-scales", type=int, default=4)
    ap.add_argument("--lam-point", type=float, default=4.0,
                    help="Pixel-domain threshold (units of noise sigma).")
    ap.add_argument("--lam-diffuse", type=float, default=4.0,
                    help="Starlet-band threshold (units of band sigma).")
    ap.add_argument("--mu-kin", type=float, default=0.1)
    ap.add_argument("--flow", choices=["tvl1", "farneback"], default="tvl1")
    args = ap.parse_args()

    cube_path = Path(args.cube)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube, gt_gals, gt_types, gt_pos = load_cube(cube_path)
    n_ch, H, W = cube.shape
    print(f"loaded {cube_path.name}: cube {cube.shape}, "
          f"{len(gt_types)} galaxies ({gt_types})")

    # Robust noise estimate via MAD on the finest starlet detail band of
    # the brightest channel. Synthetic cubes can be effectively noise-
    # free; in that case fall back to a fraction of the peak so the
    # threshold has any meaning at all.
    from galcubecraft_sourceid.mca_optical_flow import starlet_forward
    chan_energy = (cube ** 2).reshape(n_ch, -1).sum(axis=1)
    n_ref = int(np.argmax(chan_energy))
    w_ref = starlet_forward(cube[n_ref], n_scales=1)[0]
    sigma = float(1.4826 * np.median(np.abs(w_ref - np.median(w_ref))))
    peak = float(np.abs(cube).max())
    floor = 1e-3 * peak
    if sigma < floor:
        print(f"sigma {sigma:.2e} below {floor:.2e}; using signal-based floor")
        sigma = floor
    print(f"using sigma = {sigma:.4g} (reference channel {n_ref}, peak {peak:.3g})")

    t0 = time.time()
    flow = estimate_flow_field(cube, method=args.flow)
    print(f"flow ({args.flow}) computed in {time.time() - t0:.1f}s, "
          f"shape={flow.shape}, median|flow|={float(np.median(np.abs(flow))):.3f} px")

    cfg = MCAConfig(
        n_scales=args.n_scales,
        n_iter=args.n_iter,
        lam_point=args.lam_point * sigma,
        lam_diffuse=args.lam_diffuse * sigma,
        mu_kin=args.mu_kin,
        verbose=True,
    )
    t0 = time.time()
    res = mca_decompose_cube(cube.astype(np.float64), flow=flow, config=cfg)
    print(f"MCA finished in {time.time() - t0:.1f}s")

    # Quality summaries.
    total_in   = float(cube.sum())
    flux_point = float(res.point.sum())
    flux_diff  = float(res.diffuse.sum())
    flux_res   = float(res.residual.sum())
    rms_res    = float(np.sqrt(np.mean(res.residual ** 2)))
    print(f"input flux  = {total_in:.3e}")
    print(f"point flux  = {flux_point:.3e}  ({flux_point / total_in:.2%})")
    print(f"diffuse flux= {flux_diff:.3e}  ({flux_diff / total_in:.2%})")
    print(f"residual    : sum={flux_res:.3e}  rms={rms_res:.3e} "
          f"({rms_res / sigma:.2f} sigma)")

    # Compare reconstructed (point + diffuse) cube to the union of ground-
    # truth galaxy cubes — diffuse-included observation minus continuum.
    gt_total = gt_gals.sum(axis=0)        # (n_ch, H, W)
    recon = res.point + res.diffuse
    num = float((gt_total * recon).sum())
    den = float(np.linalg.norm(gt_total) * np.linalg.norm(recon) + 1e-12)
    cosine = num / den
    print(f"cos(reconstruction, sum-of-galaxies) = {cosine:.4f}")

    out_path = out_dir / (cube_path.stem + "_mca.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("cube",     data=cube,         compression="gzip")
        f.create_dataset("point",    data=res.point.astype(np.float32),    compression="gzip")
        f.create_dataset("diffuse",  data=res.diffuse.astype(np.float32),  compression="gzip")
        f.create_dataset("residual", data=res.residual.astype(np.float32), compression="gzip")
        f.create_dataset("flow",     data=res.flow.astype(np.float32),     compression="gzip")
        f.attrs["sigma"] = sigma
        f.attrs["lam_point_sigma"] = args.lam_point
        f.attrs["lam_diffuse_sigma"] = args.lam_diffuse
        f.attrs["mu_kin"] = args.mu_kin
        f.attrs["n_iter"] = args.n_iter
        f.attrs["flow_method"] = args.flow
        f.attrs["cosine_vs_truth"] = cosine
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
