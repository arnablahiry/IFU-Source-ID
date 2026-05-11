"""PyTorch dataset for GalCubeCraft HDF5 cube files."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import build_heatmap

_TYPE_TO_CLASS = {"central": 0, "satellite": 1}


def _decode_types(arr) -> np.ndarray:
    out = np.empty(len(arr), dtype=np.int64)
    for i, v in enumerate(arr):
        if isinstance(v, bytes):
            v = v.decode()
        out[i] = _TYPE_TO_CLASS[str(v)]
    return out


class CubeDataset(Dataset):
    """Iterate over HDF5 cubes produced by `GalCubeCraft(save=True)`.

    Each item returns a dict with:
      - 'cube':           FloatTensor (1, n_ch, n_y, n_x), observed cube (diffuse included).
      - 'galaxy_cubes':   FloatTensor (max_n_gals, n_ch, n_y, n_x) — clean per-galaxy
                          diffuse-free targets, zero-padded for slots past this
                          cube's n_gals.
      - 'galaxy_valid':   FloatTensor (max_n_gals,) — 1.0 for real galaxies,
                          0.0 for padded slots.
      - 'heatmap':        FloatTensor (n_classes, n_ch, n_y, n_x) — detection target.
      - 'centers_cyx':    (max_n_gals, 3) float, padded with zeros.
      - 'classes':        (max_n_gals,) int64, padded with -1.
      - 'meta':           dict of scalar metadata.
      - 'path':           file stem.
    """

    def __init__(
        self,
        root,
        heatmap_sigma: float = 1.5,
        n_classes: int = 2,
        normalise: str = "per_cube",
        max_n_gals: Optional[int] = None,
    ):
        self.root = Path(root)
        self.items = sorted(self.root.glob("cube_*.h5"))
        if not self.items:
            raise FileNotFoundError(f"No cube_*.h5 files found under {self.root}")
        self.heatmap_sigma = heatmap_sigma
        self.n_classes = n_classes
        self.normalise = normalise

        if max_n_gals is None:
            max_n_gals = 0
            for p in self.items:
                with h5py.File(p, "r") as f:
                    max_n_gals = max(max_n_gals, int(f.attrs["n_gals"]))
        self.max_n_gals = int(max_n_gals)

    def __len__(self):
        return len(self.items)

    def _normalise(self, cube: np.ndarray) -> np.ndarray:
        if self.normalise == "per_cube":
            peak = float(cube.max())
            return cube / peak if peak > 0 else cube
        return cube

    def __getitem__(self, idx: int):
        path = self.items[idx]
        with h5py.File(path, "r") as f:
            cube = f["cube"][:].astype(np.float32)                         # (n_ch, n_y, n_x)
            positions = f["galaxies/positions_xyz_px"][:]                  # (n_gals, 3) (x, y, z)
            channel_idx = f["galaxies/channel_index"][:].astype(np.int64)  # (n_gals,)
            classes = _decode_types(f["galaxies/types"][:])                # (n_gals,)
            per_gal = f["galaxies/cubes"][:].astype(np.float32)            # (n_gals, n_ch, n_y, n_x)
            meta = {
                "grid_size": int(f.attrs["grid_size"]),
                "n_channels": int(f.attrs["n_channels"]),
                "n_gals": int(f.attrs["n_gals"]),
                "n_satellites": int(f.attrs["n_satellites"]),
                "spatial_resolution_kpc_per_px": float(f.attrs["spatial_resolution_kpc_per_px"]),
                "spectral_resolution_km_s": float(f.attrs["spectral_resolution_km_s"]),
                "fov_kpc": float(f.attrs["fov_kpc"]),
                "diffuse_emission": bool(f.attrs["diffuse_emission"]),
            }

        centers_cyx = np.stack([
            channel_idx.astype(np.float32),
            positions[:, 1].astype(np.float32),
            positions[:, 0].astype(np.float32),
        ], axis=1)

        heatmap = build_heatmap(
            cube.shape, centers_cyx, classes,
            sigma=self.heatmap_sigma, n_classes=self.n_classes,
        )

        # Normalise observed cube; apply the same scale to per-galaxy targets so
        # they remain directly comparable to the model input.
        if self.normalise == "per_cube":
            peak = float(cube.max())
            if peak > 0:
                cube = cube / peak
                per_gal = per_gal / peak

        # Pad per-galaxy targets and labels to a fixed slot count.
        M = self.max_n_gals
        n_g = per_gal.shape[0]
        if n_g > M:
            per_gal = per_gal[:M]
            centers_cyx = centers_cyx[:M]
            classes = classes[:M]
            n_g = M
        pad_gal = np.zeros((M, *cube.shape), dtype=np.float32)
        pad_gal[:n_g] = per_gal
        pad_centers = np.zeros((M, 3), dtype=np.float32)
        pad_centers[:n_g] = centers_cyx
        pad_classes = np.full((M,), -1, dtype=np.int64)
        pad_classes[:n_g] = classes
        valid = np.zeros((M,), dtype=np.float32)
        valid[:n_g] = 1.0

        return {
            "cube": torch.from_numpy(cube).unsqueeze(0),
            "galaxy_cubes": torch.from_numpy(pad_gal),
            "galaxy_valid": torch.from_numpy(valid),
            "heatmap": torch.from_numpy(heatmap),
            "centers_cyx": torch.from_numpy(pad_centers),
            "classes": torch.from_numpy(pad_classes),
            "meta": meta,
            "path": path.stem,
        }
