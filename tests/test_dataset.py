"""End-to-end smoke test: generate HDF5 cubes, load via CubeDataset."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")


def _generate_sample(tmp):
    try:
        from GalCubeCraft.core import GalCubeCraft
    except ImportError:
        pytest.skip("GalCubeCraft is not installed in this environment")
    g = GalCubeCraft(
        n_gals=3, n_cubes=2, grid_size=48, n_spectral_slices=20,
        seed=0, verbose=False, save=True, fname=tmp,
    )
    g.generate_cubes()


def test_dataset_roundtrip():
    from galcubecraft_sourceid.dataset import CubeDataset

    with tempfile.TemporaryDirectory() as tmp:
        _generate_sample(tmp)
        ds = CubeDataset(tmp, heatmap_sigma=1.0)
        assert len(ds) == 2
        item = ds[0]
        assert item["cube"].ndim == 4                 # (1, n_ch, n_y, n_x)
        assert item["heatmap"].shape[0] == 2          # central + satellite channels
        centers = item["centers_cyx"].numpy()
        classes = item["classes"].numpy()
        assert centers.shape[0] == classes.shape[0]
        # At least one central is always present.
        assert (classes == 0).any()
        # Heatmap peak at each GT centre should be ~1.
        hm = item["heatmap"].numpy()
        for (c, y, x), cls in zip(centers, classes):
            ci, yi, xi = int(round(c)), int(round(y)), int(round(x))
            assert hm[cls, ci, yi, xi] > 0.9
        # Metadata is populated.
        m = item["meta"]
        for key in [
            "grid_size", "n_channels", "n_gals", "n_satellites",
            "spatial_resolution_kpc_per_px", "spectral_resolution_km_s",
            "fov_kpc", "diffuse_emission",
        ]:
            assert key in m

        # Per-galaxy source-separation targets.
        gal_cubes = item["galaxy_cubes"].numpy()
        valid = item["galaxy_valid"].numpy()
        M = ds.max_n_gals
        assert gal_cubes.shape == (M, *item["cube"].shape[1:])
        assert valid.shape == (M,)
        assert int(valid.sum()) == m["n_gals"]
        # Padded slots are exactly zero; active slots carry flux.
        assert gal_cubes[int(valid.sum()):].sum() == 0.0
        assert gal_cubes[: int(valid.sum())].sum() > 0.0


def test_inference_decode():
    import numpy as np
    from galcubecraft_sourceid.inference import decode_peaks, match_detections
    hm = np.zeros((2, 20, 32, 32), dtype=np.float32)
    hm[0, 10, 16, 16] = 1.0
    hm[1, 14, 20, 12] = 0.8
    dets = decode_peaks(hm, threshold=0.1)
    assert len(dets) == 2
    gt = np.array([[10, 16, 16], [14, 20, 12]], dtype=float)
    gt_cls = np.array([0, 1])
    matches, unmatched = match_detections(dets, gt, gt_cls, tol=1.5)
    assert len(matches) == 2 and not unmatched
