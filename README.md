# GalCubeCraft-SourceID

3D source identification on synthetic IFU spectral cubes produced by
[`GalCubeCraft`](../GalCubeCraft). Given a cube `(n_vel, n_y, n_x)` that
contains a central galaxy, some satellites, and diffuse emission
(halo + bridges + tidal tails), recover each galaxy as a **3D point**
`(channel, y, x)` with a class label (`central` / `satellite`), separating
it from the diffuse background.

## Problem

Each training example is a single HDF5 file `cube_<i>.h5` produced by
`GalCubeCraft(save=True)`. Layout:

```
/cube                           float32  (n_ch, n_y, n_x)   — beam-convolved spectral cube
/channel_velocities_km_s        float64  (n_ch,)            — channel centres
/galaxies/positions_xyz_px      int      (n_gals, 3)        — (x, y, z) in cube pixels
/galaxies/types                 str      (n_gals,)          — 'central' / 'satellite'
/galaxies/Re_px, Se, hz_px,     float    (n_gals,)          — disk parameters
    sersic_n, inclination_x_deg,
    inclination_y_deg,
    sigma_vz_km_s, v0_km_s,
    systemic_vel_km_s,
    channel_index,               int     (n_gals,)           — nearest spectral channel
    distance_to_central_px,      float   (n_gals,)
    distance_to_central_kpc      float   (n_gals,)
/galaxies/cubes                 float32  (n_gals, n_ch, n_y, n_x)
                                          — clean, diffuse-free per-galaxy spectral cubes
                                            in the final FOV (beam- and spectral-smoothed
                                            to match `/cube`).
/beam                           group    attrs: bmin_px, bmaj_px, bpa_deg
/diffuse_params                 group    attrs: one per halo/bridge/tail knob

File-level attrs: grid_size, n_channels, n_gals, n_satellites,
spatial_resolution_kpc_per_px, fov_kpc, spectral_resolution_km_s,
diffuse_emission.
```

For each galaxy the ground-truth 3D coordinate in **cube space** is taken
directly from the file:

```
ch_i = galaxies/channel_index[i]
y_i  = galaxies/positions_xyz_px[i, 1]
x_i  = galaxies/positions_xyz_px[i, 0]
class_i ∈ {central, satellite}  (from galaxies/types[i])
```

The diffuse emission (halo/bridges/tails) is *not* a target — the model must
learn to ignore it as background. Ground-truth is only the discrete galaxy
coordinates.

## Approach

Two complementary heads over a shared 3D U-Net backbone (use either or
both, depending on whether you want detection, source separation, or a
joint model):

1. **Source-separation head** (`SeparationUNet3D`): outputs `max_n_gals`
   non-negative cubes of shape `(n_ch, n_y, n_x)` — one clean cube per
   galaxy slot. Trained directly against `/galaxies/cubes` from the HDF5
   file with a masked MSE (`masked_separation_loss`) that ignores padded
   slots. This is the primary target for *"separate the observed cube
   into per-galaxy contributions"*.
2. **Center-heatmap head** (`UNet3D` with 2 channels: `central`,
   `satellite`): 3D Gaussian peak targets, focal-MSE loss. Useful as an
   auxiliary task and for downstream peak decoding via
   `decode_peaks` (skimage `peak_local_max` + class-aware NMS).

At inference, a separation model returns `max_n_gals` clean cubes — sum
of which should reconstruct the galaxy-only (diffuse-free) image; the
residual between the input cube and the sum is effectively the learned
diffuse field.

## Install

```bash
pip install -e .
```

Requires PyTorch 2+, NumPy, h5py, scikit-image (for peak_local_max).

## End-to-end quickstart

The simplest path is `pipeline.py`, which runs **generate → train →
evaluate → analyse** in sequence and skips stages whose outputs already
exist:

```bash
python scripts/pipeline.py --root experiments/sep_v1 \
    --n-cubes 2000 --epochs 30 --batch-size 2 --base 16
```

This produces an experiment directory with everything needed:

```
experiments/sep_v1/
├── pipeline.log
├── data/
│   ├── cube_*.h5
│   ├── config.json
│   └── generate_cubes.log
└── train/
    ├── train.log
    ├── metrics.jsonl
    ├── config.json
    ├── best.pt / last.pt
    ├── eval/
    │   ├── eval.log
    │   ├── per_cube_metrics.csv
    │   ├── summary.json
    │   └── samples/<stem>.npz
    └── analysis/
        ├── analyse.log
        ├── loss_curves.png
        ├── eval_distributions.png
        ├── sample_<stem>.png
        └── report.md
```

If you want to run the stages individually:

1. **Generate training data** (2000 cubes, 2–5 galaxies each, default
   `(64, 96, 96)` cubes):
   ```bash
   python scripts/generate_cubes.py --out experiments/sep_v1/data --n 2000 \
       --min-gals 2 --max-gals 5 --grid-size 96 --channels 64 --seed 0
   ```

2. **Train source separation** with rich per-epoch logging:
   ```bash
   python scripts/train_separation.py --data experiments/sep_v1/data \
       --out experiments/sep_v1/train --epochs 30 --batch-size 2 --base 16
   ```

3. **Evaluate** the best-val checkpoint on the held-out test split (the
   training script splits the data three ways — train / val / test —
   and writes the index lists to `<run>/splits.json`; `--val-fraction`
   and `--test-fraction` default to 0.1 each, so 80% train / 10% val /
   10% test). Use `--split val` or `--split train` to evaluate on those
   instead.
   ```bash
   python scripts/evaluate.py --data experiments/sep_v1/data \
       --run experiments/sep_v1/train --split test
   ```

4. **Analyse** the run (plots + markdown report):
   ```bash
   python scripts/analyse.py --run experiments/sep_v1/train
   ```

## Usage

Source-separation training (primary task):

```python
import torch
from torch.utils.data import DataLoader
from galcubecraft_sourceid import CubeDataset, SeparationUNet3D, masked_separation_loss

ds = CubeDataset('/path/to/cubes')
dl = DataLoader(ds, batch_size=2, shuffle=True)
model = SeparationUNet3D(max_n_gals=ds.max_n_gals, base=16).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for batch in dl:
    cube = batch['cube'].cuda()                  # (B, 1, C, Y, X)
    target = batch['galaxy_cubes'].cuda()        # (B, M, C, Y, X)
    valid = batch['galaxy_valid'].cuda()         # (B, M)
    pred = model(cube)
    loss = masked_separation_loss(pred, target, valid)
    opt.zero_grad(); loss.backward(); opt.step()
```

Detection (heatmap) training:

```python
from galcubecraft_sourceid.train import train
train('/path/to/cubes', epochs=30, batch_size=2)
```

## Data layout

Point `CubeDataset` at the directory produced by `GalCubeCraft` with
`save=True`. The loader globs `cube_*.h5` and reads everything it needs
from each HDF5 file.

## Repository layout

```
src/galcubecraft_sourceid/
    __init__.py
    dataset.py     # Cube + catalog loader, heatmap target builder
    targets.py     # Heatmap / segmentation target helpers
    models.py      # 3D U-Net
    train.py       # Training loop
    inference.py   # Peak extraction + NMS for evaluation
    metrics.py     # Detection metrics (precision/recall, distance error)
tests/
    test_dataset.py
```
