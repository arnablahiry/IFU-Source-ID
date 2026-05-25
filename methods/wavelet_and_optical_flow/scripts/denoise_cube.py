#!/usr/bin/env python
"""Denoise any FITS spectral cube using pycs Denoiser2D1D IST.

The denoised data replaces the primary HDU array and is saved as a new FITS
file alongside the input (suffix _denoised_ist.fits).

Run with the cosmostat conda env:
    /Users/arnablahiry/miniforge3/envs/cosmostat/bin/python \
        scripts/wavelet/denoise_cube.py <cube.fits> [options]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, '/Users/arnablahiry/repos/cosmostat')
from pycs.sparsity.sparse3d import Denoiser2D1D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('cube',             help='Input FITS file (any shape squeezable to n_ch×H×W)')
    ap.add_argument('--out',            default=None,
                    help='Output FITS path (default: <input>_denoised_ist.fits)')
    ap.add_argument('--threshold',      type=float, default=5.0,
                    help='Detection threshold in σ (default: 5)')
    ap.add_argument('--thresh-increm',  type=float, default=2.0,
                    help='Extra σ added to finest scale (default: 2)')
    ap.add_argument('--num-iter',       type=int,   default=20,
                    help='IST iterations for reweight and debias (default: 20)')
    ap.add_argument('--patience',       type=int,   default=3,
                    help='Convergence patience (default: 3)')
    args = ap.parse_args()

    in_path  = Path(args.cube)
    out_path = Path(args.out) if args.out else in_path.with_name(
        in_path.stem + '_denoised_ist.fits')

    print(f'Loading  {in_path}')
    with fits.open(in_path) as hdul:
        hdr  = hdul[0].header.copy()
        data = np.ascontiguousarray(np.squeeze(hdul[0].data), dtype=np.float32)

    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim != 3:
        raise ValueError(f'Cannot interpret shape {data.shape} as (n_ch, H, W)')

    nz, ny, nx = data.shape
    print(f'Cube shape : {data.shape}   (nz, ny, nx)')
    print(f'Flux range : [{data.min():.4e}, {data.max():.4e}]')
    print(f'Noise (std): {data.std():.4e}')

    num_scales_2d = int(math.floor(math.log2(min(ny, nx))))
    num_scales_1d = int(math.floor(math.log2(nz)))
    print(f'Max scales 2D : {num_scales_2d}  (spatial {ny}×{nx})')
    print(f'Max scales 1D : {num_scales_1d}  (spectral {nz} channels)')

    denoiser = Denoiser2D1D(threshold_type='soft', verbose=True, plot=False)
    result = denoiser.denoise(
        x=data,
        y=data,
        method='iterative',
        threshold_level=args.threshold,
        threshold_increment_high_freq=args.thresh_increm,
        num_scales_2d=num_scales_2d,
        num_scales_1d=num_scales_1d,
        noise_cube=None,
        positivity=True,
        positivity_final=True,
        num_iter_reweight=args.num_iter,
        num_iter_debias=args.num_iter,
        patience=args.patience,
    )

    denoised = result[0].astype(np.float32)
    print(f'\nDenoised shape : {denoised.shape}')
    print(f'Denoised range : [{denoised.min():.4e}, {denoised.max():.4e}]')

    hdr['HISTORY'] = f'IST-denoised by denoise_cube.py (thresh={args.threshold}sigma)'
    fits.writeto(str(out_path), denoised, hdr, overwrite=True)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
