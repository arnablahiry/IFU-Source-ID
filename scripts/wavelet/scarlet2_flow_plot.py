import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_cube(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "cube" not in f:
            raise KeyError(f"Expected dataset 'cube' in {path}")
        return f["cube"][...].astype(np.float32)


def load_sources(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "source_cubes" not in f:
            raise KeyError(f"Expected dataset 'source_cubes' in {path}")
        return f["source_cubes"][...].astype(np.float32)


def make_result_figure(original_cube: np.ndarray, source_cubes: np.ndarray, out_path: Path) -> None:
    n_sources = source_cubes.shape[0]
    original_mom0 = np.sum(original_cube, axis=0)
    source_mom0 = [np.sum(source_cubes[i], axis=0) for i in range(n_sources)]
    reconstructed_cube = np.sum(source_cubes, axis=0)
    residual_cube = original_cube - reconstructed_cube
    residual_mom0 = np.sum(residual_cube, axis=0)

    original_spec = np.sum(original_cube, axis=(1, 2))
    source_specs = [np.sum(source_cubes[i], axis=(1, 2)) for i in range(n_sources)]
    reconstructed_spec = np.sum(reconstructed_cube, axis=(1, 2))
    residual_spec = np.sum(residual_cube, axis=(1, 2))

    fig = plt.figure(figsize=(4.2 * (n_sources + 2), 9.5))
    fig.suptitle("Scarlet2 + Flow Separation Results", fontsize=18)

    # Top row: spatial moments.
    n_top = n_sources + 3
    ax = fig.add_subplot(2, n_top, 1)
    im = ax.imshow(original_mom0, cmap="magma")
    ax.set_title("Input\nMoment 0")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, mom0 in enumerate(source_mom0, start=2):
        ax = fig.add_subplot(2, n_top, i)
        im = ax.imshow(mom0, cmap="inferno")
        ax.set_title(f"Source {i - 2}\nMoment 0")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(2, n_top, n_top)
    im = ax.imshow(reconstructed_cube.sum(axis=0), cmap="viridis")
    ax.set_title("Reconstructed\nMoment 0")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(2, n_top, n_top + 1)
    im = ax.imshow(residual_mom0, cmap="coolwarm")
    ax.set_title("Residual\nMoment 0")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: spectral profiles.
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(original_spec, color="black", lw=2.0, label="Input total")
    ax.plot(reconstructed_spec, color="tab:green", lw=1.8, ls="--", label="Reconstructed total")
    ax.plot(residual_spec, color="tab:red", lw=1.2, ls=":", label="Residual total")

    cmap = plt.get_cmap("tab10")
    for i, spec in enumerate(source_specs):
        ax.plot(spec, lw=1.6, color=cmap(i % 10), label=f"Source {i}")

    ax.set_title("Integrated Spectral Profiles")
    ax.set_xlabel("Spectral Channel")
    ax.set_ylabel("Summed Flux")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scarlet2 + flow source separation results")
    parser.add_argument("--cube", default="/Users/arnablahiry/repos/GalCubeCraft-SourceID/experiments/sep_v1/data/cube_1006.h5", help="Original cube HDF5 path containing dataset 'cube'")
    parser.add_argument("--res", default="/Users/arnablahiry/repos/GalCubeCraft-SourceID/experiments/mca_v1/scarlet2_flow_test/decomposed_sources.h5", help="Result HDF5 path containing dataset 'source_cubes'")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to sibling of --res")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    res_path = Path(args.res)
    out_path = Path(args.out) if args.out else res_path.with_name("scarlet2_flow_results.png")

    original_cube = load_cube(cube_path)
    source_cubes = load_sources(res_path)

    if original_cube.shape != source_cubes.shape[1:]:
        raise ValueError(
            f"Shape mismatch: cube has {original_cube.shape}, sources have {source_cubes.shape[1:]}"
        )

    make_result_figure(original_cube, source_cubes, out_path)
    print(f"Saved result plot to: {out_path}")


if __name__ == "__main__":
    main()
