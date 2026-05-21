import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_decomposed_results(original_path, result_path):
    # Load original data
    with h5py.File(original_path, "r") as f:
        orig_cube = f["cube"][:].astype(np.float32)
    
    # Load decomposed data
    with h5py.File(result_path, "r") as f:
        decomposed = f["source_cubes"][:] # Shape: (3, S, N, N)

    n_sources = decomposed.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"MCA Decomposition Results: {Path(original_path).name}", fontsize=16)

    # 1. Plot Original Moment 0 (Integrated intensity over all channels)
    ax0 = plt.subplot(2, n_sources + 1, 1)
    mom0_orig = np.sum(orig_cube, axis=0)
    im0 = ax0.imshow(mom0_orig, cmap='viridis')
    ax0.set_title("Original Cube\n(Moment 0)")
    plt.colorbar(im0, ax=ax0)

    # 2. Plot Each Decomposed Source Moment 0
    for i in range(n_sources):
        ax = plt.subplot(2, n_sources + 1, i + 2)
        mom0_src = np.sum(decomposed[i], axis=0)
        im = ax.imshow(mom0_src, cmap='magma')
        ax.set_title(f"Source {i}\n(Moment 0)")
        plt.colorbar(im, ax=ax)

    # 3. Plot Spectral Profiles (Flux vs Channel)
    ax_spec = plt.subplot(2, 1, 2)
    
    # Original total spectrum
    orig_spec = np.sum(orig_cube, axis=(1, 2))
    ax_spec.plot(orig_spec, label="Total Input", color='black', alpha=0.3, linestyle='--')

    # Generate dynamic colors using a colormap
    cmap = plt.get_cmap('tab10') 
    
    for i in range(n_sources):
        src_spec = np.sum(decomposed[i], axis=(1, 2))
        # This ensures we don't run out of colors regardless of source count
        color = cmap(i % 10) 
        ax_spec.plot(src_spec, label=f"Source {i}", color=color, lw=2)

    ax_spec.set_xlabel("Spectral Channel")
    ax_spec.set_ylabel("Integrated Flux")
    ax_spec.set_title("Spectral Profiles (Source Separation)")
    ax_spec.legend(loc='upper right', ncol=min(n_sources, 5))
    ax_spec.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = Path(result_path).parent / "separation_plot.png"
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="experiments/sep_v1/data/cube_1006.h5")
    ap.add_argument("--res", default="experiments/mca_v1/decomposed_sources.h5")
    args = ap.parse_args()

    plot_decomposed_results(args.cube, args.res)