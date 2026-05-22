"""Visual test of wavelet_detect + flow_tracker on a real IFU cube.

Accepts HDF5, FITS, .npy, or .npz cubes.

Produces:
  out/detection_flow.gif      — 4-panel per-channel animation:
                                  diffuse component | significance + detections |
                                  real cube + source masks | flow magnitude + quivers
  out/wavelet_scales.png      — starlet detail bands + diffuse for one channel.
  out/diffuse_mosaic.png      — per-channel diffuse (coarse) maps.
  out/significance_mosaic.png — per-channel significance (compact-source) maps.
  out/flow_fields.png         — masked flow quivers for a grid of channel pairs.
  out/flow_magnitude.png      — flow magnitude heatmap mosaic.
  out/source_spectra.png      — per-source integrated flux spectrum.
  out/source_mosaic.png       — peak-flux channel image for each track.

Usage:
    cd scripts/wavelet
    # Synthetic HDF5 cube:
    python test_pipeline.py --cube ../../data/all_cubes/cube_10001.h5 --show-gt

    # Real W2246 FITS cube (active channels auto-detected):
    python test_pipeline.py \\
        --cube /mnt/home/alahiry/data/obvs_data/CROPPED_ONLY_SPATIAL_W2246_C2_125.fits \\
        --out ../../experiments/wavelet_w2246 \\
        --scales 5 --detail-scales 0,1,2 --k-sigma 3.0 --min-area 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_dilation

# ---- make sure the wavelet scripts are importable from this directory ----
sys.path.insert(0, str(Path(__file__).parent))
from scripts.wavelet.legacy2.wavelet_detect import detect_cube, starlet_transform, load_cube, active_channels
from scripts.wavelet.legacy2.flow_tracker import compute_masked_flow, build_tracks, assemble_source_cubes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm01(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def source_colormap(n: int) -> list:
    cmap = plt.colormaps["tab10"] if n <= 10 else plt.colormaps["tab20"]
    return [cmap(i % cmap.N) for i in range(n)]


# ---------------------------------------------------------------------------
# Plot 1: Starlet scales for one channel (with diffuse separation)
# ---------------------------------------------------------------------------

def plot_wavelet_scales(cube: np.ndarray, channel: int, out: Path,
                        n_scales: int = 5,
                        detail_scales: tuple[int, ...] = (0, 1, 2)) -> None:
    from scripts.wavelet.legacy2.wavelet_detect import starlet_coarse, mad_noise
    img = cube[channel].astype(np.float64)
    coeffs = starlet_transform(img, n_scales=n_scales)
    diffuse = starlet_coarse(img, n_scales=n_scales)
    residual = img - diffuse   # compact component

    noise0 = mad_noise(coeffs[0])

    # Rows: top = all detail bands + coarse; bottom = diffuse/compact decomp
    ncols = n_scales + 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.5))
    fig.suptitle(f"Starlet decomposition — channel {channel}\n"
                 f"Detection uses fine scales {list(detail_scales)} only",
                 fontsize=12)

    axes[0, 0].imshow(norm01(img), cmap="inferno", origin="lower")
    axes[0, 0].set_title("Original", fontsize=9)

    for j in range(n_scales):
        band = coeffs[j]
        vmax = max(abs(band.max()), abs(band.min())) + 1e-12
        used = j in detail_scales
        axes[0, j + 1].imshow(band, cmap="RdBu_r", origin="lower",
                               vmin=-vmax, vmax=vmax)
        label = f"Scale {j}" + (" ✓" if used else " (diffuse)")
        axes[0, j + 1].set_title(label, fontsize=8,
                                  color="green" if used else "gray")

    axes[0, -1].imshow(norm01(coeffs[-1]), cmap="inferno", origin="lower")
    axes[0, -1].set_title("Coarse residual\n(diffuse)", fontsize=8, color="gray")

    # Bottom row: original / diffuse / compact residual / significance
    axes[1, 0].imshow(norm01(img), cmap="gray", origin="lower")
    axes[1, 0].set_title("Original", fontsize=9)

    axes[1, 1].imshow(norm01(diffuse), cmap="magma", origin="lower")
    axes[1, 1].set_title("Diffuse (coarse)", fontsize=9)

    axes[1, 2].imshow(residual, cmap="RdBu_r", origin="lower",
                      vmin=-abs(residual).max(), vmax=abs(residual).max())
    axes[1, 2].set_title("Compact (original − diffuse)", fontsize=9)

    # Significance map from fine scales only
    from scripts.wavelet.legacy2.wavelet_detect import _PROPAGATION
    k_sigma = 3.0
    sig = np.zeros_like(img)
    for j in detail_scales:
        if j < n_scales:
            e_j = _PROPAGATION[min(j, len(_PROPAGATION) - 1)]
            plane = coeffs[j]
            sig += np.where(plane > k_sigma * noise0 * e_j, plane, 0.0)
    axes[1, 3].imshow(sig, cmap="hot", origin="lower")
    axes[1, 3].set_title(f"Significance\n(k={k_sigma}σ, scales {list(detail_scales)})", fontsize=8)

    for j in range(4, ncols):
        axes[1, j].axis("off")

    for row in axes:
        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "wavelet_scales.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved wavelet_scales.png")


# ---------------------------------------------------------------------------
# Plot 2a: Diffuse-vs-compact mosaic
# ---------------------------------------------------------------------------

def plot_diffuse_mosaic(dets, cube: np.ndarray, out: Path,
                        max_panels: int = 12) -> None:
    n_proc = dets.n_channels()
    step = max(1, n_proc // max_panels)
    indices = list(range(0, n_proc, step))[:max_panels]

    ncols = min(6, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols

    # Two sub-rows per panel: diffuse and compact
    fig, axes = plt.subplots(nrows * 2, ncols,
                              figsize=(2.8 * ncols, 3.0 * nrows * 2))
    axes = axes.reshape(nrows * 2, ncols)
    fig.suptitle("Diffuse component (top) vs compact significance (bottom)", fontsize=12)

    diff_vmax = float(np.percentile(dets.diffuse_maps, 99.5)) + 1e-12
    sig_vmax  = float(dets.significance.max()) + 1e-12

    for panel, idx in enumerate(indices):
        ch = dets.channel_list[idx]
        row0 = (panel // ncols) * 2
        col  = panel % ncols

        # Top: diffuse
        axes[row0, col].imshow(dets.diffuse_maps[idx], cmap="magma",
                                origin="lower", vmin=0, vmax=diff_vmax)
        axes[row0, col].set_title(f"ch {ch}  diffuse", fontsize=8)
        axes[row0, col].axis("off")

        # Bottom: significance with source centroids
        axes[row0 + 1, col].imshow(norm01(cube[ch]), cmap="gray",
                                    origin="lower", alpha=0.5)
        axes[row0 + 1, col].imshow(dets.significance[idx] / sig_vmax,
                                    cmap="hot", origin="lower", alpha=0.8)
        for reg in dets.channel_regions[idx]:
            axes[row0 + 1, col].plot(reg.x, reg.y, "c+", ms=5, mew=1.1)
        n_det = len(dets.channel_regions[idx])
        axes[row0 + 1, col].set_title(f"ch {ch}  {n_det} src", fontsize=8)
        axes[row0 + 1, col].axis("off")

    # Hide unused
    for panel in range(len(indices), ncols * nrows):
        row0 = (panel // ncols) * 2
        col  = panel % ncols
        axes[row0, col].axis("off")
        axes[row0 + 1, col].axis("off")

    fig.tight_layout()
    fig.savefig(out / "diffuse_mosaic.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved diffuse_mosaic.png")


# ---------------------------------------------------------------------------
# Plot 2b: Significance mosaic
# ---------------------------------------------------------------------------

def plot_significance_mosaic(dets, cube: np.ndarray, out: Path,
                              max_panels: int = 16) -> None:
    n_proc = dets.n_channels()
    step = max(1, n_proc // max_panels)
    indices = list(range(0, n_proc, step))[:max_panels]

    ncols = min(8, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Per-channel compact-source significance maps", fontsize=12)

    sig_global_max = dets.significance.max() + 1e-12

    for panel, idx in enumerate(indices):
        ch = dets.channel_list[idx]
        ax = axes[panel]
        sig = dets.significance[idx]
        ax.imshow(norm01(cube[ch]), cmap="gray", origin="lower", alpha=0.6)
        ax.imshow(sig / sig_global_max, cmap="hot", origin="lower",
                  alpha=0.75, vmin=0, vmax=1)
        n_det = len(dets.channel_regions[idx])
        ax.set_title(f"ch {ch}  ({n_det} src)", fontsize=8)
        for reg in dets.channel_regions[idx]:
            ax.plot(reg.x, reg.y, "c+", ms=6, mew=1.2)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "significance_mosaic.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved significance_mosaic.png")


# ---------------------------------------------------------------------------
# Plot 3: Flow fields mosaic
# ---------------------------------------------------------------------------

def plot_flow_fields(flow: np.ndarray, dets, cube: np.ndarray, out: Path,
                     max_panels: int = 12, quiver_step: int = 6) -> None:
    n_pairs = flow.shape[0]
    step = max(1, n_pairs // max_panels)
    indices = list(range(0, n_pairs, step))[:max_panels]

    ncols = min(6, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Masked optical-flow fields (source pixels only)", fontsize=12)

    H, W = cube.shape[1:]
    ys = np.arange(0, H, quiver_step)
    xs = np.arange(0, W, quiver_step)
    X, Y = np.meshgrid(xs, ys)

    for panel, idx in enumerate(indices):
        ch_ref = dets.channel_list[idx]
        ch_tgt = dets.channel_list[idx + 1]
        ax = axes[panel]

        ref_img = norm01(cube[ch_ref])
        ax.imshow(ref_img, cmap="gray", origin="lower")

        # Source mask overlay
        mask_ref = dets.union_mask(idx)
        mask_tgt = dets.union_mask(idx + 1)
        joint = mask_ref & mask_tgt
        overlay = np.zeros((*ref_img.shape, 4), dtype=np.float32)
        overlay[joint, 0] = 0.2; overlay[joint, 2] = 0.9; overlay[joint, 3] = 0.4
        ax.imshow(overlay, origin="lower")

        # Quivers — amplify tiny sub-pixel displacements for visibility
        v = flow[idx, 0][ys[:, None], xs[None, :]]
        u = flow[idx, 1][ys[:, None], xs[None, :]]
        mag = np.hypot(u, v)
        nonzero = mag > 1e-4
        if nonzero.any():
            amp = quiver_step * 0.8 / (mag[nonzero].max() + 1e-12)
            ax.quiver(X[nonzero], Y[nonzero],
                      u[nonzero] * amp, v[nonzero] * amp,
                      mag[nonzero], cmap="plasma",
                      angles="xy", scale_units="xy", scale=1,
                      width=0.008, headwidth=3, headlength=4, alpha=0.9)

        ax.set_title(f"ch {ch_ref}→{ch_tgt}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "flow_fields.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved flow_fields.png")


# ---------------------------------------------------------------------------
# Plot 4: Flow magnitude heatmap
# ---------------------------------------------------------------------------

def plot_flow_magnitude(flow: np.ndarray, dets, out: Path,
                        max_panels: int = 12) -> None:
    n_pairs = flow.shape[0]
    step = max(1, n_pairs // max_panels)
    indices = list(range(0, n_pairs, step))[:max_panels]

    ncols = min(6, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Flow magnitude |v| (px/channel, source pixels only)", fontsize=12)

    mag_all = np.hypot(flow[:, 0], flow[:, 1])
    vmax = float(np.percentile(mag_all[mag_all > 0], 99)) if (mag_all > 0).any() else 1.0

    for panel, idx in enumerate(indices):
        ch_ref = dets.channel_list[idx]
        ch_tgt = dets.channel_list[idx + 1]
        mag = mag_all[idx]
        im = axes[panel].imshow(mag, cmap="plasma", origin="lower",
                                vmin=0, vmax=vmax)
        axes[panel].set_title(f"ch {ch_ref}→{ch_tgt}", fontsize=8)
        axes[panel].axis("off")
        plt.colorbar(im, ax=axes[panel], fraction=0.046, pad=0.04)

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "flow_magnitude.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved flow_magnitude.png")


# ---------------------------------------------------------------------------
# Plot 5: Per-source spectra
# ---------------------------------------------------------------------------

def plot_source_spectra(tracks, dets, cube: np.ndarray, out: Path,
                        gt_pos=None, top_n: int = 20) -> None:
    n_tracks = len(tracks)
    if n_tracks == 0:
        print("  no tracks — skipping source_spectra.png")
        return

    # Show top-N by total flux so the plot stays readable
    shown = tracks[:min(top_n, n_tracks)]
    colors = source_colormap(len(shown))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title(
        f"Per-source integrated flux spectrum  (top {len(shown)} of {n_tracks} tracks)",
        fontsize=12,
    )

    chs = dets.channel_list
    for tid, track in enumerate(shown):
        spectrum = track.flux[:len(chs)]
        ax.plot(chs, spectrum, color=colors[tid], lw=1.8, label=f"src {tid}")

    # GT lines drawn after so y-limits are stable
    if gt_pos is not None:
        ylim = ax.get_ylim()
        for g, (gx, gy, gz) in enumerate(gt_pos):
            ax.axvline(gz, color="gold", lw=1.2, ls="--", alpha=0.8)
            ax.text(gz, ylim[1] * 0.97, f"G{g}",
                    ha="center", va="top", fontsize=8,
                    color="goldenrod", fontweight="bold")

    ax.set_xlabel("Channel index")
    ax.set_ylabel("Detection flux (significance units)")
    ax.legend(fontsize=8, ncol=min(4, len(shown)), loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "source_spectra.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved source_spectra.png")


# ---------------------------------------------------------------------------
# Plot 6: Source mosaic (peak channel per track)
# ---------------------------------------------------------------------------

def plot_source_mosaic(tracks, dets, cube: np.ndarray, out: Path,
                       top_n: int = 25, cutout_px: int = 160) -> None:
    """One panel per source: zoomed cutout at peak channel + footprint overlay."""
    n_tracks = len(tracks)
    if n_tracks == 0:
        print("  no tracks — skipping source_mosaic.png")
        return

    shown = tracks[:min(top_n, n_tracks)]
    n_shown = len(shown)
    ncols = min(5, n_shown)
    nrows = (n_shown + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(
        f"Source peak channel — zoomed cutout (gray) + detected footprint (cyan)\n"
        f"[{n_shown} sources]",
        fontsize=11,
    )
    colors = source_colormap(n_shown)
    H, W = cube.shape[1:]
    half = cutout_px // 2

    for tid, track in enumerate(shown):
        ax = axes[tid]
        best_pidx = int(np.argmax(track.flux))
        ch = dets.channel_list[best_pidx]
        img = cube[ch].astype(np.float32)

        # Centroid (full-image coords)
        cy, cx = track.trajectory[best_pidx]
        cy_i, cx_i = int(round(cy)), int(round(cx))

        # Crop window (clamped to image bounds)
        r0 = max(0, cy_i - half); r1 = min(H, cy_i + half)
        c0 = max(0, cx_i - half); c1 = min(W, cx_i + half)

        crop = img[r0:r1, c0:c1]
        ax.imshow(norm01(crop), cmap="gray", origin="lower",
                  extent=[c0, c1, r0, r1])

        # Per-channel active footprint (significant pixels within territory)
        m = track.mask(best_pidx)
        if m is not None:
            m_crop = m[r0:r1, c0:c1]
            # Show only the detected-emission footprint, dilated 1px for visibility
            m_vis = binary_dilation(m_crop, iterations=1)
            overlay = np.zeros((*crop.shape, 4), dtype=np.float32)
            overlay[m_vis, 0] = 0.0
            overlay[m_vis, 1] = 0.9
            overlay[m_vis, 2] = 0.9
            overlay[m_vis, 3] = 0.5
            ax.imshow(overlay, origin="lower",
                      extent=[c0, c1, r0, r1])

        # Centroid marker
        ax.plot(cx, cy, "o", ms=7, color=colors[tid],
                mec="white", mew=1.2, zorder=5)

        snr = track.snr[best_pidx]
        area = track.area[best_pidx]
        ax.set_title(f"src {tid}  ch={ch}  SNR={snr:.1f}  {area}px", fontsize=8)
        ax.set_xlim(c0, c1); ax.set_ylim(r0, r1)
        ax.axis("off")

    for ax in axes[n_shown:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "source_mosaic.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved source_mosaic.png")


# ---------------------------------------------------------------------------
# GIF: per-channel animation (4 panels)
# ---------------------------------------------------------------------------

def make_gif(
    cube: np.ndarray,
    dets,
    flow: np.ndarray,
    tracks,
    out: Path,
    fps: int = 5,
    quiver_step: int = 8,
    gt_pos=None,
) -> None:
    """Four-panel per-channel animation:
      Panel 1 — diffuse component (coarse starlet residual).
      Panel 2 — compact-source significance map with detection centroids.
      Panel 3 — real cube slice with detected source masks (white) and
                 per-track coloured edges.
      Panel 4 — flow magnitude + quivers (source pixels only).
    """
    n_proc = dets.n_channels()
    H, W = cube.shape[1:]
    colors = source_colormap(len(tracks))

    cube_norm = norm01(cube, lo_pct=0.5, hi_pct=99.8)
    sig_max  = dets.significance.max() + 1e-12
    diff_max = float(np.percentile(dets.diffuse_maps, 99.5)) + 1e-12

    fig = plt.figure(figsize=(18, 4.8))
    gs = GridSpec(1, 4, figure=fig, wspace=0.05)
    ax_diff = fig.add_subplot(gs[0, 0])
    ax_sig  = fig.add_subplot(gs[0, 1])
    ax_real = fig.add_subplot(gs[0, 2])
    ax_flow = fig.add_subplot(gs[0, 3])
    for ax in (ax_diff, ax_sig, ax_real, ax_flow):
        ax.axis("off")

    ax_diff.set_title("Diffuse (coarse)", fontsize=10)
    ax_sig.set_title("Compact significance", fontsize=10)
    ax_real.set_title("Detected sources only", fontsize=10)
    ax_flow.set_title("Flow magnitude + quivers", fontsize=10)

    ys = np.arange(0, H, quiver_step)
    xs = np.arange(0, W, quiver_step)
    Xq, Yq = np.meshgrid(xs, ys)

    frames = []
    for idx in range(n_proc):
        ch = dets.channel_list[idx]
        sig  = dets.significance[idx]
        diff = dets.diffuse_maps[idx]
        regs = dets.channel_regions[idx]

        # Panel 1: diffuse
        diff_rgb = plt.colormaps["magma"](
            np.clip(diff / diff_max, 0, 1)
        )[..., :3].astype(np.float32)

        # Panel 2: significance blended with grey cube
        sig_rgb = plt.colormaps["hot"](sig / sig_max)[..., :3].astype(np.float32)
        gray = cube_norm[ch][..., None]
        sig_img = 0.3 * np.repeat(gray, 3, axis=-1) + 0.7 * sig_rgb

        # Panel 3: black background, only source pixels shown in gray + coloured edges
        real_img = np.zeros((H, W, 3), dtype=np.float32)
        gray_ch = cube_norm[ch]
        for tid, track in enumerate(tracks):
            m = track.mask(idx)
            if m is None or not m.any():
                continue
            real_img[m, 0] = gray_ch[m]
            real_img[m, 1] = gray_ch[m]
            real_img[m, 2] = gray_ch[m]
            edge = binary_dilation(m, iterations=1) & ~m
            c = colors[tid]
            real_img[edge, 0] = c[0]
            real_img[edge, 1] = c[1]
            real_img[edge, 2] = c[2]

        # Panel 4: flow magnitude
        if idx < flow.shape[0]:
            mag = np.hypot(flow[idx, 0], flow[idx, 1])
            mag_norm = (mag / (float(np.percentile(mag[mag > 0], 99)) + 1e-9)
                        if (mag > 0).any() else mag)
            flow_rgb = plt.colormaps["plasma"](
                np.clip(mag_norm, 0, 1)
            )[..., :3].astype(np.float32)
        else:
            flow_rgb = np.zeros((H, W, 3), dtype=np.float32)

        frames.append((diff_rgb, sig_img, real_img, flow_rgb, sig, regs, idx, ch))

    # Init imshow artists
    im_diff = ax_diff.imshow(frames[0][0], origin="lower", animated=True)
    im_sig  = ax_sig.imshow(frames[0][1],  origin="lower", animated=True)
    im_real = ax_real.imshow(frames[0][2], origin="lower", animated=True)
    im_flow = ax_flow.imshow(frames[0][3], origin="lower", animated=True)

    scat_sig  = ax_sig.scatter([], [], marker="+", s=80, c="cyan",
                                linewidths=1.5, zorder=5)
    scat_real = ax_real.scatter([], [], marker="o", s=30,
                                 c=[], cmap="tab10", zorder=6,
                                 edgecolors="white", linewidths=0.8)

    title_text = fig.suptitle("", fontsize=11, y=1.01)

    # Channel progress bar
    ch_bar_ax = fig.add_axes([0.05, 0.98, 0.9, 0.012])
    ch_bar_ax.set_xlim(dets.channel_list[0], dets.channel_list[-1])
    ch_bar_ax.set_ylim(0, 1)
    ch_bar_ax.axis("off")
    ch_line, = ch_bar_ax.plot([], [], "|", ms=14, color="crimson", mew=2.5)

    if gt_pos is not None:
        for gx, gy, gz in gt_pos:
            ch_bar_ax.axvline(gz, color="gold", lw=1.2, alpha=0.8)

    # Initialise quiver with fixed-size grid (prevents set_UVC size mismatch)
    n_pts = Xq.size
    quiv = ax_flow.quiver(Xq.ravel(), Yq.ravel(),
                          np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts),
                          cmap="cool", angles="xy", scale_units="xy", scale=1,
                          width=0.006, headwidth=3, headlength=4, alpha=0.85,
                          clim=(0, 3))

    def _update(frame_idx):
        diff_rgb, sig_img, real_img, flow_rgb, sig, regs, pidx, ch = frames[frame_idx]

        im_diff.set_data(diff_rgb)
        im_sig.set_data(sig_img)
        im_real.set_data(real_img)
        im_flow.set_data(flow_rgb)

        # Centroids on significance panel
        if regs:
            scat_sig.set_offsets(np.column_stack([[r.x for r in regs],
                                                  [r.y for r in regs]]))
        else:
            scat_sig.set_offsets(np.empty((0, 2)))

        # Per-track dots on real panel
        dot_xy, dot_c = [], []
        for tid, track in enumerate(tracks):
            ty, tx = track.trajectory[pidx]
            if not np.isnan(ty):
                dot_xy.append([tx, ty])
                dot_c.append(tid / max(len(tracks) - 1, 1))
        if dot_xy:
            scat_real.set_offsets(np.array(dot_xy))
            scat_real.set_array(np.array(dot_c))
        else:
            scat_real.set_offsets(np.empty((0, 2)))

        # Quivers — amplify sub-pixel displacements for visibility
        if pidx < flow.shape[0]:
            v_q = flow[pidx, 0][ys[:, None], xs[None, :]].ravel()
            u_q = flow[pidx, 1][ys[:, None], xs[None, :]].ravel()
            mag_q = np.hypot(u_q, v_q)
            peak = mag_q.max()
            if peak > 1e-6:
                amp = quiver_step * 0.8 / peak
                u_q = u_q * amp
                v_q = v_q * amp
        else:
            v_q = np.zeros(n_pts)
            u_q = np.zeros(n_pts)
            mag_q = np.zeros(n_pts)
        quiv.set_UVC(u_q, v_q, mag_q)

        ch_line.set_data([ch, ch], [0, 1])
        title_text.set_text(
            f"Channel {ch} ({pidx+1}/{n_proc}) | "
            f"detections: {len(regs)} | tracks: {len(tracks)}"
        )
        return im_diff, im_sig, im_real, im_flow, scat_sig, scat_real, quiv, ch_line, title_text

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames),
        interval=1000 // fps, blit=False,
    )

    gif_path = out / "detection_flow.gif"
    ani.save(str(gif_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  saved detection_flow.gif  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cube", required=True,
                    help="Cube file: .h5/.hdf5, .fits/.fit, .npy, or .npz")
    ap.add_argument("--out", default="/tmp/wavelet_test", help="Output directory")
    ap.add_argument("--scales", type=int, default=5,
                    help="Total starlet scales (default: 5)")
    ap.add_argument("--detail-scales", type=str, default="0,1,2",
                    help="Fine scales used for compact-source detection (default: 0,1,2)")
    ap.add_argument("--no-subtract-diffuse", action="store_true",
                    help="Skip diffuse subtraction before detection")
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--max-area", type=int, default=None,
                    help="Max blob area in px; blobs larger than this are diffuse (default: no limit)")
    ap.add_argument("--peak-min-distance", type=int, default=20,
                    help="Min pixel separation between source peaks (default: 20)")
    ap.add_argument("--peak-threshold-rel", type=float, default=0.05,
                    help="Peak threshold relative to global significance max (default: 0.05)")
    ap.add_argument("--channels", type=str, default=None,
                    help="Comma-sep channel indices; default: auto-detect active")
    ap.add_argument("--active-threshold", type=float, default=0.05,
                    help="Fraction of peak flux to define active channels (default: 0.05)")
    ap.add_argument("--flow-method", choices=["tvl1", "farneback"], default="tvl1")
    ap.add_argument("--max-match-dist", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--show-gt", action="store_true",
                    help="Show GT positions if available in HDF5 (simulation cubes only)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detail_scales = tuple(int(s) for s in args.detail_scales.split(","))

    print(f"\n{'='*60}")
    print(f"  IFU wavelet+flow visual test")
    print(f"  cube          : {args.cube}")
    print(f"  output        : {out}")
    print(f"  scales        : {args.scales}  detail={list(detail_scales)}")
    print(f"  subtract_diffuse: {not args.no_subtract_diffuse}")
    print(f"{'='*60}\n")

    # Load cube — works for h5, fits, npy, npz
    print("Loading cube ...")
    cube = load_cube(args.cube)
    n_ch, H, W = cube.shape
    print(f"  shape: {cube.shape}  flux range [{cube.min():.3e}, {cube.max():.3e}]")

    # GT positions only meaningful for synthetic HDF5 cubes
    gt_pos = None
    if args.show_gt and str(args.cube).endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(args.cube, "r") as f:
            if "galaxies" in f:
                gt_pos = f["galaxies/positions_xyz_px"][:]

    if args.channels is not None:
        channel_list = [int(c) for c in args.channels.split(",")]
    else:
        channel_list = active_channels(cube, threshold_frac=args.active_threshold)
        print(f"  auto-selected {len(channel_list)} active channels "
              f"(ch {channel_list[0]}–{channel_list[-1]})")

    rep_ch = channel_list[len(channel_list) // 2]

    # --- Stage 1: detection
    print("\nStage 1: wavelet detection (global blobs → per-channel footprints) ...")
    dets = detect_cube(
        cube, channel_list=channel_list,
        n_scales=args.scales, k_sigma=args.k_sigma,
        min_area=args.min_area,
        max_area=args.max_area,
        detail_scales=detail_scales,
        subtract_diffuse=not args.no_subtract_diffuse,
        peak_min_distance=args.peak_min_distance,
        peak_threshold_rel=args.peak_threshold_rel,
    )
    print(f"  → {dets.n_sources()} global sources detected"
          f"  (tracking across {dets.n_channels()} channels)")

    # --- Stage 2: flow
    print(f"\nStage 2: masked optical flow ({args.flow_method}) ...")
    flow = compute_masked_flow(cube, dets, method=args.flow_method)
    nonzero = int((np.abs(flow).sum(axis=(1,2,3)) > 0).sum())
    med_mag = float(np.median(np.abs(flow[flow != 0]))) if (flow != 0).any() else 0.0
    print(f"  → {nonzero}/{flow.shape[0]} pairs with non-zero flow  "
          f"median |v|={med_mag:.3f} px")

    # --- Stage 3: tracks
    print("\nStage 3: track linking ...")
    tracks = build_tracks(dets, flow, max_match_dist=args.max_match_dist)
    print(f"  → {len(tracks)} tracks")

    # --- Stage 4: source cubes
    print("\nStage 4: assembling source cubes ...")
    src_cubes = assemble_source_cubes(cube, tracks, dets)
    print(f"  → source_cubes shape: {src_cubes.shape}")

    # --- Plots
    print("\nGenerating plots ...")

    plot_wavelet_scales(cube, channel=rep_ch, out=out,
                        n_scales=args.scales, detail_scales=detail_scales)
    plot_diffuse_mosaic(dets, cube, out=out)
    plot_significance_mosaic(dets, cube, out=out)
    plot_flow_fields(flow, dets, cube, out=out, quiver_step=8)
    plot_flow_magnitude(flow, dets, out=out)
    plot_source_spectra(tracks, dets, cube, out=out, gt_pos=gt_pos)
    plot_source_mosaic(tracks, dets, cube, out=out)

    print("\nGenerating GIF ...")
    make_gif(cube, dets, flow, tracks, out=out, fps=args.fps, gt_pos=gt_pos)

    print(f"\nDone.  All outputs in: {out}")
    for f in ["wavelet_scales.png", "diffuse_mosaic.png", "significance_mosaic.png",
              "flow_fields.png", "flow_magnitude.png",
              "source_spectra.png", "source_mosaic.png", "detection_flow.gif"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
