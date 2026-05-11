"""Morphological Component Analysis (MCA) for 3D spectral cubes,
coupled to a per-channel optical-flow kinematic prior.

The cube is assumed to have shape (n_ch, H, W). Successive spectral
slices are linked by a dense displacement field estimated with TV-L1
(skimage) or Farneback (OpenCV, optional). A flux-preserving forward
warp transports a slice along that field. MCA then decomposes every
slice into a sparse "point-source" component (Dirac dictionary) and a
sparse "diffuse" component (Starlet / Isotropic Undecimated Wavelet
Transform) using ISTA, with an extra quadratic term that ties each
component to the warped version of its neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import convolve1d

try:
    from skimage.registration import optical_flow_tvl1
    _HAS_TVL1 = True
except Exception:
    _HAS_TVL1 = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ----------------------------------------------------------------------
# Optical flow
# ----------------------------------------------------------------------
def estimate_flow_field(
    cube: np.ndarray,
    method: str = "tvl1",
    **kwargs,
) -> np.ndarray:
    """Estimate per-channel-pair displacement fields.

    Parameters
    ----------
    cube : (n_ch, H, W) float32/64
    method : 'tvl1' (skimage) or 'farneback' (OpenCV).

    Returns
    -------
    flow : (n_ch-1, 2, H, W). flow[n] maps slice n -> slice n+1;
        component 0 is row-displacement (v), component 1 is column (u),
        matching skimage's convention.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (n_ch, H, W); got {cube.shape}")
    n_ch, H, W = cube.shape
    flow = np.zeros((n_ch - 1, 2, H, W), dtype=np.float32)

    method = method.lower()
    if method == "tvl1":
        if not _HAS_TVL1:
            raise ImportError("skimage.registration.optical_flow_tvl1 unavailable")
        for n in range(n_ch - 1):
            v, u = optical_flow_tvl1(cube[n], cube[n + 1], **kwargs)
            flow[n, 0] = v.astype(np.float32)
            flow[n, 1] = u.astype(np.float32)
    elif method == "farneback":
        if not _HAS_CV2:
            raise ImportError("cv2 unavailable; install opencv-python")
        defaults = dict(pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        defaults.update(kwargs)
        for n in range(n_ch - 1):
            a = _to_uint8(cube[n])
            b = _to_uint8(cube[n + 1])
            f = cv2.calcOpticalFlowFarneback(a, b, None, **defaults)
            flow[n, 0] = f[..., 1]  # row component
            flow[n, 1] = f[..., 0]  # col component
    else:
        raise ValueError(f"unknown method: {method}")
    return flow


def _to_uint8(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float64)
    lo, hi = np.percentile(a, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Flux-preserving warp
# ----------------------------------------------------------------------
def warp_flux_preserving(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward-scatter warp that conserves total flux.

    Each source pixel (i, j) deposits its value into target coordinate
    (i + dv, j + du) using bilinear weights. Because every source pixel
    distributes a unit of weight, sum(image_warped) == sum(image)
    (up to flux that flows out of the field of view).

    flow : (2, H, W) — flow[0] = dv (row), flow[1] = du (col).
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    H, W = image.shape
    if flow.shape != (2, H, W):
        raise ValueError(f"flow shape {flow.shape} != (2,{H},{W})")

    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    tgt_r = rows + flow[0]
    tgt_c = cols + flow[1]

    r0 = np.floor(tgt_r).astype(np.int64)
    c0 = np.floor(tgt_c).astype(np.int64)
    fr = tgt_r - r0
    fc = tgt_c - c0

    out = np.zeros_like(image, dtype=np.float64)
    src = image.astype(np.float64)

    for dr, dc, w in (
        (0, 0, (1 - fr) * (1 - fc)),
        (0, 1, (1 - fr) * fc),
        (1, 0, fr * (1 - fc)),
        (1, 1, fr * fc),
    ):
        rr = r0 + dr
        cc = c0 + dc
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        np.add.at(out, (rr[valid], cc[valid]), (src * w)[valid])
    return out.astype(image.dtype)


def warp_adjoint(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Adjoint of `warp_flux_preserving` w.r.t. the source image.

    With y = W x and W bilinear-scatter, the adjoint W^T pulls samples
    from the warped grid back to the source — i.e. backward bilinear
    sampling at the same target coordinates. Exposed because the
    kinematic prior's gradient needs it.
    """
    H, W = image.shape
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    tgt_r = rows + flow[0]
    tgt_c = cols + flow[1]

    r0 = np.floor(tgt_r).astype(np.int64)
    c0 = np.floor(tgt_c).astype(np.int64)
    fr = tgt_r - r0
    fc = tgt_c - c0

    out = np.zeros_like(image, dtype=np.float64)
    img = image.astype(np.float64)

    for dr, dc, w in (
        (0, 0, (1 - fr) * (1 - fc)),
        (0, 1, (1 - fr) * fc),
        (1, 0, fr * (1 - fc)),
        (1, 1, fr * fc),
    ):
        rr = np.clip(r0 + dr, 0, H - 1)
        cc = np.clip(c0 + dc, 0, W - 1)
        valid = ((r0 + dr) >= 0) & ((r0 + dr) < H) & \
                ((c0 + dc) >= 0) & ((c0 + dc) < W)
        out += np.where(valid, img[rr, cc] * w, 0.0)
    return out.astype(image.dtype)


# ----------------------------------------------------------------------
# Starlet (IUWT) transform — à trous with B3-spline filter
# ----------------------------------------------------------------------
_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _atrous_convolve2d(img: np.ndarray, step: int) -> np.ndarray:
    # à trous: insert (step-1) zeros between filter taps, separable.
    h = np.zeros((step * 4 + 1,), dtype=np.float64)
    h[::step] = _B3
    out = convolve1d(img, h, axis=0, mode="reflect")
    out = convolve1d(out, h, axis=1, mode="reflect")
    return out


def starlet_forward(image: np.ndarray, n_scales: int = 4) -> np.ndarray:
    """Isotropic Undecimated Wavelet (Starlet) transform.

    Returns coeffs of shape (n_scales + 1, H, W). The last plane is the
    coarse residual; the first n_scales planes are detail bands.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    coeffs = np.empty((n_scales + 1, *image.shape), dtype=np.float64)
    c = image.astype(np.float64)
    for j in range(n_scales):
        c_next = _atrous_convolve2d(c, step=2 ** j)
        coeffs[j] = c - c_next
        c = c_next
    coeffs[-1] = c
    return coeffs


def starlet_inverse(coeffs: np.ndarray) -> np.ndarray:
    """Starlet reconstruction. The transform is a tight frame with the
    trivial reconstruction `sum over all bands`."""
    return coeffs.sum(axis=0)


def starlet_band_stds(image_shape, n_scales: int = 4) -> np.ndarray:
    """Per-band std of starlet coefficients of unit-variance white noise.

    Used to scale per-band thresholds (k * sigma * b_j). Estimated once
    by Monte Carlo for the requested image shape.
    """
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(image_shape)
    w = starlet_forward(noise, n_scales=n_scales)
    return w[:-1].std(axis=(1, 2))


# ----------------------------------------------------------------------
# Soft thresholding
# ----------------------------------------------------------------------
def soft_threshold(x: np.ndarray, thresh) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


# ----------------------------------------------------------------------
# MCA solver
# ----------------------------------------------------------------------
@dataclass
class MCAConfig:
    n_scales: int = 4
    n_iter: int = 60
    step: float = 1.0           # ISTA step; data-fit operator has spectral norm 1.
    lam_point: float = 0.05     # Dirac (point-source) threshold.
    lam_diffuse: float = 0.05   # Starlet threshold (multiplied by per-band std).
    mu_kin: float = 0.1         # Weight of the kinematic-prior term.
    decreasing_thresholds: bool = True  # Anneal thresholds across iterations.
    threshold_floor: float = 0.2        # Final fraction of the initial threshold.
    detail_only_diffuse: bool = True    # Don't threshold the coarse band.
    drop_coarse_in_diffuse: bool = False  # If True, coarse band -> background, not diffuse.
    verbose: bool = False


@dataclass
class MCAResult:
    point: np.ndarray            # (n_ch, H, W)
    diffuse: np.ndarray          # (n_ch, H, W)
    residual: np.ndarray         # (n_ch, H, W)
    flow: np.ndarray             # (n_ch-1, 2, H, W)
    history: list = field(default_factory=list)


def mca_decompose_cube(
    cube: np.ndarray,
    flow: Optional[np.ndarray] = None,
    flow_method: str = "tvl1",
    config: Optional[MCAConfig] = None,
) -> MCAResult:
    """Separate a (n_ch, H, W) cube into point-source + diffuse cubes.

    Model per channel n:  cube[n] = point[n] + diffuse[n] + r[n].
    Sparsity: point[n]      sparse in pixel basis (Dirac).
              diffuse[n]    sparse in starlet detail bands.
    Coupling: ||point[n+1]   - W_n point[n]||^2  (and same for diffuse)
              with W_n the flux-preserving warp along the optical flow
              from channel n to n+1.

    Solved by ISTA with one block-coordinate sweep per iteration.
    """
    cfg = config or MCAConfig()
    if cube.ndim != 3:
        raise ValueError(f"cube must be (n_ch,H,W); got {cube.shape}")
    cube = cube.astype(np.float64)
    n_ch, H, W = cube.shape

    if flow is None:
        flow = estimate_flow_field(cube, method=flow_method)
    if flow.shape != (n_ch - 1, 2, H, W):
        raise ValueError("flow shape does not match cube")

    band_stds = starlet_band_stds((H, W), n_scales=cfg.n_scales)

    point = np.zeros_like(cube)
    diffuse = np.zeros_like(cube)

    history = []
    for it in range(cfg.n_iter):
        # Anneal thresholds: start large to suppress junk, finish at lam_*.
        if cfg.decreasing_thresholds:
            t = it / max(cfg.n_iter - 1, 1)
            scale = (1 - t) * (1.0 / cfg.threshold_floor) + t * 1.0
        else:
            scale = 1.0
        lam_p = cfg.lam_point * scale
        lam_d = cfg.lam_diffuse * scale

        residual = cube - point - diffuse

        # Kinematic-prior gradient for a stack `s` of shape (n_ch,H,W):
        # E_kin = mu * sum_n ||s[n+1] - W_n s[n]||^2.
        # dE/ds[n]   includes -2 mu W_n^T (s[n+1] - W_n s[n])     (forward link)
        # dE/ds[n+1] includes  2 mu     (s[n+1] - W_n s[n])       (backward link)
        def kin_grad(stack):
            g = np.zeros_like(stack)
            for n in range(n_ch - 1):
                w_sn = warp_flux_preserving(stack[n], flow[n])
                diff = stack[n + 1] - w_sn
                g[n + 1] += 2.0 * cfg.mu_kin * diff
                g[n]     -= 2.0 * cfg.mu_kin * warp_adjoint(diff, flow[n])
            return g

        # ---- Point-source block (Dirac dictionary => threshold in pixel basis)
        grad_p = -2.0 * residual + kin_grad(point)
        point_tmp = point - cfg.step * 0.5 * grad_p
        point = soft_threshold(point_tmp, cfg.step * 0.5 * lam_p)
        # Point sources are non-negative emission; clip negatives.
        np.maximum(point, 0.0, out=point)

        # ---- Diffuse block (Starlet => threshold in wavelet domain)
        residual = cube - point - diffuse
        grad_d = -2.0 * residual + kin_grad(diffuse)
        diffuse_tmp = diffuse - cfg.step * 0.5 * grad_d

        for n in range(n_ch):
            w = starlet_forward(diffuse_tmp[n], n_scales=cfg.n_scales)
            for j in range(cfg.n_scales):
                w[j] = soft_threshold(w[j], cfg.step * 0.5 * lam_d * band_stds[j])
            if cfg.drop_coarse_in_diffuse:
                w[-1] = 0.0
            diffuse[n] = starlet_inverse(w)

        if cfg.verbose:
            r = cube - point - diffuse
            history.append(float(np.linalg.norm(r)))
            if it % 10 == 0 or it == cfg.n_iter - 1:
                print(f"[mca] it={it:3d} ||r||={history[-1]:.4f} "
                      f"lam_p={lam_p:.3g} lam_d={lam_d:.3g}")

    residual = cube - point - diffuse
    return MCAResult(point=point, diffuse=diffuse, residual=residual,
                     flow=flow, history=history)


# ----------------------------------------------------------------------
# Multi-source separation: K source cubes, each with its own kinematic
# track, separated by spatial-mask assignment + starlet denoising.
# ----------------------------------------------------------------------
def _bilinear_sample(field: np.ndarray, y: float, x: float) -> float:
    """Sample a 2D field at non-integer (y, x) with edge clamping."""
    H, W = field.shape
    y = np.clip(y, 0, H - 1.001)
    x = np.clip(x, 0, W - 1.001)
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    fy = y - y0; fx = x - x0
    return float((1 - fy) * (1 - fx) * field[y0,     x0]
                 + (1 - fy) * fx       * field[y0,     x0 + 1]
                 + fy       * (1 - fx) * field[y0 + 1, x0]
                 + fy       * fx       * field[y0 + 1, x0 + 1])


def _smooth1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing along the leading axis (per-channel)."""
    if sigma <= 0:
        return arr
    radius = int(np.ceil(3 * sigma))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    n = arr.shape[0]
    pad = np.concatenate([arr[:1].repeat(radius, 0), arr,
                          arr[-1:].repeat(radius, 0)], axis=0)
    out = np.zeros_like(arr)
    for i, w in enumerate(k):
        out += w * pad[i:i + n]
    return out


def detect_source_peaks(cube: np.ndarray, K=None, min_distance=5,
                         detail_scales=(0, 1, 2), threshold_rel=0.02,
                         n_scales=4):
    """Detect peaks in a high-pass starlet view of the cube's moment-0.

    Compact sources embedded in a brighter galaxy's halo are *not* local
    maxima of the raw moment-0 — they sit on a smooth gradient. Summing
    only the fine starlet detail bands removes the halo and exposes the
    embedded peaks. `K` caps the number of returned peaks; if None, all
    peaks above `threshold_rel` are returned.
    """
    from skimage.feature import peak_local_max

    m0 = cube.sum(0)
    coeffs = starlet_forward(m0, n_scales=n_scales)
    high = sum(coeffs[j] for j in detail_scales)
    high = np.maximum(high, 0.0)
    peaks = peak_local_max(
        high,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=K if K is not None else np.inf,
    )
    return peaks


def propagate_centroids(initial_peaks, flow, n_ref):
    """Integrate the optical-flow field forward and backward from a
    reference channel to give each source its trajectory.

    initial_peaks : (K, 2) row,col at channel n_ref.
    flow          : (n_ch-1, 2, H, W); flow[n] maps n -> n+1.
    """
    n_pairs = flow.shape[0]
    n_ch = n_pairs + 1
    K = len(initial_peaks)
    traj = np.zeros((n_ch, K, 2), dtype=np.float64)
    traj[n_ref] = initial_peaks.astype(np.float64)
    for n in range(n_ref, n_ch - 1):
        for k in range(K):
            cy, cx = traj[n, k]
            v = _bilinear_sample(flow[n, 0], cy, cx)
            u = _bilinear_sample(flow[n, 1], cy, cx)
            traj[n + 1, k] = (cy + v, cx + u)
    for n in range(n_ref, 0, -1):
        for k in range(K):
            cy, cx = traj[n, k]
            # Inverse step approximated by sampling forward flow at
            # current location (small-displacement regime).
            v = _bilinear_sample(flow[n - 1, 0], cy, cx)
            u = _bilinear_sample(flow[n - 1, 1], cy, cx)
            traj[n - 1, k] = (cy - v, cx - u)
    return traj


def _gaussian_masks(traj, mask_sigma, H, W):
    """Per-channel, per-source soft mask at each centroid.

    Returns (K, n_ch, H, W) raw (un-normalized) Gaussians.
    """
    n_ch, K, _ = traj.shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    out = np.zeros((K, n_ch, H, W), dtype=np.float32)
    inv2s2 = 1.0 / (2.0 * mask_sigma ** 2)
    for n in range(n_ch):
        for k in range(K):
            cy, cx = traj[n, k]
            out[k, n] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) * inv2s2)
    return out


def _flux_centroid(img, fallback):
    """Flux-weighted centroid; falls back to `fallback` if total flux is
    negligible (e.g. an off-line channel)."""
    total = float(img.sum())
    if total < 1e-12 or not np.isfinite(total):
        return fallback
    H, W = img.shape
    yy, xx = np.mgrid[:H, :W]
    return (float((yy * img).sum() / total),
            float((xx * img).sum() / total))


@dataclass
class MultiSourceConfig:
    K: Optional[int] = None              # None => auto-detect.
    mask_sigma: float = 6.0              # Default Gaussian-template σ (px).
    per_source_sigmas: Optional[np.ndarray] = None  # Per-source σ override.
    n_iter: int = 5                      # NNLS + residual iterations.
    starlet_denoise: bool = False
    n_scales: int = 4
    denoise_k_sigma: float = 3.0
    centroid_smooth_sigma: float = 0.0   # Spectral smoothing of trajectories.
    min_peak_distance: int = 5
    peak_threshold_rel: float = 0.02
    detail_scales: tuple = (0, 1, 2)
    distribute_residual: bool = True     # Spread the NNLS residual via masks.
    nonneg: bool = True
    verbose: bool = False


@dataclass
class MultiSourceResult:
    sources: np.ndarray        # (K, n_ch, H, W)
    trajectories: np.ndarray   # (n_ch, K, 2) in (row, col)
    spectra: np.ndarray        # (K, n_ch) per-source spectrum
    sigmas: np.ndarray         # (K,) per-source mask sigma in px
    flow: np.ndarray           # (n_ch-1, 2, H, W)
    peaks: np.ndarray          # (K, 2) detected in moment-0
    residual: np.ndarray       # (n_ch, H, W)


def estimate_source_sigmas(cube, peaks, max_sigma_px=20,
                            detail_scales=(0, 1, 2), n_scales=4):
    """Estimate the spatial σ of each source from a high-pass moment-0
    image. Using high-pass starlet bands rather than raw moment-0
    suppresses the broad halo of any dominant neighbour, so the σ of a
    compact embedded source isn't inflated by the brighter galaxy it
    sits inside.

    Caps σ at half the distance to the nearest neighbouring peak so the
    Gaussian template never bleeds into another source's territory.
    """
    m0 = cube.sum(0)
    coeffs = starlet_forward(m0, n_scales=n_scales)
    high = sum(coeffs[j] for j in detail_scales)
    high = np.maximum(high, 0.0)
    H, W = m0.shape
    K = len(peaks)
    sigmas = np.zeros(K, dtype=np.float64)
    for k in range(K):
        py, px = peaks[k]
        if K > 1:
            dists = np.linalg.norm(peaks - peaks[k], axis=1)
            dists[k] = np.inf
            cap = 0.5 * dists.min()
        else:
            cap = max_sigma_px
        cap = min(cap, max_sigma_px)
        r = int(min(cap, max_sigma_px))
        y0 = max(0, py - r); y1 = min(H, py + r + 1)
        x0 = max(0, px - r); x1 = min(W, px + r + 1)
        patch = high[y0:y1, x0:x1]
        if patch.size == 0 or patch.max() <= 0:
            sigmas[k] = max(2.0, 0.4 * cap)
            continue
        # Threshold the patch above a fraction of its peak so only
        # the source's own coherent core feeds the second moment.
        thr = 0.2 * patch.max()
        weights = np.where(patch >= thr, patch - thr, 0.0)
        wsum = weights.sum() + 1e-12
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        cy, cx = py - y0, px - x0
        var = ((yy - cy) ** 2 + (xx - cx) ** 2) * weights
        sigma_est = float(np.sqrt(var.sum() / wsum / 2))
        sigmas[k] = float(np.clip(sigma_est, 1.5, cap))
    return sigmas


def _gaussian_template(cy, cx, sigma, H, W):
    """Unit-amplitude (peak=1) 2D Gaussian template."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def separate_sources_kinematic(
    cube: np.ndarray,
    flow: Optional[np.ndarray] = None,
    flow_method: str = "tvl1",
    config: Optional[MultiSourceConfig] = None,
) -> MultiSourceResult:
    """Separate a (n_ch, H, W) cube into K per-source cubes that each
    follow their own kinematic track.

    Pipeline:
      1. peak-detect K sources in the smoothed moment-0
      2. estimate dense optical flow between successive channels
      3. propagate each peak forward/backward through the cube along
         the flow → per-source trajectory in (channel, row, col)
      4. build soft Gaussian masks at the per-channel centroids;
         partition each pixel's flux among sources by mask weight
      5. (optional) starlet-soft-threshold each source's slice
      6. re-estimate centroids from flux-weighted centers and smooth
         the trajectory along the spectral axis; iterate
    """
    cfg = config or MultiSourceConfig()
    if cube.ndim != 3:
        raise ValueError("cube must be (n_ch, H, W)")
    cube = cube.astype(np.float64)
    n_ch, H, W = cube.shape

    if flow is None:
        flow = estimate_flow_field(cube, method=flow_method)

    peaks = detect_source_peaks(
        cube, K=cfg.K,
        min_distance=cfg.min_peak_distance,
        threshold_rel=cfg.peak_threshold_rel,
    )
    if len(peaks) == 0:
        raise RuntimeError("no source peaks detected; lower peak_threshold_rel")
    K = len(peaks)
    if cfg.verbose:
        print(f"[multisource] detected K={K} peaks at\n{peaks}")

    # Reference = brightest channel; propagate trajectories along flow.
    n_ref = int(np.argmax((cube ** 2).reshape(n_ch, -1).sum(1)))
    traj = propagate_centroids(peaks, flow, n_ref=n_ref)
    if cfg.centroid_smooth_sigma > 0:
        traj = _smooth1d(traj, cfg.centroid_smooth_sigma)

    # Per-source σ, scaled to each source's own extent.
    if cfg.per_source_sigmas is not None:
        sigmas = np.asarray(cfg.per_source_sigmas, dtype=np.float64)
        if sigmas.shape != (K,):
            raise ValueError(f"per_source_sigmas must have shape ({K},)")
    else:
        sigmas = estimate_source_sigmas(cube, peaks)
        # Floor at the user-supplied default so we never go below it.
        sigmas = np.maximum(sigmas, cfg.mask_sigma * 0.5)
    if cfg.verbose:
        print(f"[multisource] per-source σ (px) = {sigmas}")

    # Per-channel non-negative least-squares fit:
    #   cube[n] ≈ Σ_k a_k(n) * T_k[n]
    # where T_k[n] is a unit-peak Gaussian at traj[n, k] with σ = sigmas[k].
    # NNLS gives a non-negative spectrum a_k(n) per source. Sources with
    # no real flux at a channel get a_k(n)=0 — this is what stops the
    # central's halo getting absorbed into nearby satellites.
    from scipy.optimize import nnls

    spectra = np.zeros((K, n_ch), dtype=np.float64)
    sources = np.zeros((K, n_ch, H, W), dtype=np.float64)
    templates = np.zeros((K, n_ch, H, W), dtype=np.float32)
    for n in range(n_ch):
        for k in range(K):
            templates[k, n] = _gaussian_template(traj[n, k, 0], traj[n, k, 1],
                                                 sigmas[k], H, W)

    for it in range(cfg.n_iter):
        for n in range(n_ch):
            T = templates[:, n].reshape(K, -1).T            # (H*W, K)
            y = cube[n].reshape(-1)
            try:
                a, _ = nnls(T, y, maxiter=200 * K)
            except Exception:
                a = np.zeros(K)
            spectra[:, n] = a
            for k in range(K):
                sources[k, n] = a[k] * templates[k, n]

        # Distribute the per-channel NNLS residual onto whichever source
        # explains it best at each pixel. Without this step the cube's
        # non-Gaussian halo (mostly the bright extended source) lands in
        # `residual` rather than in any source. With it, each pixel's
        # leftover flux is shared by mask weight × spectrum amplitude.
        if cfg.distribute_residual:
            for n in range(n_ch):
                model = sources[:, n].sum(0)
                r = cube[n] - model
                weights = templates[:, n] * spectra[:, n][:, None, None]
                wsum = weights.sum(0) + 1e-12
                for k in range(K):
                    sources[k, n] += (weights[k] / wsum) * r
            if cfg.nonneg:
                np.maximum(sources, 0.0, out=sources)

        # Optional per-source per-channel starlet thresholding.
        if cfg.starlet_denoise:
            for k in range(K):
                for n in range(n_ch):
                    sl = sources[k, n]
                    if not np.any(sl):
                        continue
                    w = starlet_forward(sl, n_scales=cfg.n_scales)
                    for j in range(cfg.n_scales):
                        sj = float(np.std(w[j]))
                        if sj > 0:
                            w[j] = soft_threshold(
                                w[j], cfg.denoise_k_sigma * sj * 0.1)
                    sources[k, n] = starlet_inverse(w)
            if cfg.nonneg:
                np.maximum(sources, 0.0, out=sources)

        if cfg.verbose:
            recon = sources.sum(0)
            r = cube - recon
            print(f"[multisource] it={it:2d} ||r||={np.linalg.norm(r):.4f} "
                  f"recon flux={recon.sum():.3e}")

    residual = cube - sources.sum(0)
    return MultiSourceResult(sources=sources, trajectories=traj,
                             spectra=spectra, sigmas=sigmas,
                             flow=flow, peaks=peaks, residual=residual)


# ----------------------------------------------------------------------
# Smoke test on a synthetic cube
# ----------------------------------------------------------------------
def _synthetic_cube(n_ch=12, H=64, W=64, seed=0):
    rng = np.random.default_rng(seed)
    cube = np.zeros((n_ch, H, W), dtype=np.float64)
    yy, xx = np.mgrid[:H, :W]

    # Diffuse galaxy: 2D Gaussian drifting linearly across channels.
    for n in range(n_ch):
        cy = 20 + 0.4 * n
        cx = 20 + 0.6 * n
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2))
        cube[n] += 1.0 * g

    # Two point sources, each drifting on a different track.
    for cy0, cx0, vy, vx, amp in [(45, 15, -0.3, 0.5, 5.0),
                                  (50, 50,  0.2, -0.4, 3.5)]:
        for n in range(n_ch):
            cy = cy0 + vy * n
            cx = cx0 + vx * n
            r0, c0 = int(round(cy)), int(round(cx))
            if 0 <= r0 < H and 0 <= c0 < W:
                cube[n, r0, c0] += amp

    cube += 0.02 * rng.standard_normal(cube.shape)
    return cube


if __name__ == "__main__":
    cube = _synthetic_cube()
    print("cube:", cube.shape, "total flux:", cube.sum())

    flow = estimate_flow_field(cube, method="tvl1")
    print("flow:", flow.shape, "median |v|:", float(np.median(np.abs(flow))))

    res = mca_decompose_cube(
        cube, flow=flow,
        config=MCAConfig(n_iter=40, lam_point=0.4, lam_diffuse=2.0,
                         mu_kin=0.1, verbose=True),
    )
    print("point flux  :", res.point.sum())
    print("diffuse flux:", res.diffuse.sum())
    print("residual rms:", float(np.sqrt(np.mean(res.residual ** 2))))
