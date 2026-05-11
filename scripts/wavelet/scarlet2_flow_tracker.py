import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scarlet2 as sc
from scipy.ndimage import center_of_mass, gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.registration import optical_flow_ilk, optical_flow_tvl1


def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("scarlet2_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(out_dir / "processing.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def robust_scale(image: np.ndarray) -> float:
    med = np.median(image)
    mad = np.median(np.abs(image - med)) + 1e-8
    return 1.4826 * mad


def normalize01(image: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(image, 1.0))
    hi = float(np.percentile(image, 99.0))
    return np.clip((image - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def detect_seed_centers(image: np.ndarray, n_peaks: int, min_distance: int = 3) -> list[tuple[float, float]]:
    smooth = gaussian_filter(image, sigma=1.0)
    sigma = robust_scale(smooth)
    thresh = np.median(smooth) + 2.0 * sigma

    peaks = peak_local_max(
        smooth,
        min_distance=min_distance,
        threshold_abs=thresh,
        num_peaks=max(n_peaks, 1),
        exclude_border=False,
    )

    if len(peaks) == 0:
        y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
        return [(float(y), float(x))]

    return [(float(y), float(x)) for y, x in peaks]


def detect_seed_centers_wavelet(
    image: np.ndarray,
    n_peaks: int,
    scales: int = 4,
    k_sigma: float = 3.0,
    use_scale: int = 2,
    min_area: int = 10,
    min_separation: int = 0,
    thresh: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Reproduce scarlet-style wavelet detection logic using scarlet2 primitives:
    1) starlet transform
    2) multiscale support mask via sigma thresholding on each scale
    3) detect positive footprints on chosen scale
    4) use footprint peaks as source seeds
    """
    img = np.asarray(image, dtype=np.float32)
    sigma = robust_scale(img)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(img) + 1e-8)

    coeffs = np.asarray(sc.wavelets.starlet_transform(img, scales=scales), dtype=np.float32)
    support = np.zeros_like(coeffs, dtype=np.float32)
    for si in range(coeffs.shape[0] - 1):
        # Approximate multiresolution support: keep statistically significant coeffs per scale.
        support[si] = (np.abs(coeffs[si]) > (k_sigma * sigma)).astype(np.float32)
    support[-1] = 1.0

    detect = support * coeffs
    detect[detect < 0] = 0

    # Notebook uses 1-based scale notion; convert to 0-based index.
    scale_idx = int(np.clip(use_scale - 1, 0, detect.shape[0] - 1))
    plane = detect[scale_idx]
    binary = plane > max(thresh, 0.0)

    labeled = label(binary)
    regs = [r for r in regionprops(labeled, intensity_image=plane) if r.area >= min_area]

    centers: list[tuple[float, float]] = []
    for reg in regs:
        y0, x0, y1, x1 = reg.bbox
        patch = plane[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        centers.append((float(y0 + py), float(x0 + px)))

    if len(centers) == 0:
        return detect_seed_centers(image, n_peaks=n_peaks, min_distance=3)

    # Deduplicate and rank by underlying channel intensity.
    uniq = list({(int(round(y)), int(round(x))) for y, x in centers})
    uniq = [
        (float(np.clip(y, 0, img.shape[0] - 1)), float(np.clip(x, 0, img.shape[1] - 1)))
        for y, x in uniq
    ]
    uniq.sort(key=lambda yx: img[int(yx[0]), int(yx[1])], reverse=True)

    return uniq[: max(1, n_peaks)]


def local_positive_morphology(image: np.ndarray, cy: float, cx: float, size: int = 13) -> np.ndarray:
    half = size // 2
    h, w = image.shape
    yi = int(np.clip(round(cy), 0, h - 1))
    xi = int(np.clip(round(cx), 0, w - 1))

    y0 = max(0, yi - half)
    y1 = min(h, yi + half + 1)
    x0 = max(0, xi - half)
    x1 = min(w, xi + half + 1)

    patch = np.clip(image[y0:y1, x0:x1], 0.0, None).astype(np.float32)
    if patch.size == 0 or patch.sum() <= 0:
        patch = np.ones((max(3, y1 - y0), max(3, x1 - x0)), dtype=np.float32)

    patch /= patch.sum() + 1e-8
    return patch


def component_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)

    aa = aa - aa.mean()
    bb = bb - bb.mean()
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0

    return float(np.dot(aa.ravel(), bb.ravel()) / (na * nb))


def compute_flow(prev_img: np.ndarray, curr_img: np.ndarray, fast: bool = False) -> np.ndarray:
    p = normalize01(prev_img)
    c = normalize01(curr_img)

    if fast:
        flow = optical_flow_ilk(p, c)
    else:
        flow = optical_flow_tvl1(p, c, num_warp=8, num_iter=12)

    # skimage returns (2, H, W): [v_y, v_x]
    return flow.astype(np.float32)


def flow_predict(center_yx: tuple[float, float], flow: np.ndarray) -> tuple[float, float]:
    y, x = center_yx
    h, w = flow.shape[1:]
    yi = int(np.clip(round(y), 0, h - 1))
    xi = int(np.clip(round(x), 0, w - 1))
    vy = float(flow[0, yi, xi])
    vx = float(flow[1, yi, xi])
    return y + vy, x + vx


@dataclass
class ComponentObs:
    channel: int
    comp_index: int
    center_y: float
    center_x: float
    flux: float
    model: np.ndarray


@dataclass
class Track:
    track_id: int
    parent_id: int | None
    observations: list[ComponentObs]

    def last(self) -> ComponentObs:
        return self.observations[-1]

    @property
    def total_flux(self) -> float:
        return float(sum(o.flux for o in self.observations))



def fit_scarlet_channel(channel_image: np.ndarray, centers: list[tuple[float, float]], max_iter: int = 100) -> list[ComponentObs]:
    h, w = channel_image.shape

    data2d = channel_image.astype(np.float32)
    data2d = data2d - float(np.percentile(data2d, 1.0))
    data2d = np.clip(data2d, 0.0, None)
    data = data2d[None, :, :]
    sigma = robust_scale(channel_image)
    weights = np.full_like(data, 1.0 / (sigma * sigma + 1e-8), dtype=np.float32)

    obs = sc.Observation(data=data, weights=weights)
    frame = sc.Frame(sc.Box(data.shape))

    components: list[ComponentObs] = []

    with sc.Scene(frame) as scene:
        for cy, cx in centers:
            try:
                spectrum, morphology = sc.init.from_gaussian_moments(
                    obs,
                    center=np.array([cy, cx]),
                    min_snr=1.5,
                    min_corr=0.8,
                    min_value=1e-6,
                    max_value=1 - 1e-6,
                )
            except Exception:
                morphology = local_positive_morphology(data2d, cy, cx, size=13)
                spectrum = np.array([max(float(data2d[int(np.clip(round(cy), 0, h - 1)), int(np.clip(round(cx), 0, w - 1))]), 1e-3)], dtype=np.float32)

            sc.Source(center=np.array([cy, cx]), spectrum=spectrum, morphology=morphology)

        for idx, src in enumerate(scene.sources):
            model3d = np.array(scene.evaluate_source(src), dtype=np.float32)
            model2d = model3d[0]
            model2d = np.clip(model2d, 0.0, None)

            flux = float(model2d.sum())
            if flux <= 0:
                continue

            cy, cx = center_of_mass(model2d)
            if not np.isfinite(cy) or not np.isfinite(cx):
                cy, cx = centers[min(idx, len(centers) - 1)]

            components.append(
                ComponentObs(
                    channel=-1,
                    comp_index=idx,
                    center_y=float(cy),
                    center_x=float(cx),
                    flux=flux,
                    model=model2d,
                )
            )

    return components


def build_cost_matrix(
    active_tracks: list[Track],
    candidates: list[ComponentObs],
    flow: np.ndarray,
    max_motion_px: float,
    w_dist: float,
    w_shape: float,
    w_flux: float,
) -> np.ndarray:
    n_t = len(active_tracks)
    n_c = len(candidates)
    cost = np.full((n_t, n_c), 1e6, dtype=np.float32)

    for i, tr in enumerate(active_tracks):
        last = tr.last()
        py, px = flow_predict((last.center_y, last.center_x), flow)

        for j, c in enumerate(candidates):
            dist = np.hypot(c.center_y - py, c.center_x - px)
            if dist > max_motion_px:
                continue

            dist_cost = dist / (max_motion_px + 1e-8)
            shape_sim = component_similarity(last.model, c.model)
            shape_cost = 1.0 - max(-1.0, min(1.0, shape_sim))
            flux_cost = abs(np.log((c.flux + 1e-8) / (last.flux + 1e-8)))

            cost[i, j] = w_dist * dist_cost + w_shape * shape_cost + w_flux * flux_cost

    return cost


def track_components(
    all_components: list[list[ComponentObs]],
    flow_fields: list[np.ndarray],
    max_motion_px: float,
    split_radius_px: float,
    split_flux_frac: float,
) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    next_id = 0

    # Initialize from first non-empty channel
    start_channel = None
    for s, comps in enumerate(all_components):
        if len(comps) > 0:
            start_channel = s
            break

    if start_channel is None:
        return tracks

    for c in all_components[start_channel]:
        c.channel = start_channel
        tracks[next_id] = Track(track_id=next_id, parent_id=None, observations=[c])
        next_id += 1

    for s in range(start_channel + 1, len(all_components)):
        candidates = all_components[s]
        for k in range(len(candidates)):
            candidates[k].channel = s

        if len(candidates) == 0:
            continue

        active_ids = [tid for tid, tr in tracks.items() if tr.last().channel == s - 1]
        active_tracks = [tracks[tid] for tid in active_ids]

        if len(active_tracks) == 0:
            for c in candidates:
                tracks[next_id] = Track(track_id=next_id, parent_id=None, observations=[c])
                next_id += 1
            continue

        cost = build_cost_matrix(
            active_tracks,
            candidates,
            flow_fields[s - 1],
            max_motion_px=max_motion_px,
            w_dist=0.45,
            w_shape=0.40,
            w_flux=0.15,
        )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_candidates = set()

        for r, cidx in zip(row_ind, col_ind, strict=False):
            if cost[r, cidx] >= 1e5:
                continue
            tid = active_ids[r]
            tracks[tid].observations.append(candidates[cidx])
            matched_tracks.add(tid)
            matched_candidates.add(cidx)

        # Split handling: unmatched candidates near already matched parent prediction become child tracks
        matched_parent_ids = list(matched_tracks)
        for cidx, comp in enumerate(candidates):
            if cidx in matched_candidates:
                continue

            best_parent = None
            best_dist = 1e9
            for pid in matched_parent_ids:
                parent_last = tracks[pid].last()
                py, px = parent_last.center_y, parent_last.center_x
                d = np.hypot(comp.center_y - py, comp.center_x - px)
                if d < best_dist:
                    best_dist = d
                    best_parent = pid

            if (
                best_parent is not None
                and best_dist <= split_radius_px
                and comp.flux >= split_flux_frac * tracks[best_parent].last().flux
            ):
                tracks[next_id] = Track(track_id=next_id, parent_id=best_parent, observations=[comp])
                next_id += 1
            else:
                tracks[next_id] = Track(track_id=next_id, parent_id=None, observations=[comp])
                next_id += 1

    return tracks


def aggregate_to_roots(tracks: dict[int, Track]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for tid in tracks:
        children[tid] = []

    for tid, tr in tracks.items():
        if tr.parent_id is not None and tr.parent_id in children:
            children[tr.parent_id].append(tid)

    root_map: dict[int, list[int]] = {}
    for tid, tr in tracks.items():
        if tr.parent_id is None:
            root_map[tid] = [tid]

    changed = True
    while changed:
        changed = False
        for root, members in list(root_map.items()):
            grown = list(members)
            for m in members:
                grown.extend(children.get(m, []))
            grown_unique = sorted(set(grown))
            if len(grown_unique) != len(members):
                root_map[root] = grown_unique
                changed = True

    return root_map


def build_source_cubes(
    tracks: dict[int, Track],
    cube_shape: tuple[int, int, int],
    n_sources: int,
    merge_splits: bool,
) -> np.ndarray:
    s, h, w = cube_shape

    if merge_splits:
        roots = aggregate_to_roots(tracks)
        root_flux = {}
        for rid, members in roots.items():
            root_flux[rid] = sum(tracks[mid].total_flux for mid in members)

        selected_roots = [rid for rid, _ in sorted(root_flux.items(), key=lambda kv: kv[1], reverse=True)[:n_sources]]

        source_cubes = np.zeros((n_sources, s, h, w), dtype=np.float32)
        for out_idx, rid in enumerate(selected_roots):
            members = roots[rid]
            for tid in members:
                for obs in tracks[tid].observations:
                    source_cubes[out_idx, obs.channel] += obs.model
        return source_cubes

    ranked_tracks = sorted(tracks.values(), key=lambda t: t.total_flux, reverse=True)[:n_sources]
    source_cubes = np.zeros((n_sources, s, h, w), dtype=np.float32)
    for out_idx, tr in enumerate(ranked_tracks):
        for obs in tr.observations:
            source_cubes[out_idx, obs.channel] += obs.model
    return source_cubes


def main() -> None:
    ap = argparse.ArgumentParser(description="scarlet2 + optical flow tracker for non-stationary cube sources")
    ap.add_argument("--cube", required=True, help="Input HDF5 cube path with dataset name 'cube'")
    ap.add_argument("--out", default="experiments/mca_v1/scarlet2_flow", help="Output directory")
    ap.add_argument("--n_sources", type=int, default=4, help="Number of output sources")
    ap.add_argument("--seeds_per_channel", type=int, default=8, help="Initial scarlet seeds per channel")
    ap.add_argument("--use_wavelet_detection", action="store_true", help="Use scarlet2 wavelet+footprint-like detection for per-channel seeds")
    ap.add_argument("--wavelet_scales", type=int, default=4, help="Number of starlet scales for scarlet1 detection")
    ap.add_argument("--wavelet_k", type=float, default=3.0, help="K threshold for scarlet2 multiscale support")
    ap.add_argument("--wavelet_scale", type=int, default=2, help="1-based wavelet scale to extract footprints from")
    ap.add_argument("--footprint_min_area", type=int, default=10, help="Minimum footprint area for scarlet2 wavelet detection")
    ap.add_argument("--fit_iter", type=int, default=120, help="Scarlet fit iterations per channel")
    ap.add_argument("--fast", action="store_true", help="Use ILK flow (faster) instead of TV-L1")
    ap.add_argument("--max_motion_px", type=float, default=8.0, help="Max expected per-channel motion")
    ap.add_argument("--split_radius_px", type=float, default=4.0, help="Radius for split child association")
    ap.add_argument("--split_flux_frac", type=float, default=0.15, help="Min child flux relative to parent")
    ap.add_argument("--merge_splits", action="store_true", help="Merge split child tracks into root source")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    log.info("Loading cube: %s", args.cube)
    with h5py.File(args.cube, "r") as f:
        cube = f["cube"][...].astype(np.float32)

    s, h, w = cube.shape
    log.info("Cube shape: (S=%d, H=%d, W=%d)", s, h, w)
    if args.use_wavelet_detection:
        log.info("Seed detection: scarlet2 wavelet+footprint-like extraction (scale=%d)", args.wavelet_scale)

    # 1) Decompose each spectral channel with scarlet2
    all_components: list[list[ComponentObs]] = []
    for ch in range(s):
        img = cube[ch]
        if args.use_wavelet_detection:
            centers = detect_seed_centers_wavelet(
                img,
                n_peaks=args.seeds_per_channel,
                scales=args.wavelet_scales,
                k_sigma=args.wavelet_k,
                use_scale=args.wavelet_scale,
                min_area=args.footprint_min_area,
                min_separation=0,
                thresh=0.0,
            )
        else:
            centers = detect_seed_centers(img, n_peaks=args.seeds_per_channel, min_distance=3)
        try:
            comps = fit_scarlet_channel(img, centers=centers, max_iter=args.fit_iter)
        except Exception as exc:
            log.warning("Channel %d scarlet2 fit failed: %s", ch, exc)
            comps = []

        # Keep only strong components for stability
        if comps:
            fluxes = np.array([c.flux for c in comps], dtype=np.float32)
            fcut = np.percentile(fluxes, 20)
            comps = [c for c in comps if c.flux >= float(fcut)]

        all_components.append(comps)

        if ch % 8 == 0 or ch == s - 1:
            log.info("Channel %d: %d components", ch, len(comps))

    # 2) Optical flow between consecutive channels
    flow_fields: list[np.ndarray] = []
    for ch in range(s - 1):
        flow_fields.append(compute_flow(cube[ch], cube[ch + 1], fast=args.fast))

    # 3) Track identities over channels (+ split lineage)
    if len(all_components) == 0 or not any(len(c) > 0 for c in all_components):
        raise RuntimeError("No components detected in any channel; increase --seeds_per_channel or relax thresholds")

    tracks = track_components(
        all_components=all_components,
        flow_fields=flow_fields,
        max_motion_px=args.max_motion_px,
        split_radius_px=args.split_radius_px,
        split_flux_frac=args.split_flux_frac,
    )

    # 4) Build output source cubes
    source_cubes = build_source_cubes(
        tracks=tracks,
        cube_shape=cube.shape,
        n_sources=args.n_sources,
        merge_splits=args.merge_splits,
    )

    out_h5 = out_dir / "decomposed_sources.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("source_cubes", data=source_cubes, compression="gzip")

    # Save lineage summary
    lines = ["track_id,parent_id,n_obs,total_flux,channels"]
    for tid, tr in sorted(tracks.items(), key=lambda kv: kv[0]):
        ch_list = [str(o.channel) for o in tr.observations]
        lines.append(
            f"{tid},{tr.parent_id if tr.parent_id is not None else -1},{len(tr.observations)},{tr.total_flux:.6g}," + "|".join(ch_list)
        )

    (out_dir / "tracks.csv").write_text("\n".join(lines) + "\n")

    log.info("Done. Saved: %s", out_h5)
    log.info("Tracks: %d", len(tracks))


if __name__ == "__main__":
    main()
