import argparse
import h5py
import numpy as np
import logging
import sys
from pathlib import Path
import cv2
import pywt
from tqdm import tqdm
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mca_decomposer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    
    # File handler
    fh = logging.FileHandler(out_dir / "processing.log", mode="w")
    fh.setFormatter(fmt)
    # Stream handler (console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def load_cube(path: Path, logger):
    logger.info(f"Loading data from {path}")
    with h5py.File(path, "r") as f:
        cube = f["cube"][...].astype(np.float32)
        logger.info(f"Cube loaded. Shape: {cube.shape}")
    return cube

def mca_threshold(slice_data, sigma_clip=2.0):
    """
    Separates compact sources from diffuse background using Wavelet Sparsity.
    Changed 'starlet' to 'db1' (Haar) for compatibility.
    """
    # Use 'db1' or 'sym2' which are standard in PyWavelets
    wavelet_name = 'db1' 
    
    # Perform 2D multi-level decomposition
    coeffs = pywt.wavedec2(slice_data, wavelet_name, level=3)
    
    std = np.std(slice_data)
    thresh = std * sigma_clip
    
    # Thresholding detail coefficients (the sparse parts)
    # coeffs[0] is the approximation (low-frequency), we keep it
    new_coeffs = [coeffs[0]] 
    for level in coeffs[1:]:
        # Apply threshold to (horizontal, vertical, diagonal) tuples
        new_coeffs.append(tuple(pywt.threshold(c, thresh, mode='hard') for c in level))
    
    # Reconstruct the "clean" image
    clean = pywt.waverec2(new_coeffs, wavelet_name)
    
    # Ensure the output matches the original shape and is non-negative
    clean = clean[:slice_data.shape[0], :slice_data.shape[1]]
    return np.clip(clean, 0, None)

def compute_flow_field(prev_slice, curr_slice, use_fast=False):
    """
    Compute 2D optical flow between consecutive spectral channels.
    
    Args:
        prev_slice: Previous spectral channel (2D array)
        curr_slice: Current spectral channel (2D array)
        use_fast: If True, use ILK (faster); else use TV-L1 (more accurate)
    
    Returns:
        flow: 2D optical flow field (H, W, 2)
    """
    # Normalize to 0-1 range for flow computation
    prev_norm = (prev_slice - prev_slice.min()) / (prev_slice.max() - prev_slice.min() + 1e-8)
    curr_norm = (curr_slice - curr_slice.min()) / (curr_slice.max() - curr_slice.min() + 1e-8)
    
    if use_fast:
        flow = optical_flow_ilk(prev_norm, curr_norm)
    else:
        flow = optical_flow_tvl1(prev_norm, curr_norm, num_warp=10, num_iter=15)
    
    return flow

def warp_mask_by_flow(mask, flow, dilation_radius=2):
    """
    Warp a mask to predicted location using optical flow.
    
    Args:
        mask: Binary mask (H, W)
        flow: Optical flow field (2, H, W) from optical_flow_tvl1/ilk
        dilation_radius: Expand warped region to account for flow uncertainty
    
    Returns:
        warped_mask: Warped binary mask
    """
    h, w = mask.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flow is in (2, H, W) format: flow[0] = u (x), flow[1] = v (y)
    # Transpose to (H, W, 2) for easier indexing
    flow_xy = np.stack([flow[0], flow[1]], axis=-1)
    
    # Apply flow displacement
    x_new = (x + flow_xy[..., 0]).astype(np.float32)
    y_new = (y + flow_xy[..., 1]).astype(np.float32)
    
    # Remap mask to new positions
    warped = cv2.remap(mask.astype(np.float32), x_new, y_new, 
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Binarize and dilate to account for flow uncertainty
    binary = warped > 0.5
    if dilation_radius > 0:
        binary = binary_dilation(binary, disk(dilation_radius))
    
    return binary

def detect_and_match_sources(curr_slice, prev_masks, flow, n_sources, logger):
    """
    Detect sources in current slice and match to previous locations using flow.
    Uses overlap-based optimal assignment to prevent greedy matching errors.
    
    Args:
        curr_slice: Current spectral channel (2D array)
        prev_masks: List of previous masks (or None)
        flow: Optical flow field from previous to current channel
        n_sources: Number of sources to track
        logger: Logger instance
    
    Returns:
        matched_masks: List of matched masks for current channel
    """
    # Detect compact sources with MCA
    clean = mca_threshold(curr_slice, sigma_clip=1.0)
    
    # Detect all candidate regions
    threshold = np.clip(clean.max() * 0.1, 0.01, None)
    binary = clean > threshold
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=curr_slice)
    
    if not regions:
        # No regions detected, keep previous masks if available
        return prev_masks
    
    # Compute overlap matrix: (n_sources, n_regions)
    overlap_matrix = np.zeros((n_sources, len(regions)))
    
    for i, prev_mask in enumerate(prev_masks):
        if prev_mask is not None:
            # Predict where source i should be
            pred_mask = warp_mask_by_flow(prev_mask, flow, dilation_radius=1)
            
            # Compute overlap with all detected regions
            for j, region in enumerate(regions):
                region_mask = labeled == region.label
                overlap = np.sum(region_mask & pred_mask)
                overlap_matrix[i, j] = overlap
    
    # Greedy matching: assign highest-overlap pairs first
    # Create list of (overlap, source_idx, region_idx)
    pairs = []
    for i in range(n_sources):
        for j in range(len(regions)):
            if overlap_matrix[i, j] > 0:
                pairs.append((overlap_matrix[i, j], i, j))
    
    pairs.sort(reverse=True)  # Sort by overlap descending
    
    matched_masks = [None] * n_sources
    assigned_sources = set()
    assigned_regions = set()
    
    # Assign highest-overlap pairs, avoiding conflicts
    for overlap, i, j in pairs:
        if i not in assigned_sources and j not in assigned_regions:
            matched_masks[i] = labeled == regions[j].label
            assigned_sources.add(i)
            assigned_regions.add(j)
    
    # Birth new sources from unassigned, high-intensity regions
    unassigned_regions = [j for j in range(len(regions)) if j not in assigned_regions]
    unassigned_sources = [i for i in range(n_sources) if i not in assigned_sources]
    
    # Sort unassigned regions by intensity (descending)
    region_intensities = [regions[j].intensity_mean for j in unassigned_regions]
    sorted_indices = np.argsort(region_intensities)[::-1]
    
    for idx, region_j in enumerate([unassigned_regions[i] for i in sorted_indices]):
        if idx < len(unassigned_sources):
            source_i = unassigned_sources[idx]
            matched_masks[source_i] = labeled == regions[region_j].label
    
    return matched_masks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="/Users/arnablahiry/repos/GalCubeCraft-SourceID/experiments/sep_v1/data/cube_1006.h5")
    ap.add_argument("--out", default="experiments/mca_v1")
    ap.add_argument("--n_sources", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="Use ILK flow (faster) instead of TV-L1")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logging(out_path)
    log.info("Starting MCA Decomposition Pipeline")

    try:
        full_cube = load_cube(Path(args.cube), log)
    except Exception as e:
        log.error(f"Failed to load cube: {e}")
        return

    S, N, _ = full_cube.shape
    source_cubes = np.zeros((args.n_sources, S, N, N), dtype=np.float32)
    
    # Initialize masks to None
    prev_masks = [None] * args.n_sources
    prev_slice = None
    
    log.info(f"Processing {S} spectral channels with Optical Flow + MCA...")

    for s in tqdm(range(S), desc="Spectral Channels"):
        current_slice = full_cube[s]
        
        # First channel: initialize sources without flow
        if s == 0:
            clean = mca_threshold(current_slice, sigma_clip=1.0)
            binary = clean > (clean.max() * 0.1 if clean.max() > 0 else 0.01)
            labeled = label(binary)
            regions = sorted(regionprops(labeled, intensity_image=current_slice),
                            key=lambda x: x.intensity_mean, reverse=True)
            
            for i in range(min(args.n_sources, len(regions))):
                prev_masks[i] = labeled == regions[i].label
        else:
            # Compute optical flow from previous to current channel
            flow = compute_flow_field(prev_slice, current_slice, use_fast=args.fast)
            
            # Detect and match sources using flow prediction
            current_masks = detect_and_match_sources(
                current_slice, prev_masks, flow, args.n_sources, log
            )
            prev_masks = current_masks
        
        # 3. Assign flux from the ORIGINAL slice using matched masks
        for i in range(args.n_sources):
            if prev_masks[i] is not None:
                # Use original intensity to preserve physics
                source_cubes[i, s] = current_slice * prev_masks[i]
        
        prev_slice = current_slice
        # Periodic log update
        if s % 50 == 0 and s > 0:
            log.info(f"Checkpoint: Channel {s} complete.")

    log.info(f"Decomposition complete. Saving to {out_path / 'decomposed_sources.h5'}")
    with h5py.File(out_path / "decomposed_sources.h5", "w") as f:
        f.create_dataset("source_cubes", data=source_cubes, compression="gzip")
    
    log.info("Done.")

if __name__ == "__main__":
    main()