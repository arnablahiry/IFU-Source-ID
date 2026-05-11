import argparse
import h5py
import numpy as np
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.signal import correlate
from scipy.ndimage import center_of_mass
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.measure import label, regionprops

try:
    import scarlet
    HAS_SCARLET = True
except ImportError:
    HAS_SCARLET = False
    print("Warning: Scarlet not installed. Run: pip install scarlet")


def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("scarlet_tracker")
    logger.setLevel(logging.DEBUG)
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


def compute_flow_cube(cube, use_fast=False):
    """
    Compute optical flow between consecutive channels.
    
    Args:
        cube: (S, H, W) spectral cube
        use_fast: Use ILK (faster) vs TV-L1 (more accurate)
    
    Returns:
        flow_fields: List of (H, W, 2) flow fields
    """
    S, H, W = cube.shape
    flow_fields = []
    
    for s in range(S - 1):
        prev_slice = cube[s].astype(np.float32)
        curr_slice = cube[s + 1].astype(np.float32)
        
        # Normalize
        prev_norm = (prev_slice - prev_slice.min()) / (prev_slice.max() - prev_slice.min() + 1e-8)
        curr_norm = (curr_slice - curr_slice.min()) / (curr_slice.max() - curr_slice.min() + 1e-8)
        
        if use_fast:
            flow = optical_flow_ilk(prev_norm, curr_norm)
        else:
            flow = optical_flow_tvl1(prev_norm, curr_norm, num_warp=10, num_iter=15)
        
        # Convert (2, H, W) to (H, W, 2)
        flow_xy = np.stack([flow[0], flow[1]], axis=-1)
        flow_fields.append(flow_xy)
    
    return flow_fields


def scarlet_decompose_channel(channel_data, n_sources, logger):
    """
    Decompose a single channel using Scarlet.
    
    Args:
        channel_data: (H, W) 2D image
        n_sources: Number of sources to decompose
        logger: Logger instance
    
    Returns:
        components: List of dicts with morphology, position, flux
    """
    if not HAS_SCARLET:
        raise ImportError("Scarlet not installed")
    
    # Normalize
    data = channel_data / (channel_data.max() + 1e-8)
    
    try:
        # Create Scarlet scene
        scene = scarlet.Scene([data], backgrounds=[None])
        scene.fit(max_iter=100, e_rel=1e-3)
        
        components = []
        for src_idx, source in enumerate(scene.sources):
            # Get morphology (spatial profile)
            morphology = source.morphology.data  # (H, W)
            
            # Get center of mass
            try:
                y, x = center_of_mass(morphology)
            except:
                # Fallback
                props = regionprops(label(morphology > morphology.max() * 0.1))
                if props:
                    y, x = props[0].centroid
                else:
                    y, x = np.array(morphology.shape) / 2
            
            # Get flux
            flux = np.sum(source.sed.data) if hasattr(source, 'sed') else np.sum(source)
            
            components.append({
                'idx': src_idx,
                'morphology': morphology.astype(np.float32),
                'x': float(x),
                'y': float(y),
                'flux': float(flux),
                'source_obj': source
            })
        
        logger.debug(f"Decomposed into {len(components)} components")
        return components
    
    except Exception as e:
        logger.warning(f"Scarlet decomposition failed: {e}")
        return []


def correlate_morphologies(morph1, morph2, max_shift=5):
    """
    Compute correlation between two morphologies with alignment.
    
    Args:
        morph1: (H, W) morphology 1
        morph2: (H, W) morphology 2
        max_shift: Max pixels to search for alignment
    
    Returns:
        correlation: Correlation score (0-1)
    """
    # Normalize
    m1 = morph1 / (np.sum(morph1) + 1e-8)
    m2 = morph2 / (np.sum(morph2) + 1e-8)
    
    # Pad to same size
    h1, w1 = m1.shape
    h2, w2 = m2.shape
    max_h = max(h1, h2) + 2 * max_shift
    max_w = max(w1, w2) + 2 * max_shift
    
    m1_pad = np.zeros((max_h, max_w))
    m2_pad = np.zeros((max_h, max_w))
    
    off = max_shift
    m1_pad[off:off+h1, off:off+w1] = m1
    m2_pad[off:off+h2, off:off+w2] = m2
    
    # Cross-correlation
    corr = correlate(m1_pad, m2_pad, mode='valid')
    max_corr = corr.max()
    
    # Normalize by norms
    norm1 = np.sqrt(np.sum(m1_pad**2))
    norm2 = np.sqrt(np.sum(m2_pad**2))
    
    if norm1 > 0 and norm2 > 0:
        return max_corr / (norm1 * norm2)
    return 0.0


def match_components_between_frames(prev_comps, curr_comps, flow=None,
                                    morphology_weight=0.7, flow_weight=0.3):
    """
    Match components between two consecutive channels.
    
    Cost combines:
    - Morphology similarity (same source has similar PSF)
    - Flow-predicted position (optical flow guidance)
    - Flux continuity
    
    Args:
        prev_comps: List of components from previous channel
        curr_comps: List of components from current channel
        flow: (H, W, 2) optical flow field
        morphology_weight: Weight for morphology similarity
        flow_weight: Weight for flow prediction
    
    Returns:
        matches: List of (prev_idx, curr_idx) tuples
    """
    n_prev = len(prev_comps)
    n_curr = len(curr_comps)
    
    if n_prev == 0 or n_curr == 0:
        return []
    
    # Compute cost matrix
    cost_matrix = np.full((n_prev, n_curr), 1e8)
    
    for i, prev_comp in enumerate(prev_comps):
        prev_morph = prev_comp['morphology']
        prev_pos = np.array([prev_comp['x'], prev_comp['y']])
        prev_flux = prev_comp['flux'] + 1e-8
        
        for j, curr_comp in enumerate(curr_comps):
            curr_morph = curr_comp['morphology']
            curr_pos = np.array([curr_comp['x'], curr_comp['y']])
            curr_flux = curr_comp['flux'] + 1e-8
            
            # 1. Morphology similarity (normalized cross-correlation)
            morph_sim = correlate_morphologies(prev_morph, curr_morph)
            morph_cost = 1.0 - morph_sim
            
            # 2. Flow-predicted distance
            if flow is not None:
                py, px = int(np.clip(prev_pos[1], 0, flow.shape[0]-1)), \
                         int(np.clip(prev_pos[0], 0, flow.shape[1]-1))
                flow_u = flow[py, px, 0]
                flow_v = flow[py, px, 1]
                pred_pos = prev_pos + np.array([flow_u, flow_v])
                flow_dist = np.linalg.norm(curr_pos - pred_pos)
                flow_cost = flow_dist / (5.0 + flow_dist)  # Normalize
            else:
                flow_cost = np.linalg.norm(curr_pos - prev_pos) / 20.0
                flow_cost = min(flow_cost, 1.0)
            
            # 3. Flux continuity (log-space)
            flux_ratio = np.log(curr_flux / prev_flux)
            flux_cost = min(np.abs(flux_ratio), 2.0) / 2.0  # Cap at 2
            
            # Combined cost
            total_cost = (morphology_weight * morph_cost + 
                         flow_weight * flow_cost + 
                         (1.0 - morphology_weight - flow_weight) * flux_cost)
            
            cost_matrix[i, j] = total_cost
    
    # Greedy matching: assign highest-priority pairs
    matches = []
    used_curr = set()
    used_prev = set()
    
    # Sort by cost
    costs_flat = cost_matrix.flatten()
    sorted_indices = np.argsort(costs_flat)
    
    for flat_idx in sorted_indices:
        i, j = np.unravel_index(flat_idx, cost_matrix.shape)
        cost = cost_matrix[i, j]
        
        if i not in used_prev and j not in used_curr and cost < 1.5:
            matches.append((i, j))
            used_prev.add(i)
            used_curr.add(j)
    
    return matches


def build_component_tracks(frame_matches, component_info):
    """
    Link frame-to-frame matches into continuous tracks.
    
    Args:
        frame_matches: List of (channel, [(prev_idx, curr_idx), ...])
        component_info: List of components per channel
    
    Returns:
        tracks: Dict of track_id -> [(channel, comp_idx), ...]
    """
    S = len(component_info)
    
    tracks = {}  # track_id -> [(channel, comp_idx), ...]
    track_id_counter = 0
    visited = set()
    
    # Build nodes and edges
    nodes = {}  # (channel, idx) -> node_id
    node_counter = 0
    
    for ch in range(S):
        for comp_idx in range(len(component_info[ch])):
            nodes[(ch, comp_idx)] = node_counter
            node_counter += 1
    
    # Forward pass: extend chains
    for ch, matches in frame_matches:
        for prev_idx, curr_idx in matches:
            prev_node = (ch, prev_idx)
            curr_node = (ch + 1, curr_idx)
            
            # Check if prev_node already in track
            track_id = None
            for tid, track in tracks.items():
                if prev_node in track:
                    track_id = tid
                    break
            
            if track_id is None:
                # Start new track
                track_id = track_id_counter
                tracks[track_id] = [prev_node]
                track_id_counter += 1
            
            # Extend track
            if curr_node not in tracks[track_id]:
                tracks[track_id].append(curr_node)
    
    # Add unmatched components as singleton tracks
    for ch in range(S):
        for comp_idx in range(len(component_info[ch])):
            node = (ch, comp_idx)
            if node not in visited:
                is_in_track = False
                for track in tracks.values():
                    if node in track:
                        is_in_track = True
                        break
                
                if not is_in_track:
                    track_id = track_id_counter
                    tracks[track_id] = [node]
                    track_id_counter += 1
    
    return tracks


def reconstruct_source_cubes(tracks, component_info, H, W, S, n_sources):
    """
    Reconstruct source cubes from tracks.
    Assign track_id to physical source_id based on continuity and brightness.
    
    Args:
        tracks: Dict of track_id -> [(channel, comp_idx), ...]
        component_info: List of components per channel
        H, W, S: Cube dimensions
        n_sources: Number of expected unique sources
    
    Returns:
        source_cubes: (n_sources, S, H, W) array
    """
    source_cubes = np.zeros((n_sources, S, H, W), dtype=np.float32)
    
    # Sort tracks by total flux (brightest first)
    track_fluxes = {}
    for track_id, track in tracks.items():
        total_flux = 0
        for ch, comp_idx in track:
            if comp_idx < len(component_info[ch]):
                total_flux += component_info[ch][comp_idx]['flux']
        track_fluxes[track_id] = total_flux
    
    sorted_tracks = sorted(track_fluxes.items(), key=lambda x: x[1], reverse=True)
    
    # Assign tracks to source indices
    for src_idx, (track_id, _) in enumerate(sorted_tracks[:n_sources]):
        track = tracks[track_id]
        
        for ch, comp_idx in track:
            if comp_idx < len(component_info[ch]):
                comp = component_info[ch][comp_idx]
                # Use morphology as mask
                morph_binary = comp['morphology'] > comp['morphology'].max() * 0.1
                
                # Normalize morphology and scale by flux
                morph_norm = comp['morphology'] / (comp['morphology'].max() + 1e-8)
                
                # Place in cube
                y_start = max(0, int(comp['y']) - morph_norm.shape[0] // 2)
                x_start = max(0, int(comp['x']) - morph_norm.shape[1] // 2)
                
                y_end = min(H, y_start + morph_norm.shape[0])
                x_end = min(W, x_start + morph_norm.shape[1])
                
                morph_h = y_end - y_start
                morph_w = x_end - x_start
                
                source_cubes[src_idx, ch, y_start:y_end, x_start:x_end] = \
                    morph_norm[:morph_h, :morph_w] * comp['flux']
    
    return source_cubes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default="/Users/arnablahiry/repos/GalCubeCraft-SourceID/experiments/sep_v1/data/cube_1006.h5")
    ap.add_argument("--out", default="experiments/mca_v1")
    ap.add_argument("--n_sources", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="Use ILK flow (faster) instead of TV-L1")
    ap.add_argument("--morph_weight", type=float, default=0.7, help="Morphology weight in matching")
    ap.add_argument("--flow_weight", type=float, default=0.3, help="Flow weight in matching")
    args = ap.parse_args()
    
    if not HAS_SCARLET:
        print("Error: Scarlet not installed. Run: pip install scarlet")
        return
    
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logging(out_path)
    log.info("Starting Scarlet+Flow Non-Stationary Source Tracking")
    
    try:
        full_cube = load_cube(Path(args.cube), log)
    except Exception as e:
        log.error(f"Failed to load cube: {e}")
        return
    
    S, H, W = full_cube.shape
    
    # Compute optical flow fields
    log.info("Computing optical flow fields...")
    flow_fields = compute_flow_cube(full_cube, use_fast=args.fast)
    log.info(f"Computed {len(flow_fields)} flow fields")
    
    # Stage 1: Scarlet decomposition per channel
    log.info("Decomposing channels with Scarlet...")
    component_info = []
    
    for s in tqdm(range(S), desc="Scarlet decomposition"):
        try:
            comps = scarlet_decompose_channel(full_cube[s], args.n_sources, log)
            component_info.append(comps)
        except Exception as e:
            log.warning(f"Channel {s} decomposition failed: {e}")
            component_info.append([])
    
    log.info(f"Decomposition complete: {sum(len(c) for c in component_info)} total components")
    
    # Stage 2: Match components across channels
    log.info("Matching components across channels...")
    frame_matches = []
    
    for s in tqdm(range(S - 1), desc="Component matching"):
        if not component_info[s] or not component_info[s + 1]:
            frame_matches.append((s, []))
            continue
        
        flow = flow_fields[s]
        matches = match_components_between_frames(
            component_info[s], 
            component_info[s + 1],
            flow=flow,
            morphology_weight=args.morph_weight,
            flow_weight=args.flow_weight
        )
        
        frame_matches.append((s, matches))
        log.debug(f"Channel {s}->{s+1}: {len(matches)} matches")
    
    # Stage 3: Build tracks
    log.info("Building component tracks...")
    tracks = build_component_tracks(frame_matches, component_info)
    log.info(f"Found {len(tracks)} component tracks")
    
    # Stage 4: Reconstruct source cubes
    log.info("Reconstructing source cubes...")
    source_cubes = reconstruct_source_cubes(
        tracks, component_info, H, W, S, args.n_sources
    )
    
    # Save results
    log.info(f"Saving to {out_path / 'decomposed_sources.h5'}")
    with h5py.File(out_path / "decomposed_sources.h5", "w") as f:
        f.create_dataset("source_cubes", data=source_cubes, compression="gzip")
        f.create_dataset("n_tracks", data=np.array(len(tracks)))
    
    # Save track info
    track_info = []
    for track_id, track in tracks.items():
        track_info.append({
            'track_id': track_id,
            'length': len(track),
            'channels': [t[0] for t in track],
            'components': [t[1] for t in track]
        })
    
    with open(out_path / "tracks.txt", "w") as f:
        f.write("Track ID | Length | Channels | Components\n")
        for info in track_info:
            f.write(f"{info['track_id']} | {info['length']} | {info['channels']} | {info['components']}\n")
    
    log.info("Done!")


if __name__ == "__main__":
    main()
