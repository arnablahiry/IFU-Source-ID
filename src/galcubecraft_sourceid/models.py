"""Small 3D U-Net for voxel-wise heatmap regression."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
        nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Minimal 3D U-Net with 3 downsampling levels.

    Input:  (B, in_channels, D, H, W)
    Output: (B, out_channels, D, H, W) — per-voxel sigmoid heatmaps.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base: int = 16):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv3d(base, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input so every axis is divisible by 8 (three clean 2× poolings).
        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            p = (8 - d % 8) % 8
            pad_amt.append(p)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x = F.pad(x, pad_tuple)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = torch.sigmoid(self.head(d1))
        # Crop back to original spatial shape.
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return out[tuple(slices)]


class SeparationUNet3D(UNet3D):
    """U-Net that outputs `max_n_gals` non-negative per-galaxy cubes.

    Final activation is softplus instead of sigmoid, since the clean per-galaxy
    targets are non-negative flux cubes rather than [0, 1] heatmaps. Works with
    `masked_separation_loss` below.
    """

    def __init__(self, max_n_gals: int, in_channels: int = 1, base: int = 16):
        super().__init__(in_channels=in_channels, out_channels=max_n_gals, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            pad_amt.append((8 - d % 8) % 8)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x = F.pad(x, pad_tuple)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = F.softplus(self.head(d1))
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return out[tuple(slices)]


def masked_separation_loss(
    pred: torch.Tensor,      # (B, M, C, Y, X)
    target: torch.Tensor,    # (B, M, C, Y, X)
    valid: torch.Tensor,     # (B, M)
) -> torch.Tensor:
    """Per-slot MSE, masked by `valid` so padded galaxy slots are ignored."""
    w = valid[:, :, None, None, None]
    denom = w.sum() * pred.shape[2] * pred.shape[3] * pred.shape[4] + 1e-8
    return (((pred - target) ** 2) * w).sum() / denom


class EmbeddingUNet3D(UNet3D):
    """Per-voxel instance-embedding net for source segmentation.
    
    Updated to support CoordConv by allowing variable in_channels (e.g., 4).
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 8, base: int = 16):
        # Pass in_channels to the parent UNet3D class
        super().__init__(in_channels=in_channels, out_channels=embedding_dim, base=base)
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check to ensure input matches expected channels (e.g., 4 for CoordConv)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            pad_amt.append((8 - d % 8) % 8)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
            
        x_pad = F.pad(x, pad_tuple)
        
        # Standard U-Net forward pass
        e1 = self.enc1(x_pad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.head(d1)   # raw embeddings
        
        # Crop back to original dimensions
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return out[tuple(slices)]


def add_coord_channels(cube: torch.Tensor) -> torch.Tensor:
    """Concatenate normalised (z, y, x) coordinate grids in [-1, 1] to the cube.

    Lets the network use absolute spatial position as a feature — useful when
    sources are similar in texture but differ only by location (CoordConv,
    Liu et al. 2018). Input (B, 1, Z, Y, X) → output (B, 4, Z, Y, X).
    """
    B, _, Z, Y, X = cube.shape
    device, dtype = cube.device, cube.dtype
    z = torch.linspace(0.0, 1.0, Z, device=device, dtype=dtype).view(1, 1, Z, 1, 1)
    y = torch.linspace(0.0, 1.0, Y, device=device, dtype=dtype).view(1, 1, 1, Y, 1)
    x = torch.linspace(0.0, 1.0, X, device=device, dtype=dtype).view(1, 1, 1, 1, X)
    z_ch = z.expand(B, 1, Z, Y, X)
    y_ch = y.expand(B, 1, Z, Y, X)
    x_ch = x.expand(B, 1, Z, Y, X)
    return torch.cat([cube, z_ch, y_ch, x_ch], dim=1)


def voxel_instance_labels(
    galaxy_cubes: torch.Tensor,   # (M, C, Y, X) — clean per-galaxy targets
    valid: torch.Tensor,          # (M,) — 1 for real galaxies
    threshold_frac: float = 0.1,
) -> torch.Tensor:
    """Per-voxel source labels for instance-segmentation training.

    Voxel v is labeled k iff galaxy_cubes[k][v] >= threshold_frac * max(galaxy_cubes[k])
    AND k = argmax_j galaxy_cubes[j][v] (so overlap voxels go to the dominant
    source). Voxels below threshold for every source are labeled -1 (background
    / diffuse), and excluded from the variance term in the discriminative loss.
    Returns int64 (C, Y, X).
    """
    M = galaxy_cubes.shape[0]
    src_max = galaxy_cubes.reshape(M, -1).amax(dim=1)         # (M,)
    thresh = (src_max * threshold_frac).view(M, 1, 1, 1)
    above = (galaxy_cubes >= thresh) & (valid.view(M, 1, 1, 1) > 0)
    any_above = above.any(dim=0)
    masked = torch.where(above, galaxy_cubes, torch.full_like(galaxy_cubes, -1.0))
    argmax = masked.argmax(dim=0).to(torch.int64)
    labels = torch.where(any_above, argmax, torch.full_like(argmax, -1))
    return labels


def discriminative_loss(
    emb: torch.Tensor,        # (B, D, C, Y, X) — raw embeddings
    labels: torch.Tensor,     # (B, C, Y, X)   int64, -1 = background
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1e-3,
) -> torch.Tensor:
    """De Brabandere et al. 2017 discriminative loss for instance embeddings.

    L_var: pulls each source's voxel embeddings within `delta_v` of the source mean.
    L_dist: pushes pairs of source means at least `2*delta_d` apart.
    L_reg: keeps means near origin.
    Background voxels (label = -1) are ignored — they don't contribute to means
    or to the variance term. Per-sample loop over instances; cheap because
    n_instances per cube is small (≤ max_n_gals).
    """
    B, D = emb.shape[:2]
    losses = []
    for b in range(B):
        e = emb[b].reshape(D, -1)            # (D, V)
        lab = labels[b].reshape(-1)          # (V,)
        unique = torch.unique(lab)
        unique = unique[unique >= 0]
        if unique.numel() == 0:
            continue
        means = []
        var_terms = []
        for k in unique.tolist():
            mask = (lab == k)
            voxels = e[:, mask]              # (D, n_k)
            mu = voxels.mean(dim=1)
            means.append(mu)
            d = (voxels - mu[:, None]).norm(dim=0)
            var_terms.append((torch.relu(d - delta_v) ** 2).mean())
        means_t = torch.stack(means, dim=0)   # (K, D)
        L_var = torch.stack(var_terms).mean()
        K = means_t.shape[0]
        if K > 1:
            pair = (means_t[:, None] - means_t[None, :]).norm(dim=2)
            offdiag = ~torch.eye(K, dtype=torch.bool, device=pair.device)
            L_dist = (torch.relu(2 * delta_d - pair[offdiag]) ** 2).mean()
        else:
            L_dist = emb.new_zeros(())
        L_reg = means_t.norm(dim=1).mean()
        losses.append(alpha * L_var + beta * L_dist + gamma * L_reg)
    if not losses:
        return emb.sum() * 0.0
    return torch.stack(losses).mean()


def position_query_volume(
    center_cyx: torch.Tensor,   # (3,) in (channel, y, x)
    vol_shape: tuple[int, int, int],
    sigma: float,
) -> torch.Tensor:
    """3D Gaussian volume centered at `center_cyx`. Returns (1, C, Y, X)."""
    C, Y, X = vol_shape
    dev, dt = center_cyx.device, center_cyx.dtype
    cc = torch.arange(C, device=dev, dtype=dt).view(C, 1, 1)
    yy = torch.arange(Y, device=dev, dtype=dt).view(1, Y, 1)
    xx = torch.arange(X, device=dev, dtype=dt).view(1, 1, X)
    cz, cy, cx = center_cyx[0], center_cyx[1], center_cyx[2]
    d2 = (cc - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma)).unsqueeze(0)


class ExtractorUNet3D(UNet3D):
    """Single-source extractor: (cube, position prior) → that source's clean cube.

    Stage 2 of the detect-then-extract pipeline. Input is two channels:
    raw cube + Gaussian "where to look" prior. Output is a single non-negative
    cube — the flux belonging to whichever source sits at the queried position.
    No slot ambiguity, since one forward pass extracts exactly one source.
    """

    def __init__(self, base: int = 16):
        super().__init__(in_channels=2, out_channels=1, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            pad_amt.append((8 - d % 8) % 8)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x_pad = F.pad(x, pad_tuple)
        e1 = self.enc1(x_pad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = F.softplus(self.head(d1))
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return out[tuple(slices)]


class MaskedSeparationUNet3D(UNet3D):
    """Soft-mask separation: pred[k] = softmax_over_slots(logits)[k] * input.

    The trunk produces M+1 logit channels (M galaxy slots + 1 diffuse slot);
    softmax across the slot axis turns them into per-voxel ownership weights
    that sum to 1, so `Σ pred = input` exactly. Slots compete for each voxel,
    which mechanically prevents slot collapse — two identical slots split
    every voxel 50/50, which can never match a single target with full flux.
    The diffuse slot acts as the "everything else" bucket and absorbs flux
    that doesn't belong to any galaxy.
    """

    def __init__(self, max_n_gals: int, base: int = 16):
        super().__init__(in_channels=1, out_channels=max_n_gals + 1, base=base)
        self.n_galaxy_slots = max_n_gals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            pad_amt.append((8 - d % 8) % 8)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x_pad = F.pad(x, pad_tuple)
        e1 = self.enc1(x_pad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        mask = F.softmax(logits, dim=1)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return mask[tuple(slices)] * x   # broadcasts over the slot axis


def _pairwise_voxel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise per-voxel MSE between every pred slot and every target slot.

    Uses the (a-b)^2 = a^2 - 2ab + b^2 identity so we never materialise the
    (B, M, M, C, Y, X) pairwise-difference tensor.
    """
    B, M = pred.shape[:2]
    nvox = pred.shape[2] * pred.shape[3] * pred.shape[4]
    p = pred.reshape(B, M, -1)
    t = target.reshape(B, M, -1)
    p2 = (p * p).mean(dim=2)                                  # (B, M)
    t2 = (t * t).mean(dim=2)                                  # (B, M)
    pt = torch.einsum("bik,bjk->bij", p, t) / nvox            # (B, M, M)
    return p2.unsqueeze(2) + t2.unsqueeze(1) - 2.0 * pt       # (B, M, M)


def hungarian_separation_loss(
    pred: torch.Tensor,    # (B, M, C, Y, X)
    target: torch.Tensor,  # (B, M, C, Y, X)
    valid: torch.Tensor,   # (B, M)
    rebalance: bool = True,
    empty_weight: float = 0.5,
) -> torch.Tensor:
    """Permutation-invariant separation loss via min-cost slot assignment.

    For each sample, predicted slots are matched to *valid* target slots by a
    Hungarian assignment on per-voxel MSE cost. Unmatched predicted slots are
    pushed toward zero with `empty_weight`. With `rebalance=True`, each matched
    pair's MSE is divided by the target's mean energy so faint satellites and
    bright centrals contribute comparable gradients (Fix 3).
    """
    B, M = pred.shape[:2]
    costs = _pairwise_voxel_mse(pred, target).detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()

    losses = []
    for b in range(B):
        n_valid = int(valid_np[b].sum())
        unmatched = set(range(M))
        if n_valid > 0:
            rows, cols = linear_sum_assignment(costs[b, :, :n_valid])
            for r, t_idx in zip(rows.tolist(), cols.tolist()):
                pair = ((pred[b, r] - target[b, t_idx]) ** 2).mean()
                if rebalance:
                    # RMS of the target — softer than mean(target^2) so faint
                    # slots get amplified gradient without exploding the loss.
                    # Floor prevents division by ~0 for nearly-empty slots.
                    scale = (target[b, t_idx] ** 2).mean().sqrt().detach().clamp(min=1e-2)
                    pair = pair / scale
                losses.append(pair)
                unmatched.discard(r)
        for i in unmatched:
            losses.append(empty_weight * (pred[b, i] ** 2).mean())

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def hungarian_separation_loss_with_diffuse(
    pred: torch.Tensor,    # (B, M+1, C, Y, X) — final channel is the diffuse slot
    target: torch.Tensor,  # (B, M, C, Y, X)
    valid: torch.Tensor,   # (B, M)
    cube: torch.Tensor,    # (B, 1, C, Y, X) — observed input
    rebalance: bool = True,
    empty_weight: float = 0.5,
    diffuse_weight: float = 1.0,
    recon_weight: float = 0.1,
) -> torch.Tensor:
    """Hungarian galaxy loss + explicit diffuse target + reconstruction term.

    The last predicted channel is supervised against `cube - Σ targets`, giving
    diffuse a place to live so it stops leaking into satellite slots. The
    reconstruction term ties the sum of all output slots to the observed cube.
    """
    M = target.shape[1]
    pred_gal = pred[:, :M]
    pred_diff = pred[:, M : M + 1]

    loss_gal = hungarian_separation_loss(
        pred_gal, target, valid, rebalance=rebalance, empty_weight=empty_weight,
    )
    diff_target = (cube - target.sum(dim=1, keepdim=True)).clamp(min=0.0)
    loss_diff = ((pred_diff - diff_target) ** 2).mean()
    loss_recon = ((pred.sum(dim=1, keepdim=True) - cube) ** 2).mean()
    return loss_gal + diffuse_weight * loss_diff + recon_weight * loss_recon


def focal_mse_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Focal-weighted MSE: emphasise hard positives without ignoring background."""
    weight = torch.pow(torch.abs(target - pred), beta).detach()
    return ((pred - target) ** 2 * (weight + 1e-3)).mean()
