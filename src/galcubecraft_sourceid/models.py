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
    delta_bg: float = 3.0,
    zeta: float = 0.1,
) -> torch.Tensor:
    """De Brabandere et al. 2017 discriminative loss for instance embeddings.

    L_var:  pulls each source's voxel embeddings within `delta_v` of the source mean.
    L_dist: pushes pairs of source means at least `2*delta_d` apart.
    L_reg:  keeps means near origin.
    L_bg:   pushes background voxel embeddings at least `delta_bg` away from every
            source mean. Without this, background/diffuse voxels (excluded from
            the variance term) get arbitrary embeddings that land inside source
            clusters at eval time, creating spurious detections.
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

        bg_mask = (lab == -1)
        if bg_mask.any() and zeta > 0:
            bg_vox = e[:, bg_mask]                             # (D, n_bg)
            # distance from each bg voxel to each source mean: (K, n_bg)
            d_bg = (bg_vox[None, :, :] - means_t[:, :, None]).norm(dim=1)
            # penalise bg voxels that are too close to any source mean
            L_bg = (torch.relu(delta_bg - d_bg.min(dim=0).values) ** 2).mean()
        else:
            L_bg = emb.new_zeros(())

        losses.append(alpha * L_var + beta * L_dist + gamma * L_reg + zeta * L_bg)
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
        self.last_mask: torch.Tensor = None  # written during forward

    def forward(self, x: torch.Tensor, valid_slots: torch.Tensor = None) -> torch.Tensor:
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
        if valid_slots is not None:
            inf_mask = (valid_slots < 0.5)[:, :, None, None, None]  # (B, M, 1, 1, 1)
            logits[:, :self.n_galaxy_slots].masked_fill_(inf_mask, float("-inf"))
        mask = F.softmax(logits, dim=1)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        self.last_mask = mask[tuple(slices)]
        return self.last_mask * x


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
    cross_weight: float = 0.0,
) -> torch.Tensor:
    """Permutation-invariant separation loss via min-cost slot assignment.

    For each sample, predicted slots are matched to *valid* target slots by a
    Hungarian assignment on per-voxel MSE cost. Unmatched predicted slots are
    pushed toward zero with `empty_weight`. With `rebalance=True`, each matched
    pair's MSE is divided by the target's mean energy so faint satellites and
    bright centrals contribute comparable gradients.

    When `cross_weight > 0`, an additional cross-contamination term penalises
    each matched slot for containing flux that belongs to *other* galaxies:
    for matched pair (slot k → galaxy t), penalise pred[k] at voxels where
    galaxy j≠t is bright. This directly discourages multiple sources per slot.
    """
    B, M = pred.shape[:2]
    costs = _pairwise_voxel_mse(pred, target).detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()

    losses = []
    for b in range(B):
        n_valid = int(valid_np[b].sum())
        unmatched = set(range(M))
        sample_losses = []
        cross_terms = []
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
                sample_losses.append(pair)
                unmatched.discard(r)
                if cross_weight > 0:
                    for j in range(n_valid):
                        if j == t_idx:
                            continue
                        t_max = target[b, j].max().detach().clamp(min=1e-8)
                        other_mask = (target[b, j] / t_max).detach()
                        cross_terms.append((pred[b, r] * other_mask).pow(2).mean())
        for i in unmatched:
            sample_losses.append(empty_weight * (pred[b, i] ** 2).mean())
        if sample_losses:
            total = torch.stack(sample_losses).mean()
            if cross_weight > 0 and cross_terms:
                total = total + cross_weight * torch.stack(cross_terms).mean()
            losses.append(total)

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
    cross_weight: float = 0.0,
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
        cross_weight=cross_weight,
    )
    diff_target = (cube - target.sum(dim=1, keepdim=True)).clamp(min=0.0)
    loss_diff = ((pred_diff - diff_target) ** 2).mean()
    loss_recon = ((pred.sum(dim=1, keepdim=True) - cube) ** 2).mean()
    return loss_gal + diffuse_weight * loss_diff + recon_weight * loss_recon


def focal_mse_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Focal-weighted MSE: emphasise hard positives without ignoring background."""
    weight = torch.pow(torch.abs(target - pred), beta).detach()
    return ((pred - target) ** 2 * (weight + 1e-3)).mean()


class TwoStageUNet3D(nn.Module):
    """Shared-encoder U-Net with two parallel decoders.

    Stage 1 — foreground decoder: sigmoid mask separating any-galaxy voxels
    from diffuse. Supervised by BCE against the union of all valid galaxy
    footprints.

    Stage 2 — slot decoder: softmax over M galaxy slots applied *only* to
    foreground voxels, separating individual galaxies.

    Output (B, M+1, C, Y, X) matches MaskedSeparationUNet3D so the same loss
    functions apply. Stores fg_mask as `self.last_fg_mask` for BCE supervision.

        pred_gal[k]  = slot_weight[k] * fg_mask * input
        pred_diff    = (1 - fg_mask) * input
    """

    def __init__(self, max_n_gals: int, base: int = 16):
        super().__init__()
        self.n_galaxy_slots = max_n_gals
        # Shared encoder
        self.enc1 = _conv_block(1, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Foreground decoder
        self.fg_up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.fg_dec3 = _conv_block(base * 8, base * 4)
        self.fg_up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.fg_dec2 = _conv_block(base * 4, base * 2)
        self.fg_up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.fg_dec1 = _conv_block(base * 2, base)
        self.fg_head = nn.Conv3d(base, 1, kernel_size=1)
        # Slot decoder
        self.slot_up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.slot_dec3 = _conv_block(base * 8, base * 4)
        self.slot_up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.slot_dec2 = _conv_block(base * 4, base * 2)
        self.slot_up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.slot_dec1 = _conv_block(base * 2, base)
        self.slot_head = nn.Conv3d(base, max_n_gals, kernel_size=1)

        self.last_fg_mask: torch.Tensor = None  # written during forward for loss access

    def forward(self, x: torch.Tensor, valid_slots: torch.Tensor = None) -> torch.Tensor:
        orig = x.shape[2:]
        pad_amt, pad_tuple = [], []
        for d in orig:
            pad_amt.append((8 - d % 8) % 8)
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x_pad = F.pad(x, pad_tuple)

        # Shared encoder
        e1 = self.enc1(x_pad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))

        # Foreground decoder
        fg_d3 = self.fg_dec3(torch.cat([self.fg_up3(bn), e3], dim=1))
        fg_d2 = self.fg_dec2(torch.cat([self.fg_up2(fg_d3), e2], dim=1))
        fg_d1 = self.fg_dec1(torch.cat([self.fg_up1(fg_d2), e1], dim=1))
        fg_logits = self.fg_head(fg_d1)

        # Slot decoder
        sl_d3 = self.slot_dec3(torch.cat([self.slot_up3(bn), e3], dim=1))
        sl_d2 = self.slot_dec2(torch.cat([self.slot_up2(sl_d3), e2], dim=1))
        sl_d1 = self.slot_dec1(torch.cat([self.slot_up1(sl_d2), e1], dim=1))
        slot_logits = self.slot_head(sl_d1)

        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        fg_logits = fg_logits[tuple(slices)]
        slot_logits = slot_logits[tuple(slices)]

        if valid_slots is not None:
            inf_mask = (valid_slots < 0.5)[:, :, None, None, None]
            slot_logits.masked_fill_(inf_mask, float("-inf"))

        fg_mask = torch.sigmoid(fg_logits)            # (B, 1, C, Y, X)
        slot_weights = F.softmax(slot_logits, dim=1)  # (B, M, C, Y, X)

        self.last_fg_mask = fg_mask

        pred_gal = slot_weights * fg_mask * x         # (B, M, C, Y, X)
        pred_diff = (1.0 - fg_mask) * x               # (B, 1, C, Y, X)
        return torch.cat([pred_gal, pred_diff], dim=1) # (B, M+1, C, Y, X)


class PositionGuidedMaskedSeparationUNet3D(MaskedSeparationUNet3D):
    """MaskedSeparationUNet3D with per-slot positional Gaussian bias.

    At forward time, a 3D Gaussian volume is computed for each galaxy slot
    centered at its known position (`centers_cyx`, shape (B, M, 3) in
    (channel, y, x) pixel coordinates). The Gaussian is added as a logit
    bias before the softmax, biasing slot k to take ownership of voxels near
    galaxy k. The model can override this bias — it is a soft prior, not a
    hard constraint.

    At training time: pass GT `centers_cyx` from the batch.
    At test time: pass centers predicted by a separate detection model.
    When `centers_cyx=None`, degrades to plain `MaskedSeparationUNet3D`.
    """

    def __init__(self, max_n_gals: int, base: int = 16,
                 center_sigma: float = 1.5, bias_scale: float = 30.0):
        super().__init__(max_n_gals=max_n_gals, base=base)
        self.center_sigma = center_sigma
        self.bias_scale = bias_scale

    def forward(self, x: torch.Tensor,
                valid_slots: torch.Tensor = None,
                centers_cyx: torch.Tensor = None) -> torch.Tensor:
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
        b_feat = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b_feat), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        logits = logits[tuple(slices)]

        M = self.n_galaxy_slots
        B = x.shape[0]

        # Inject per-slot positional Gaussian bias into galaxy logits.
        if centers_cyx is not None:
            gauss_bias = torch.zeros(B, M, *orig, device=x.device, dtype=x.dtype)
            for bi in range(B):
                n_valid = int(valid_slots[bi].sum().item()) if valid_slots is not None else M
                for m in range(n_valid):
                    g = position_query_volume(centers_cyx[bi, m], orig, self.center_sigma)
                    gauss_bias[bi, m] = g.squeeze(0)
            logits[:, :M] = logits[:, :M] + self.bias_scale * gauss_bias

        if valid_slots is not None:
            inf_mask = (valid_slots < 0.5)[:, :, None, None, None]
            logits[:, :M].masked_fill_(inf_mask, float("-inf"))

        mask = F.softmax(logits, dim=1)
        self.last_mask = mask
        return mask * x


def mask_entropy_loss(model: "MaskedSeparationUNet3D") -> torch.Tensor:
    """Per-voxel softmax entropy penalty over all slots.

    The softmax mask assigns each voxel a probability distribution over slots.
    High entropy = flux spread evenly (bad: hallucination).
    Low entropy = one slot clearly owns each voxel (good: clean separation).
    Minimising this pushes toward hard slot assignment without breaking Σ=input.
    """
    mask = model.last_mask                    # (B, M+1, C, Y, X)
    return -(mask * (mask.clamp(min=1e-8).log())).sum(dim=1).mean()


def two_stage_separation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    cube: torch.Tensor,
    model: "TwoStageUNet3D",
    fg_threshold: float = 0.05,
    fg_weight: float = 1.0,
    cross_weight: float = 0.0,
) -> torch.Tensor:
    """Hungarian separation loss + explicit foreground BCE.

    The BCE term supervises stage-1 directly: the fg_mask should be 1 wherever
    any valid galaxy has flux above `fg_threshold * cube.max()`, and 0 elsewhere
    (diffuse). This gives the foreground decoder a clean, unambiguous signal
    independent of the slot-assignment stage.
    """
    L_sep = hungarian_separation_loss_with_diffuse(
        pred, target, valid, cube, recon_weight=0.0, cross_weight=cross_weight,
    )
    # GT foreground: union of all valid galaxy footprints
    valid_exp = valid[:, :, None, None, None]
    galaxy_signal = (target * valid_exp).amax(dim=1, keepdim=True)
    cube_max = cube.flatten(2).amax(dim=2)[:, :, None, None, None].clamp(min=1e-8)
    gt_fg = (galaxy_signal > fg_threshold * cube_max).float()
    L_fg = F.binary_cross_entropy(model.last_fg_mask, gt_fg)
    return L_sep + fg_weight * L_fg


class SegMaskUNet3D(nn.Module):
    """Direct supervised segmentation of per-galaxy binary masks.

    Input:  observed cube (B, 1, C, Y, X) concatenated with M Gaussian
            position priors (B, M, C, Y, X) → (B, M+1, C, Y, X).
    Output: M sigmoid masks (B, M, C, Y, X) — one per galaxy slot.

    GT mask for slot k: voxels where galaxy_cubes[k] exceeds a flux
    threshold. Loss is weighted BCE so background (the majority class)
    doesn't dominate. No Hungarian matching, no softmax competition,
    no permutation ambiguity — slot k always corresponds to the galaxy
    whose Gaussian prior was placed at center k.

    At inference: pred_flux[k] = mask[k] * cube extracts galaxy k.
    Centers come from a detector at test time; GT centers during training.
    """

    def __init__(self, max_n_gals: int, base: int = 16,
                 center_sigma: float = 1.5):
        super().__init__()
        self.max_n_gals = max_n_gals
        self.center_sigma = center_sigma
        in_ch = max_n_gals + 1  # cube + M Gaussian priors
        self.enc1 = _conv_block(in_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv3d(base, max_n_gals, kernel_size=1)

    def forward(self, cube: torch.Tensor,
                centers_cyx: torch.Tensor,
                valid_slots: torch.Tensor = None) -> torch.Tensor:
        """
        cube:        (B, 1, C, Y, X)
        centers_cyx: (B, M, 3)  — (channel, y, x) pixel coords
        valid_slots: (B, M)     — 1.0 for real slots, 0.0 for padding
        Returns:     (B, M, C, Y, X) sigmoid masks
        """
        B, _, *orig = cube.shape
        M = self.max_n_gals

        # Build per-slot Gaussian position priors.
        priors = torch.zeros(B, M, *orig, device=cube.device, dtype=cube.dtype)
        for b in range(B):
            n_valid = int(valid_slots[b].sum().item()) if valid_slots is not None else M
            for m in range(n_valid):
                priors[b, m] = position_query_volume(
                    centers_cyx[b, m], tuple(orig), self.center_sigma
                ).squeeze(0)

        x = torch.cat([cube, priors], dim=1)  # (B, M+1, C, Y, X)

        # Pad to multiple of 8.
        pad_tuple = []
        pad_amt = [(8 - d % 8) % 8 for d in orig]
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x = F.pad(x, pad_tuple)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)

        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        masks = torch.sigmoid(logits[tuple(slices)])  # (B, M, C, Y, X)

        # Zero out padding slots so they never contribute to loss or output.
        if valid_slots is not None:
            masks = masks * valid_slots[:, :, None, None, None]

        return masks


def _peaks_to_centers(heatmap: torch.Tensor, max_n_gals: int,
                      threshold: float = 0.2) -> torch.Tensor:
    """Extract up to max_n_gals peak locations from a sigmoid heatmap.

    heatmap: (n_classes, C, Y, X) on any device.
    Returns: (max_n_gals, 3) float tensor of (channel, y, x) coords,
             padded with zeros for unfilled slots, and
             (max_n_gals,) valid mask.
    """
    import numpy as np
    from skimage.feature import peak_local_max

    h_np = heatmap.detach().cpu().numpy()
    combined = h_np.max(axis=0)  # (C, Y, X) — merge classes
    peaks = peak_local_max(combined, min_distance=2, threshold_abs=threshold,
                           num_peaks=max_n_gals)
    # Sort by score descending so strongest source = slot 0.
    scores = [combined[tuple(p)] for p in peaks]
    order = sorted(range(len(peaks)), key=lambda i: -scores[i])
    peaks = [peaks[i] for i in order]

    centers = torch.zeros(max_n_gals, 3, dtype=heatmap.dtype, device=heatmap.device)
    valid = torch.zeros(max_n_gals, dtype=heatmap.dtype, device=heatmap.device)
    for i, (c, y, x) in enumerate(peaks[:max_n_gals]):
        centers[i] = torch.tensor([float(c), float(y), float(x)])
        valid[i] = 1.0
    return centers, valid


class JointDetSegUNet3D(nn.Module):
    """Joint detection + segmentation U-Net: one forward pass, no external centers.

    Shared encoder → two decoders:
      1. Detection decoder → sigmoid heatmap (B, n_classes, C, Y, X).
         Supervised by GT Gaussian heatmaps from `build_heatmap`.
      2. Seg decoder → M binary masks (B, M, C, Y, X).
         Conditioned on per-slot Gaussian position priors (M, C, Y, X)
         injected at the final decoder layer before the 1×1 head.
         Supervised by GT binary masks from `galaxy_cubes`.

    Training: GT centers are used to build Gaussian priors, with optional
    Gaussian noise (`center_noise`) to bridge the train/test gap — at test
    time the model uses its own detected peaks as centers.

    Inference (centers_cyx=None): detection decoder runs first, peaks are
    extracted, Gaussian priors are built, then the seg decoder runs.
    Single forward pass, no external inputs beyond the observed cube.
    """

    def __init__(self, max_n_gals: int, n_classes: int = 2, base: int = 16,
                 center_sigma: float = 1.5, center_noise: float = 2.0):
        super().__init__()
        self.max_n_gals = max_n_gals
        self.n_classes = n_classes
        self.center_sigma = center_sigma
        self.center_noise = center_noise

        # Shared encoder
        self.enc1 = _conv_block(1, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Detection decoder
        self.det_up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.det_dec3 = _conv_block(base * 8, base * 4)
        self.det_up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.det_dec2 = _conv_block(base * 4, base * 2)
        self.det_up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.det_dec1 = _conv_block(base * 2, base)
        self.det_head = nn.Conv3d(base, n_classes, kernel_size=1)

        # Seg decoder — head takes (base + max_n_gals) channels: decoder
        # features + M Gaussian prior maps concatenated at full resolution.
        self.seg_up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.seg_dec3 = _conv_block(base * 8, base * 4)
        self.seg_up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.seg_dec2 = _conv_block(base * 4, base * 2)
        self.seg_up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.seg_dec1 = _conv_block(base * 2, base)
        self.seg_head = nn.Conv3d(base + max_n_gals, max_n_gals, kernel_size=1)

    def _build_priors(self, centers_cyx: torch.Tensor,
                      valid_slots: torch.Tensor,
                      vol_shape: tuple) -> torch.Tensor:
        """Build (B, M, *vol_shape) Gaussian prior volume."""
        B, M = centers_cyx.shape[:2]
        priors = torch.zeros(B, M, *vol_shape,
                             device=centers_cyx.device, dtype=centers_cyx.dtype)
        for b in range(B):
            n_v = int(valid_slots[b].sum().item())
            for m in range(n_v):
                priors[b, m] = position_query_volume(
                    centers_cyx[b, m], vol_shape, self.center_sigma
                ).squeeze(0)
        return priors

    def forward(self, cube: torch.Tensor,
                centers_cyx: torch.Tensor = None,
                valid_slots: torch.Tensor = None) -> tuple:
        """
        cube:        (B, 1, C, Y, X)
        centers_cyx: (B, M, 3) — GT centers for training; None at inference.
        valid_slots: (B, M)    — 1.0 for real slots; None at inference.

        Returns: (masks, heatmap)
          masks:   (B, M, C, Y, X) sigmoid binary masks
          heatmap: (B, n_classes, C, Y, X) sigmoid detection heatmap
        """
        orig = tuple(cube.shape[2:])
        B = cube.shape[0]

        # Pad to multiple of 8.
        pad_tuple, pad_amt = [], [(8 - d % 8) % 8 for d in orig]
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x = F.pad(cube, pad_tuple)

        # Shared encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))

        # Detection decoder
        dd3 = self.det_dec3(torch.cat([self.det_up3(bn), e3], dim=1))
        dd2 = self.det_dec2(torch.cat([self.det_up2(dd3), e2], dim=1))
        dd1 = self.det_dec1(torch.cat([self.det_up1(dd2), e1], dim=1))
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        heatmap = torch.sigmoid(self.det_head(dd1)[tuple(slices)])  # (B, n_cls, C, Y, X)

        # Build Gaussian position priors.
        if centers_cyx is not None:
            # Training: use GT centers, optionally perturb them.
            ctrs = centers_cyx.clone()
            if self.training and self.center_noise > 0:
                noise = torch.randn_like(ctrs) * self.center_noise
                ctrs = ctrs + noise
            v = valid_slots if valid_slots is not None else torch.ones(
                B, self.max_n_gals, device=cube.device)
            priors = self._build_priors(ctrs, v, orig)
        else:
            # Inference: extract peaks from heatmap.
            priors = torch.zeros(B, self.max_n_gals, *orig,
                                 device=cube.device, dtype=cube.dtype)
            inf_valid = torch.zeros(B, self.max_n_gals, device=cube.device)
            for b in range(B):
                ctrs_b, val_b = _peaks_to_centers(heatmap[b], self.max_n_gals)
                priors[b] = self._build_priors(
                    ctrs_b.unsqueeze(0), val_b.unsqueeze(0), orig
                )[0]
                inf_valid[b] = val_b
            valid_slots = inf_valid

        # Seg decoder
        sd3 = self.seg_dec3(torch.cat([self.seg_up3(bn), e3], dim=1))
        sd2 = self.seg_dec2(torch.cat([self.seg_up2(sd3), e2], dim=1))
        sd1 = self.seg_dec1(torch.cat([self.seg_up1(sd2), e1], dim=1))
        sd1_crop = sd1[tuple(slices)]
        seg_feat = torch.cat([sd1_crop, priors], dim=1)  # (B, base+M, C, Y, X)
        masks = torch.sigmoid(self.seg_head(seg_feat))    # (B, M, C, Y, X)

        if valid_slots is not None:
            masks = masks * valid_slots[:, :, None, None, None]

        return masks, heatmap


def joint_det_seg_loss(
    masks: torch.Tensor,     # (B, M, C, Y, X)
    heatmap: torch.Tensor,   # (B, n_classes, C, Y, X)
    galaxy_cubes: torch.Tensor,
    valid: torch.Tensor,
    gt_heatmap: torch.Tensor,  # (B, n_classes, C, Y, X) from build_heatmap
    fg_threshold: float = 0.05,
    det_weight: float = 1.0,
) -> torch.Tensor:
    """BCE segmentation loss + MSE detection heatmap loss."""
    L_seg = seg_mask_loss(masks, galaxy_cubes, valid, fg_threshold=fg_threshold)
    L_det = F.mse_loss(heatmap, gt_heatmap)
    return L_seg + det_weight * L_det


class InstanceSegUNet3D(UNet3D):
    """Per-voxel instance segmentation: cube → (M+1)-class label map.

    Output channels: 0..M-1 are galaxy classes, M is background/diffuse.
    GT comes from voxel_instance_labels (argmax per voxel over galaxy_cubes).
    Loss is weighted cross-entropy — no centers, no Hungarian, no priors,
    no permutation problem. Identical to standard semantic segmentation.

    Inference:
        logits = model(cube)                          # (1, M+1, C, Y, X)
        label_map = logits.argmax(dim=1)              # (1, C, Y, X)
        pred_k = (label_map == k).float() * cube      # flux for galaxy k
    """

    def __init__(self, max_n_gals: int, base: int = 16):
        super().__init__(in_channels=1, out_channels=max_n_gals + 1, base=base)
        self.max_n_gals = max_n_gals

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        orig = x.shape[2:]
        pad_tuple, pad_amt = [], [(8 - d % 8) % 8 for d in orig]
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
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return logits[tuple(slices)]   # raw logits (B, M+1, C, Y, X)


def instance_seg_loss(
    logits: torch.Tensor,       # (B, M+1, C, Y, X) raw logits
    galaxy_cubes: torch.Tensor, # (B, M, C, Y, X)
    valid: torch.Tensor,        # (B, M)
    fg_threshold: float = 0.05,
    bg_weight: float = 0.1,
) -> torch.Tensor:
    """Weighted cross-entropy against per-voxel instance labels.

    GT label map: voxel v → galaxy k if galaxy k dominates there (argmax),
    → M (background class) if no galaxy is above threshold.
    bg_weight downweights the majority background class.
    """
    B, Mp1 = logits.shape[:2]
    M = Mp1 - 1
    losses = []
    for b in range(B):
        n_valid = int(valid[b].sum().item())
        if n_valid == 0:
            continue
        # GT instance labels: 0..M-1 for galaxies, M for background.
        raw = voxel_instance_labels(
            galaxy_cubes[b], valid[b], threshold_frac=fg_threshold
        )  # (C, Y, X), values 0..M-1 or -1
        gt = raw.clone()
        gt[gt == -1] = M  # remap background from -1 → M

        # Class weights: galaxy voxels get weight 1.0, background gets bg_weight.
        weight = torch.ones(Mp1, device=logits.device, dtype=logits.dtype)
        weight[M] = bg_weight

        losses.append(F.cross_entropy(
            logits[b].unsqueeze(0),  # (1, M+1, C, Y, X)
            gt.unsqueeze(0).to(logits.device),          # (1, C, Y, X)
            weight=weight,
        ))

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def seg_mask_loss(
    masks: torch.Tensor,    # (B, M, C, Y, X) predicted sigmoid masks
    galaxy_cubes: torch.Tensor,  # (B, M, C, Y, X) GT per-galaxy flux
    valid: torch.Tensor,    # (B, M)
    fg_threshold: float = 0.05,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Weighted BCE between predicted masks and GT binary segmentation masks.

    GT mask[k] = 1 where galaxy k's flux > fg_threshold * galaxy k's peak.
    pos_weight upweights foreground voxels (which are the minority class).
    Only valid slots contribute to the loss.
    """
    # GT binary mask per slot: voxel belongs to galaxy k if it's above threshold.
    slot_peak = galaxy_cubes.flatten(2).amax(dim=2)[:, :, None, None, None].clamp(min=1e-8)
    gt_masks = (galaxy_cubes >= fg_threshold * slot_peak).float()  # (B, M, C, Y, X)

    w = valid[:, :, None, None, None]  # (B, M, 1, 1, 1)
    # Weighted BCE: foreground voxels are rare, upweight them.
    pw = torch.full_like(gt_masks, pos_weight)
    loss = F.binary_cross_entropy_with_logits(
        masks.logit(eps=1e-6), gt_masks,
        pos_weight=pw,
        reduction="none",
    )
    denom = w.sum() * masks.shape[2] * masks.shape[3] * masks.shape[4] + 1e-8
    return (loss * w).sum() / denom


# ---------------------------------------------------------------------------
# v7 — BinarySegUNet3D: M independent binary masks, one loss per galaxy
# ---------------------------------------------------------------------------

class BinarySegUNet3D(UNet3D):
    """Cube → M independent sigmoid masks, one per galaxy slot.

    No flux conservation, no position priors, no centers at inference.
    Galaxies are ordered by descending total flux so slot assignment is
    consistent across cubes (slot 0 = brightest, slot M-1 = faintest).

    Inference:
        masks = model(cube)                         # (1, M, C, Y, X) in [0,1]
        pred_k = masks[:, k] * cube                 # flux for galaxy k
    """

    def __init__(self, max_n_gals: int, base: int = 16):
        super().__init__(in_channels=1, out_channels=max_n_gals, base=base)
        self.max_n_gals = max_n_gals

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        orig = x.shape[2:]
        pad_amt = [(8 - d % 8) % 8 for d in orig]
        pad_tuple: list[int] = []
        for p in reversed(pad_amt):
            pad_tuple.extend([0, p])
        x_pad = F.pad(x, pad_tuple)
        e1 = self.enc1(x_pad)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig]
        return torch.sigmoid(out[tuple(slices)])   # (B, M, C, Y, X) in [0,1]


def binary_seg_loss(
    masks: torch.Tensor,          # (B, M, C, Y, X) predicted sigmoid masks
    galaxy_cubes: torch.Tensor,   # (B, M, C, Y, X) GT per-galaxy flux
    valid: torch.Tensor,          # (B, M) 1.0 for real galaxies
    fg_threshold: float = 0.05,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """Per-galaxy binary cross-entropy — separate loss per satellite.

    For each valid galaxy slot k:
        GT_k[v] = 1  if galaxy_cubes[k,v] >= fg_threshold * peak(galaxy_cubes[k])
        loss_k   = BCE(masks[k], GT_k, pos_weight)

    Galaxies are assumed pre-sorted by descending flux before this call so
    slot ordering is consistent across cubes.  Padding slots (valid=0) are
    excluded from the loss entirely.
    """
    # Sort GT galaxies by descending total flux so slot 0 = brightest.
    flux = galaxy_cubes.flatten(2).sum(dim=2)        # (B, M)
    order = flux.argsort(dim=1, descending=True)     # (B, M)
    galaxy_cubes = galaxy_cubes.gather(
        1, order.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        .expand_as(galaxy_cubes))
    valid = valid.gather(1, order)

    slot_peak = galaxy_cubes.flatten(2).amax(dim=2)[:, :, None, None, None].clamp(min=1e-8)
    gt = (galaxy_cubes >= fg_threshold * slot_peak).float()   # (B, M, C, Y, X)

    w  = valid[:, :, None, None, None]               # (B, M, 1, 1, 1)
    pw = torch.full_like(gt, pos_weight)
    loss = F.binary_cross_entropy_with_logits(
        masks.logit(eps=1e-6), gt,
        pos_weight=pw,
        reduction="none",
    )
    denom = w.sum() * gt.shape[2] * gt.shape[3] * gt.shape[4] + 1e-8
    return (loss * w).sum() / denom
