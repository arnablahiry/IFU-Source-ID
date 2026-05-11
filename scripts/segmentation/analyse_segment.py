"""Analyse a completed `train_segment` run + `evaluate_segment` output.

Reads:
    <run>/metrics.jsonl              -- training curves
    <run>/eval/per_cube_metrics.csv  -- per-cube eval metrics
    <run>/eval/summary.json
    <run>/eval/samples/*.npz         -- per-cube embeddings + labels

Writes:
    <run>/analysis/loss_curves.png        -- val_loss + intra/inter cluster geometry
    <run>/analysis/eval_distributions.png -- matched_iou, flux_rel_err, n_pred vs n_gt
    <run>/analysis/sample_*.png           -- per-cube segmentation montage
    <run>/analysis/report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path


def _setup_logging(out_dir):
    logger = logging.getLogger("analyse_segment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "analyse.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _read_jsonl(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k == "path":
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def plot_training_curves(metrics, delta_v, delta_d, out_path):
    import matplotlib.pyplot as plt
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [m["train_loss"] for m in metrics], "-o", label="train", color="tab:blue", ms=4)
    axes[0].plot(epochs, [m["val_loss"] for m in metrics], "-o", label="val", color="tab:orange", ms=4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("discriminative loss"); axes[0].set_yscale("log")
    axes[0].legend(); axes[0].set_title("training / validation loss"); axes[0].grid(alpha=0.3)

    intra = [m.get("intra_cluster_spread", float("nan")) for m in metrics]
    inter = [m.get("inter_cluster_min_sep", float("nan")) for m in metrics]
    axes[1].plot(epochs, intra, "-o", color="tab:red", ms=4, label="intra-cluster spread")
    axes[1].axhline(delta_v, color="tab:red", ls="--", alpha=0.6, label=f"δ_v={delta_v}")
    axes[1].plot(epochs, inter, "-o", color="tab:green", ms=4, label="inter-cluster min sep")
    axes[1].axhline(2 * delta_d, color="tab:green", ls="--", alpha=0.6, label=f"2δ_d={2 * delta_d}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("embedding distance")
    axes[1].set_title("cluster geometry (val)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_eval_distributions(rows, out_path):
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    iou = [r["matched_iou_mean"] for r in rows
           if isinstance(r.get("matched_iou_mean"), float) and not math.isnan(r["matched_iou_mean"])]
    axes[0, 0].hist(iou, bins=20, color="tab:blue", alpha=0.7)
    if iou:
        axes[0, 0].axvline(sum(iou) / len(iou), color="tab:red", ls="--",
                           label=f"mean={sum(iou) / len(iou):.3f}")
        axes[0, 0].legend()
    axes[0, 0].set_title("matched_iou_mean (Hungarian, pred vs GT masks)")
    axes[0, 0].set_xlabel("IoU"); axes[0, 0].grid(alpha=0.3)

    fr = [r["flux_relative_error"] for r in rows
          if isinstance(r.get("flux_relative_error"), float) and not math.isnan(r["flux_relative_error"])]
    axes[0, 1].hist(fr, bins=20, color="tab:purple", alpha=0.7)
    if fr:
        axes[0, 1].axvline(sum(fr) / len(fr), color="tab:red", ls="--",
                           label=f"mean={sum(fr) / len(fr):.3g}")
        axes[0, 1].legend()
    axes[0, 1].set_title("flux_relative_error  (per-source mask × input vs target)")
    axes[0, 1].set_xlabel("|Σpred-Σtgt| / Σ|tgt|"); axes[0, 1].grid(alpha=0.3)

    # Predicted vs ground-truth instance count.
    n_pred = [r["n_pred"] for r in rows]
    n_gt = [r["n_gt"] for r in rows]
    axes[1, 0].scatter(n_gt, n_pred, alpha=0.4)
    lim = max(max(n_pred + [0]), max(n_gt + [0])) + 1
    axes[1, 0].plot([0, lim], [0, lim], "k--", alpha=0.5)
    axes[1, 0].set_xlabel("n_gt"); axes[1, 0].set_ylabel("n_pred")
    axes[1, 0].set_title("predicted vs ground-truth instance count"); axes[1, 0].grid(alpha=0.3)

    # diff distribution
    diffs = [int(p - g) for p, g in zip(n_pred, n_gt)]
    axes[1, 1].hist(diffs, bins=range(min(diffs) - 1, max(diffs) + 2), color="tab:orange", alpha=0.8)
    axes[1, 1].set_title("n_pred - n_gt"); axes[1, 1].set_xlabel("count difference")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _color_label_volume(labels, mom_axis=0):
    """mom-0 project a per-voxel label volume into a 2D RGB image.

    For each (y, x) take the label of the voxel with the largest counted source
    along the channel axis (most-common non-bg label). Background → gray.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    Y, X = labels.shape[1], labels.shape[2]
    out_lab = -np.ones((Y, X), dtype=np.int64)
    # Per (y, x), pick the most-common non-background label across channels.
    for y in range(Y):
        for x in range(X):
            col = labels[:, y, x]
            col = col[col >= 0]
            if col.size == 0:
                continue
            vals, counts = np.unique(col, return_counts=True)
            out_lab[y, x] = vals[counts.argmax()]
    cmap = plt.get_cmap("tab10")
    img = np.full((Y, X, 3), 0.85, dtype=np.float32)
    uniq = np.unique(out_lab)
    uniq = uniq[uniq >= 0]
    for u in uniq:
        rgb = cmap(int(u) % 10)[:3]
        img[out_lab == u] = rgb
    return img, out_lab


def _embedding_pca_image(emb, fg_mask):
    """Project D-dim embedding to RGB via PCA on the foreground voxels."""
    import numpy as np

    D, C, Y, X = emb.shape
    flat = emb.reshape(D, -1)
    fg_flat = fg_mask.reshape(-1)
    fg_e = flat[:, fg_flat]
    if fg_e.shape[1] < 4:
        return np.full((Y, X, 3), 0.85, dtype=np.float32)
    mu = fg_e.mean(axis=1, keepdims=True)
    centered = fg_e - mu
    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = (U[:, :3].T @ centered)        # (3, n_fg)
    # Normalise to [0, 1] per axis for RGB.
    proj = (proj - proj.min(axis=1, keepdims=True))
    proj = proj / (proj.max(axis=1, keepdims=True) + 1e-9)
    rgb_flat = np.full((3, Y * X * C), 0.85, dtype=np.float32)
    rgb_flat[:, fg_flat] = proj
    rgb = rgb_flat.reshape(3, C, Y, X)
    # Mom-0 over the channel axis (mean of color over channels containing fg).
    fg = fg_mask.astype(np.float32)
    weight = fg.sum(axis=0) + 1e-9
    img = (rgb * fg_mask.astype(np.float32)[None]).sum(axis=1) / weight
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    bg = ~np.any(fg_mask, axis=0)
    img[bg] = 0.85
    return img


def plot_sample(npz_path, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    inp = d["input"]                 # (C, Y, X)
    emb = d["embedding"]              # (D, C, Y, X)
    pred_lab = d["pred_labels"]       # (C, Y, X)
    gt_lab = d["gt_labels"]           # (C, Y, X)
    target = d["target"]              # (M, C, Y, X)
    valid = d["valid"]
    matches = d["matches"] if "matches" in d.files else np.zeros((0, 3), dtype=np.float32)
    n_g = int(valid.sum())

    fg_mask_input = inp >= 0.05 * inp.max()
    pca_rgb = _embedding_pca_image(emb, fg_mask_input)
    gt_rgb, _ = _color_label_volume(gt_lab)
    pred_rgb, _ = _color_label_volume(pred_lab)

    inp_m0 = inp.sum(0)
    # Index matches by GT id so pred column j aligns with target column j.
    # matches rows are (pred_r, gt_c, iou); GT indices not present in matches
    # have no corresponding pred (e.g., pred under-counted). Unmatched preds
    # (extra clusters) get appended after the matched columns.
    matched_pred_set = set(int(m[0]) for m in matches)
    match_by_gt = {int(c): (int(r), float(iou)) for r, c, iou in matches}
    pred_ids = sorted(set(int(v) for v in pred_lab.flatten() if v >= 0))
    extra_preds = [p for p in pred_ids if p not in matched_pred_set]
    n_cols = max(4, n_g + len(extra_preds))

    fig, axes = plt.subplots(4, n_cols, figsize=(2.6 * n_cols, 9.5), constrained_layout=True)

    def _imshow(ax, img, title, cmap="viridis", colorbar=True):
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img, origin="lower")
        else:
            im = ax.imshow(img, origin="lower", cmap=cmap)
            if colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    _imshow(axes[0, 0], inp_m0, "input (mom0)")
    _imshow(axes[0, 1], gt_rgb, "GT labels (most-common per col)")
    _imshow(axes[0, 2], pred_rgb, "pred labels (kept sources)")
    if "diffuse_mask" in d.files:
        diffuse_flux = inp * d["diffuse_mask"]
        _imshow(axes[0, 3], diffuse_flux.sum(0), "predicted diffuse (mom0)", cmap="magma")
    else:
        _imshow(axes[0, 3], pca_rgb, "embedding (PCA→RGB)")
    for j in range(4, n_cols):
        axes[0, j].set_visible(False)

    # row 1: per-GT-source target moment-0 (columns 0..n_g-1 = GT 0..n_g-1)
    for j in range(n_cols):
        if j < n_g:
            _imshow(axes[1, j], target[j].sum(0), f"target source {j}")
        else:
            axes[1, j].set_visible(False)

    def _pred_panel(ax_pred, ax_outline, j_col, pred_id, title_pred, title_outline):
        mask3d = (pred_lab == pred_id)
        pred_flux = inp * mask3d
        _imshow(ax_pred, pred_flux.sum(0), title_pred)
        mask2d = mask3d.any(axis=0).astype(np.float32)
        base = (inp_m0 - inp_m0.min()) / (inp_m0.max() - inp_m0.min() + 1e-9)
        rgb = np.stack([base, base, base], axis=-1)
        cmap = plt.get_cmap("tab10")
        color = np.array(cmap(int(pred_id) % 10)[:3])
        rgb[mask2d > 0] = 0.5 * rgb[mask2d > 0] + 0.5 * color
        _imshow(ax_outline, rgb, title_outline, colorbar=False)

    # rows 2 + 3: pred panels aligned to GT columns. Column j shows the pred
    # matched to GT j (from Hungarian); empty if no pred matched to GT j.
    # Then any extra unmatched preds are appended after column n_g.
    for j in range(n_cols):
        if j < n_g:
            if j in match_by_gt:
                pred_id, iou = match_by_gt[j]
                _pred_panel(axes[2, j], axes[3, j], j, pred_id,
                            f"pred {pred_id} → GT{j}  IoU={iou:.2f}",
                            f"pred {pred_id} extent")
            else:
                axes[2, j].set_visible(False); axes[3, j].set_visible(False)
        elif (j - n_g) < len(extra_preds):
            pred_id = extra_preds[j - n_g]
            _pred_panel(axes[2, j], axes[3, j], j, pred_id,
                        f"extra pred {pred_id}  (no GT match)",
                        f"pred {pred_id} extent")
        else:
            axes[2, j].set_visible(False); axes[3, j].set_visible(False)

    fig.suptitle(npz_path.stem)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def write_report(run_dir, metrics, eval_rows, summary, out_path, sample_pngs):
    lines = [f"# Segmentation analysis — `{run_dir.name}`\n"]
    if metrics:
        last = metrics[-1]
        lines.append("## Training")
        lines.append(f"- Epochs trained: **{last['epoch']}**")
        lines.append(f"- Final train loss: **{last['train_loss']:.4e}**")
        lines.append(f"- Final val loss: **{last['val_loss']:.4e}**")
        best_val = min(m["val_loss"] for m in metrics)
        best_ep = next(m["epoch"] for m in metrics if m["val_loss"] == best_val)
        lines.append(f"- Best val loss: **{best_val:.4e}** at epoch {best_ep}")
        lines.append(f"- Final intra-cluster spread: **{last.get('intra_cluster_spread', float('nan')):.3f}**")
        lines.append(f"- Final inter-cluster min sep: **{last.get('inter_cluster_min_sep', float('nan')):.3f}**\n")
        lines.append("![training curves](loss_curves.png)\n")
    if summary:
        lines.append("## Validation evaluation")
        lines.append(f"- Cubes evaluated: {summary.get('n_cubes', 0)}")
        lines.append(f"- Seeding mode: **{summary.get('seeds', 'unknown')}**")
        for key in ["matched_iou_mean", "per_source_mse", "flux_relative_error", "n_pred", "n_gt"]:
            s = summary.get(key)
            if s:
                lines.append(f"- `{key}`: mean={s['mean']:.4g}  median={s['median']:.4g}  "
                             f"std={s['std']:.4g}  min={s['min']:.4g}  max={s['max']:.4g}")
        lines.append("\n![eval distributions](eval_distributions.png)\n")
    if sample_pngs:
        lines.append("## Qualitative samples")
        for p in sample_pngs:
            lines.append(f"![{p.stem}]({p.name})\n")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None,
                    help="Analysis output dir (default: <run>/analysis or <eval-dir>/../../analysis/<eval-dir name>)")
    ap.add_argument("--eval-dir", default=None,
                    help="Eval directory to analyse. Default: <run>/eval. Use this to analyse "
                         "per-epoch outputs at <run>/eval/epoch_N/.")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    eval_dir = Path(args.eval_dir).resolve() if args.eval_dir else run_dir / "eval"
    out_dir = Path(args.out).resolve() if args.out else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    cfg = json.loads((run_dir / "config.json").read_text())
    delta_v = float(cfg.get("delta_v", 0.5))
    delta_d = float(cfg.get("delta_d", 1.5))

    log.info("Reading metrics from %s", run_dir / "metrics.jsonl")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    log.info("  %d epochs", len(metrics))

    log.info("Reading eval from %s", eval_dir)
    eval_rows = _read_csv(eval_dir / "per_cube_metrics.csv")
    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    log.info("  %d eval rows", len(eval_rows))

    if metrics:
        plot_training_curves(metrics, delta_v, delta_d, out_dir / "loss_curves.png")
        log.info("  wrote loss_curves.png")
    if eval_rows:
        plot_eval_distributions(eval_rows, out_dir / "eval_distributions.png")
        log.info("  wrote eval_distributions.png")

    sample_pngs: list = []
    samples_dir = eval_dir / "samples"
    if samples_dir.exists():
        for npz_path in sorted(samples_dir.glob("*.npz")):
            png_path = out_dir / f"sample_{npz_path.stem}.png"
            try:
                plot_sample(npz_path, png_path)
                sample_pngs.append(png_path)
            except Exception as e:
                log.warning("  sample plot failed for %s: %s", npz_path.name, e)
        log.info("  wrote %d sample montages", len(sample_pngs))

    report_path = out_dir / "report.md"
    write_report(run_dir, metrics, eval_rows, summary, report_path, sample_pngs)
    log.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
