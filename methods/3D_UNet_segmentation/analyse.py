"""Analyse a completed train + eval run.

Reads:
    <run>/metrics.jsonl                  -- per-epoch training curves
    <run>/eval/per_cube_metrics.csv      -- per-cube validation metrics
    <run>/eval/summary.json              -- aggregated stats
    <run>/eval/samples/*.npz             -- qualitative example arrays

Writes:
    <run>/analysis/loss_curves.png
    <run>/analysis/flux_conservation.png
    <run>/analysis/eval_distributions.png
    <run>/analysis/sample_*.png          -- moment-0 montages
    <run>/analysis/report.md             -- markdown summary
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path


def _setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("analyse")
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


def _read_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k, v in r.items():
            if k in ("path",):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def plot_training_curves(metrics, out_path):
    import matplotlib.pyplot as plt
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [m["train_loss"] for m in metrics], "-o", label="train", color="tab:blue", ms=4)
    axes[0].plot(epochs, [m["val_loss"] for m in metrics], "-o", label="val", color="tab:orange", ms=4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_yscale("log")
    axes[0].legend(); axes[0].set_title("training / validation loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [m["val_flux_relative_error"] for m in metrics], "-o", color="tab:red", ms=4)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("flux_relative_error")
    axes[1].set_yscale("log"); axes[1].set_title("flux conservation (val)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_eval_distributions(rows, out_path):
    import matplotlib.pyplot as plt
    if not rows:
        return
    cols = ["per_slot_mse", "flux_relative_error",
            "residual_fraction_of_input", "mean_peak_distance_px"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key in zip(axes.flat, cols):
        vals = [r[key] for r in rows
                if r.get(key) is not None and isinstance(r[key], float) and not math.isnan(r[key])]
        if not vals:
            ax.set_visible(False); continue
        ax.hist(vals, bins=20, color="tab:blue", alpha=0.7)
        ax.axvline(sum(vals) / len(vals), color="tab:red", ls="--", label=f"mean={sum(vals)/len(vals):.3g}")
        ax.set_title(key); ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_sample(npz_path, out_path):
    import numpy as np
    import matplotlib.pyplot as plt
    d = np.load(npz_path)
    inp, pred, target, valid = d["input"], d["pred"], d["target"], d["valid"]
    n_g = int(valid.sum())
    inp_m0 = inp.sum(0)
    pred_total_m0 = pred.sum((0, 1))
    target_total_m0 = target.sum((0, 1))
    residual_m0 = inp_m0 - pred_total_m0

    n_cols = max(4, n_g)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 7.4),
                             constrained_layout=True)

    def _imshow(ax, img, title, cmap="viridis"):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _imshow(axes[0, 0], inp_m0, "input (mom0)")
    _imshow(axes[0, 1], target_total_m0, "Σ target slots")
    _imshow(axes[0, 2], pred_total_m0, "Σ pred slots")
    _imshow(axes[0, 3], residual_m0, "input − Σ pred (≈ diffuse)", cmap="magma")
    for j in range(4, n_cols):
        axes[0, j].set_visible(False)

    for j in range(n_cols):
        if j < n_g:
            _imshow(axes[1, j], target[j].sum(0), f"target slot {j}")
            _imshow(axes[2, j], pred[j].sum(0), f"pred slot {j}")
        else:
            axes[1, j].set_visible(False); axes[2, j].set_visible(False)

    fig.suptitle(npz_path.stem)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def write_report(run_dir, metrics, eval_rows, summary, out_path, sample_pngs):
    lines = []
    lines.append(f"# Analysis report — `{run_dir.name}`\n")
    if metrics:
        last = metrics[-1]
        lines.append("## Training")
        lines.append(f"- Epochs trained: **{last['epoch']}**")
        lines.append(f"- Final train loss: **{last['train_loss']:.4e}**")
        lines.append(f"- Final val loss: **{last['val_loss']:.4e}**")
        best_val = min(m["val_loss"] for m in metrics)
        best_epoch = next(m["epoch"] for m in metrics if m["val_loss"] == best_val)
        lines.append(f"- Best val loss: **{best_val:.4e}** at epoch {best_epoch}")
        lines.append(f"- Final flux relative error: **{last['val_flux_relative_error']:.3e}**\n")
        lines.append("![training curves](loss_curves.png)\n")
    if summary:
        lines.append("## Validation evaluation")
        lines.append(f"- Cubes evaluated: {summary.get('n_cubes', 0)}")
        for key in ["per_slot_mse", "flux_relative_error",
                    "residual_fraction_of_input", "mean_peak_distance_px"]:
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
    ap.add_argument("--run", required=True, help="Train run directory")
    ap.add_argument("--out", default=None, help="Analysis output dir (default: <run>/analysis)")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir)

    log.info("Reading training metrics from %s", run_dir / "metrics.jsonl")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    log.info("  %d epochs", len(metrics))

    eval_dir = run_dir / "eval"
    log.info("Reading evaluation results from %s", eval_dir)
    eval_rows = _read_csv(eval_dir / "per_cube_metrics.csv")
    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    log.info("  %d eval rows", len(eval_rows))

    if metrics:
        plot_training_curves(metrics, out_dir / "loss_curves.png")
        log.info("  wrote loss_curves.png and flux conservation panel")
    if eval_rows:
        plot_eval_distributions(eval_rows, out_dir / "eval_distributions.png")
        log.info("  wrote eval_distributions.png")

    sample_pngs = []
    samples_dir = eval_dir / "samples"
    if samples_dir.exists():
        for npz_path in sorted(samples_dir.glob("*.npz")):
            png_path = out_dir / f"sample_{npz_path.stem}.png"
            plot_sample(npz_path, png_path)
            sample_pngs.append(png_path)
        log.info("  wrote %d sample montages", len(sample_pngs))

    report_path = out_dir / "report.md"
    write_report(run_dir, metrics, eval_rows, summary, report_path, sample_pngs)
    log.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
