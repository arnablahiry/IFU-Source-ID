"""End-to-end pipeline: generate → train → evaluate → analyse.

Each stage writes to a subdirectory of the experiment root:

    <root>/data/        -- generated HDF5 cubes + generate_cubes.log
    <root>/train/       -- train.log, metrics.jsonl, best.pt, last.pt, config.json
    <root>/eval/        -- per_cube_metrics.csv, summary.json, samples/, eval.log
    <root>/analysis/    -- loss_curves.png, eval_distributions.png, sample_*.png, report.md

Each stage is skipped if its sentinel output already exists, unless
`--force` is given. Pass `--skip-generate` / `--skip-train` /
`--skip-eval` to bypass individual stages explicitly.

Example:

    python scripts/pipeline.py --root experiments/sep_v1 \\
        --n-cubes 2000 --epochs 30 --batch-size 2 --base 16

Stage commands are also written to `<root>/pipeline.log` so the run is
fully reproducible.
"""
from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


def _setup_logging(root: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(root / "pipeline.log", mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _run(cmd, log: logging.Logger):
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log.error("Command failed (exit %d) after %.1fs", proc.returncode, time.time() - t0)
        sys.exit(proc.returncode)
    log.info("Completed in %.1fs", time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", required=True, help="Experiment root directory")
    # Generation
    ap.add_argument("--n-cubes", type=int, default=2000)
    ap.add_argument("--min-gals", type=int, default=2)
    ap.add_argument("--max-gals", type=int, default=5)
    ap.add_argument("--grid-size", type=int, default=96)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--resolution", default="resolved", choices=["all", "resolved", "unresolved"])
    ap.add_argument("--gen-seed", type=int, default=0)
    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1,
                    help="Held-out test fraction (used by the eval stage).")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=20)
    # Eval
    ap.add_argument("--eval-samples", type=int, default=8)
    # Stage controls
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-analyse", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if their outputs already exist")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    train_dir = root / "train"
    eval_dir = train_dir / "eval"
    analysis_dir = train_dir / "analysis"
    log = _setup_logging(root)
    log.info("=" * 72)
    log.info("Pipeline root: %s", root)
    log.info("Args: %s", vars(args))
    pipeline_t0 = time.time()

    # 1. Generate
    gen_done = (data_dir / "config.json").exists() and any(data_dir.glob("cube_*.h5"))
    if args.skip_generate:
        log.info("[gen] skipped via flag.")
    elif gen_done and not args.force:
        log.info("[gen] outputs already present at %s — skipping", data_dir)
    else:
        log.info("[gen] generating %d cubes", args.n_cubes)
        _run([
            sys.executable, str(SCRIPTS_DIR / "generate_cubes.py"),
            "--out", str(data_dir),
            "--n", str(args.n_cubes),
            "--min-gals", str(args.min_gals),
            "--max-gals", str(args.max_gals),
            "--grid-size", str(args.grid_size),
            "--channels", str(args.channels),
            "--resolution", args.resolution,
            "--seed", str(args.gen_seed),
        ], log)

    # 2. Train
    train_done = (train_dir / "best.pt").exists() and (train_dir / "metrics.jsonl").exists()
    if args.skip_train:
        log.info("[train] skipped via flag.")
    elif train_done and not args.force:
        log.info("[train] checkpoints already present at %s — skipping", train_dir)
    else:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_separation.py"),
            "--data", str(data_dir),
            "--out", str(train_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--base", str(args.base),
            "--lr", str(args.lr),
            "--num-workers", str(args.num_workers),
            "--val-fraction", str(args.val_fraction),
            "--test-fraction", str(args.test_fraction),
            "--seed", str(args.train_seed),
            "--log-every", str(args.log_every),
        ]
        if args.device:
            cmd += ["--device", args.device]
        log.info("[train] training for %d epochs", args.epochs)
        _run(cmd, log)

    # 3. Evaluate
    eval_done = (eval_dir / "summary.json").exists() and (eval_dir / "per_cube_metrics.csv").exists()
    if args.skip_eval:
        log.info("[eval] skipped via flag.")
    elif eval_done and not args.force:
        log.info("[eval] outputs already present at %s — skipping", eval_dir)
    else:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "evaluate.py"),
            "--data", str(data_dir),
            "--run", str(train_dir),
            "--out", str(eval_dir),
            "--n-samples", str(args.eval_samples),
        ]
        if args.device:
            cmd += ["--device", args.device]
        log.info("[eval] running inference on validation split")
        _run(cmd, log)

    # 4. Analyse
    analysis_done = (analysis_dir / "report.md").exists()
    if args.skip_analyse:
        log.info("[analyse] skipped via flag.")
    elif analysis_done and not args.force:
        log.info("[analyse] outputs already present at %s — skipping", analysis_dir)
    else:
        log.info("[analyse] generating plots and report")
        _run([
            sys.executable, str(SCRIPTS_DIR / "analyse.py"),
            "--run", str(train_dir),
            "--out", str(analysis_dir),
        ], log)

    log.info("Pipeline complete in %.1fs", time.time() - pipeline_t0)
    log.info("Outputs:")
    log.info("  data     : %s", data_dir)
    log.info("  train    : %s", train_dir)
    log.info("  eval     : %s", eval_dir)
    log.info("  analysis : %s", analysis_dir)


if __name__ == "__main__":
    main()
