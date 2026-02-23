"""Evaluate a trained PINN checkpoint against the FD reference.

Usage:
    python scripts/evaluate_checkpoint.py \
        --config configs/zerilli_l2_fd_refined.yaml \
        --checkpoint outputs/pinn/zerilli_l2_fd_refined/checkpoints/window_5/model-final-5000.pt
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import build_model, eval_on_grid
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/pinn/<name>/eval)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]

    # --- FD baseline ---
    print("[Eval] Computing FD reference...")
    fd = solve_fd(cfg)
    x, t, phi_fd = fd["x"], fd["t"], fd["phi"]
    print(f"[Eval] FD grid: x={len(x)} pts, t={len(t)} pts, phi shape={phi_fd.shape}")

    # --- Rebuild model and load checkpoint ---
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model, data = build_model(cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.net.load_state_dict(ckpt["model_state_dict"])
    else:
        # Might be a raw state dict
        model.net.load_state_dict(ckpt)

    # Compile the model (DeepXDE requires this before predict)
    model.compile("adam", lr=1e-3)

    # --- Evaluate ---
    print("[Eval] Evaluating PINN on FD grid...")
    phi_pinn = eval_on_grid(model, x=x, t=t, dtype=cfg["pinn"]["dtype"])

    metrics = {
        "RMSD": float(rmsd(phi_fd, phi_pinn)),
        "MAD": float(mad(phi_fd, phi_pinn)),
        "RL2": float(rl2(phi_fd, phi_pinn)),
    }

    print(f"[Eval] RMSD: {metrics['RMSD']:.6e}")
    print(f"[Eval] MAD:  {metrics['MAD']:.6e}")
    print(f"[Eval] RL2:  {metrics['RL2']:.6e}")

    # --- Save outputs ---
    outdir = args.outdir or os.path.join("outputs", "pinn", name, "eval")
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"), x=x, t=t, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # --- Plots ---
    times = [10.0, 20.0, 30.0, 40.0]
    pot_title = cfg["physics"]["potential"].title()
    ell = cfg["physics"]["l"]

    plot_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots — {pot_title} potential (l={ell})"
    )
    plot_abs_diff_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference — {pot_title} potential (l={ell})"
    )
    plot_snapshots_zoomed(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) — {pot_title} potential (l={ell})",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error — {pot_title} (l={ell})",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) — {pot_title} (l={ell})",
        xlim=(-20.0, 60.0),
    )

    # Ringdown overlay at xq
    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x - xq)))
    plot_ringdown_overlay(
        t, phi_fd[:, ix], phi_pinn[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown — {pot_title} (l={ell})",
        xq=xq,
    )

    print(f"[Eval] All outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
